"""
call_handler.py

Manages real-time voice call sessions between the user and the AI agent.

Architecture:
  Browser mic (PCM16, 16kHz) → Socket.IO → CallSession → Voxtral/Mistral STT
  → Mistral Medium (Chat) → Voxtral/Mistral TTS → Socket.IO → Browser speaker

Key behaviors:
  - STT runs when an utterance is collected
  - AI waits for a silence gap (~1.5s) before responding
  - User can interrupt AI mid-speech (stops playback, resumes listening)
  - AI never interrupts the user
"""

import asyncio
import base64
import json
import os
import math
import struct
import traceback
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from mistralai.client import Mistral
from sqlmodel import Session, select

from database import engine
from models import Case, CaseMessage

load_dotenv()

# --- Constants ---

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# STT model for transcription (Voxtral Mini Speech-to-Text)
STT_MODEL = "voxtral-mini-latest"

# TTS model for generating AI speech (Voxtral Mini Text-to-Speech)
TTS_MODEL = "voxtral-mini-tts-2603"

# Chat model for generating AI text responses
CHAT_MODEL = "mistral-large-latest"

# Preset voice ID for TTS output
TTS_VOICE_ID = "gb_jane_neutral"

# System prompt for the voice agent — tailored for spoken conversation
VOICE_SYSTEM_PROMPT = """You are Lexis, a senior legal AI assistant having a live voice conversation with a lawyer about their case.

RULES FOR VOICE RESPONSES:
- Keep responses SHORT and conversational (2-4 sentences max)
- Speak naturally as if in a phone call — no markdown, no bullet points, no headers
- Use simple language a person can follow by ear
- If asked a complex question, give a brief answer first, then ask if they want more detail
- Reference the case documents and context you've been given
- Be confident but honest — say "I'd need to check that" if unsure
- Use transitional phrases like "So basically...", "The key issue here is...", "What I'm seeing is..."

TRIGGERING BACKGROUND RESEARCH:
- If the user explicitly asks or commands you to "run research", "do some research", "start background research", "start another research", or similar, you MUST trigger the background research job.
- To do this, simply append the exact token '[TRIGGER_RESEARCH]' at the very end of your response text (e.g., 'I will start that background research right now. [TRIGGER_RESEARCH]').
- Explain conversationally that you are starting the background research crew and that you'll let them know the moment you find anything. Keep it natural!
"""


class CallSession:
    """
    Each call session:
    1. Loads the case context from the DB
    2. Collects user audio chunks from Socket.IO
    3. Detects end of speech, sends audio to Mistral for transcription (STT)
    4. Generates an AI response from Mistral Chat
    5. Converts AI response to speech via Mistral TTS
    6. Sends base64 audio back to the client via Socket.IO
    """

    def __init__(self, sio, sid: str, case_id: int):
        # Socket.IO server instance and client session ID
        self.sio = sio
        self.sid = sid
        self.case_id = case_id

        # Mistral client
        self.client = Mistral(api_key=MISTRAL_API_KEY)

        # Case context loaded from the DB
        self.case_context = ""

        # Conversation history for the voice call
        # Each item: {"role": "user" | "assistant", "content": "..."}
        self.conversation_history = []

        # Async queue for incoming audio chunks from the browser
        self.audio_queue = asyncio.Queue()

        # Tracks whether the AI audio is currently playing on the client
        self.is_ai_speaking = False

        # Flag to signal that the user interrupted the AI
        self.interrupted = False

        # Flag to signal that the session should stop
        self.is_active = False

        # Accumulates partial transcription text for the current utterance
        self.current_transcript = ""

        # Background task for processing the audio stream
        self._stream_task = None

        # Rolling history of RMS values for dynamic noise thresholding
        self.rms_history = []
        self.speech_threshold = 400.0  # Initial default threshold

        # Echo-guard: timestamp when AI audio playback started on the client.
        # We ignore interruption attempts for a grace period after this to
        # prevent the AI's own speaker output from triggering a false cut-off.
        self._speaking_start_time: Optional[float] = None
        self._interruption_streak = 0  # consecutive high-RMS chunks counter

    async def start(self):
        """Load case context and start the audio processing loop."""
        # Load the case context from the database
        self.case_context = await asyncio.get_event_loop().run_in_executor(
            None, self._load_case_context
        )

        self.is_active = True
 
        # Start the background audio processing task
        self._stream_task = asyncio.create_task(self._process_audio_stream())

        print(f"[call] Session started for case {self.case_id}, sid={self.sid}")

    def _load_case_context(self) -> str:
        """
        Loads the case context from the database (runs in thread executor).
        Combines the case description with any vault document summaries.
        """
        with Session(engine) as session:
            case = session.get(Case, self.case_id)
            if not case:
                return "No case context available."

            context_parts = []

            # Add the case description
            if case.context:
                context_parts.append(f"Case Description:\n{case.context}")

            # Add PDF file names for reference
            pdfs_raw = (getattr(case, 'pdf_paths_json', None) or '').strip() or '[]'
            pdf_paths = json.loads(pdfs_raw)
            if pdf_paths:
                pdf_names = [p.split("/")[-1] for p in pdf_paths]
                context_parts.append(f"Uploaded Documents: {', '.join(pdf_names)}")

            # Add URLs for reference
            urls_raw = (getattr(case, 'urls_json', None) or '').strip() or '[]'
            urls = json.loads(urls_raw)
            if urls:
                context_parts.append(f"Reference URLs: {', '.join(urls)}")

            return "\n\n".join(context_parts) if context_parts else "No case context available."

    async def handle_audio_chunk(self, chunk: bytes):
        """
        Called when the client sends a binary audio chunk.
        Puts the chunk into the async queue for processing.
        """
        if not self.is_active:
            return

        # Calculate RMS energy of the incoming chunk
        rms = self._calculate_rms(chunk)

        # If AI is speaking, check for genuine user interruption
        if self.is_ai_speaking:
            # Grace period: ignore the first 1.5s after AI audio starts
            # to let echo cancellation settle and avoid self-interruption
            ECHO_GRACE_SECONDS = 1.5
            elapsed = (asyncio.get_event_loop().time() - self._speaking_start_time
                       if self._speaking_start_time else 0.0)

            if elapsed > ECHO_GRACE_SECONDS and rms >= self.speech_threshold:
                self._interruption_streak += 1
                # Require 3 consecutive loud chunks (~300ms) to confirm
                # this is real user speech, not a transient echo spike
                if self._interruption_streak >= 3:
                    print(f"[call] User interrupted AI (RMS={rms:.1f}, streak={self._interruption_streak})")
                    self._interruption_streak = 0
                    await self.handle_interruption()
            else:
                self._interruption_streak = 0

            # Do not queue chunks while the AI is speaking to prevent echo from triggering false VAD transcriptions
            return

        await self.audio_queue.put(chunk)

    def _calculate_rms(self, chunk: bytes) -> float:
        """Calculates root-mean-square (RMS) energy of signed 16-bit PCM samples."""
        if len(chunk) < 2:
            return 0.0
        num_samples = len(chunk) // 2
        # Unpack as signed 16-bit integers ('h')
        try:
            samples = struct.unpack(f"{num_samples}h", chunk)
            sum_squares = sum(s * s for s in samples)
            return math.sqrt(sum_squares / num_samples)
        except Exception:
            return 0.0

    async def handle_interruption(self):
        """Signals to the client that the user interrupted the AI speaking."""
        self.interrupted = True
        self.is_ai_speaking = False

        # Drain the audio queue to discard any buffered user speaking during interruption
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        await self.sio.emit("call_interrupted", {
            "case_id": self.case_id,
        }, to=self.sid)

        print(f"[call] User interrupted AI for case {self.case_id}")

    async def _process_audio_stream(self):
        """
        Background task: continuously reads audio chunks from the queue,
        detects end of speech, transcribes, and triggers AI responses.
        """
        try:
            while self.is_active:
                # Collect audio chunks for a short window, then transcribe
                audio_data = await self._collect_audio_window()

                if not audio_data or not self.is_active:
                    continue

                # Transcribe the collected audio using non-streaming STT
                transcript = await self._transcribe_audio(audio_data)

                if not transcript or not transcript.strip():
                    continue

                self.current_transcript = transcript

                # Send the transcript to the client for display
                await self.sio.emit("call_transcript", {
                    "case_id": self.case_id,
                    "text": transcript,
                    "is_final": True,
                }, to=self.sid)

                print(f"[call] Transcription for case {self.case_id}: {transcript}")

                # Generate conversational response
                await self._generate_and_speak(transcript)

        except asyncio.CancelledError:
            pass
        except Exception as error:
            print(f"[call] Voice processing loop crashed: {error}")
            traceback.print_exc()
            await self.sio.emit("call_error", {
                "case_id": self.case_id,
                "message": f"Voice processing error: {str(error)}",
            }, to=self.sid)

    def _update_threshold(self, rms: float):
        """
        Dynamically updates the ambient noise floor and speech threshold.
        Only adds to history if the RMS is below an absolute max threshold,
        preventing speech from inflating the ambient noise estimate.
        """
        # If the chunk is loud (likely speech), do not add it to ambient noise floor
        if rms >= 300.0:
            return

        self.rms_history.append(rms)
        # Keep only the last 30 chunks (~3 seconds of ambient audio)
        if len(self.rms_history) > 30:
            self.rms_history.pop(0)

        # Estimate background noise floor by taking the average of the lowest 5 RMS values
        sorted_rms = sorted(self.rms_history)
        lowest_vals = sorted_rms[:min(5, len(sorted_rms))]
        
        if lowest_vals:
            noise_floor = sum(lowest_vals) / len(lowest_vals)
        else:
            noise_floor = 100.0

        # Cap the noise floor at a safe maximum to prevent user speech from locking them out
        noise_floor = min(200.0, noise_floor)

        # Dynamically calculate the speech threshold:
        # Cap at 400.0 so quiet/normal speech is always detected and never locked out
        self.speech_threshold = min(400.0, max(250.0, noise_floor + 150.0))

    async def _collect_audio_window(self) -> Optional[bytes]:
        """
        Collects audio chunks from the queue until there's a silence gap.
        Uses dynamic RMS-based Voice Activity Detection (VAD) to find when user starts and stops speaking.
        """
        chunks = []
        silence_timeout = 1.5  # seconds of silence before considering speech done
        chunk_timeout = 0.1    # how long to wait for each individual chunk

        # Step 1: Wait until the user starts speaking (RMS above threshold)
        print("[call] Waiting for active speech (user to start speaking)...")
        quiet_chunk_count = 0
        while self.is_active:
            try:
                chunk = await self.audio_queue.get()
                rms = self._calculate_rms(chunk)
                
                # Update dynamic threshold using only ambient noise chunks (before speech starts)
                self._update_threshold(rms)
                
                # Check if this chunk is above our dynamically adjusted speech threshold
                if rms >= self.speech_threshold:
                    chunks.append(chunk)
                    print(f"[call] Speech detected (RMS={rms:.1f} >= threshold={self.speech_threshold:.1f})! Starting collection...")
                    break
                else:
                    quiet_chunk_count += 1
                    if quiet_chunk_count % 30 == 0:  # Print every ~3 seconds (30 * 100ms chunks)
                        print(f"[call] Listening... (current ambient RMS={rms:.1f}, threshold={self.speech_threshold:.1f})")
            except asyncio.CancelledError:
                return None
            except Exception:
                return None

        # Step 2: Keep collecting chunks until silence gap (using the FROZEN speech threshold)
        silence_elapsed = 0.0
        frozen_threshold = self.speech_threshold
        max_duration = 15.0  # safety cap: max 15 seconds of audio per utterance
        duration_elapsed = 0.0

        while silence_elapsed < silence_timeout and duration_elapsed < max_duration and self.is_active:
            try:
                chunk = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=chunk_timeout
                )
                chunks.append(chunk)
                rms = self._calculate_rms(chunk)
                
                # Reset silence timer on new audio energy above the frozen threshold
                if rms >= frozen_threshold:
                    silence_elapsed = 0.0
                else:
                    silence_elapsed += chunk_timeout
                
                duration_elapsed += chunk_timeout
            except asyncio.TimeoutError:
                silence_elapsed += chunk_timeout
                duration_elapsed += chunk_timeout
            except asyncio.CancelledError:
                return None

        if not chunks:
            return None

        # Combine all chunks into a single bytes object
        combined = b"".join(chunks)
        print(f"[call] Finished collecting utterance. Accumulated {len(chunks)} chunks, total size={len(combined)} bytes.")
        return combined

    def _add_wav_header(self, pcm_data: bytes) -> bytes:
        """Prepends a 44-byte WAV header to raw PCM16 (16kHz mono) data."""
        import struct
        sample_rate = 16000
        bits_per_sample = 16
        num_channels = 1
        byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
        block_align = num_channels * (bits_per_sample // 8)
        data_size = len(pcm_data)

        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,
            b'WAVE',
            b'fmt ',
            16,
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_size
        )
        return header + pcm_data

    async def _transcribe_audio(self, audio_data: bytes) -> str:
        """
        Sends audio data to Mistral's STT API for transcription.
        """
        try:
            # Prepend WAV header since Mistral requires a decodable audio format
            wav_bytes = self._add_wav_header(audio_data)

            # Call Mistral's STT API
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.audio.transcriptions.complete(
                    model=STT_MODEL,
                    file={
                        "file_name": "audio.wav",
                        "content": wav_bytes,
                    },
                )
            )

            return result.text if result and result.text else ""

        except Exception as error:
            print(f"[call] STT error for case {self.case_id}: {error}")
            traceback.print_exc()
            return ""

    async def _generate_and_speak(self, user_text: str):
        """
        Takes the user's transcribed text, generates an AI response,
        converts it to speech, and sends both back to the client.
        """
        # Reset interruption flag
        self.interrupted = False

        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_text,
        })

        try:
            # Generate AI text response using Mistral chat
            ai_text = await self._chat_completion(user_text)

            # Check if user interrupted during chat completion
            if self.interrupted:
                print(f"[call] AI response cancelled (interrupted) for case {self.case_id}")
                return

            # Check if a background research task needs to be triggered
            if "[TRIGGER_RESEARCH]" in ai_text:
                print(f"[call] Research trigger requested for case {self.case_id}")
                ai_text = ai_text.replace("[TRIGGER_RESEARCH]", "").strip()
                from ai.background import enqueue_research
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, enqueue_research, self.case_id)

            # Add AI response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_text,
            })

            # Convert response text to speech (TTS)
            audio_base64 = await self._generate_speech(ai_text)

            # Check if user interrupted during speech generation
            if self.interrupted:
                print(f"[call] AI audio playback cancelled (interrupted) for case {self.case_id}")
                self.is_ai_speaking = False
                return

            # Send AI text response to the client
            await self.sio.emit("call_ai_text", {
                "case_id": self.case_id,
                "text": ai_text,
            }, to=self.sid)

            # Persist both user and AI messages to the DB so they appear
            # in the Intelligence Stream chat panel after the call ends
            user_msg_id = await self._save_voice_message_to_chat("user", user_text)
            ai_msg_id = await self._save_voice_message_to_chat("ai", ai_text)

            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

            # Emit the user's voice transcript to the chat panel
            await self.sio.emit("ai_response", {
                "case_id": self.case_id,
                "message": {
                    "id": user_msg_id,
                    "role": "user",
                    "content": f"🎙️ {user_text}",
                    "timestamp": timestamp,
                    "citation": None,
                    "source": "voice",
                },
            }, to=self.sid)

            # Emit the AI response to the chat panel
            await self.sio.emit("ai_response", {
                "case_id": self.case_id,
                "message": {
                    "id": ai_msg_id,
                    "role": "ai",
                    "content": ai_text,
                    "timestamp": timestamp,
                    "citation": None,
                    "source": "voice",
                },
            }, to=self.sid)

            # Send AI audio separately (only if TTS succeeded)
            if audio_base64:
                # Drain the audio queue to discard any chunks accumulated during the LLM/TTS generation phase
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                # Mark AI as speaking so we block interruptions or trigger VAD cut-offs
                self.is_ai_speaking = True
                self._speaking_start_time = asyncio.get_event_loop().time()
                self._interruption_streak = 0
                await self.sio.emit("call_ai_audio", {
                    "case_id": self.case_id,
                    "audio_data": audio_base64,
                }, to=self.sid)
            else:
                self.is_ai_speaking = False
                print(f"[call] TTS returned no audio for case {self.case_id}, skipping playback")

            print(f"[call] AI response sent for case {self.case_id}: {ai_text[:80]}...")

        except Exception as error:
            print(f"[call] Response generation error for case {self.case_id}: {error}")
            traceback.print_exc()
            await self.sio.emit("call_error", {
                "case_id": self.case_id,
                "message": "Failed to generate AI response.",
            }, to=self.sid)
            self.is_ai_speaking = False

    async def _chat_completion(self, user_text: str) -> str:
        """
        Sends the conversation history + case context to Mistral chat
        and returns the AI's text response.
        """
        # Load the latest case context and background task status dynamically
        latest_context = await asyncio.get_event_loop().run_in_executor(
            None, self._load_case_context
        )

        # Build the messages array for the chat API
        messages = [
            {
                "role": "system",
                "content": f"{VOICE_SYSTEM_PROMPT}\n\n--- CASE CONTEXT & RESEARCH STATUS ---\n{latest_context}",
            }
        ]

        # Add conversation history (keep last 20 messages to avoid token limits)
        recent_history = self.conversation_history[-20:]
        for msg in recent_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Run the chat completion in a thread executor (blocking API call)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.chat.complete(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=300,  # Keep responses short for voice
            )
        )

        # Extract the response text
        if response and response.choices:
            return response.choices[0].message.content
        return "I'm having trouble getting an answer right now."

    async def _generate_speech(self, text: str) -> Optional[str]:
        """
        Calls Mistral's TTS API to convert text response into base64-encoded audio (mp3 format).
        Returns base64-encoded mp3 audio, or None on failure.
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.audio.speech.complete(
                    model=TTS_MODEL,
                    input=text,
                    voice_id=TTS_VOICE_ID,
                    response_format="mp3",
                )
            )

            if response and response.audio_data:
                return response.audio_data

            return None

        except Exception as error:
            print(f"[call] TTS error for case {self.case_id}: {error}")
            traceback.print_exc()
            return None

    async def _save_voice_message_to_chat(self, role: str, content: str) -> str:
        """
        Persists a voice call message to the CaseMessage table so it
        shows up in the Intelligence Stream chat panel.
        Returns the message ID as a string.
        """
        def _save():
            msg = CaseMessage(
                case_id=self.case_id,
                role=role,
                content=content,
            )
            with Session(engine) as db:
                db.add(msg)
                db.commit()
                db.refresh(msg)
                return str(msg.id)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _save)

    def mark_ai_done_speaking(self):
        """Called when the client confirms AI audio finished playing."""
        self.is_ai_speaking = False

    async def stop(self):
        """Clean up the session and cancel background tasks."""
        self.is_active = False

        # Cancel the audio processing task
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        # Drain the audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        print(f"[call] Session stopped for case {self.case_id}, sid={self.sid}")


# --- Active Sessions Registry ---
# Maps socket sid → CallSession instance
active_call_sessions: dict[str, CallSession] = {}
