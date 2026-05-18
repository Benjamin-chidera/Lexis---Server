"""
chat_handler.py

Handles incoming chat messages from the Socket.IO client.

chat_graph.invoke() is synchronous (LangChain/Ollama use blocking HTTP calls).
We run it in a thread executor to avoid blocking the async event loop.

ProgressEmitter lets synchronous graph nodes emit WebSocket "stage_update"
events in real time (analyst → researcher → strategist) using
asyncio.run_coroutine_threadsafe so they can schedule async emits from threads.
"""

import json
import asyncio
from datetime import datetime
from sqlmodel import Session, select

from database import engine
from models import Case, CaseMessage
from .graph import chat_graph


class ProgressEmitter:
    """
    Wraps sio/sid so synchronous graph nodes (running in a thread executor)
    can emit real-time stage progress events to the client without awaiting.
    """

    def __init__(self, sio, sid: str, case_id: int, loop):
        self._sio = sio
        self._sid = sid
        self._case_id = case_id
        self._loop = loop

    def emit_stage(self, stage: str, message: str) -> None:
        asyncio.run_coroutine_threadsafe(
            self._sio.emit(
                "stage_update",
                {"case_id": self._case_id, "stage": stage, "message": message},
                to=self._sid,
            ),
            self._loop,
        )


def get_timestamp() -> str:
    now = datetime.utcnow()
    return now.strftime("%H:%M")


async def process_chat_message(case_id: int, user_content: str, sio, sid: str) -> None:
    """
    Handles a single chat message from the user:

    1. Saves the user message to the DB
    2. Emits "ai_typing" so the UI shows a loading indicator immediately
    3. Loads the case data and chat history from the DB
    4. Runs the LangGraph pipeline in a thread (non-blocking)
    5. Saves the AI response to the DB
    6. Emits "ai_response" back to the client
    """

    # Step 1: Save the user's message to the database
    user_message = CaseMessage(
        case_id=case_id,
        role="user",
        content=user_content,
    )
    with Session(engine) as session:
        session.add(user_message)
        session.commit()
        session.refresh(user_message)
        user_message_id = str(user_message.id)

    # Step 2: Tell the client the AI is working — do this BEFORE the slow call
    # This is why we must NOT block the event loop below
    await sio.emit("ai_typing", {"case_id": case_id}, to=sid)

    try:
        # Step 3: Load the case and its conversation history from the DB
        with Session(engine) as session:
            case = session.get(Case, case_id)

            if not case:
                await sio.emit("ai_typing_done", {"case_id": case_id}, to=sid)
                return

            past_messages = session.exec(
                select(CaseMessage)
                .where(CaseMessage.case_id == case_id)
                .order_by(CaseMessage.created_at)
            ).all()

            # Build conversation history, excluding the message we just saved
            chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in past_messages
                if str(msg.id) != user_message_id
            ]

            pdfs_raw = getattr(case, 'pdf_paths_json', '[]') or '[]'
            pdf_paths = json.loads(pdfs_raw)
            
            urls_raw = getattr(case, 'urls_json', '[]') or '[]'
            urls = json.loads(urls_raw)
            
            imgs_raw = getattr(case, 'image_paths_json', '[]') or '[]'
            image_paths = json.loads(imgs_raw)
            
            context = case.context

        # Build the initial state for the LangGraph pipeline
        loop = asyncio.get_event_loop()
        emitter = ProgressEmitter(sio, sid, case_id, loop)

        initial_state = {
            "case_id": case_id,
            "context": context,
            "pdf_paths": pdf_paths,
            "urls": urls,
            "image_paths": image_paths,
            "chat_history": chat_history,
            "current_query": user_content,
            "query_route": "",
            "vault_chunks": [],
            "web_results": "",
            # Analyst, Researcher, Strategist pipeline — each node fills these in
            "analyst_findings": "",
            "researcher_findings": "",
            "emitter": emitter,
            "response": "",
            "citation": None,
        }

        # Step 4: Run the 3-node pipeline in a thread executor.
        #
        # Flow: analyst_node → researcher_node → strategist_node
        #
        # WHY thread executor: LangChain/Ollama HTTP calls are synchronous (blocking).
        # Running them directly in this async coroutine would freeze the entire
        # event loop and prevent heartbeats and other socket events from firing.
        # run_in_executor() offloads the blocking work to a thread pool.
        final_state = await loop.run_in_executor(
            None,           # Use the default ThreadPoolExecutor
            chat_graph.invoke,
            initial_state,
        )

        ai_response_text = final_state.get("response", "")
        citation = final_state.get("citation", None)

        # Step 5: Save the AI response to the database
        citation_json = json.dumps(citation) if citation else None

        ai_message = CaseMessage(
            case_id=case_id,
            role="ai",
            content=ai_response_text,
            citation_json=citation_json,
        )
        with Session(engine) as session:
            session.add(ai_message)
            session.commit()
            session.refresh(ai_message)
            ai_message_id = str(ai_message.id)

        # Step 6: Send the response back to the client in real time
        await sio.emit(
            "ai_response",
            {
                "case_id": case_id,
                "message": {
                    "id": ai_message_id,
                    "role": "ai",
                    "content": ai_response_text,
                    "timestamp": get_timestamp(),
                    "citation": citation,
                },
            },
            to=sid,
        )

    except Exception as error:
        print(f"[chat_handler] Error processing message for case {case_id}: {str(error)}")

        # Stop the typing indicator
        await sio.emit("ai_typing_done", {"case_id": case_id}, to=sid)

        # Send a fallback error message so the UI doesn't hang
        await sio.emit(
            "ai_response",
            {
                "case_id": case_id,
                "message": {
                    "id": "error",
                    "role": "ai",
                    "content": (
                        "I encountered an error while processing your request. "
                        "Please make sure Ollama is running with granite3.2-vision:latest pulled.\n\n"
                        f"Detail: {str(error)}"
                    ),
                    "timestamp": get_timestamp(),
                    "citation": None,
                },
            },
            to=sid,
        )
