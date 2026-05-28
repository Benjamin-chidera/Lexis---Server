"""
chat_handler.py

Handles incoming chat messages from the Socket.IO client.
Runs the synchronous chat_graph.invoke() in a thread executor to avoid blocking the event loop.
"""

import json
import asyncio
from datetime import datetime
from sqlmodel import Session, select

from database import engine
from models import Case, CaseMessage
from .graph import chat_graph
from ai.background import enqueue_research


class ProgressEmitter:
    """
    Wraps sio/sid so synchronous graph nodes (running in a thread executor)
    can emit real-time stage progress events to the client.
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
    """Returns the current UTC timestamp formatted as ISO 8601 string."""
    now = datetime.utcnow()
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


async def process_chat_message(case_id: int, user_content: str, sio, sid: str) -> None:
    """
    Handles a single chat message from the user:
    1. Saves the user message to the DB
    2. Emits "ai_typing" to show loading indicator
    3. Loads the case data and history
    4. Runs LangGraph pipeline in a background thread
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

    # Step 2: Show typing indicator
    await sio.emit("ai_typing", {"case_id": case_id}, to=sid)

    try:
        # Step 3: Load the case and its history from the DB
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

            # Build conversation history, excluding current user message
            chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in past_messages
                if str(msg.id) != user_message_id
            ]

            pdfs_raw = (getattr(case, 'pdf_paths_json', None) or '').strip() or '[]'
            pdf_paths = json.loads(pdfs_raw)

            urls_raw = (getattr(case, 'urls_json', None) or '').strip() or '[]'
            urls = json.loads(urls_raw)

            imgs_raw = (getattr(case, 'image_paths_json', None) or '').strip() or '[]'
            image_paths = json.loads(imgs_raw)
            
            context = case.context

        # Build progress emitter
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
            "analyst_findings": "",
            "researcher_findings": "",
            "emitter": emitter,
            "response": "",
            "citation": None,
        }

        # Step 4: Run the LangGraph pipeline in a thread executor
        final_state = await loop.run_in_executor(
            None,
            chat_graph.invoke,
            initial_state,
        )

        ai_response_text = final_state.get("response", "")
        citation = final_state.get("citation", None)

        # Check if research trigger is requested
        if "[TRIGGER_RESEARCH]" in ai_response_text:
            print(f"[chat_handler] Research trigger requested for case {case_id}")
            ai_response_text = ai_response_text.replace("[TRIGGER_RESEARCH]", "").strip()
            try:
                enqueue_research(case_id)
            except Exception as e:
                print(f"[chat_handler] Error enqueuing research: {e}")

        # Step 5: Save AI response to DB
        ai_message = CaseMessage(
            case_id=case_id,
            role="ai",
            content=ai_response_text,
            citation=json.dumps(citation) if citation else None,
        )
        with Session(engine) as session:
            session.add(ai_message)
            session.commit()
            session.refresh(ai_message)
            ai_message_id = str(ai_message.id)

        # Step 6: Emit AI response back to the client
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

        # Emit fallback error message
        await sio.emit(
            "ai_response",
            {
                "case_id": case_id,
                "message": {
                    "id": "error",
                    "role": "ai",
                    "content": (
                        "I encountered an error while processing your request.\n\n"
                        f"Detail: {str(error)}"
                    ),
                    "timestamp": get_timestamp(),
                    "citation": None,
                },
            },
            to=sid,
        )
