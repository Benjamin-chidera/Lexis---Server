import json
from fastapi import APIRouter
from sqlmodel import Session, select
from database import engine
from models import CaseMessage

router = APIRouter(prefix="/api", tags=["messages"])


@router.get("/cases/{case_id}/messages")
def get_case_messages(case_id: int):
    """
    Returns all chat messages for a case, ordered oldest first.
    Called when the user opens a case modal to restore conversation history.
    """
    with Session(engine) as session:
        messages = session.exec(
            select(CaseMessage)
            .where(CaseMessage.case_id == case_id)
            .order_by(CaseMessage.created_at)
        ).all()

    result = []
    for msg in messages:
        # Parse citation JSON back into an object if it exists
        citation = None
        if msg.citation_json:
            try:
                citation = json.loads(msg.citation_json)
            except Exception:
                citation = None

        result.append({
            "id": str(msg.id),
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.created_at.strftime("%H:%M"),
            "citation": citation,
        })

    return result
