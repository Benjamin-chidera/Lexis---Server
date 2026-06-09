from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select
from database import engine
from models import Alert, Case

router = APIRouter(prefix="/api", tags=["alerts"])


@router.get("/alerts")
def get_alerts():
    """Returns all alerts ordered newest first, with the parent case name."""
    with Session(engine) as session:
        alerts = session.exec(
            select(Alert).order_by(Alert.created_at.desc())
        ).all()

        # Build a lookup of case_id → case name so we can label each alert
        case_ids = set(a.case_id for a in alerts if a.case_id is not None)
        case_names: dict[int, str] = {}
        if case_ids:
            cases = session.exec(
                select(Case).where(Case.id.in_(case_ids))
            ).all()
            for c in cases:
                # Use the first 40 chars of the context as the name, same as the cases route
                if c.context:
                    case_names[c.id] = c.context[:40] + ("..." if len(c.context) > 40 else "")
                else:
                    case_names[c.id] = f"Case #{c.id}"

    return [
        {
            "id": a.id,
            "case_id": a.case_id,
            "case_name": case_names.get(a.case_id, "Unknown Case") if a.case_id else None,
            "title": a.title,
            "summary": a.summary,
            "ai_reasoning": a.ai_reasoning,
            "severity": a.severity,
            "status": a.status,
            "review_status": a.review_status,
            "created_at": a.created_at.isoformat(),
        }
        for a in alerts
    ]


@router.patch("/alerts/{alert_id}/read")
def mark_alert_read(alert_id: int):
    """Marks a single alert as read."""
    with Session(engine) as session:
        alert = session.get(Alert, alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        alert.status = "read"
        session.add(alert)
        session.commit()

    return {"ok": True}


@router.patch("/alerts/{alert_id}/accept")
def accept_alert(alert_id: int):
    """
    Marks a research alert as accepted by the attorney.
    The accept/reject buttons will no longer be shown for this alert.
    """
    with Session(engine) as session:
        alert = session.get(Alert, alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        alert.review_status = "accepted"
        session.add(alert)
        session.commit()

    return {"ok": True}


class RejectAlertRequest(BaseModel):
    reason: str


@router.patch("/alerts/{alert_id}/reject")
def reject_alert(alert_id: int, req: RejectAlertRequest):
    """
    Rejects a research alert: stores the reason, archives the alert,
    appends the feedback to the case context, and re-queues research
    so the AI agent addresses the attorney's concerns.
    """
    with Session(engine) as session:
        alert = session.get(Alert, alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        # Persist the rejection decision
        alert.review_status = "rejected"
        alert.rejection_reason = req.reason
        alert.status = "archived"
        session.add(alert)
        session.commit()

        # Append the rejection feedback to the parent case context
        # so the AI agent uses it in the next research pass
        if alert.case_id:
            case = session.get(Case, alert.case_id)
            if case:
                feedback_block = (
                    "\n\n[ATTORNEY FEEDBACK — Previous research was rejected]\n"
                    f"Reason: {req.reason}\n"
                    "Please address this feedback and avoid the same issues "
                    "in the next research pass."
                )
                case.context = (case.context or "") + feedback_block
                case.status = "pending"
                session.add(case)
                session.commit()

                # Re-enqueue research for this case with the updated context
                from ai.background import enqueue_research
                enqueue_research(case.id)

    return {"ok": True, "message": "Research re-queued with feedback"}


@router.patch("/alerts/archive-all")
def archive_all_alerts():
    """Archives all alerts that are currently unread."""
    with Session(engine) as session:
        unread_alerts = session.exec(
            select(Alert).where(Alert.status == "unread")
        ).all()

        for alert in unread_alerts:
            alert.status = "archived"
            session.add(alert)

        session.commit()

    return {"archived_count": len(unread_alerts)}
