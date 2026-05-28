from fastapi import APIRouter, HTTPException
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
