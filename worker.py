"""
worker.py

RQ worker entry point for background legal research.

Run with:
    source .venv/bin/activate
    rq worker --url redis://localhost:6379 legal

The worker picks up jobs from the "legal" queue, runs CrewAI research,
saves results to Postgres (Neon), and publishes a Redis pub/sub message
so the FastAPI server can push a socket notification to connected clients.

Resume logic:
    When the server restarts (e.g. laptop woken from sleep), the lifespan
    handler in main.py queries Postgres for any Case with status "pending"
    or "processing" and re-enqueues those case IDs. This means interrupted
    research is never abandoned — it will run once the worker is back online.
"""

import os
import json
import redis
import time

from dotenv import load_dotenv
from sqlmodel import Session

load_dotenv()

from database import engine
from models import Case, Alert
from ai.crew import run_research_crew

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Channel name the FastAPI server listens to for socket.io relay
PUBSUB_CHANNEL = "research_done"


def research_job(case_id: int) -> None:
    """
    RQ job: run background legal research for a single case.

    Called by the RQ worker process. Updates Case.status in Postgres
    at each stage so the server can recover on restart.
    """
    print(f"[worker] Starting research for case {case_id}")

    redis_conn = redis.from_url(REDIS_URL)

    # Mark as in-progress so the server knows this job is active
    with Session(engine) as session:
        case = session.get(Case, case_id)
        if not case:
            print(f"[worker] Case {case_id} not found — aborting")
            return
        case.status = "processing"
        session.add(case)
        session.commit()
        context = case.context 

    if not context.strip():
        print(f"[worker] Case {case_id} has no context — marking complete")
        _set_status(case_id, "complete")
        return

    max_retries = 3
    retry_delay = 15  # seconds backoff factor
    final_error = None

    for attempt in range(1, max_retries + 1):
        try:
            research_result, ai_reasoning = run_research_crew(context)
            severity = "strategic" if len(research_result) > 400 else "routine"
            title = f"Research Ready: Case #{case_id}" 
            summary = research_result
            status = "complete"
            final_error = None
            break  # Success! Break the retry loop
        except Exception as error:
            print(f"[worker] Research attempt {attempt}/{max_retries} failed for case {case_id}: {error}")
            final_error = error
            if attempt < max_retries:
                sleep_time = retry_delay * attempt
                print(f"[worker] Waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
            else:
                # Final failure
                title = f"Research Error: Case #{case_id}"
                summary = f"Background research encountered an error and could not complete after {max_retries} attempts.\n\n**Error:** {str(error)}"
                ai_reasoning = ""
                severity = "routine"
                status = "failed"

    # Always update the case status
    with Session(engine) as session:
        case = session.get(Case, case_id)
        case.status = status
        session.add(case)
        session.commit()

        case_name = (case.context[:40] + "...") if case.context and len(case.context) > 40 else (case.context or f"Case #{case_id}")

        if status == "failed":
            # Do NOT create an alert for errors, just send an error event
            error_data = {
                "type": "research_error",
                "case_name": case_name,
                "message": summary
            }
            redis_conn.publish(PUBSUB_CHANNEL, json.dumps(error_data))
            print(f"[worker] Research job finished for case {case_id} (status={status})")
            if final_error:
                raise final_error
        else:
            new_alert = Alert(
                case_id=case_id,
                title=title,
                summary=summary,
                ai_reasoning=ai_reasoning or None,
                severity=severity,
                status="unread",
            )
            session.add(new_alert)
            session.commit()
            session.refresh(new_alert)

            alert_data = {
                "type": "new_alert",
                "id": new_alert.id,
                "case_id": new_alert.case_id,
                "case_name": case_name,
                "title": new_alert.title,
                "summary": new_alert.summary,
                "ai_reasoning": new_alert.ai_reasoning,
                "severity": new_alert.severity,
                "status": new_alert.status,
                "created_at": new_alert.created_at.isoformat(),
            }

            # Notify the FastAPI server so it can push a socket event
            redis_conn.publish(PUBSUB_CHANNEL, json.dumps(alert_data))
            print(f"[worker] Research job finished for case {case_id} (status={status})")


def _set_status(case_id: int, status: str) -> None:
    with Session(engine) as session:
        case = session.get(Case, case_id)
        if case:
            case.status = status
            session.add(case)
            session.commit()
