import json
from typing import List
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from sqlmodel import Session, select
from pydantic import BaseModel
from database import get_session
from models import Case, Alert
from ai.vector_store import ingest_pdf_into_vector_store, ingest_url_into_vector_store
from ai.summarizer import generate_url_summary
from auth import get_current_user

router = APIRouter(tags=["cases"])


@router.get("/api/cases")
def get_cases(
    session: Session = Depends(get_session),
    current_user: dict = Depends(get_current_user)
):
    cases = session.exec(
        select(Case).where(Case.user_id == current_user["user_id"])
    ).all()

    result = []
    for c in cases:
        # map backend status to frontend CaseStatus
        status_map = {
            "pending": "active",
            "active": "active",
            "closed": "closed",
            "archived": "archived",
            "success": "success",
            "abandoned": "abandoned"
        }
        frontend_status = status_map.get(c.case_result_status, "active")

        vault = []

        # 1. Parse URLs
        try:
            urls_raw = getattr(c, 'urls_json', '[]') or '[]'
            urls = json.loads(urls_raw)

            for u in urls:
                vault.append({
                    "id": f"url-{len(vault)}",
                    "type": "url",
                    "name": u,
                    "url": u,
                    "addedAt": c.created_at.strftime("%H:%M")
                })
        except Exception as e:
            print(f"[cases] Error parsing URLs for case {c.id}: {e}")

        # 2. Parse PDFs
        try:
            pdfs_raw = getattr(c, 'pdf_paths_json', '[]') or '[]'
            pdfs = json.loads(pdfs_raw)
            for p in pdfs:
                filename = p.split("/")[-1] if "/" in p else p
                vault.append({
                    "id": f"pdf-{len(vault)}",
                    "type": "pdf",
                    "name": filename,
                    "url": f"http://localhost:8000/api/files/{filename}",
                    "addedAt": c.created_at.strftime("%H:%M")
                })
        except Exception as e:
            print(f"[cases] Error parsing PDFs for case {c.id}: {e}")

        # 3. Parse Images
        try:
            imgs_raw = getattr(c, 'image_paths_json', '[]') or '[]'
            images = json.loads(imgs_raw)
            for img in images:
                filename = img.split("/")[-1] if "/" in img else img
                vault.append({
                    "id": f"image-{len(vault)}",
                    "type": "image",
                    "name": filename,
                    "url": f"http://localhost:8000/api/files/{filename}",
                    "addedAt": c.created_at.strftime("%H:%M")
                })
        except Exception as e:
            print(f"[cases] Error parsing Images for case {c.id}: {e}")

        name = "New Case"
        if c.context:
            name = c.context[:40] + ("..." if len(c.context) > 40 else "")
        else:
            name = f"Case #{c.id}"

        # Check if this case has at least one accepted alert
        has_accepted_alert = session.exec(
            select(Alert).where(Alert.case_id == c.id, Alert.review_status == "accepted")
        ).first() is not None

        result.append({
            "id": str(c.id),
            "name": name,
            "caseType": "Legal Research",
            "attorney": "AI Assistant",
            "openedDate": c.created_at.strftime("%b %d, %Y"),
            "case_result_status": frontend_status,
            "case_result_reason": c.case_result_reason,
            "researchStatus": c.status,
            "canResolve": has_accepted_alert,
            "messages": [],
            "vault": vault
        })

    return result


@router.post("/api/cases/{case_id}/reindex")
def reindex_case(
    case_id: int, 
    background_tasks: BackgroundTasks, 
    session: Session = Depends(get_session),
    current_user: dict = Depends(get_current_user)
):
    """
    Re-ingests all PDFs and URLs for an existing case into the vector store.

    Use this for cases that were created before the vector store was added,
    or if the vector store data was deleted and needs to be rebuilt.

    Example: POST /api/cases/1/reindex
    """
    case = session.get(Case, case_id)
    if not case or case.user_id != current_user["user_id"]:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    pdfs_raw = getattr(case, 'pdf_paths_json', '[]') or '[]'
    pdf_paths = json.loads(pdfs_raw)
    urls_raw = getattr(case, 'urls_json', '[]') or '[]'
    urls = json.loads(urls_raw)

    total_files = len(pdf_paths) + len(urls)

    if total_files == 0:
        return {"message": "No files to index for this case.", "total_files": 0}

    # Ingest each file in the background so the API responds immediately
    for pdf_path in pdf_paths:
        background_tasks.add_task(ingest_pdf_into_vector_store, case_id, pdf_path)

    for url in urls:
        background_tasks.add_task(ingest_url_into_vector_store, case_id, url)

    return {
        "message": f"Reindexing {total_files} file(s) for case {case_id}. This runs in the background.",
        "pdfs": len(pdf_paths),
        "urls": len(urls),
    }


class AddURLRequest(BaseModel):
    url: str

@router.post("/api/cases/{case_id}/add-url")
def add_url_to_case(
    case_id: int, 
    req: AddURLRequest, 
    background_tasks: BackgroundTasks, 
    session: Session = Depends(get_session),
    current_user: dict = Depends(get_current_user)
):
    """
    Adds a new URL to an existing case, triggers background ingestion,
    and re-queues background research so the agent uses the new content.
    """
    case = session.get(Case, case_id)
    if not case or case.user_id != current_user["user_id"]:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    urls_raw = getattr(case, 'urls_json', '[]') or '[]'
    urls = json.loads(urls_raw)
    urls.append(req.url)
    case.urls_json = json.dumps(urls)
    case.status = "pending"
    session.add(case)
    session.commit()

    background_tasks.add_task(ingest_url_into_vector_store, case_id, req.url)

    from ai.background import enqueue_research
    enqueue_research(case_id)

    return {"message": f"URL added and ingestion started for case {case_id}", "url": req.url}


class AddContextRequest(BaseModel):
    context: str

@router.post("/api/cases/{case_id}/add-context")
def add_context_to_case(
    case_id: int, 
    req: AddContextRequest, 
    session: Session = Depends(get_session),
    current_user: dict = Depends(get_current_user)
):
    """
    Appends additional context notes to an existing case and re-queues
    background research so the agent uses the updated context.
    """
    case = session.get(Case, case_id)
    if not case or case.user_id != current_user["user_id"]:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    existing = case.context or ""
    separator = "\n\n" if existing.strip() else ""
    case.context = existing + separator + req.context.strip()
    case.status = "pending"
    session.add(case)
    session.commit()

    from ai.background import enqueue_research
    enqueue_research(case_id)

    return {"message": f"Context updated and research re-queued for case {case_id}"}

class UpdateCaseStatusRequest(BaseModel):
    case_result_status: str
    case_result_reason: str = ""

@router.patch("/api/cases/{case_id}/status")
def update_case_status(
    case_id: int, 
    req: UpdateCaseStatusRequest, 
    session: Session = Depends(get_session),
    current_user: dict = Depends(get_current_user)
):
    """
    Updates the status and result reason for a specific case.
    Only allowed when at least one alert for the case has been accepted.
    """

    case = session.get(Case, case_id)
    if not case or case.user_id != current_user["user_id"]:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    # Check if there is at least one accepted alert for this case
    accepted_alert = session.exec(
        select(Alert).where(Alert.case_id == case_id, Alert.review_status == "accepted")
    ).first()

    if not accepted_alert:
        raise HTTPException(
            status_code=400,
            detail="Cannot change case status until at least one alert has been accepted."
        )

    case.case_result_status = req.case_result_status
    case.case_result_reason = req.case_result_reason
    session.add(case)
    session.commit()

    return {"message": f"Case {case_id} status updated"}


@router.post("/api/cases/{case_id}/retry-research")
def retry_research(
    case_id: int, 
    session: Session = Depends(get_session),
    current_user: dict = Depends(get_current_user)
):
    """
    Retries failed background research for a case.

    Resets the case status from "failed" back to "pending" and re-enqueues
    the research job onto the RQ queue.
    """
    case = session.get(Case, case_id)
    if not case or case.user_id != current_user["user_id"]:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    if case.status not in ("failed", "complete"):
        raise HTTPException(
            status_code=400,
            detail=f"Case {case_id} has status '{case.status}' — only failed or complete cases can be retried.",
        )

    case.status = "pending"
    session.add(case)
    session.commit()

    from ai.background import enqueue_research
    enqueue_research(case_id)

    return {"message": f"Research re-queued for case {case_id}"}
