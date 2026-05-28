"""
upload.py

Handles file uploads (PDFs and images) from the client.

Files are uploaded to Cloudinary and the returned URL is stored in the database.
After saving, the file is immediately ingested into the case's Chroma vector store
so it's available for semantic search during chat.

Ingestion runs in a background thread so the HTTP response returns instantly
and the user doesn't have to wait for the embedding process to complete.
"""

import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import List, Optional

from cloudinary_client import upload_pdf, upload_image

# Image file extensions we accept
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
}

router = APIRouter(prefix="/api", tags=["upload"])


def ingest_pdf_in_background(case_id: int, file_url: str):
    """
    Called in a background thread after a PDF is uploaded to Cloudinary.
    Ingests the PDF into the case's Chroma vector store.
    """
    if case_id is None:
        return

    try:
        from ai.vector_store import ingest_pdf_into_vector_store
        chunk_count = ingest_pdf_into_vector_store(case_id, file_url)
        print(f"[upload] PDF ingested for case {case_id}: {chunk_count} chunks")
    except Exception as error:
        print(f"[upload] PDF ingestion failed for case {case_id}: {str(error)}")


def ingest_image_in_background(case_id: int, file_url: str):
    """
    Called in a background thread after an image is uploaded to Cloudinary.
    Describes the image with the vision model and stores the description in Chroma.
    """
    if case_id is None:
        return

    try:
        from ai.vector_store import ingest_image_into_vector_store
        chunk_count = ingest_image_into_vector_store(case_id, file_url)
        print(f"[upload] Image ingested for case {case_id}: {chunk_count} chunks")
    except Exception as error:
        print(f"[upload] Image ingestion failed for case {case_id}: {str(error)}")


@router.post("/upload-pdfs")
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    case_id: Optional[int] = Form(default=None),
):
    """
    Accepts one or more PDF files from the client.
    Uploads each file to Cloudinary and stores the returned URL in the database.
    Triggers background ingestion into the vector store if a case_id is provided.
    Returns a list of the Cloudinary URLs.
    """
    saved_paths = []

    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400,
                detail=f"'{file.filename}' is not a PDF file.",
            )

        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{file.filename}"

        file_bytes = await file.read()
        file_url = upload_pdf(file_bytes, safe_filename)

        saved_paths.append(file_url)

        if case_id is not None:
            from database import engine
            from sqlmodel import Session
            from models import Case
            import json

            with Session(engine) as session:
                case = session.get(Case, case_id)
                if case:
                    pdfs_raw = getattr(case, 'pdf_paths_json', '[]') or '[]'
                    current_pdfs = json.loads(pdfs_raw)
                    current_pdfs.append(file_url)
                    case.pdf_paths_json = json.dumps(current_pdfs)
                    case.status = "pending"
                    session.add(case)
                    session.commit()

            background_tasks.add_task(ingest_pdf_in_background, case_id, file_url)

    if case_id is not None and saved_paths:
        from ai.background import enqueue_research
        enqueue_research(case_id)

    return {"saved_paths": saved_paths}


@router.post("/upload-images")
async def upload_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    case_id: Optional[int] = Form(default=None),
):
    """
    Accepts one or more image files from the client.
    Uploads each file to Cloudinary and stores the returned URL in the database.
    Triggers background ingestion (vision description → Chroma) if a case_id is provided.
    Returns a list of the Cloudinary URLs.
    """
    saved_paths = []

    for file in files:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"'{file.filename}' is not a supported image type. Allowed: JPEG, PNG, WebP, GIF.",
            )

        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{file.filename}"

        file_bytes = await file.read()
        file_url = upload_image(file_bytes, safe_filename)

        saved_paths.append(file_url)

        if case_id is not None:
            from database import engine
            from sqlmodel import Session
            from models import Case
            import json

            with Session(engine) as session:
                case = session.get(Case, case_id)
                if case:
                    imgs_raw = getattr(case, 'image_paths_json', '[]') or '[]'
                    current_images = json.loads(imgs_raw)
                    current_images.append(file_url)
                    case.image_paths_json = json.dumps(current_images)
                    case.status = "pending"
                    session.add(case)
                    session.commit()

            background_tasks.add_task(ingest_image_in_background, case_id, file_url)

    if case_id is not None and saved_paths:
        from ai.background import enqueue_research
        enqueue_research(case_id)

    return {"saved_paths": saved_paths}
