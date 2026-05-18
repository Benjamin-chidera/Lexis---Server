"""
upload.py

Handles file uploads (PDFs and images) from the client.

After saving a file to disk, we immediately ingest it into the case's
Chroma vector store so it's available for semantic search during chat.

Ingestion runs in a background thread so the HTTP response returns instantly
and the user doesn't have to wait for the embedding process to complete.
"""

import os
import time
import asyncio
import aiofiles
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import List, Optional

# PDFs and images are saved here, inside the server directory
UPLOADS_FOLDER = "uploads"

# Image file extensions we accept
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
}

router = APIRouter(prefix="/api", tags=["upload"])


def ensure_uploads_folder_exists():
    if not os.path.exists(UPLOADS_FOLDER):
        os.makedirs(UPLOADS_FOLDER)


def ingest_pdf_in_background(case_id: int, file_path: str):
    """
    Called in a background thread after a PDF is saved.
    Ingests the PDF into the case's Chroma vector store.
    """
    if case_id is None:
        # No case ID supplied — skip ingestion (will ingest when case is created)
        return

    try:
        from ai.vector_store import ingest_pdf_into_vector_store
        chunk_count = ingest_pdf_into_vector_store(case_id, file_path)
        print(f"[upload] PDF ingested for case {case_id}: {chunk_count} chunks")
    except Exception as error:
        print(f"[upload] PDF ingestion failed for case {case_id}: {str(error)}")


def ingest_image_in_background(case_id: int, file_path: str):
    """
    Called in a background thread after an image is saved.
    Describes the image with llava and stores the description in Chroma.
    """
    if case_id is None:
        return

    try:
        from ai.vector_store import ingest_image_into_vector_store
        chunk_count = ingest_image_into_vector_store(case_id, file_path)
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
    Saves each file to the uploads/ folder on disk.
    Triggers background ingestion into the vector store if a case_id is provided.
    Returns a list of the saved file paths.
    """
    ensure_uploads_folder_exists()

    saved_paths = []

    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400,
                detail=f"'{file.filename}' is not a PDF file.",
            )

        # Prefix with timestamp to avoid filename collisions
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOADS_FOLDER, safe_filename)

        file_bytes = await file.read()
        async with aiofiles.open(file_path, "wb") as output_file:
            await output_file.write(file_bytes)

        saved_paths.append(file_path)

        # Ingest into vector store in the background (non-blocking)
        if case_id is not None:
            # Update database record
            from database import engine
            from sqlmodel import Session
            from models import Case
            import json

            with Session(engine) as session:
                case = session.get(Case, case_id)
                if case:
                    pdfs_raw = getattr(case, 'pdf_paths_json', '[]') or '[]'
                    current_pdfs = json.loads(pdfs_raw)
                    current_pdfs.append(file_path)
                    case.pdf_paths_json = json.dumps(current_pdfs)
                    session.add(case)
                    session.commit()

            background_tasks.add_task(ingest_pdf_in_background, case_id, file_path)

    return {"saved_paths": saved_paths}


@router.post("/upload-images")
async def upload_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    case_id: Optional[int] = Form(default=None),
):
    """
    Accepts one or more image files from the client.
    Saves each file to the uploads/ folder on disk.
    Triggers background ingestion (llava description → Chroma) if a case_id is provided.
    Returns a list of the saved file paths.
    """
    ensure_uploads_folder_exists()

    saved_paths = []

    for file in files:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"'{file.filename}' is not a supported image type. Allowed: JPEG, PNG, WebP, GIF.",
            )

        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOADS_FOLDER, safe_filename)

        file_bytes = await file.read()
        async with aiofiles.open(file_path, "wb") as output_file:
            await output_file.write(file_bytes)

        saved_paths.append(file_path)

        # Describe and ingest the image in the background
        if case_id is not None:
            # Update database record
            from database import engine
            from sqlmodel import Session
            from models import Case
            import json

            with Session(engine) as session:
                case = session.get(Case, case_id)
                if case:
                    imgs_raw = getattr(case, 'image_paths_json', '[]') or '[]'
                    current_images = json.loads(imgs_raw)
                    current_images.append(file_path)
                    case.image_paths_json = json.dumps(current_images)
                    session.add(case)
                    session.commit()

            background_tasks.add_task(ingest_image_in_background, case_id, file_path)

    return {"saved_paths": saved_paths}
