import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

UPLOADS_FOLDER = "uploads"

router = APIRouter(prefix="/api", tags=["files"])


@router.get("/files/{filename}")
def serve_upload(filename: str):
    """
    Serves a file (PDF or Image) from the uploads folder.
    """
    file_path = os.path.join(UPLOADS_FOLDER, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    # FileResponse will automatically guess the media_type (content-type)
    # based on the file extension (e.g. .pdf -> application/pdf, .png -> image/png)
    return FileResponse(file_path)
