import os
import json
import urllib.parse
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse, RedirectResponse
from sqlmodel import Session, select
from database import get_session
from models import Case

UPLOADS_FOLDER = "uploads"

router = APIRouter(prefix="/api", tags=["files"])


@router.api_route("/files/{filename}", methods=["GET", "HEAD"])
def serve_upload(filename: str, session: Session = Depends(get_session)):
    """
    Serves a file (PDF or Image) from the uploads folder,
    or redirects to Cloudinary if stored in the cloud.
    """
    file_path = os.path.join(UPLOADS_FOLDER, filename)

    if os.path.exists(file_path):
        return FileResponse(file_path)

    # Search cases for a matching Cloudinary URL
    decoded_filename = urllib.parse.unquote(filename)
    cases = session.exec(select(Case)).all()
    for case in cases:
        for json_field in [case.pdf_paths_json, case.image_paths_json]:
            try:
                paths = json.loads(json_field or '[]')
                for path in paths:
                    path_basename = path.split("/")[-1] if "/" in path else path
                    decoded_path_basename = urllib.parse.unquote(path_basename)
                    if (
                        filename == path_basename
                        or decoded_filename == decoded_path_basename
                        or filename in path
                        or decoded_filename in path
                    ):
                        return RedirectResponse(url=path, status_code=302)
            except Exception:
                continue

    raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
