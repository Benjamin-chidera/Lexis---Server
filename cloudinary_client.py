import os
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)


def upload_pdf(file_bytes: bytes, filename: str) -> str:
    result = cloudinary.uploader.upload(
        file_bytes,
        resource_type="raw",
        folder="legal-assistant/pdfs",
        public_id=filename,
        overwrite=True,
    )
    return result["secure_url"]


def upload_image(file_bytes: bytes, filename: str) -> str:
    result = cloudinary.uploader.upload(
        file_bytes,
        resource_type="image",
        folder="legal-assistant/images",
        public_id=filename,
        overwrite=True,
    )
    return result["secure_url"]
