"""
image_handler.py

Sends an image to the vision model (nvidia/nemotron-3.5-bolt-e4b) and returns a plain-text
description. This description is then stored in the vector DB so the AI can
answer questions about image content (e.g. court diagrams, scanned exhibits).
"""

import base64
import io
import os
import requests
from PIL import Image
from .model_providers import CHAT_REQUEST_TIMEOUT_SECONDS, build_nvidia_chat_model

# Model to use for image understanding (fast and supports multimodal vision)
VISION_MODEL = "nvidia/nemotron-3.5-bolt-e4b"
IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 30
JPEG_QUALITY = 85


def convert_image_bytes_to_jpeg(raw_image_bytes: bytes) -> bytes:
    """
    Converts any image format (WebP, PNG, TIFF, JPEG) into RGB JPEG bytes.
    Why: NVIDIA NIM's vision guard rejects non-JPEG bytes when labeled as data:image/jpeg.
    """
    with Image.open(io.BytesIO(raw_image_bytes)) as image:
        rgb_image = image.convert("RGB")
        output_buffer = io.BytesIO()
        rgb_image.save(output_buffer, format="JPEG", quality=JPEG_QUALITY)
        return output_buffer.getvalue()


def describe_image(image_path: str) -> str:
    """
    Reads an image from disk or remote URL, sends it to the vision model,
    and returns a plain English description of what the image contains.

    Returns an error string if the file is missing or the model call fails.
    """
    if image_path.startswith("http"):
        try:
            response = requests.get(image_path, timeout=IMAGE_DOWNLOAD_TIMEOUT_SECONDS)
            response.raise_for_status()
            raw_image_bytes = response.content
        except Exception as error:
            return f"[Error downloading image: {str(error)}]"
    elif not os.path.exists(image_path):
        return f"[Image file not found: {image_path}]"
    else:
        with open(image_path, "rb") as image_file:
            raw_image_bytes = image_file.read()

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return "[Error: NVIDIA_API_KEY environment variable is not set]"

    prompt = (
        "You are a legal document analyst. Describe everything visible in this image "
        "in detail. If it contains text, transcribe it exactly. If it contains a chart, "
        "table, diagram, or photograph, describe what it shows. Be thorough."
    )

    try:
        jpeg_bytes = convert_image_bytes_to_jpeg(raw_image_bytes)
        image_base64 = base64.b64encode(jpeg_bytes).decode("utf-8")

        llm = build_nvidia_chat_model(
            model_name=VISION_MODEL,
            temperature=0,
            timeout_seconds=CHAT_REQUEST_TIMEOUT_SECONDS,
        )

        response = llm.invoke(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ],
                }
            ]
        )
        return response.content.strip()

    except Exception as error:
        return f"[Error describing image: {str(error)}]"
