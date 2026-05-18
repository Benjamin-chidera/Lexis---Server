"""
image_handler.py

Sends an image to the Ollama granite3.2-vision:latest multimodal model and returns a plain-text
description. This description is then stored in the vector DB so the AI can
answer questions about image content (e.g. court diagrams, scanned exhibits).

Ollama must be running with the granite3.2-vision:latest model loaded:
  ollama run granite3.2-vision:latest
"""

import base64
import os
import requests
from .model_providers import OLLAMA_BASE_URL, MODEL

# The Ollama REST endpoint for chat completions
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/chat"

# Model to use for image understanding
VISION_MODEL = MODEL


def describe_image(image_path: str) -> str:
    """
    Reads an image from disk, sends it to the granite3.2-vision:latest model via Ollama,
    and returns a plain English description of what the image contains.

    Returns an error string if the file is missing or the model call fails.
    """
    if not os.path.exists(image_path):
        return f"[Image file not found: {image_path}]"

    # Read the image bytes and encode as base64
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "You are a legal document analyst. Describe everything visible in this image "
        "in detail. If it contains text, transcribe it exactly. If it contains a chart, "
        "table, diagram, or photograph, describe what it shows. Be thorough."
    )

    # Build the request payload for Ollama's chat API
    request_body = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64],
            }
        ],
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=request_body, timeout=120)
        response.raise_for_status()

        data = response.json()
        description = data["message"]["content"]

        return description.strip()

    except requests.exceptions.ConnectionError:
        return "[Error: Could not connect to Ollama. Is it running? Run: ollama run llava]"

    except Exception as error:
        return f"[Error describing image: {str(error)}]"
