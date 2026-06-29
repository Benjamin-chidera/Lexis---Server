"""
image_handler.py

Sends an image to the Nvidia Vision model (llama-3.1-nemotron-nano-vl-8b-v1) and returns a plain-text
description. This description is then stored in the vector DB so the AI can
answer questions about image content (e.g. court diagrams, scanned exhibits).
"""

import base64
import os
import requests
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Model to use for image understanding
VISION_MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"


def describe_image(image_path: str) -> str:
    """
    Reads an image from disk or remote URL, sends it to the Nvidia vision model,
    and returns a plain English description of what the image contains.

    Returns an error string if the file is missing or the model call fails.
    """
    if image_path.startswith("http"):
        try:
            response = requests.get(image_path, timeout=30)
            response.raise_for_status()
            image_bytes = response.content
        except Exception as error:
            return f"[Error downloading image: {str(error)}]"
    elif not os.path.exists(image_path):
        return f"[Image file not found: {image_path}]"
    else:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return "[Error: NVIDIA_API_KEY environment variable is not set]"

    prompt = (
        "You are a legal document analyst. Describe everything visible in this image "
        "in detail. If it contains text, transcribe it exactly. If it contains a chart, "
        "table, diagram, or photograph, describe what it shows. Be thorough."
    )

    try:
        llm = ChatNVIDIA(
            model=VISION_MODEL,
            nvidia_api_key=api_key
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
