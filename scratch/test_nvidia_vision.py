import os
import base64
import requests
from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA

try:
    # Download a real image
    url = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"
    img_data = requests.get(url).content
    b64_image = base64.b64encode(img_data).decode("utf-8")
    
    llm = ChatNVIDIA(
        model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1", 
        nvidia_api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    print("Invoking Nvidia vision model with real image...")
    response = llm.invoke(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What text is written in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                ],
            }
        ]
    )
    print("Success! Response:", response.content)
except Exception as e:
    print("Failed to invoke Nvidia vision model:", e)
