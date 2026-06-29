import os
from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA

try:
    chat = ChatNVIDIA(nvidia_api_key=os.getenv("NVIDIA_API_KEY"))
    models = chat.available_models
    print("Available Nvidia Models:")
    for model in sorted(models, key=lambda x: x.id):
         print(f" - {model.id}")
except Exception as e:
    print("Error querying Nvidia models:", e)
