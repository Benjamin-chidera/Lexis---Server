"""
ai/providers.py

Centralized configuration for AI models (Chat, Embeddings, Vision).
By importing from this file, we ensure all components use the same models.

This also makes it trivial to switch from local Ollama models to cloud
providers (OpenAI, Anthropic, etc.) in the future by updating this one file.
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
import os
from dotenv import load_dotenv

load_dotenv() 

# --- MODEL CONFIGURATION ---

# 1. Main Chat Model (Used for Analyst, Strategist, Summarizer)
# Must be pulled in Ollama: ollama pull granite3.2-vision:latest
# CHAT_MODEL_NAME = "granite3.2-vision:latest"

# MODEL = "gpt-4-turbo"
# MODEL = "granite3.2-vision:latest"
# MODEL = "mistral-large-latest"
# MODEL = "mistral-small-latest"
# MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
MODEL = "deepseek-ai/deepseek-v4-pro"

# 2. Embedding Model (Used for Vector Store)
# Must be pulled in Ollama: ollama pull nomic-embed-text
# EMBEDDING_MODEL_NAME = "text-embedding-3-large"
# EMBEDDING_MODEL_NAME = "mistral-embed"
# EMBEDDING_MODEL_NAME = "nomic-embed-text"
# EMBEDDING_MODEL_NAME = "NV-Embed-QA"
EMBEDDING_MODEL_NAME = "nvidia/llama-nemotron-embed-1b-v2"

# 3. Vision Model (Used for Image Handling)
# VISION_MODEL_NAME = "granite3.2-vision:latest"

# Ollama Connection Settings
OLLAMA_BASE_URL = "http://localhost:11434"


def get_chat_model(temperature=0):
    """
    Returns a configured LangChain Chat model instance.
    Standardized across the whole project.
    """
    return ChatNVIDIA(
        model=MODEL,
        # api_key=os.getenv("MISTRAL_API_KEY"), 
        #         api_key=os.getenv("OPENAI_API_KEY"), 
        api_key=os.getenv("NVIDIA_API_KEY"),  

        temperature=temperature,
    )
# def get_chat_model(temperature=0):

  
def get_embeddings():
    """
    Returns a configured LangChain Embeddings instance.
    Used by the Chroma vector store.
    """
    return NVIDIAEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        api_key=os.getenv("NVIDIA_API_KEY"), 
    )


def get_crew_llm(timeout=1200):
    """
    Returns a configured CrewAI LLM instance.
    LiteLLM requires the provider prefix (e.g. 'mistral/') in the model name.
    """
    from crewai import LLM
    return LLM(
        model=f"nvidia_nim/{MODEL}",
        api_key=os.getenv("NVIDIA_API_KEY"),
        timeout=timeout
    )
