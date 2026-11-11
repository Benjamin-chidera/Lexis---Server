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
import requests
from dotenv import load_dotenv
 
load_dotenv() 

# Map NVIDIA_API_KEY to NVIDIA_NIM_API_KEY so LiteLLM/CrewAI can authenticate correctly
if os.getenv("NVIDIA_API_KEY") and not os.getenv("NVIDIA_NIM_API_KEY"):
    os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# --- MODEL CONFIGURATION ---

# 1. Main Chat Model (Used for Analyst, Strategist, Summarizer, deep research)
MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

# Used automatically if the main chat model fails or times out
FALLBACK_MODEL = "nvidia/nemotron-3.5-bolt-e4b"

# Voice calls need very fast answers, so they use the small, fast model
VOICE_MODEL = "nvidia/nemotron-3.5-bolt-e4b"


# 2. Embedding Model (Used for Vector Store)
# Must be pulled in Ollama: ollama pull nomic-embed-text
# EMBEDDING_MODEL_NAME = "text-embedding-3-large"
# EMBEDDING_MODEL_NAME = "mistral-embed"
# EMBEDDING_MODEL_NAME = "nomic-embed-text"
# EMBEDDING_MODEL_NAME = "NV-Embed-QA"
# EMBEDDING_MODEL_NAME = "nvidia/llama-nemotron-embed-1b-v2"
EMBEDDING_MODEL_NAME = "nvidia/nemotron-3-embed-1b"

# 3. Vision Model (Used for Image Handling)
# VISION_MODEL_NAME = "granite3.2-vision:latest"

# Ollama Connection Settings
OLLAMA_BASE_URL = "http://localhost:11434"

# Max seconds to wait for NVIDIA to start answering before giving up.
# Why: ChatNVIDIA's own `timeout` field only controls polling after a 202 response,
# so without this a hung NVIDIA endpoint blocks the chat for minutes.
CHAT_REQUEST_TIMEOUT_SECONDS = 60
# A live voice call cannot wait a minute, so it gives up much sooner
VOICE_REQUEST_TIMEOUT_SECONDS = 10


def build_session_class_with_timeout(timeout_seconds: float):
    """Returns a requests.Session class that applies a default timeout to every request."""

    class SessionWithDefaultTimeout(requests.Session):
        def request(self, *args, **kwargs):
            kwargs.setdefault("timeout", timeout_seconds)
            return super().request(*args, **kwargs)

    return SessionWithDefaultTimeout


def build_nvidia_chat_model(model_name: str, temperature: float, timeout_seconds: float, callbacks=None):
    """Creates a ChatNVIDIA model whose HTTP calls (normal and streaming) fail after the timeout."""
    chat_model = ChatNVIDIA(
        model=model_name,
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=temperature,
        callbacks=callbacks,
    )
    chat_model._client.get_session_fn = build_session_class_with_timeout(timeout_seconds)
    return chat_model


def get_chat_model(temperature=0):
    """
    Returns a configured LangChain Chat model instance.
    Standardized across the whole project.

    If the main model errors or times out, the fallback model answers instead.
    """
    from ai.monitoring import get_langfuse_handler
    handler = get_langfuse_handler()
    callbacks = [handler] if handler else None

    main_model = build_nvidia_chat_model(MODEL, temperature, CHAT_REQUEST_TIMEOUT_SECONDS, callbacks)
    fallback_model = build_nvidia_chat_model(FALLBACK_MODEL, temperature, CHAT_REQUEST_TIMEOUT_SECONDS, callbacks)
    return main_model.with_fallbacks([fallback_model])


def get_voice_chat_model(temperature=0.6):
    """Returns the fast chat model used for live voice calls (fails fast so the call never goes silent)."""
    return build_nvidia_chat_model(VOICE_MODEL, temperature, VOICE_REQUEST_TIMEOUT_SECONDS)

  
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
    """
    from crewai import LLM
    return LLM(
        model=f"nvidia_nim/{MODEL}",
        api_key=os.getenv("NVIDIA_API_KEY"),
        timeout=timeout,
        max_tokens=8192,
    )
