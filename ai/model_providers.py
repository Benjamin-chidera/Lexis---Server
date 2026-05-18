"""
ai/providers.py

Centralized configuration for AI models (Chat, Embeddings, Vision).
By importing from this file, we ensure all components use the same models.

This also makes it trivial to switch from local Ollama models to cloud
providers (OpenAI, Anthropic, etc.) in the future by updating this one file.
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  
import os
from dotenv import load_dotenv

load_dotenv() 

# --- MODEL CONFIGURATION ---

# 1. Main Chat Model (Used for Analyst, Strategist, Summarizer)
# Must be pulled in Ollama: ollama pull granite3.2-vision:latest
# CHAT_MODEL_NAME = "granite3.2-vision:latest"

MODEL = "gpt-4-turbo"
# MODEL = "granite3.2-vision:latest"

# 2. Embedding Model (Used for Vector Store)
# Must be pulled in Ollama: ollama pull nomic-embed-text
EMBEDDING_MODEL_NAME = "text-embedding-3-larger"
# EMBEDDING_MODEL_NAME = "nomic-embed-text"

# 3. Vision Model (Used for Image Handling)
# VISION_MODEL_NAME = "granite3.2-vision:latest"

# Ollama Connection Settings
OLLAMA_BASE_URL = "http://localhost:11434"


def get_chat_model(temperature=0):
    """
    Returns a configured LangChain Chat model instance.
    Standardized across the whole project.
    """
    return ChatOpenAI(
        model=MODEL,
        api_key=os.getenv("OPENAI_API_KEY"), 
        temperature=temperature,
    )
# def get_chat_model(temperature=0):
#     """
#     Returns a configured LangChain Chat model instance.
#     Standardized across the whole project.
#     """
#     return ChatOllama(
#         model=MODEL,
#         base_url=OLLAMA_BASE_URL,
#         temperature=temperature,
#     )


def get_embeddings():
    """
    Returns a configured LangChain Embeddings instance.
    Used by the Chroma vector store.
    """
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        api_key=os.getenv("OPENAI_API_KEY"), 
    )
# def get_embeddings():
#     """
#     Returns a configured LangChain Embeddings instance.
#     Used by the Chroma vector store.
#     """
#     return OllamaEmbeddings(
#         model=EMBEDDING_MODEL_NAME,
#         base_url=OLLAMA_BASE_URL,
#     )


def get_crew_llm(timeout=1200):
    """
    Returns a configured CrewAI LLM instance.
    LiteLLM requires the 'ollama/' prefix for local models.
    """
    from crewai import LLM
    return LLM(
        # model=f"ollama/{MODEL}",
        model=MODEL,
        # base_url=OLLAMA_BASE_URL,
        api_key=os.getenv("OPENAI_API_KEY"), # Dummy key for local Ollama
        # api_key="ollama", # Dummy key for local Ollama
        timeout=timeout
    )
