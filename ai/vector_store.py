"""
vector_store.py

Manages a per-case Chroma vector store using LangChain's Chroma integration.
Every case gets its own isolated collection named "case_{case_id}".
Documents are split into chunks and embedded using the nomic-embed-text model.
"""

import os
import json
import requests as http_requests

# Disable ChromaDB anonymous telemetry
os.environ["ANONYMOUS_TELEMETRY"] = "False"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from .model_providers import get_embeddings
from .image_handler import describe_image

# Where Chroma will store its data on disk
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# Text chunking settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def get_collection_name(case_id: int) -> str:
    """Each case gets its own isolated Chroma collection."""
    return f"case_{case_id}"


def get_vector_store(case_id: int) -> Chroma:
    """
    Initializes/loads a persistent Chroma vector store for a specific case.
    Uses the embeddings from get_embeddings().
    """
    embeddings = get_embeddings()
    collection_name = get_collection_name(case_id)
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def split_text_into_chunks(text: str, source_label: str) -> list:
    """Splits plain text into structured LangChain Documents with source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = splitter.create_documents([text], metadatas=[{"source": source_label}])
    return docs


def ingest_pdf_into_vector_store(case_id: int, pdf_path: str) -> int:
    """
    Parses text from a local PDF file or remote PDF URL and adds it to the case's vector store.
    """
    import pypdf
    import io
    filename = os.path.basename(pdf_path)
    try:
        if pdf_path.startswith("http://") or pdf_path.startswith("https://"):
            response = http_requests.get(pdf_path, timeout=30)
            response.raise_for_status()
            reader = pypdf.PdfReader(io.BytesIO(response.content))
        else:
            reader = pypdf.PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        full_text = "\n".join(text_parts).strip()
        
        if not full_text:
            print(f"[vector_store] PDF '{filename}' yielded no text")
            return 0

        chunks = split_text_into_chunks(full_text, source_label=filename)
        if not chunks:
            return 0

        vector_store = get_vector_store(case_id)
        vector_store.add_documents(chunks)
        print(f"[vector_store] Added {len(chunks)} chunks from PDF '{filename}'")
        return len(chunks)
    except Exception as error:
        print(f"[vector_store] PDF ingestion failed for '{filename}': {error}")
        return 0


def ingest_url_into_vector_store(case_id: int, url: str) -> int:
    """
    Fetches raw HTML from a URL, extracts plain text, and adds it to the case's vector store.
    """
    from bs4 import BeautifulSoup
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = http_requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Strip script and style blocks
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text()
        
        # Clean up spacing
        lines = (line.strip() for line in text.splitlines())
        chunks_text = (phrase.strip() for line in lines for phrase in line.split("  "))
        full_text = "\n".join(chunk for chunk in chunks_text if chunk)
        
        if not full_text.strip():
            print(f"[vector_store] URL '{url}' yielded no text")
            return 0

        chunks = split_text_into_chunks(full_text, source_label=url)
        if not chunks:
            return 0

        vector_store = get_vector_store(case_id)
        vector_store.add_documents(chunks)
        print(f"[vector_store] Added {len(chunks)} chunks from URL '{url}'")
        return len(chunks)
    except Exception as error:
        print(f"[vector_store] URL ingestion failed for '{url}': {error}")
        return 0


def ingest_image_into_vector_store(case_id: int, image_path: str) -> int:
    """
    Uses Mistral Vision API to generate an image description,
    chunks it, and adds it to the case's vector store.
    """
    filename = os.path.basename(image_path)
    description = describe_image(image_path)

    if description.startswith("[Error") or description.startswith("[Image"):
        print(f"[vector_store] Image description failed: {description}")
        return 0

    full_text = f"[Image: {filename}]\n\n{description}"
    chunks = split_text_into_chunks(full_text, source_label=f"Image: {filename}")

    if not chunks:
        return 0

    vector_store = get_vector_store(case_id)
    vector_store.add_documents(chunks)

    print(f"[vector_store] Added {len(chunks)} chunks from image '{filename}'")
    return len(chunks)


def search_vector_store(case_id: int, query: str, top_k: int = 5) -> list:
    """
    Runs a similarity search against the case's vector store.
    Returns plain text strings with their source headers.
    """
    try:
        vector_store = get_vector_store(case_id)
        results = vector_store.similarity_search(query, k=top_k)

        if not results:
            return []

        formatted_chunks = []
        for doc in results:
            source = doc.metadata.get("source", "Unknown source")
            chunk_text = f"[{source}]\n{doc.page_content}"
            formatted_chunks.append(chunk_text)

        return formatted_chunks
    except Exception as error:
        print(f"[vector_store] Search failed for case {case_id}: {str(error)}")
        return []
