"""
vector_store.py

Manages a per-case Pinecone vector store using LangChain's Pinecone integration.
Every case gets its own isolated namespace named "case_{case_id}" inside the
"lexis-vector" index. Documents are embedded using the nvidia/llama-nemotron-embed-1b-v2 model.
"""

import os
import json
import requests as http_requests
from pinecone import Pinecone, ServerlessSpec

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from .model_providers import get_embeddings
from .image_handler import describe_image

# Pinecone Index name
PINECONE_INDEX_NAME = "lexis-vector" 

# Text chunking settings 
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def get_collection_name(case_id: int) -> str:
    """Each case gets its own isolated Pinecone namespace."""
    return f"case_{case_id}"


def get_vector_store(case_id: int) -> PineconeVectorStore:
    """
    Initializes/loads the Pinecone vector store for a specific case.
    Uses namespaces for case-level data isolation.
    """
    embeddings = get_embeddings()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")

    pc = Pinecone(api_key=api_key)

    # Auto-create the index if it doesn't exist
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"[vector_store] Creating Pinecone index '{PINECONE_INDEX_NAME}' with dimension 2048...")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=2048,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        except Exception as e:
            print(f"[vector_store] Error creating index (could be creating in background): {e}")

    namespace = get_collection_name(case_id)
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace,
        pinecone_api_key=api_key
    )


def split_text_into_chunks(text: str, source_label: str) -> list:
    """Splits plain text into structured LangChain Documents with source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = splitter.create_documents([text], metadatas=[{"source": source_label}])
    return docs


def ingest_pdf_into_vector_store(case_id: int, pdf_path: str, file_bytes: bytes = None) -> int:
    """
    Parses text from a local PDF file, remote PDF URL, or raw bytes and adds it to the case's vector store.
    """
    import pypdf
    import io
    filename = os.path.basename(pdf_path)
    try:
        if file_bytes:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        elif pdf_path.startswith("http://") or pdf_path.startswith("https://"):
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
    Uses Nvidia Vision API to generate an image description,
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
