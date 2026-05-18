"""
vector_store.py

Manages a per-case Chroma vector store using LangChain's Chroma integration.

Every case gets its own isolated collection named "case_{case_id}".
Documents are split into chunks and embedded using Ollama's nomic-embed-text
model (or a fallback sentence-transformers model if Ollama isn't available).

Key functions:
  - ingest_pdf_into_vector_store(case_id, pdf_path)
  - ingest_url_into_vector_store(case_id, url)
  - ingest_image_into_vector_store(case_id, image_path)
  - search_vector_store(case_id, query, top_k) -> list of chunk strings
"""

import os
import re
import requests as http_requests

# Disable ChromaDB anonymous telemetry to silence "capture()" errors in the logs
os.environ["ANONYMOUS_TELEMETRY"] = "False"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from .model_providers import get_embeddings

# Where Chroma will store its data on disk
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# How large each chunk is and how much they overlap
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# Replaced by .providers.get_embeddings
# def get_embeddings(): ...



def get_collection_name(case_id: int) -> str:
    """Each case gets its own isolated Chroma collection."""
    return f"case_{case_id}"


def get_vector_store(case_id: int) -> Chroma:
    """
    Opens (or creates) the Chroma collection for a specific case.
    Data is persisted to disk so it survives server restarts.
    """
    collection_name = get_collection_name(case_id)
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    return vector_store


def split_text_into_chunks(text: str, source_label: str) -> list:
    """
    Splits a large block of text into smaller overlapping chunks.
    Each chunk gets a metadata dict so we can trace it back to its source.

    Returns a list of LangChain Document objects.
    """
    from langchain_core.documents import Document

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source_label}],
    )

    return chunks


def ingest_pdf_into_vector_store(case_id: int, pdf_path: str) -> int:
    """
    Reads a PDF from disk, splits its text into chunks, and stores
    them in the case's Chroma collection.

    Returns the number of chunks added, or 0 if extraction failed.
    """
    from .ingestion import extract_text_from_pdf

    filename = os.path.basename(pdf_path)
    print(f"[vector_store] Ingesting PDF '{filename}' for case {case_id}")

    raw_text = extract_text_from_pdf(pdf_path)

    # If extraction failed (error string), don't store junk
    if raw_text.startswith("["):
        print(f"[vector_store] PDF extraction failed: {raw_text}")
        return 0

    chunks = split_text_into_chunks(raw_text, source_label=f"PDF: {filename}")

    if not chunks:
        return 0

    vector_store = get_vector_store(case_id)
    vector_store.add_documents(chunks)

    print(f"[vector_store] Added {len(chunks)} chunks from '{filename}'")
    return len(chunks)


def ingest_url_into_vector_store(case_id: int, url: str) -> int:
    """
    Fetches a web page, strips HTML, splits into chunks, and stores
    them in the case's Chroma collection.
    Also generates a Markdown summary and saves it to the database.

    Returns the number of chunks added, or 0 if fetch failed.
    """
    from .ingestion import extract_text_from_url
    from .summarizer import generate_url_summary
    import json

    print(f"[vector_store] Ingesting URL '{url}' for case {case_id}")

    raw_text = extract_text_from_url(url)

    if raw_text.startswith("["):
        print(f"[vector_store] URL extraction failed: {raw_text}")
        # Even if it failed, we can generate an error summary to show the user
        summary = generate_url_summary(url, raw_text)
        chunks = []
    else:
        chunks = split_text_into_chunks(raw_text, source_label=f"URL: {url}")
        summary = generate_url_summary(url, raw_text)

        if chunks:
            vector_store = get_vector_store(case_id)
            vector_store.add_documents(chunks)
            print(f"[vector_store] Added {len(chunks)} chunks from URL '{url}'")

    # Save the summary to the database
    try:
        from database import engine
        from sqlmodel import Session
        from models import Case

        with Session(engine) as session:
            case = session.get(Case, case_id)
            if case:
                summaries_raw = getattr(case, 'url_summaries_json', '{}') or '{}'
                summaries = json.loads(summaries_raw)
                summaries[url] = summary
                case.url_summaries_json = json.dumps(summaries)
                session.add(case)
                session.commit()
                print(f"[vector_store] Saved summary for URL '{url}' to database")
    except Exception as e:
        print(f"[vector_store] Failed to save summary to database: {e}")

    return len(chunks)


def ingest_image_into_vector_store(case_id: int, image_path: str) -> int:
    """
    Uses the llava model to describe an image, then stores that description
    as text chunks in the vector store. This lets the AI answer questions
    about image content (diagrams, scanned exhibits, photographs, etc.).

    Returns the number of chunks added, or 0 if description failed.
    """
    from .image_handler import describe_image

    filename = os.path.basename(image_path)
    print(f"[vector_store] Describing image '{filename}' with llava for case {case_id}")

    description = describe_image(image_path)

    if description.startswith("[Error") or description.startswith("[Image"):
        print(f"[vector_store] Image description failed: {description}")
        return 0

    # Prefix so the LLM knows this text came from an image, not written text
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

    Returns a list of plain text strings — the most relevant chunks —
    each prefixed with its source label so the LLM can cite it.

    Returns an empty list if the collection doesn't exist yet.
    """
    try:
        vector_store = get_vector_store(case_id)

        results = vector_store.similarity_search(query, k=top_k)

        if not results:
            return []

        # Format each chunk as "SOURCE:\nCONTENT" for easy LLM reading
        formatted_chunks = []
        for doc in results:
            source = doc.metadata.get("source", "Unknown source")
            chunk_text = f"[{source}]\n{doc.page_content}"
            formatted_chunks.append(chunk_text)

        return formatted_chunks

    except Exception as error:
        print(f"[vector_store] Search failed for case {case_id}: {str(error)}")
        return []
