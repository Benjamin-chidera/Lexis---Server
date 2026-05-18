"""
state.py

Defines the data structure that flows through every node in the LangGraph pipeline.

Each node reads what it needs from the state and writes its outputs back in.
TypedDict gives us type hints without the overhead of a real class.
"""

from typing import TypedDict, List, Dict, Optional, Any, TYPE_CHECKING


class CaseState(TypedDict):
    # ── Input fields (set at the start of every run) ─────────────────────────

    # The database ID of the case being processed
    case_id: int

    # The user's text description of the case (set when the case was created)
    context: str

    # Raw file paths and URLs stored in the DB
    pdf_paths: List[str]
    urls: List[str]
    image_paths: List[str]  # NEW — image files uploaded to the vault

    # The full conversation history so the AI has context
    # Each item: {"role": "user" | "ai", "content": "..."}
    chat_history: List[Dict[str, str]]

    # The user's latest message that triggered this run
    current_query: str

    # ── Routing result (set by the route_query node) ──────────────────────────

    # One of: "vault", "web", "both"
    # Decides which retrieval path(s) to take
    query_route: str

    # ── Retrieval results (set by retrieval nodes) ────────────────────────────

    # Chunks retrieved from the Chroma vector store
    # Each item is a formatted string: "[Source: ...]\nchunk text"
    vault_chunks: List[str]

    # Text retrieved from a live web search (Tavily)
    web_results: str

    # ── Analyst output (set by analyst_node) ──────────────────────────────────

    # Vulnerabilities and direct quotes found in vault documents
    analyst_findings: str

    # ── Researcher output (set by researcher_node) ────────────────────────────

    # Real-world lawsuits, FTC fines, and precedents found via adversarial web search
    researcher_findings: str

    # ── Output fields (set by the strategist node) ────────────────────────────

    # The AI's final IRAC-structured answer
    response: str

    # Optional citation pointing to the source used
    citation: Optional[Dict[str, Any]]

    # ── Runtime helper (not persisted) ───────────────────────────────────────

    # ProgressEmitter instance for real-time stage WebSocket events.
    # Typed as Any to avoid circular imports with chat_handler.
    emitter: Any
