"""
graph.py

Defines the 3-node LangGraph pipeline for the chat feature:

  START → [analyst_node + researcher_node] (PARALLEL) → strategist_node → END

Node Roles:
  analyst_node    - Discovery Compliance Auditor: finds direct quotes and contradictions
                    in vault documents (PDFs, images, URLs).
  researcher_node - Adversarial Search Hunter: runs targeted Tavily queries to find
                    real lawsuits, FTC fines, GDPR violations, and class actions.
  strategist_node - Senior Strategic Partner: synthesizes Analyst + Researcher output
                    into an IRAC-structured response with 3 actionable Next Moves.

Performance Note:
  Analyst and Researcher run IN PARALLEL because they don't depend on each other.
  This cuts total pipeline time from ~3 minutes to ~90 seconds on local hardware.
"""

from langgraph.graph import StateGraph, START, END

from .state import CaseState
from .vector_store import search_vector_store
from .analyst import run_analyst
from .researcher import run_researcher
from .chat_llm import run_chat_direct  


def analyst_node(state: CaseState) -> dict:
    """
    Node 1a (runs in parallel with researcher_node). 

    Discovery Compliance Auditor:
    Searches the vault for the most relevant document chunks,
    then audits them for contradictions, policy violations, and admissions.
    Only quotes directly from documents — never invents findings.
    """
    case_id = state.get("case_id")
    user_query = state.get("current_query", "")
    emitter = state.get("emitter")
    pdf_paths = state.get("pdf_paths", [])
    urls = state.get("urls", [])
    image_paths = state.get("image_paths", []) 

    # Emit progress to the client so the UI shows the AI is working
    if emitter:
        emitter.emit_stage("analyst", "🔍 Auditing case documents for vulnerabilities...")

    print(f"[graph] analyst_node: Searching vault for case {case_id}")

    # Pull the most relevant chunks from the vector store
    chunks = search_vector_store(case_id, user_query, top_k=5)

    # If the vector store is empty but the case has files (uploaded from the home page
    # to Cloudinary), ingest them on-demand so they are immediately searchable without
    # requiring a separate vault upload.
    if not chunks and (pdf_paths or urls or image_paths):
        print(f"[graph] analyst_node: Vector store empty — ingesting case files on-demand for case {case_id}")
        from .vector_store import (
            ingest_pdf_into_vector_store,
            ingest_url_into_vector_store,
            ingest_image_into_vector_store,
        )

        for pdf_path in pdf_paths:
            ingest_pdf_into_vector_store(case_id, pdf_path)

        for url in urls:
            ingest_url_into_vector_store(case_id, url)

        for img_path in image_paths:
            ingest_image_into_vector_store(case_id, img_path)

        chunks = search_vector_store(case_id, user_query, top_k=5)

    if not chunks:
        print(f"[graph] analyst_node: No vault documents found for case {case_id}")

    # Run the Analyst against the retrieved chunks
    findings = run_analyst(chunks, user_query)

    print(f"[graph] analyst_node: Complete ({len(findings)} chars)")

    return {
        "vault_chunks": chunks,
        "analyst_findings": findings,
    }


def researcher_node(state: CaseState) -> dict:
    """
    Node 1b (runs in parallel with analyst_node).

    Adversarial Search Hunter:
    Takes the case context and builds aggressive Tavily queries using
    litigation trigger terms to find real-world precedents where the
    opponent lost or settled.
    """
    context = state.get("context", "")
    emitter = state.get("emitter")

    # Emit progress to the client
    if emitter:
        emitter.emit_stage("researcher", "🌐 Hunting for lawsuits, fines, and precedents...")

    print("[graph] researcher_node: Running adversarial web searches")

    # Run the adversarial Tavily searches
    # Pass empty string for analyst_findings since we run in parallel
    researcher_findings = run_researcher(context, "")

    print(f"[graph] researcher_node: Complete ({len(researcher_findings)} chars)")

    return {
        "researcher_findings": researcher_findings,
    }


def strategist_node(state: CaseState) -> dict:
    """
    Node 2 — Senior Strategic Partner (runs AFTER both analyst + researcher finish).

    Reads ONLY the Analyst findings and Researcher findings.
    Synthesizes them into an IRAC-structured response with 3 Next Moves.
    Will not add any fact that was not provided by the upstream nodes.
    """
    case_id = state.get("case_id")
    context = state.get("context", "")
    user_query = state.get("current_query", "")
    vault_chunks = state.get("vault_chunks", [])
    chat_history = state.get("chat_history", [])
    analyst_findings = state.get("analyst_findings", "")
    researcher_findings = state.get("researcher_findings", "")
    emitter = state.get("emitter")

    # Emit progress to the client
    if emitter:
        emitter.emit_stage("strategist", "⚖️ Building your legal strategy...")

    print("[graph] strategist_node: Synthesizing findings into IRAC response")

    # Build the vault content string for fallback reference
    if vault_chunks:
        vault_text = "\n\n".join(vault_chunks)
        vault_content = f"=== Vault Document Chunks ===\n{vault_text}"
    else:
        vault_content = "=== Vault ===\nNo documents in vault."

    # Call the Strategist with all evidence from upstream nodes
    response_text = run_chat_direct(
        context=context,
        vault_content=vault_content,
        user_query=user_query,
        chat_history=chat_history,
        analyst_findings=analyst_findings,
        researcher_findings=researcher_findings,
        case_id=case_id,
    )

    # Build citation from first vault chunk, or fall back to web research
    citation = None
    if vault_chunks:
        first_line = vault_chunks[0].split("\n")[0]
        raw_source = first_line.strip("[]")
        filename = raw_source.replace("Source:", "").replace("source:", "").strip()
        citation = {
            "filename": filename if filename else "Case Vault",
            "exhibit": "Case Vault Document",
        }
    elif researcher_findings and "No web precedents" not in researcher_findings:
        citation = {
            "filename": "Web Research",
            "exhibit": "External Sources",
        }

    print("[graph] strategist_node: Complete")

    return {
        "response": response_text,
        "citation": citation,
    }


def build_chat_graph():
    """
    Compiles the full LangGraph pipeline.

    Flow (parallel fan-out, then fan-in):
      START ──┬── analyst_node    ──┬── strategist_node ── END
              └── researcher_node ──┘
    """
    graph = StateGraph(CaseState)

    # Register all three nodes
    graph.add_node("analyst", analyst_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("strategist", strategist_node)

    # Fan-out: START triggers BOTH analyst and researcher in parallel
    graph.add_edge(START, "analyst")
    graph.add_edge(START, "researcher")

    # Fan-in: Both must finish before strategist runs
    graph.add_edge("analyst", "strategist")
    graph.add_edge("researcher", "strategist")

    # Strategist is the final node
    graph.add_edge("strategist", END)

    return graph.compile()


# Compiled once at import time — reused for every chat message
chat_graph = build_chat_graph()
