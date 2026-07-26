"""
researcher.py

Node 2 of the Analyst → Researcher → Strategist pipeline.

The Researcher is an Adversarial Search Hunter.
It takes what the Analyst found in the documents and searches the web
for REAL precedents — lawsuits, FTC fines, GDPR violations, class actions —
where the same opponent lost or settled for the same behaviour.

It does NOT search for generic information.
Every search query is designed to find legal leverage.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _build_adversarial_queries(context: str, analyst_findings: str) -> list:
    """
    Builds a list of aggressive, targeted search queries designed to find
    legal precedents and litigation history against the opponent.

    Strategy: Extract the opponent's name / company from context,
    then append litigation trigger terms to each query.
    """
    # Litigation trigger terms — these are the phrases that find lawsuits
    trigger_terms = [
        "lawsuit settlement",
        "FTC fine penalty",
        "GDPR violation fine",
        "class action lawsuit",
        "data breach penalty",
        "privacy policy violation lawsuit",
        "regulatory action",
        "consumer protection lawsuit",
    ]

    # We send the context + findings to build smart queries
    # But we also build some from raw trigger terms to guarantee coverage
    queries = []

    # Build queries using each trigger term combined with the first 200 chars of context
    # (The context usually contains the opponent name early on)
    short_context = context[:200].strip()

    for trigger in trigger_terms:
        query = f"{short_context} {trigger}"
        # Trim to a reasonable search length (Tavily works best under 300 chars)
        queries.append(query[:300])

    # Also add a direct query based on the analyst findings if they found something
    if "No documentary vulnerabilities" not in analyst_findings:
        # Build a query from the findings to find related precedents
        short_findings = analyst_findings[:200].strip()
        queries.append(f"{short_findings} legal precedent lawsuit")

    return queries


def run_researcher(context: str, analyst_findings: str) -> str:
    """
    Runs adversarial web searches to find real legal precedents.

    Args:
        context:          The attorney's case context (contains opponent name/details)
        analyst_findings: Output from the Analyst node (documented vulnerabilities)

    Returns a formatted string of real-world precedents found, or a clear
    "no precedents found" statement.
    """
    # Try to import Tavily — if not available, return a clear message
    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        return (
            "Researcher: Web search unavailable.\n\n"
            "Reason: langchain-tavily is not installed. "
            "Run: uv add langchain-tavily"
        )

    # Check API key is present
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_api_key:
        return (
            "Researcher: Web search unavailable.\n\n"
            "Reason: TAVILY_API_KEY is not set in the .env file."
        )

    # Build the adversarial search queries
    queries = _build_adversarial_queries(context, analyst_findings)

    # Run searches — we limit to the first 3 queries to keep it fast
    # Each query costs an API call, so we pick the most useful ones
    queries_to_run = queries[:3]

    print(f"[researcher] Running {len(queries_to_run)} adversarial searches...")

    all_results = []

    for query in queries_to_run: 
        try:
            search_tool = TavilySearch(
                max_results=2,             # 2 results per query is enough
                search_depth="advanced",   # Deeper search for legal content
                include_domains=[
                    "bailii.org",          # UK & Irish case law (gold standard)
                    "legislation.gov.uk",  # Official UK statutes
                    "scotcourts.gov.uk",   # Scottish court judgments
                    "gov.uk",              # UK government regulatory guidance
                ],
            )
            raw_results = search_tool.invoke(query)

            if raw_results:
                # Format the results clearly so the Strategist can read them
                all_results.append(f"### Search: \"{query}\"\n{raw_results}")

        except Exception as error:
            print(f"[researcher] Search failed for query '{query[:60]}': {error}")
            continue

    if not all_results:
        return (
            "No web precedents found.\n\n"
            "Reason: All adversarial search queries returned no results. "
            "The researcher could not locate lawsuits or regulatory actions "
            "matching the case context."
        )

    # Combine all search results into one block for the Strategist
    combined = "\n\n---\n\n".join(all_results)
    print(f"[researcher] Research complete. Total results length: {len(combined)} chars")
    return combined
