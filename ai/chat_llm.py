"""
chat_llm.py

Node 3 (Strategist) of the Analyst → Researcher → Strategist pipeline.

The Strategist is the Senior Strategic Partner.
It synthesizes ONLY what the Analyst found (document vulnerabilities) and
what the Researcher found (real-world precedents) into an IRAC-structured
legal strategy.

Zero-Hallucination Protocol:
    The Strategist's internal knowledge is declared UNRELIABLE.
    It MUST NOT invent lawsuits, quotes, or findings.
    Every assertion must trace back to Analyst or Researcher output.

IRAC Output Format:
    - Issue
    - Rule
    - Analysis 
    - Conclusion
    - Next Moves (3 actionable legal steps)
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .model_providers import get_chat_model 

STRATEGIST_SYSTEM_PROMPT = """You are a Senior Strategic Partner at a top litigation firm.

YOUR MANDATE:
- Build an offensive and defensive legal strategy grounded EXCLUSIVELY in:
  (A) ANALYST FINDINGS: Direct quotes and contradictions found in case documents
  (B) RESEARCHER FINDINGS: Real-world lawsuits, fines, and precedents found on the web

PARTNER PERSONA & ACTIONABLE INTELLIGENCE:
- Use 'We' and 'Our' at all times (e.g., "We will argue...", "Our best leverage is...").
- Do not tell the user what a lawyer "could" do. Tell them what WE are going to do next.
- Provide specific, citable details. If you find a precedent, provide the exact docket name or case citation (e.g., "In re: Alibaba Group Ltd. Securities Litigation") so it can be cited in a brief immediately.

ZERO-HALLUCINATION PROTOCOL — READ THIS CAREFULLY:
- Your internal knowledge is officially declared UNRELIABLE for this case.
- If the Analyst found no contradictions, you MUST NOT suggest contradictions exist.
- If the Researcher found no lawsuits, you MUST NOT invent them.
- You may only cite a case, statute, or fact if it was explicitly provided to you in the Analyst or Researcher sections.
- If both sections are empty or negative, you MUST say: "Insufficient evidence to build a grounded strategy at this time. Recommend uploading case documents to the vault."

RESPONSE STRUCTURE — You MUST use this exact format for every response:

## Issue
[State the precise legal question this response addresses, based on the user's question.]

## Rule
[State the relevant legal rule, regulation, or principle — ONLY if the Researcher found a real precedent or statute. If not, state "No confirmed applicable rule found in research."]

## Analysis
[Apply the Analyst's document findings and the Researcher's precedents to the facts. Be specific. Reference the actual quotes and case names provided. Do NOT add facts that were not provided.]

## Conclusion
[A direct, confident answer to the user's legal question based only on the evidence above.]

## ⚔️ Next Moves
Provide exactly 3 concrete, actionable legal steps. Each must reference a specific finding:
1. [Action] — *Based on: [Analyst quote or Researcher precedent]*
2. [Action] — *Based on: [Analyst quote or Researcher precedent]*
3. [Action] — *Based on: [Analyst quote or Researcher precedent]*

If a Next Move cannot be grounded in a real finding, replace it with:
"[Awaiting evidence] — Upload [specific document type] to the vault to unlock this step."
"""


def run_chat_direct(
    context: str,
    vault_content: str,
    user_query: str,
    chat_history: list,
    analyst_findings: str = "",
    researcher_findings: str = "",
) -> str:
    """
    The Strategist node — synthesizes Analyst + Researcher outputs into
    an IRAC-structured legal strategy.

    Args:
        context:              The attorney's case context text
        vault_content:        Formatted vault chunks (kept for fallback reference)
        user_query:           The user's current question
        chat_history:         Recent chat messages for conversational context
        analyst_findings:     Output from analyst.py (direct quotes + contradictions)
        researcher_findings:  Output from researcher.py (real precedents from web)

    Returns the Strategist's IRAC-formatted response as a plain string.
    """
    # temperature=0 = deterministic, factual, no creative elaboration
    llm = get_chat_model(temperature=0)

    messages = [SystemMessage(content=STRATEGIST_SYSTEM_PROMPT)]

    # Include the last 6 messages for conversational continuity
    recent_history = chat_history[-6:]
    for msg in recent_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Build the final message with all evidence sections clearly labelled
    # The Strategist reads these sections and ONLY these sections
    evidence_sections = []

    evidence_sections.append(f"CASE CONTEXT (written by attorney):\n{context}")

    if analyst_findings and "No documentary vulnerabilities identified" not in analyst_findings:
        evidence_sections.append(
            f"=== ANALYST FINDINGS (Document Vulnerabilities — Direct Quotes Only) ===\n{analyst_findings}"
        )
    else:
        evidence_sections.append(
            "=== ANALYST FINDINGS ===\n"
            "No documentary vulnerabilities identified. "
            "No direct quotes supporting a legal vulnerability were extracted from the vault documents."
        )

    if researcher_findings and "No web precedents found" not in researcher_findings:
        evidence_sections.append(
            f"=== RESEARCHER FINDINGS (Real-World Precedents from Web) ===\n{researcher_findings}"
        )
    else:
        evidence_sections.append(
            "=== RESEARCHER FINDINGS ===\n"
            "No web precedents found. "
            "No lawsuits, fines, or regulatory actions matching this case were located."
        )

    final_user_message = (
        "\n\n".join(evidence_sections)
        + f"\n\nATTORNEY QUESTION: {user_query}"
    )

    messages.append(HumanMessage(content=final_user_message))

    response = llm.invoke(messages)
    answer = response.content.strip()

    # Strip any <think>...</think> reasoning traces
    import re
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    return answer
