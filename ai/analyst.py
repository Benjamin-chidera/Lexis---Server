"""
analyst.py

Node 1 of the Analyst → Researcher → Strategist pipeline.

The Analyst acts as a hostile Discovery Compliance Auditor.
Its only job is to find contradictions, policy violations, and admissions
DIRECTLY quoted from the case vault documents (PDFs, images, URLs).

Anti-Hallucination Rule: If no document supports a finding, it says so.
It never invents quotes or references documents it hasn't read.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from .model_providers import get_chat_model

# The Analyst's persona and strict operating rules
ANALYST_SYSTEM_PROMPT = """You are a Discovery Compliance Auditor working for the plaintiff's legal team.

Your ONLY job is to find WEAPONS in documents — contradictions, policy violations, and damaging admissions.

STRICT RULES you must follow:
1. Read every document excerpt provided to you carefully and with suspicion.
2. For each finding, you MUST provide:
   - A DIRECT QUOTE from the document (copy the exact words)
   - The FILENAME or SOURCE of that quote (e.g., "contract.pdf", "https://example.com")
   - The CONTRADICTION: What does this quote contradict? (e.g., another policy claim, a legal standard)
3. Do NOT summarize what documents say in general terms. Find the EXCEPTION, the LOOPHOLE, the ADMISSION.
4. If a document says "We protect your data", search the same document for technical exceptions that undo that promise.
5. Look specifically for:
   - Unilateral modification clauses (they can change terms without notice)
   - Data sharing with "third parties" or "affiliates" that contradicts privacy claims
   - Liability limitation clauses that cap compensation unfairly
   - Arbitration clauses that waive the right to sue
   - Vague definitions of key terms that could be exploited against the user
   - Any admission that they have done, or may do, the harmful thing the case is about

ANTI-HALLUCINATION PROTOCOL:
- You are FORBIDDEN from mentioning any document you were not given.
- You are FORBIDDEN from inventing quotes. Every quote must be verbatim from the text provided.
- If you cannot find a direct quote to support a finding, you MUST NOT include that finding.
- If no vulnerabilities are found, you MUST state: "No documentary vulnerabilities identified."

OUTPUT FORMAT:
For each vulnerability found:
## Vulnerability #[N]: [Short Title]
- **Source:** [Filename or URL]
- **Direct Quote:** "[exact verbatim text from the document]"
- **Why This Is a Weapon:** [One clear sentence explaining the legal contradiction or risk]

If nothing found:
## No documentary vulnerabilities identified.
No direct quotes supporting a legal vulnerability could be extracted from the provided documents.
"""


def run_analyst(vault_chunks: list, user_query: str) -> str:
    """
    Runs the Discovery Compliance Auditor against the vault documents.

    Args:
        vault_chunks: List of text chunks retrieved from the vector store.
                      Each chunk is formatted as "[Source: ...]\nchunk text"
        user_query:   The user's question — used to focus the audit on what matters.

    Returns a formatted string of vulnerabilities found, or a "none found" statement.
    """
    if not vault_chunks:
        return "No documentary vulnerabilities identified.\n\nReason: No documents are present in the case vault."

    # Combine all vault chunks into one evidence block for the auditor
    evidence_block = "\n\n---\n\n".join(vault_chunks)

    # Lower temperature = more precise, less creative (good for auditing)
    llm = get_chat_model(temperature=0)

    messages = [
        SystemMessage(content=ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"CASE QUESTION (what the attorney is trying to prove or defend against):\n"
            f"{user_query}\n\n"
            f"=== DOCUMENTS TO AUDIT ===\n\n"
            f"{evidence_block}"
        )),
    ]

    response = llm.invoke(messages)
    findings = response.content.strip()

    # Strip any reasoning traces the model might output
    import re
    findings = re.sub(r"<think>.*?</think>", "", findings, flags=re.DOTALL).strip()

    print(f"[analyst] Completed audit. Findings length: {len(findings)} chars")
    return findings
