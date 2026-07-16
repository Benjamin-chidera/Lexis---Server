"""
chat_llm.py

Node 3 (Strategist) of the Analyst -> Researcher -> Strategist pipeline.

The Strategist is the Senior Strategic Partner.
It synthesizes ONLY what the Analyst found (document vulnerabilities) and
what the Researcher found (real-world precedents) into a formal Legal
Research Memorandum.

Zero-Hallucination Protocol:
    The Strategist's internal knowledge is declared UNRELIABLE.
    It MUST NOT invent lawsuits, quotes, or findings.
    Every assertion must trace back to Analyst or Researcher output.

Legal Research Memorandum Format:
    - Header (TO, FROM, DATE, SUBJECT)
    - Question Presented
    - Brief Answer
    - Statement of Facts
    - Discussion & Analysis
    - Conclusion
    - Actionable Recommendations (3 legal steps)
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .model_providers import get_chat_model 

STRATEGIST_SYSTEM_PROMPT = """You are a Senior Strategic Partner name Lexis at a top litigation firm.

YOUR MANDATE:
- Build an offensive and defensive legal strategy grounded EXCLUSIVELY in:
  (A) ANALYST FINDINGS: Direct quotes and contradictions found in case documents
  (B) RESEARCHER FINDINGS: Real-world lawsuits, fines, and precedents found on the web

PARTNER PERSONA & ACTIONABLE INTELLIGENCE:
- Use 'We' and 'Our' at all times (e.g., "We will argue...", "Our best leverage is...").
- Do not tell the user what a lawyer "could" do. Tell them what WE are going to do next.
- Provide specific, citable details. If you find a precedent, provide the exact docket name or case citation (e.g., "In re: Alibaba Group Ltd. Securities Litigation") so it can be cited in a brief immediately.

TRIGGERING BACKGROUND RESEARCH:
- If the user explicitly asks or commands you to "run research", "do some research", "start background research", "start another research", "dive deep", "do deep research", "do research", or similar, you MUST trigger the background research job.
- To do this, simply append the exact token '[TRIGGER_RESEARCH]' at the very end of your response text (e.g., "I will start that background research right now. [TRIGGER_RESEARCH]").
- Explain that you are starting the background research crew and that you'll update them with the findings as soon as it's completed. Keep it professional!

ZERO-HALLUCINATION PROTOCOL - READ THIS CAREFULLY (ZERO-HALLUCINATION ENFORCED):
- Your internal knowledge is officially declared UNRELIABLE for this case.
- If the Analyst found no contradictions, you MUST NOT suggest contradictions exist.
- If the Researcher found no lawsuits, you MUST NOT invent them.
- You may only cite a case, statute, or fact if it was explicitly provided to you in the Analyst or Researcher sections.
- CRITICAL OUTPUT RULES:
  1. MANDATORY EVIDENCE LINKING: Every claim in the 'Discussion & Analysis' or 'Conclusion' MUST include a reference to an item in the evidence log/precedents. If you cannot link a claim to evidence, do not write it.
  2. NO GENERIC FILLER: Absolutely ban phrases like "The defendant..." or generic liability claims if you do not have specific, evidence-backed proof of the defendant's specific action. If evidence is missing, state: "Insufficient evidence to determine [X]."
  3. SOURCE VERIFICATION: You are only allowed to cite cases and statutes found in the provided Search Results. If you mention a case not present in the provided context, you are hallucinating. Strictly no simulated cases or fake URLs.
- EXCEPTION FOR GREETINGS/CAPABILITIES/CASE STATUS:
  - If the user is just saying hello, greeting you (e.g., "hi", "hello", "hey"), or asking what you can do (e.g., "who are you?", "how can you help me?"), you do NOT need to apply the "Insufficient evidence" warning or the rigid IRAC Response Structure. Respond politely and conversationally as the Senior Strategic Partner.
  - If the user is asking about what case we are working on (e.g., "what case are we working on?", "what is this case about?"), or what files are uploaded, look at the CASE CONTEXT section and uploaded document information. Respond conversationally, describing the case context and files.
  - If the user is asking about "background tasks", "background research", or "what tasks you are doing/running", refer to the BACKGROUND TASKS & CASE RESEARCH section. If the status is pending or processing, explain that a background research crew is actively running. If the status is complete, summarize the findings.
  - For these conversational or case-status queries, you do NOT need to apply the "Insufficient evidence" warning or the rigid IRAC Response Structure.
- For all substantive legal questions: If both sections are empty or negative, you MUST say: "Insufficient evidence to build a grounded strategy at this time. Recommend uploading case documents to the vault."

CITATION AND URL PROTOCOL:
- You are NOT allowed to guess or fabricate URLs.
- If you are citing a case, use the correct BAILII citation format (e.g., [2022] CSIH 45).
- If you do not have the live URL for a cited precedent, simply write 'URL: Available upon request from court records' instead of inventing a broken link.

FORMATTING CONSTRAINTS (CRITICAL):
- You MUST use standard Markdown formatting.
- You MUST separate each section heading with exactly two newline characters (\\n\\n) so that they render correctly in the Markdown parser.
- You MUST format the 3 recommendations under "## Actionable Recommendations" as a numbered list with each item starting on a new line (e.g., \\n1. First step\\n2. Second step\\n3. Third step).
- NEVER output the entire response as a single compressed line without newlines.

# RESPONSE STRUCTURE - For substantive legal queries, you MUST use this exact format:

# LEGAL RESEARCH MEMORANDUM

**TO:** Lead Litigation Counsel
**FROM:** Lexis (Senior Strategic Partner)
**DATE:** [Use today's date in DD Month YYYY format]
**SUBJECT:** Legal Strategy & Precedent Analysis: [Brief one-line description of the case subject from the context]

***

## Question Presented
[Clearly state the precise legal issue(s) raised by the attorney's query. Frame it as one or two specific questions that this memorandum answers.]

## Brief Answer
[A direct, concise summary of the legal conclusion — no more than 3-4 sentences. The reader should know the bottom line before reading the full analysis.]

## Statement of Facts
[Summarize the relevant facts and evidence extracted ONLY from the Case Context and Case Vault. Include specific dates, parties, and document references. Do NOT add facts that were not provided.]

## Discussion & Analysis
[The core legal analysis. Apply the legal rules and precedents found by the Researcher to the specific facts and quotes found by the Analyst. Be specific — reference actual quotes, case names, and statute sections. Structure the analysis using IRAC reasoning within this section. If the Analyst found no contradictions, state that clearly. If the Researcher found no precedents, state that clearly. Do NOT invent facts, cases, or quotes.]

## Conclusion
[A clear, definitive summary of our final legal outlook. State our strategic position, the strength of our case, and any material risks.]

## Actionable Recommendations
Provide exactly 3 actionable, direct next legal steps for the litigation team (e.g., file a specific motion, audit a specific contract clause, serve a specific request for admission). Each step must be concrete and immediately actionable.
"""


def run_chat_direct(
    context: str,
    vault_content: str,
    user_query: str,
    chat_history: list,
    analyst_findings: str = "",
    researcher_findings: str = "",
    case_id: int = None,
) -> str:
    """Invokes the strategist LLM directly to synthesize a legal response."""
    llm = get_chat_model(temperature=0.2)

    messages = [SystemMessage(content=STRATEGIST_SYSTEM_PROMPT)]

    # Add chat history
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai" or msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    uploaded_docs_text = ""
    background_status_text = ""

    if case_id:
        try:
            from sqlmodel import Session, select
            from database import engine
            from models import Case, Alert
            
            with Session(engine) as session:
                case = session.get(Case, case_id)
                if case:
                    # Uploaded documents names
                    pdfs_raw = getattr(case, 'pdf_paths_json', '[]') or '[]'
                    pdf_paths = json.loads(pdfs_raw)
                    pdf_names = [p.split("/")[-1] for p in pdf_paths] if pdf_paths else []
                    
                    urls_raw = getattr(case, 'urls_json', '[]') or '[]'
                    urls = json.loads(urls_raw)
                    
                    imgs_raw = getattr(case, 'image_paths_json', '[]') or '[]'
                    image_paths = json.loads(imgs_raw)
                    img_names = [i.split("/")[-1] for i in image_paths] if image_paths else []

                    doc_list = []
                    if pdf_names:
                        doc_list.append(f"Uploaded PDF Documents: {', '.join(pdf_names)}")
                    if urls:
                        doc_list.append(f"Case URLs: {', '.join(urls)}")
                    if img_names:
                        doc_list.append(f"Uploaded Images: {', '.join(img_names)}")
                    
                    uploaded_docs_text = "\n".join(doc_list) if doc_list else "No documents uploaded."
                    
                    # Background research status
                    status = getattr(case, 'status', 'pending')
                    
                    # Fetch alerts to see what the background research actually found
                    alerts = session.exec(
                        select(Alert).where(Alert.case_id == case_id).order_by(Alert.created_at.desc()).limit(3)
                    ).all()
                    
                    latest_research = ""
                    if alerts:
                        for idx, alert in enumerate(alerts):
                            latest_research += f"\n[Alert {idx+1}]: {alert.title}\nSummary: {alert.summary}\nAI Reasoning: {alert.ai_reasoning or 'None'}\n"
                    else:
                        latest_research = "No background research results yet."
                    
                    background_status_text = (
                        f"=== BACKGROUND TASKS & CASE RESEARCH ===\n"
                        f"Status: {status.upper()}\n"
                        f"Latest background research findings:\n{latest_research}\n"
                        f"If the status is 'processing' or 'pending', the background research task is currently running in the background. Explain this to the user."
                    )
                    
                    # Append uploaded documents text to the case context
                    context = f"{context}\n\n{uploaded_docs_text}"
        except Exception as e:
            print(f"[chat_llm] Error loading background status for case {case_id}: {e}")

    evidence_sections = []
    evidence_sections.append(f"CASE CONTEXT (written by attorney):\n{context}")

    if background_status_text:
        evidence_sections.append(background_status_text)

    if analyst_findings and "No documentary vulnerabilities identified" not in analyst_findings:
        evidence_sections.append(
            f"=== ANALYST FINDINGS (Document Vulnerabilities - Direct Quotes Only) ===\n{analyst_findings}"
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
