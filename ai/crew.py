from pydantic import BaseModel, Field, field_validator
from typing import List
from crewai import Agent, Task, Crew, Process
from .model_providers import get_crew_llm
import litellm


# ---------------------------------------------------------------------------
# Mistral Message Order Fix
#
# Mistral's API strictly requires:
#   1. Messages must alternate roles (user -> assistant -> user -> ...).
#   2. The LAST message must have role "user" or "tool".
#
# CrewAI's internal tool-calling loop sometimes produces consecutive
# messages with the same role (e.g. two "assistant" messages in a row),
# which causes a 400 error: "Expected last role User or Tool".
#
# We monkey-patch both litellm.completion and litellm.acompletion
# to sanitize messages before they reach the Mistral API.
# ---------------------------------------------------------------------------
def _fix_messages(messages: list) -> list:
    """
    Return a sanitized copy of the message list that satisfies Mistral's
    strict role-alternation rules:
      - Merge consecutive same-role messages (except 'tool' messages,
        which carry distinct tool_call_ids and must stay separate).
      - Ensure the final message is never role='assistant'.
    """
    if not messages or len(messages) < 2:
        return messages

    merged = []
    for msg in messages:
        msg_copy = dict(msg)
        if not merged:
            merged.append(msg_copy)
            continue

        last_msg = merged[-1]
        # Never merge 'tool' messages — each has a unique tool_call_id
        if msg_copy["role"] == last_msg["role"] and msg_copy["role"] != "tool":
            content1 = last_msg.get("content") or ""
            content2 = msg_copy.get("content") or ""
            last_msg["content"] = (str(content1) + "\n\n" + str(content2)).strip()

            # Preserve any tool_calls from the merged message
            if "tool_calls" in msg_copy:
                if "tool_calls" not in last_msg:
                    last_msg["tool_calls"] = []
                calls = msg_copy["tool_calls"]
                if isinstance(calls, list):
                    last_msg["tool_calls"].extend(calls)
        else:
            merged.append(msg_copy)

    # If the last message is 'assistant', Mistral will reject it.
    # Append a user message to satisfy the constraint.
    if merged and merged[-1]["role"] == "assistant":
        merged.append({"role": "user", "content": "Please continue."})

    return merged


# Monkey-patch litellm.completion AND litellm.acompletion once.
# The guard prevents double-patching if this module is imported more than once.
if not getattr(litellm, "_mistral_msg_fix_applied", False):
    # Patch synchronous completion (used by CrewAI's default path)
    _orig_completion = litellm.completion

    def _patched_completion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _fix_messages(kwargs["messages"])
        return _orig_completion(*args, **kwargs)

    litellm.completion = _patched_completion

    # Patch async completion (in case CrewAI uses the async path)
    _orig_acompletion = litellm.acompletion

    async def _patched_acompletion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _fix_messages(kwargs["messages"])
        return await _orig_acompletion(*args, **kwargs)

    litellm.acompletion = _patched_acompletion
    litellm._mistral_msg_fix_applied = True


# Centralized LLM for all agents
llm = get_crew_llm()


# ---------------------------------------------------------------------------
# Pydantic output model for the background research task.
# Forces the LLM to return a structured, validated object instead of
# a free-form string. This is what stops it from skipping citations.
# ---------------------------------------------------------------------------
class EvidenceItem(BaseModel):
    # A single piece of evidence found by the agent
    title: str = Field(description="Short title of the finding")
    summary: str = Field(description="Complete, detailed summary of why this is relevant to the case")
    source_url: str = Field(
        description=(
            "The FULL, complete URL starting with https://. "
            "NEVER truncate. NEVER use partial URLs. "
            "Example: 'https://www.reuters.com/full/path/to/article'. "
            "If no URL is available, write 'URL: Available upon request from court records'. "
            "Do NOT guess or invent a broken link."
        )
    )
    evidence_type: str = Field(description="One of: URL, PDF Record, Court Docket, Media/Image")

    @field_validator('source_url')
    @classmethod
    def url_must_be_complete(cls, v: str) -> str:
        # Strip whitespace to avoid hidden truncation
        v = v.strip()

        # If it looks like a truncated URL (no http, not a docket, and not court records warning), flag it
        if v and not v.startswith('http') and 'No.' not in v and 'Docket' not in v and 'Available upon request' not in v:
            # Prefix a warning so it's visible in the UI
            return f"[INCOMPLETE URL — check source] {v}"

        return v


class LeverageStrategy(BaseModel):
    """Tactical leverage analysis produced after the liability summary and evidence log."""
    settlement_trigger: str = Field(
        description=(
            "The single most powerful fact from the Evidence Log that creates "
            "maximum settlement pressure on the opposing party."
        )
    )
    barriers_to_defense: str = Field(
        description=(
            "The opponent's most likely defense, followed by the direct "
            "counter-argument grounded in the statutes or evidence cited."
        )
    )
    next_tactical_move: str = Field(
        description=(
            "The exact recommended next step to force a win — e.g. file an "
            "interim injunction, issue a preservation order, send a Pre-Action "
            "Protocol letter citing the specific regulatory notice."
        )
    )


class ResearchOutput(BaseModel):
    # The full structured output of the research task
    liability_summary: str = Field(description="High-level summary of the opponent's legal vulnerabilities")
    evidence_log: List[EvidenceItem] = Field(description="List of all evidence items found, each with a source URL")
    leverage_strategy: LeverageStrategy = Field(description="Tactical Leverage Analysis: settlement trigger, barriers to defense, and next tactical move")
    source_index: List[str] = Field(description="A complete list of ALL relevant URLs found during the search for further reading")


def run_chat_crew(context: str, vault_content: str, user_query: str) -> str:
    """ 
    Runs a CrewAI crew to answer the user's question about the case.

    The crew has one agent (Lexis AI) that reads the provided documents
    and context, then generates a cited response.

    Returns the AI's response as a plain string.
    """
  
    # Load Tavily search tool if configured
    search_tools = []
    try:
        from crewai_tools import TavilySearchResults
        search_tools = [TavilySearchResults()]
    except Exception:
        try:
            from langchain_tavily import TavilySearch
            from crewai.tools import tool as crewai_tool 
            _tavily = TavilySearch(max_results=3)

            @crewai_tool("Tavily Web Search")
            def tavily_search(query: str) -> str:
                """Search the web for legal information using Tavily."""
                return str(_tavily.invoke(query))

            search_tools = [tavily_search]
        except Exception:
            pass

    lexis_agent = Agent(
        role="Lexis AI Legal Assistant",
        goal=(
            "Answer the user's legal question accurately using the provided "
            "case documents. For every key fact you state, add a citation in "
            "brackets like [Source: contract.pdf, Page 3] or [Source: pacer.gov]."
        ),
        backstory=(
            "You are Lexis AI, an expert civil litigation research assistant. "
            "You have been given extracted text from case documents and web pages. "
            "You answer questions clearly and always cite the source of each fact. "
            "[LEGAL CLASSIFICATION PROTOCOL]\n"
            "CIVIL VS. CRIMINAL: If you identify a precedent or statute that is criminal "
            "in nature (e.g., Computer Misuse Act 1990), do not discard it. Instead, label "
            "it explicitly under a section titled 'CRIMINAL LEVERAGE OPPORTUNITIES.' "
            "You must clearly distinguish between Civil Remedies (Injunctions, Damages) and "
            "Criminal Liability (Police Referral, Prosecution). "
            "When citing criminal statutes, clearly state: 'This is a criminal statute. It "
            "establishes potential grounds for a police report, which could be used as leverage "
            "to accelerate a civil settlement.' "
            "ZERO HALLUCINATION: You are strictly prohibited from fabricating cases. If you "
            "cite a criminal statute, quote the actual text of the Act. Do not invent cases "
            "like 'R v Smith' to fit the statute. If a fact or citation is not in the "
            "provided documents, you must state: 'The provided sources do not contain this "
            "information.'\n\n"
            "CITATION AND URL PROTOCOL:\n"
            "- You are NOT allowed to guess or fabricate URLs. If you cite a case, use the correct BAILII citation format (e.g., [2022] CSIH 45). "
            "If you do not have the live URL returned by the search tool, simply write 'URL: Available upon request from court records' instead of inventing a broken link."
        ),
        llm=llm,
        tools=search_tools,
        verbose=False,
    )

    answer_task = Task(
        description=(
            f"Case Context (written by the attorney):\n{context}\n\n" 
            f"Case Documents:\n{vault_content}\n\n"
            f"User Question: {user_query}\n\n"
            "First, review the Case Documents to see if the answer is there. "
            "If the documents do not fully answer the question, or if you need external legal facts, "
            "use your search tool to find the required information online. "
            "Provide a clear, detailed answer. "
            "Cite your sources using [Source: ...] inline after each key fact. "
            "If you used the web search, cite the URL.\n\n"
            "[LEGAL CLASSIFICATION PROTOCOL]\n"
            "1. CIVIL VS. CRIMINAL DISTINCTION: If you identify a precedent or statute that is "
            "criminal in nature (e.g., Computer Misuse Act 1990), do NOT discard it. Instead, "
            "label it explicitly under a section titled 'CRIMINAL LEVERAGE OPPORTUNITIES.' "
            "Clearly distinguish between Civil Remedies (Injunctions, Damages) and Criminal "
            "Liability (Police Referral, Prosecution).\n"
            "2. ADVISORY NOTE: When citing criminal statutes, clearly state: 'This is a criminal "
            "statute. It establishes potential grounds for a police report, which could be used "
            "as leverage to accelerate a civil settlement.'\n"
            "3. ZERO HALLUCINATION: You are strictly prohibited from fabricating cases. If you "
            "cite a criminal statute (e.g., Computer Misuse Act 1990), quote the actual text of "
            "the Act. Do not invent cases like 'R v Smith' to fit the statute. If a fact or "
            "citation is not in the provided documents or search results, state: 'The provided "
            "sources do not contain this information.'\n"
            "4. SELF-REVIEW: Before finalising your response, review every citation "
            "(Case Name, Year, Court). If you cannot verify that the citation exists in "
            "the provided documents or search results, delete it from your response entirely. "
            "An uncited fact is better than a fabricated citation.\n"
            "5. CITATION & URL PROTOCOL: You are NOT allowed to guess or fabricate URLs. If you cite a case, "
            "use the correct BAILII citation format (e.g., [2022] CSIH 45). If you do not have the live URL, "
            "simply write 'URL: Available upon request from court records' instead of inventing a broken link."
        ),
        expected_output=(
            "A well-reasoned answer to the user's legal question with inline "
            "citations pointing back to the source documents. Every citation must "
            "be verifiable from the provided case documents or search results. "
            "Criminal statutes must be clearly labeled under CRIMINAL LEVERAGE OPPORTUNITIES. "
            "No fabricated cases."
        ),
        agent=lexis_agent,
    )

    crew = Crew(
        agents=[lexis_agent],
        tasks=[answer_task],
        process=Process.sequential,
        verbose=False,
        max_rpm=10,
    )

    result = crew.kickoff()
    return str(result)


# ---------------------------------------------------------------------------
# Helper functions for formatting research output into Markdown
# ---------------------------------------------------------------------------

def _flatten_liability_summary(summary_dict: dict) -> str:
    """
    Converts a nested liability_summary dict (with core_findings,
    strategic_leverage, etc.) into a readable Markdown string.
    """
    parts = []

    # Handle core_findings list
    findings = summary_dict.get("core_findings", [])
    if findings:
        for finding in findings:
            if isinstance(finding, dict):
                text = finding.get("finding", "")
                implications = finding.get("legal_implications", "")
                if text:
                    parts.append(f"**Finding:** {text}")
                if implications:
                    parts.append(f"*Legal Implications:* {implications}")
                parts.append("")  # blank line separator
            else:
                parts.append(str(finding))

    # Handle strategic_leverage section
    leverage = summary_dict.get("strategic_leverage", {})
    if leverage:
        defensive = leverage.get("defensive", [])
        offensive = leverage.get("offensive", [])

        if defensive:
            parts.append("**Defensive Strategies:**")
            for item in defensive:
                parts.append(f"- {item}")
            parts.append("")

        if offensive:
            parts.append("**Offensive Strategies:**")
            for item in offensive:
                parts.append(f"- {item}")
            parts.append("")

    # If none of the above matched, just stringify whatever we got
    if not parts:
        return str(summary_dict)

    return "\n".join(parts).strip()


def _format_research_output(output: "ResearchOutput") -> str:
    """
    Formats a validated ResearchOutput Pydantic object into clean Markdown
    with clickable source links.
    """
    lines = []
    lines.append("## Strategic Legal Memo")
    lines.append(f"\n### Liability Summary\n{output.liability_summary}")
    lines.append("\n### Evidence Log")

    for item in output.evidence_log:
        lines.append(f"\n#### [{item.evidence_type}] {item.title}")
        lines.append(f"{item.summary}")
        lines.append(f"[Source]({item.source_url})")

    # Leverage & Winning Strategy section
    leverage = output.leverage_strategy
    lines.append("\n---")
    lines.append("\n### LEVERAGE & WINNING STRATEGY")
    lines.append(f"\n**SETTLEMENT TRIGGER**\n{leverage.settlement_trigger}")
    lines.append(f"\n**BARRIERS TO DEFENSE**\n{leverage.barriers_to_defense}")
    lines.append(f"\n**NEXT TACTICAL MOVE**\n{leverage.next_tactical_move}")

    lines.append("\n### Full Source Index (For Further Reading)")
    for url in output.source_index:
        lines.append(f"- [{url}]({url})")

    return "\n".join(lines)


def _format_raw_research_dict(data: dict) -> str:
    """
    Last-resort formatter that takes a raw dict (when Pydantic validation
    fails entirely) and extracts whatever it can into readable Markdown.
    """
    # Mistral sometimes wraps the response in an outer "ResearchOutput" key
    if "ResearchOutput" in data and isinstance(data["ResearchOutput"], dict):
        data = data["ResearchOutput"]

    lines = []
    lines.append("## Strategic Legal Memo")

    # Liability summary — could be a string or a nested dict
    summary = data.get("liability_summary", "")
    if isinstance(summary, dict):
        summary = _flatten_liability_summary(summary)
    if summary:
        lines.append(f"\n### Liability Summary\n{summary}")

    # Evidence log
    evidence = data.get("evidence_log", [])
    if evidence:
        lines.append("\n### Evidence Log")
        for item in evidence:
            if isinstance(item, dict):
                # Get the type field (could be 'type' or 'evidence_type')
                evidence_type = item.get("evidence_type", item.get("type", "Evidence"))
                title = item.get("title", "Untitled")
                item_summary = item.get("summary", "")
                url = item.get("source_url", "")

                lines.append(f"\n#### [{evidence_type}] {title}")
                if item_summary:
                    lines.append(f"{item_summary}")
                if url and url.startswith("http"):
                    lines.append(f"[Source]({url})")
            else:
                lines.append(f"- {str(item)}")

    # Leverage & Winning Strategy section (raw dict fallback)
    leverage = data.get("leverage_strategy", {})
    if isinstance(leverage, dict) and leverage:
        lines.append("\n---")
        lines.append("\n### LEVERAGE & WINNING STRATEGY")
        settlement = leverage.get("settlement_trigger", "")
        barriers = leverage.get("barriers_to_defense", "")
        next_move = leverage.get("next_tactical_move", "")
        if settlement:
            lines.append(f"\n**SETTLEMENT TRIGGER**\n{settlement}")
        if barriers:
            lines.append(f"\n**BARRIERS TO DEFENSE**\n{barriers}")
        if next_move:
            lines.append(f"\n**NEXT TACTICAL MOVE**\n{next_move}")

    # Source index
    sources = data.get("source_index", [])
    if sources:
        lines.append("\n### Full Source Index (For Further Reading)")
        for url in sources:
            if isinstance(url, str) and url.startswith("http"):
                lines.append(f"- [{url}]({url})")
            else:
                lines.append(f"- {url}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model-agnostic JSON extraction helpers
#
# These functions handle the many different ways LLMs wrap structured output
# so that switching models never breaks the formatted research memo.
# ---------------------------------------------------------------------------

import json
import re
import ast


def _extract_json_from_raw_text(raw_text: str) -> dict | None:
    """
    Attempts every known extraction strategy to pull a JSON dict out of
    an LLM's raw text output. Returns the parsed dict or None.

    Handles:
      - Clean JSON
      - Markdown ```json ... ``` blocks
      - "Final Answer": "{...}" (JSON-as-escaped-string)
      - "Final Answer": { ... }  (JSON-as-dict)
      - Outer wrappers like {"ResearchOutput": {...}}
      - Brace-matching extraction as a last resort
    """
    if not raw_text:
        return None

    # --- Step 1: Strip markdown code fences ---
    cleaned = raw_text
    if "```json" in cleaned:
        # Take the content inside the LAST ```json ... ``` block
        parts = cleaned.split("```json")
        last_block = parts[-1]
        if "```" in last_block:
            cleaned = last_block.split("```")[0].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:-1]).strip()

    # --- Step 2: Try direct JSON parse ---
    parsed = _try_parse_json(cleaned)
    if parsed:
        return _unwrap_outer_keys(parsed)

    # --- Step 3: Handle "Final Answer" wrapper ---
    # Many models return: ... "Final Answer": "{...}" or "Final Answer": {...}
    final_answer_match = re.search(
        r'"Final Answer"\s*:\s*', raw_text, re.IGNORECASE
    )
    if final_answer_match:
        after_key = raw_text[final_answer_match.end():].strip()

        # Case A: The value is a JSON string (starts with quote)
        if after_key.startswith('"'):
            # Extract the string value — it may contain escaped JSON
            try:
                # Wrap in an object to let json.loads handle the escaping
                wrapper = '{"__val__": ' + after_key.rstrip().rstrip("}").rstrip(",") + "}"
                val = json.loads(wrapper).get("__val__", "")
                inner = _try_parse_json(val)
                if inner:
                    return _unwrap_outer_keys(inner)
            except Exception:
                pass

            # Brute force: find JSON substring inside the string value
            inner_json = _extract_first_json_object(after_key)
            if inner_json:
                return _unwrap_outer_keys(inner_json)

        # Case B: The value is a direct JSON object (starts with brace)
        if after_key.startswith("{"):
            inner_json = _extract_first_json_object(after_key)
            if inner_json:
                return _unwrap_outer_keys(inner_json)

    # --- Step 4: Last resort — find the first valid JSON object anywhere ---
    found = _extract_first_json_object(raw_text)
    if found:
        return _unwrap_outer_keys(found)

    return None


def _try_parse_json(text: str) -> dict | None:
    """Try json.loads, then ast.literal_eval. Returns dict or None."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    return None


def _extract_first_json_object(text: str) -> dict | None:
    """
    Finds the first '{' in text and uses brace-matching to extract
    the complete JSON object. Handles nested braces correctly.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                parsed = _try_parse_json(candidate)
                if parsed:
                    return parsed
                break

    return None


def _unwrap_outer_keys(data: dict) -> dict:
    """
    Strip known wrapper keys that models add around the actual payload.
    e.g. {"ResearchOutput": {...}} -> {...}
         {"Final Answer": {...}}  -> {...}
    """
    wrapper_keys = ["ResearchOutput", "Final Answer", "final_answer", "output"]
    for key in wrapper_keys:
        if key in data and isinstance(data[key], dict):
            data = data[key]
            break
        # Some models put the JSON as a string value under the key
        if key in data and isinstance(data[key], str):
            inner = _try_parse_json(data[key])
            if inner:
                data = inner
                break
    return data


def _normalize_to_research_output(parsed: dict) -> "ResearchOutput | None":
    """
    Takes a raw parsed dict and normalises it into a valid ResearchOutput
    Pydantic object. Handles field name variations across models.
    Returns None if the dict cannot be normalised.
    """
    try:
        normalized = dict(parsed)

        # If liability_summary is a dict instead of a string, flatten it
        if isinstance(normalized.get("liability_summary"), dict):
            normalized["liability_summary"] = _flatten_liability_summary(
                normalized["liability_summary"]
            )

        # Fix evidence items that use 'type' instead of 'evidence_type'
        evidence = normalized.get("evidence_log", [])
        if isinstance(evidence, list):
            for item in evidence:
                if isinstance(item, dict):
                    if "evidence_type" not in item and "type" in item:
                        item["evidence_type"] = item.pop("type")
                    # Some models use 'url' instead of 'source_url'
                    if "source_url" not in item and "url" in item:
                        item["source_url"] = item.pop("url")
                    # Some models use 'link' instead of 'source_url'
                    if "source_url" not in item and "link" in item:
                        item["source_url"] = item.pop("link")

        # Some models use 'sources' instead of 'source_index'
        if "source_index" not in normalized and "sources" in normalized:
            normalized["source_index"] = normalized.pop("sources")

        # Normalise leverage_strategy — some models may use alternative key names
        if "leverage_strategy" not in normalized:
            for alt_key in ["leverage", "winning_strategy", "tactical_leverage", "strategy"]:
                if alt_key in normalized and isinstance(normalized[alt_key], dict):
                    normalized["leverage_strategy"] = normalized.pop(alt_key)
                    break

        # If leverage_strategy is still missing, provide safe defaults so
        # Pydantic validation doesn't fail on older research results
        if "leverage_strategy" not in normalized:
            normalized["leverage_strategy"] = {
                "settlement_trigger": "Not available — run a new research cycle to generate tactical analysis.",
                "barriers_to_defense": "Not available — run a new research cycle to generate tactical analysis.",
                "next_tactical_move": "Not available — run a new research cycle to generate tactical analysis.",
            }

        return ResearchOutput(**normalized)
    except Exception:
        return None


def run_research_crew(case_id: int, case_context: str) -> tuple[str, str]:
    """
    Runs a background research crew that conducts adversarial legal research,
    hunting for precedents, regulatory fines, and court rulings.

    Dynamically fetches:
      - Vault evidence from the vector store (PDFs, URLs, images)
      - Attorney rejection feedback from the latest rejected Alert record

    Every finding MUST include an inline citation (URL or docket number).
    Includes a full Source Index of all sites searched for further reading.
    Returns a tuple of (strategic_memo_markdown, ai_reasoning) where
    ai_reasoning is a short explanation of why this result is relevant and
    how it can help win the case.
    """

    # -----------------------------------------------------------------------
    # 1. Fetch vault evidence from the vector store for this case
    # -----------------------------------------------------------------------
    from .vector_store import search_vector_store
    vault_chunks = search_vector_store(case_id, case_context, top_k=15)
    vault_evidence = "\n\n---\n\n".join(vault_chunks) if vault_chunks else "(No vault evidence available)"

    # -----------------------------------------------------------------------
    # 2. Fetch the latest rejection feedback from the Alert table
    # -----------------------------------------------------------------------
    rejection_reason = ""
    try:
        from sqlmodel import Session, select
        from database import engine
        from models import Alert

        with Session(engine) as db_session:
            latest_rejected = db_session.exec(
                select(Alert)
                .where(Alert.case_id == case_id)
                .where(Alert.review_status == "rejected")
                .order_by(Alert.created_at.desc())
            ).first()
            if latest_rejected and latest_rejected.rejection_reason:
                rejection_reason = latest_rejected.rejection_reason
    except Exception as e:
        print(f"[crew] Failed to fetch rejection feedback for case {case_id}: {e}")

    # -----------------------------------------------------------------------
    # 3. Build the system prompt sections
    # -----------------------------------------------------------------------
    rejection_block = ""
    if rejection_reason:
        rejection_block = (
            "\n\nATTORNEY REJECTION FEEDBACK (MANDATORY — treat as direct order from Managing Partner):\n"
            f"{rejection_reason}\n\n"
            "You MUST specifically address this feedback and correct your previous mistakes. "
            "Pivot your research strategy entirely to satisfy this exact requirement."
        )

    # Load Tavily search tool if configured
    search_tools = []
    try:
        from crewai_tools import TavilySearchResults
        search_tools = [TavilySearchResults()]
    except Exception:
        try:
            from langchain_tavily import TavilySearch
            from crewai.tools import tool as crewai_tool
            _tavily = TavilySearch(max_results=5)

            @crewai_tool("Tavily Web Search")
            def tavily_search(query: str) -> str:
                """Search the web for legal information using Tavily."""
                return str(_tavily.invoke(query))

            search_tools = [tavily_search]
        except Exception:
            pass

    researcher = Agent(
        role="Lexis Adversarial Legal Research Agent",
        goal=(
            "Synthesize internal client evidence (VAULT EVIDENCE) with public legal precedents "
            "to build an aggressive, highly accurate, and actionable legal memo for a Managing Partner. "
            "Focus on hard timelines, physical evidence (e.g., missing hazard signs from photos), "
            "and contradictions found in the vault evidence. The 'smoking guns' are in the vault. "
            "After generating the Liability Summary and Evidence Log, you MUST perform a Tactical "
            "Leverage Analysis identifying the settlement trigger, barriers to defense, and the "
            "next tactical move."
        ),
        backstory=(
            "You are an elite Senior Civil Litigation Strategist and Forensic Evidence Auditor. "
            "Your objective is to synthesize internal client evidence with public legal precedents "
            "to build an aggressive, highly accurate, and actionable legal memo.\n\n"

            "EVIDENCE SYNTHESIS PROTOCOL:\n"
            "You must read the VAULT EVIDENCE and connect it directly to the CASE CONTEXT. "
            "Look for hard timelines, physical evidence (e.g., missing hazard signs from photos), "
            "and contradictions. Do not rely solely on the case context; the 'smoking guns' are "
            "in the vault evidence.\n\n"

            "ZERO HALLUCINATION PROTOCOL:\n"
            "You are strictly prohibited from inventing case law, citations, or URLs. "
            "If you cannot verify a specific case name, year, and court using your research "
            "capabilities, do not include it. An uncited fact is better than a fabricated citation.\n\n"

            "CIVIL LAW STRICT COMPLIANCE:\n"
            "This is civil litigation. You must apply the 'balance of probabilities' standard. "
            "Do not cite Criminal Court cases (e.g., citations containing 'Crim' or 'R v.') as "
            "civil precedents.\n\n"

            "CRIMINAL LEVERAGE EXCEPTION:\n"
            "You may cite criminal statutes (e.g., Computer Misuse Act, Fraud Act) ONLY if you "
            "explicitly separate them under a 'Criminal Leverage' heading, noting that they "
            "establish grounds for a police report to be used as leverage to accelerate a civil "
            "settlement.\n\n"

            "CITATION AND URL PROTOCOL:\n"
            "- You are NOT allowed to guess or fabricate URLs.\n"
            "- If you cite a case, use the correct BAILII citation format (e.g., [2022] CSIH 45).\n"
            "- If you do not have the live URL returned by your search tool, write "
            "'URL: Available upon request from court records' instead of inventing a broken link."
        ),
        llm=llm,
        tools=search_tools,
        verbose=True,
        max_iter=5,
    )

    research_task = Task(
        description=(
            "SYSTEM ROLE: You are an elite Senior Civil Litigation Strategist and Forensic Evidence Auditor.\n\n"

            f"CASE CONTEXT (original attorney notes — DO NOT modify):\n{case_context[:800]}\n\n"

            f"VAULT EVIDENCE (extracted text from uploaded PDFs, OCR descriptions of images, "
            f"and scraped URLs — treat as absolute verified fact):\n{vault_evidence[:3000]}\n\n"

            f"{rejection_block}"

            "CORE DIRECTIVES:\n\n"

            "1. EVIDENCE SYNTHESIS: Read the VAULT EVIDENCE and connect it directly to the CASE CONTEXT. "
            "Look for hard timelines, physical evidence, and contradictions. The 'smoking guns' are in "
            "the vault evidence, not just the case context.\n\n"

            "2. ZERO HALLUCINATION: You are strictly prohibited from inventing case law, citations, or URLs. "
            "If you cannot verify a specific case name, year, and court, do not include it.\n\n"

            "3. CIVIL LAW COMPLIANCE: Apply 'balance of probabilities'. Do NOT cite Criminal Court cases "
            "(e.g., 'Crim', 'R v.') as civil precedents.\n\n"

            "4. CRIMINAL LEVERAGE EXCEPTION: You may cite criminal statutes ONLY under a separate "
            "'CRIMINAL LEVERAGE OPPORTUNITIES' heading, noting they establish grounds for a police "
            "report to accelerate a civil settlement.\n\n"

            "5. Use your Tavily Web Search tool to locate real legal findings.\n\n"

            "RELEVANCE FILTERS:\n"
            "- DROP: historical patents, physical science, entertainment, abstract definitions.\n"
            "- KEEP: active/pending lawsuits, regulatory enforcement, material corporate liabilities, "
            "data breach incidents, community organising for legal recourse.\n\n"

            "TACTICAL LEVERAGE ANALYSIS (REQUIRED):\n"
            "After the Liability Summary and Evidence Log, answer:\n"
            "1. SETTLEMENT TRIGGER: The single most powerful fact creating maximum settlement pressure.\n"
            "2. BARRIERS TO DEFENSE: The opponent's most likely defense and the direct counter-argument.\n"
            "3. NEXT TACTICAL MOVE: The exact recommended next step to force a win.\n\n"

            "SELF-REVIEW: Before finalising, review every citation. If it cannot be verified from "
            "your search results or vault evidence, delete it. An uncited fact is better than a "
            "fabricated citation.\n\n"

            "CRITICAL CONSTRAINTS:\n"
            "- Only report real search results and real URLs from Tavily.\n"
            "- NEVER fabricate lawsuits, court filings, or URLs.\n"
            "- Use BAILII citation format for cases.\n"
            "- NO placeholder URLs, truncated links, or '#' links.\n"
            "- NO disclaimers or meta-talk about simulated research."
        ),
        expected_output=(
            "A structured JSON object matching the ResearchOutput schema containing:\n"
            "1. liability_summary: A detailed, professional summary of actual legal liabilities and insights found (strictly no disclaimers or simulated text).\n"
            "2. evidence_log: A list of actual evidence items found, each with a real title, detailed summary, full complete source URL, and type.\n"
            "3. leverage_strategy: An object with three fields — settlement_trigger (the single most powerful fact creating settlement pressure), barriers_to_defense (the opponent's likely defense and the direct counter-argument), and next_tactical_move (the exact recommended next step to force a win).\n"
            "4. source_index: A complete list of all unique, real URLs found during the web search."
        ),
        agent=researcher,
        output_pydantic=ResearchOutput,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
        max_rpm=10,
    )

    import time
    
    max_retries = 3
    retry_delay = 65  # Mistral's rate limit resets every minute, wait a bit longer to be safe
    
    result = None
    for attempt in range(max_retries):
        try:
            result = crew.kickoff()
            break  # Success
        except Exception as e:
            if "RateLimitError" in str(e) or "429" in str(e):
                if attempt < max_retries - 1:
                    print(f"\\n[crew] Rate limit hit (429). Waiting {retry_delay} seconds before attempt {attempt + 2}/{max_retries}...")
                    time.sleep(retry_delay)
                    continue
            # Re-raise if it's not a rate limit error or we ran out of retries
            raise

    # ---------------------------------------------------------------------------
    # Model-agnostic output parsing
    #
    # Different LLMs return structured output in wildly different wrappers:
    #   - Clean Pydantic object (ideal)
    #   - Raw JSON string
    #   - JSON inside markdown ```json ... ``` blocks
    #   - "Final Answer": "{...}" (Gemma, Nemotron, smaller models)
    #   - "Final Answer": { ... }  (dict, not string)
    #   - Outer wrapper like {"ResearchOutput": {...}}
    #
    # The parser below tries every extraction strategy in order so that
    # switching models never breaks the formatted output.
    # ---------------------------------------------------------------------------
    import json
    import re

    output_obj = None
    raw_dict = None

    # Strategy 1: CrewAI successfully parsed into Pydantic
    if hasattr(result, 'pydantic') and result.pydantic:
        output_obj = result.pydantic
    else:
        raw_text = str(result).strip()
        parsed = _extract_json_from_raw_text(raw_text)

        if parsed:
            raw_dict = parsed
            output_obj = _normalize_to_research_output(parsed)

    # Extract the ai_reasoning before formatting. Use the liability_summary
    # because it's the high-level "why this matters" paragraph — exactly what
    # we want to surface in the Brain icon hover popup.
    if output_obj:
        memo_text = _format_research_output(output_obj)
        ai_reasoning = output_obj.liability_summary or ""
    elif raw_dict:
        memo_text = _format_raw_research_dict(raw_dict)
        raw_summary = raw_dict.get("liability_summary", "")
        if isinstance(raw_summary, dict):
            ai_reasoning = _flatten_liability_summary(raw_summary)
        else:
            ai_reasoning = str(raw_summary)
    else:
        memo_text = str(result)
        ai_reasoning = ""

    # Safety Net: remove any simulated research disclaimers or tool note lines
    disclaimer_phrases = [
        "This memo is based on simulated research",
        "simulated research due to tool limitations",
        "Links are AI-generated",
        "execute the Tavily Web Search queries",
        "Would you like me to run these queries now",
        "Proceed or refine the scope",
        "Tavily Web Search actions",
        "disclaimer"
    ]
    
    cleaned_lines = []
    in_table = False
    headers = []
    
    for line in memo_text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
            
        if any(phrase.lower() in stripped.lower() for phrase in disclaimer_phrases):
            continue
            
        # Check if line is a markdown table row (starts and ends with |)
        if stripped.startswith("|") and stripped.endswith("|"):
            parts = [p.strip() for p in stripped.split("|")[1:-1]]
            # Skip separator rows like |---|---|
            if all(all(c == '-' or c == ' ' or c == ':' for c in p) for p in parts if p):
                continue
            
            # Check if this is the header row
            if any(term in parts[0].lower() for term in ["description", "source", "case name", "pacer link"]):
                headers = parts
                in_table = True
                continue
                
            # If we are in a table and have valid parsed columns
            if in_table and len(parts) >= 2:
                desc = parts[0]
                source = parts[1]
                
                # Format beautifully as text blocks to prevent card overflow
                desc_clean = desc.replace("**", "").replace("*", "").strip()
                source_clean = source.replace("[", "").replace("]", "").replace("(#)", "").strip()
                
                # If there's an actual URL in the source, extract it
                if "http" in source_clean:
                    import re
                    urls = re.findall(r'https?://[^\s|\)]+', source_clean)
                    url = urls[0] if urls else source_clean
                    cleaned_lines.append(f"\n- **{desc_clean}**")
                    cleaned_lines.append(f"  [Source]({url})")
                else:
                    # Skip empty placeholder rows or rows with only '#' as links
                    if source_clean == "#" or not source_clean:
                        continue
                    cleaned_lines.append(f"\n- **{desc_clean}**")
                    if source_clean:
                        cleaned_lines.append(f"  Source: {source_clean}")
            continue
        else:
            in_table = False
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip(), ai_reasoning
 