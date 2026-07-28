import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# --- Mistral SDK import patch for older/newer library compatibility ---
try:
    import sys
    import mistralai
    from mistralai.client import Mistral
    setattr(sys.modules['mistralai'], 'Mistral', Mistral)
except Exception:
    pass

from pydantic import BaseModel, Field
from typing import List
from crewai import Agent, Task, Crew, Process
from .model_providers import get_crew_llm
import litellm
import json
import re
import ast
import time


# ---------------------------------------------------------------------------
# Mistral Message Order Fix
# ---------------------------------------------------------------------------
def _fix_messages(messages: list) -> list:
    if not messages or len(messages) < 2:
        return messages

    merged = []
    for msg in messages:
        msg_copy = dict(msg)
        if not merged:
            merged.append(msg_copy)
            continue

        last_msg = merged[-1]
        if msg_copy["role"] == last_msg["role"] and msg_copy["role"] != "tool":
            content1 = last_msg.get("content") or ""
            content2 = msg_copy.get("content") or ""
            last_msg["content"] = (str(content1) + "\n\n" + str(content2)).strip()

            if "tool_calls" in msg_copy:
                if "tool_calls" not in last_msg:
                    last_msg["tool_calls"] = []
                calls = msg_copy["tool_calls"]
                if isinstance(calls, list):
                    last_msg["tool_calls"].extend(calls)
        else:
            merged.append(msg_copy)

    if merged and merged[-1]["role"] == "assistant":
        merged.append({"role": "user", "content": "Please continue."})

    return merged


if not getattr(litellm, "_mistral_msg_fix_applied", False):
    _orig_completion = litellm.completion

    def _patched_completion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _fix_messages(kwargs["messages"])
        return _orig_completion(*args, **kwargs)

    litellm.completion = _patched_completion

    _orig_acompletion = litellm.acompletion

    async def _patched_acompletion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _fix_messages(kwargs["messages"])
        return await _orig_acompletion(*args, **kwargs)

    litellm.acompletion = _patched_acompletion
    litellm._mistral_msg_fix_applied = True


# Centralized LLM for all agents
llm = get_crew_llm()

# Domain whitelist for output URL filtering
ALLOWED_DOMAINS = [
    "bailii.org",
    "legislation.gov.uk",
    "scotcourts.gov.uk",
    "gov.uk",
]


def _is_allowed_domain(url: str) -> bool:
    from urllib.parse import urlparse
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower()
        return any(
            hostname == domain or hostname.endswith(f".{domain}")
            for domain in ALLOWED_DOMAINS
        )
    except Exception:
        return False


def _strip_non_whitelisted_urls(text: str) -> str:
    """
    Removes any URL from free text that does not belong to ALLOWED_DOMAINS.
    This is the last-resort sanitizer for raw LLM output that bypasses
    structured parsing — it prevents hallucinated URLs from reaching the user.
    """
    import re as _re
    url_pattern = _re.compile(r'https?://[^\s)\]>"]+', _re.IGNORECASE)

    def _replace_if_blocked(match: _re.Match) -> str:
        url = match.group(0)
        if _is_allowed_domain(url):
            return url
        return "[URL removed — not from an authoritative legal source]"

    return url_pattern.sub(_replace_if_blocked, text)


class EvidenceItem(BaseModel):
    title: str = Field(description="Short title of the finding")
    summary: str = Field(description="Complete, detailed summary of why this is relevant to the case")
    evidence_type: str = Field(
        description="Source type: 'URL' for web search, or 'PDF Record' / 'Witness Statement' / 'Maintenance Log' / 'Policy Extract' for internal vault documents"
    )
    web_url: str | None = Field(
        default=None,
        description=(
            "The exact whitelisted URL starting with https:// for external web search items. "
            "MUST be null/empty if this evidence item comes from an internal Vault document."
        )
    )
    document_name: str | None = Field(
        default=None,
        description=(
            "The exact filename or title of the internal document (e.g., 'Witness Statement - Sarah Jenkins', "
            "'Maintenance Log FLT-04.pdf'). MUST be null/empty if this evidence item comes from a web search."
        )
    )


class LeverageStrategy(BaseModel):
    settlement_trigger: str = Field(
        description="The single most powerful fact from the Evidence Log that creates maximum settlement pressure."
    )
    barriers_to_defense: str = Field(
        description="The opponent's most likely defense, followed by the direct counter-argument."
    )
    next_tactical_move: str = Field(
        description="The exact recommended next step to force a win."
    )


class ResearchOutput(BaseModel):
    liability_summary: str = Field(description="High-level summary of the opponent's legal vulnerabilities")
    evidence_log: List[EvidenceItem] = Field(description="List of all evidence items found, each with a source URL")
    leverage_strategy: LeverageStrategy = Field(description="Tactical Leverage Analysis")
    source_index: List[str] = Field(description="A complete list of ALL relevant URLs found during the search")


def run_chat_crew(context: str, vault_content: str, user_query: str) -> str:
    """
    Runs a CrewAI crew to answer the user's question about the case.
    """
    try:
        from crewai.tools import tool as crewai_tool

        @crewai_tool("Tavily Web Search")
        def tavily_search(query: str) -> str:
            """Search UK legal databases (BAILII, legislation.gov.uk, gov.uk, scotcourts.gov.uk) for legal precedents, statutes, and regulatory guidelines."""
            clean_q = query.strip()
            if len(clean_q) > 100:
                words = [w for w in re.findall(r'\b\w+\b', clean_q) if len(w) > 3 and w.lower() not in ('where', 'which', 'there', 'their', 'about', 'would', 'could', 'should', 'actual', 'please', 'find')]
                clean_q = " ".join(words[:8])
            
            results = search_tavily(clean_q)
            if not results:
                return "No web search results found on official legal domains."

            formatted = []
            for r in results:
                u = r.get("url", "")
                if u and _is_allowed_domain(u):
                    formatted.append(f"Title: {r.get('title')}\nURL: {u}\nSnippet: {r.get('content')}")
            return "\n\n".join(formatted) if formatted else "No web search results found on official legal domains."

        search_tools = [tavily_search]
    except Exception:
        search_tools = []

    lexis_agent = Agent(
        role="Lexis AI Legal Assistant",
        goal=(
            "Answer the user's legal question accurately using the provided "
            "case documents and web search results. Cite your sources using exact URLs."
        ),
        backstory=(
            "You are Lexis AI, an expert civil litigation research assistant. "
            "You answer questions clearly and always cite the source of each fact. "
            "CITATION AND URL PROTOCOL:\n"
            "- Copy the exact, full URL starting with https:// for web citations from bailii.org, legislation.gov.uk, scotcourts.gov.uk, or gov.uk.\n"
            "- Always present web search citations as clickable Markdown hyperlinks like [Source](https://www.legislation.gov.uk/...).\n"
            "- Do NOT fabricate or invent URLs outside the provided search results."
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
            "Cite your sources using exact URLs as clickable links [Source](https://...)."
        ),
        expected_output=(
            "A well-reasoned answer to the user's legal question with inline citations."
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
# CONTAMINATION GUARD
# ---------------------------------------------------------------------------
_CONTAMINATION_MARKERS = [
    "INCOMPLETE URL",
    "check source]",
    "[crew]",
    "Links are AI-generated",
]


def _scan_for_contamination(chunks: list[str], case_id: int) -> list[str]:
    clean_chunks = []
    contaminated_count = 0

    for chunk in chunks:
        if any(marker.lower() in chunk.lower() for marker in _CONTAMINATION_MARKERS):
            contaminated_count += 1
            print(
                f"[crew] CONTAMINATION DETECTED in vault chunk for case {case_id} — "
                f"this looks like a previous AI-generated memo output that got "
                f"ingested back into the vector store. First 200 chars: {chunk[:200]!r}",
                flush=True,
            )
            continue
        clean_chunks.append(chunk)

    if contaminated_count:
        print(
            f"[crew] WARNING: {contaminated_count}/{len(chunks)} vault chunks for case "
            f"{case_id} were discarded as contaminated.",
            flush=True,
        )

    return clean_chunks


def _flatten_liability_summary(summary_dict: dict) -> str:
    parts = []
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
                parts.append("")
            else:
                parts.append(str(finding))

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

    if not parts:
        return str(summary_dict)

    return "\n".join(parts).strip()


def _clean_vault_reference(source_str: str) -> str:
    from urllib.parse import unquote

    text = unquote(source_str)
    for marker in _CONTAMINATION_MARKERS:
        text = re.sub(re.escape(marker), "", text, flags=re.IGNORECASE)
    text = text.replace("[", "").replace("]", "")
    text = re.sub(r'\b\d{9,12}_', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _format_research_output(output: "ResearchOutput") -> str:
    lines = []
    lines.append("## Strategic Legal Memo")
    lines.append(f"\n### Liability Summary\n{output.liability_summary}")
    lines.append("\n### Evidence Log")

    for item in output.evidence_log:
        lines.append(f"\n#### [{item.evidence_type}] {item.title}")
        lines.append(f"{item.summary}")

        url = (item.web_url or "").strip()
        doc = (item.document_name or "").strip()

        if url and url.startswith("http"):
            # Final rendering gate: only render URLs from whitelisted domains
            if _is_allowed_domain(url):
                lines.append(f"[Source]({url})")
            else:
                print(f"[crew] _format_research_output: blocked non-whitelisted URL: {url}", flush=True)
        elif doc:
            clean_name = _clean_vault_reference(doc)
            if clean_name:
                lines.append(f"📄 Source: {clean_name}")

    leverage = output.leverage_strategy
    lines.append("\n---")
    lines.append("\n### LEVERAGE & WINNING STRATEGY")
    lines.append(f"\n**SETTLEMENT TRIGGER**\n{leverage.settlement_trigger}")
    lines.append(f"\n**BARRIERS TO DEFENSE**\n{leverage.barriers_to_defense}")
    lines.append(f"\n**NEXT TACTICAL MOVE**\n{leverage.next_tactical_move}")

    lines.append("\n### Source Index")
    for url in output.source_index:
        url_clean = url.strip()
        if url_clean.startswith("http"):
            # Final rendering gate: only render whitelisted domain URLs in the source index
            if _is_allowed_domain(url_clean):
                lines.append(f"- [{url_clean}]({url_clean})")
            else:
                print(f"[crew] _format_research_output source_index: blocked non-whitelisted URL: {url_clean}", flush=True)
        else:
            clean_name = _clean_vault_reference(url_clean)
            if clean_name:
                lines.append(f"- 📄 {clean_name}")

    return "\n".join(lines)


def _format_raw_research_dict(data: dict) -> str:
    if "ResearchOutput" in data and isinstance(data["ResearchOutput"], dict):
        data = data["ResearchOutput"]

    lines = []
    lines.append("## Strategic Legal Memo")

    summary = data.get("liability_summary", "")
    if isinstance(summary, dict):
        summary = _flatten_liability_summary(summary)
    if summary:
        lines.append(f"\n### Liability Summary\n{summary}")

    evidence = data.get("evidence_log", [])
    if evidence:
        lines.append("\n### Evidence Log")
        for item in evidence:
            if isinstance(item, dict):
                evidence_type = item.get("evidence_type", item.get("type", "Evidence"))
                title = item.get("title", "Untitled")
                item_summary = item.get("summary", "")
                url = str(item.get("web_url") or item.get("source_url") or "").strip()
                doc = str(item.get("document_name") or "").strip()
                if not doc and url and not url.startswith("http"):
                    doc = url
                    url = ""

                lines.append(f"\n#### [{evidence_type}] {title}")
                if item_summary:
                    lines.append(f"{item_summary}")
                if url and url.startswith("http"):
                    # Final rendering gate: only render whitelisted domain URLs
                    if _is_allowed_domain(url):
                        lines.append(f"[Source]({url})")
                    else:
                        print(f"[crew] _format_raw_research_dict: blocked non-whitelisted URL: {url}", flush=True)
                elif doc:
                    clean_name = _clean_vault_reference(doc)
                    if clean_name:
                        lines.append(f"📄 Source: {clean_name}")
            else:
                lines.append(f"- {str(item)}")

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

    return "\n".join(lines)


def _try_parse_json(text: str) -> dict | None:
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

    if depth > 0:
        print(
            f"[crew] TRUNCATION DETECTED: JSON object never closed (depth={depth} "
            f"at end of text). Check max_tokens ceiling in get_crew_llm().",
            flush=True,
        )
    return None


def _unwrap_outer_keys(data: dict) -> dict:
    wrapper_keys = ["ResearchOutput", "Final Answer", "final_answer", "output"]
    for key in wrapper_keys:
        if key in data and isinstance(data[key], dict):
            data = data[key]
            break
        if key in data and isinstance(data[key], str):
            inner = _try_parse_json(data[key])
            if inner:
                data = inner
                break
    return data


def _extract_json_from_raw_text(raw_text: str) -> dict | None:
    if not raw_text:
        return None

    cleaned = raw_text
    if "```json" in cleaned:
        parts = cleaned.split("```json")
        last_block = parts[-1]
        if "```" in last_block:
            cleaned = last_block.split("```")[0].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:-1]).strip()

    parsed = _try_parse_json(cleaned)
    if parsed:
        return _unwrap_outer_keys(parsed)

    final_answer_match = re.search(r'"Final Answer"\s*:\s*', raw_text, re.IGNORECASE)
    if final_answer_match:
        after_key = raw_text[final_answer_match.end():].strip()
        if after_key.startswith('"'):
            try:
                wrapper = '{"__val__": ' + after_key.rstrip().rstrip("}").rstrip(",") + "}"
                val = json.loads(wrapper).get("__val__", "")
                inner = _try_parse_json(val)
                if inner:
                    return _unwrap_outer_keys(inner)
            except Exception:
                pass
            inner_json = _extract_first_json_object(after_key)
            if inner_json:
                return _unwrap_outer_keys(inner_json)
        if after_key.startswith("{"):
            inner_json = _extract_first_json_object(after_key)
            if inner_json:
                return _unwrap_outer_keys(inner_json)

    found = _extract_first_json_object(raw_text)
    if found:
        return _unwrap_outer_keys(found)

    return None


def _normalize_to_research_output(parsed: dict) -> "ResearchOutput | None":
    try:
        normalized = dict(parsed)

        if isinstance(normalized.get("liability_summary"), dict):
            normalized["liability_summary"] = _flatten_liability_summary(normalized["liability_summary"])

        evidence = normalized.get("evidence_log", [])
        if isinstance(evidence, list):
            for item in evidence:
                if isinstance(item, dict):
                    if "title" not in item:
                        for alt in ["claim", "name", "header", "topic"]:
                            if alt in item:
                                item["title"] = item.pop(alt)
                                break
                    if "title" not in item:
                        item["title"] = "Legal Finding"

                    if "summary" not in item:
                        for alt in ["description", "details", "content", "text"]:
                            if alt in item:
                                item["summary"] = item.pop(alt)
                                break
                    if "summary" not in item:
                        item["summary"] = "No summary provided."

                    if "evidence_type" not in item and "type" in item:
                        item["evidence_type"] = item.pop("type")
                    if "evidence_type" not in item:
                        item_url = str(item.get("web_url") or item.get("source_url") or item.get("url") or "")
                        item["evidence_type"] = "URL" if item_url.startswith("http") else "PDF Record"

                    # Normalize URL and document_name fields
                    raw_src = str(item.pop("source_url", item.pop("url", item.pop("link", "")))).strip()
                    if "web_url" not in item and raw_src.startswith("http"):
                        if _is_allowed_domain(raw_src):
                            item["web_url"] = raw_src
                        else:
                            # If LLM attached a fake web URL to a vault doc, salvage it as a document reference
                            item["document_name"] = item.get("title") or "Vault Document"
                    elif "document_name" not in item and raw_src and not raw_src.startswith("http"):
                        item["document_name"] = raw_src

        if "source_index" not in normalized and "sources" in normalized:
            normalized["source_index"] = normalized.pop("sources")

        if "leverage_strategy" not in normalized:
            for alt_key in ["leverage", "winning_strategy", "tactical_leverage", "strategy"]:
                if alt_key in normalized and isinstance(normalized[alt_key], dict):
                    normalized["leverage_strategy"] = normalized.pop(alt_key)
                    break

        if "leverage_strategy" not in normalized:
            normalized["leverage_strategy"] = {
                "settlement_trigger": "Not available — run a new research cycle to generate tactical analysis.",
                "barriers_to_defense": "Not available — run a new research cycle to generate tactical analysis.",
                "next_tactical_move": "Not available — run a new research cycle to generate tactical analysis.",
            }

        return ResearchOutput(**normalized)
    except Exception:
        return None


def is_url_working(url: str) -> bool:
    import urllib.request
    import urllib.error
    from urllib.parse import urlparse

    url = url.strip().strip('()[]')
    parsed = urlparse(url)
    if not parsed.scheme or parsed.scheme not in ('http', 'https'):
        return False

    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=8.0) as response:
            if response.status >= 400:
                print(f"[crew] URL check status {response.status}: {url}", flush=True)
                return False

            html = response.read(4096).decode("utf-8", errors="ignore").lower()

            if "not on our system" in html or "bailii >> not found" in html:
                print(f"[crew] Soft 404 on BAILII: {url}", flush=True)
                return False
            if "<title>not found</title>" in html or "<title>404" in html:
                print(f"[crew] Soft 404 page: {url}", flush=True)
                return False
            if "<h1>not found</h1>" in html or "<h1>404" in html:
                print(f"[crew] Soft 404 heading: {url}", flush=True)
                return False

            return True
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return True
        print(f"[crew] HTTP Error {e.code} checking {url}", flush=True)
        return False
    except Exception as e:
        print(f"[crew] Exception checking {url}: {e}", flush=True)
        return False


def verify_urls_in_parallel(urls: list[str]) -> dict[str, bool]:
    import concurrent.futures

    unique_urls = list(set(urls))
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(is_url_working, url): url for url in unique_urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception:
                results[url] = False

    return results


def search_tavily(query: str) -> list[dict]:
    import urllib.request
    import os

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("[crew] WARNING: TAVILY_API_KEY is not set — web search is disabled, evidence will be vault-only.", flush=True)
        return []

    url = "https://api.api.tavily.com/search" if "api.api.tavily.com" in os.getenv("TAVILY_API_URL", "") else "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": 4,
        "search_depth": "advanced",
        "include_domains": ALLOWED_DOMAINS
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=6.0) as response:
            data = json.loads(response.read().decode("utf-8"))
            results = data.get("results", [])
            print(f"[crew] Tavily raw response for '{query}': {len(results)} total results before domain filter", flush=True)
            return [r for r in results if r.get("url") and _is_allowed_domain(r["url"])]
    except Exception as e:
        print(f"[crew] Tavily raw search FAILED for query '{query}': {type(e).__name__}: {e}", flush=True)
        return []


def _build_legal_queries(case_context: str, rejection_reason: str) -> list[str]:
    try:
        prompt = (
            "Given this legal case context, generate 3 short UK legal search "
            "queries (max 8 words each) that would find relevant statutes or "
            "case law on bailii.org, legislation.gov.uk, gov.uk, or scotcourts.gov.uk. "
            "Focus on the legal cause of action (e.g. 'employer negligence duty of care', "
            "'Health and Safety at Work Act', 'PUWER 1998 forklift'), not incident "
            "narrative details like names, dates, or locations. "
            "Return ONLY a JSON list of strings, nothing else.\n\n"
            f"CASE CONTEXT:\n{case_context[:1200]}\n\n"
            f"REJECTION FEEDBACK:\n{rejection_reason[:400]}"
        )
        resp = litellm.completion(
            model=llm.model if hasattr(llm, "model") else None,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        text = resp.choices[0].message.content
        queries = _try_parse_json(text) or []

        if isinstance(queries, dict):
            queries = list(queries.values())

        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        if queries:
            return queries[:3]
    except Exception as e:
        print(f"[crew] LLM query generation failed, falling back to keyword extraction: {e}", flush=True)

    words = [w for w in re.findall(r'\b\w+\b', case_context[:400]) if len(w) > 4]
    return [
        " ".join(words[:6]),
        "UK employer liability negligence breach",
        "health and safety at work act duty of care",
    ]


def run_research_crew(case_id: int, case_context: str) -> tuple[str, str]:
    from .vector_store import search_vector_store
    raw_vault_chunks = search_vector_store(case_id, case_context, top_k=15)
    vault_chunks = _scan_for_contamination(raw_vault_chunks or [], case_id)
    vault_evidence = "\n\n---\n\n".join(vault_chunks) if vault_chunks else "(No vault evidence available)"

    vault_doc_keywords = set()
    VAULT_DOC_PATTERNS = [
        "Incident Report", "Maintenance Log", "Policy Extract", "Timeline",
        "Supervisor Email", "Email Chain", "Teams Conversation", "Training Record",
        "Benchmark Case Pack", "Case Pack", "Witness Statement", "Risk Assessment",
        "CCTV", "Medical Report", "Accident Report", "H&S Policy",
        "Internal Investigation Report",
    ]
    for chunk in vault_chunks:
        for pattern in VAULT_DOC_PATTERNS:
            if pattern.lower() in chunk.lower():
                vault_doc_keywords.add(pattern.lower())
    print(f"[crew] Vault document keywords detected: {vault_doc_keywords}", flush=True)

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
        print(f"[crew] Failed to fetch rejection feedback for case {case_id}: {e}", flush=True)

    rejection_block = ""
    if rejection_reason:
        rejection_block = (
            "\n\nATTORNEY REJECTION FEEDBACK (MANDATORY):\n"
            f"{rejection_reason}\n\n"
            "You MUST specifically address this feedback."
        )

    verified_web_evidence = []
    verified_search_urls = set()

    queries_to_try = _build_legal_queries(case_context, rejection_reason)
    print(f"[crew] Generated {len(queries_to_try)} search queries: {queries_to_try}", flush=True)

    raw_tavily_results = []
    seen_urls: set[str] = set()
    MAX_HITS_PER_QUERY = 3

    for query in queries_to_try:
        print(f"[crew] Running pre-search query: '{query}'", flush=True)
        hits = search_tavily(query)
        print(f"[crew] Query '{query}' returned {len(hits)} whitelisted hits", flush=True)

        new_hits_count = 0
        for hit in hits[:MAX_HITS_PER_QUERY]:
            url = hit.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                raw_tavily_results.append(hit)
                new_hits_count += 1
        print(f"[crew]   -> {new_hits_count} new unique URLs added", flush=True)

    if not raw_tavily_results:
        print("[crew] Initial queries returned 0 hits. Running broad UK legal fallback queries...", flush=True)
        fallback_queries = [
            "UK employer liability negligence breach legislation.gov.uk",
            "health and safety at work act duty of care legislation.gov.uk",
            "Provision and Use of Work Equipment Regulations PUWER legislation.gov.uk"
        ]
        for fq in fallback_queries:
            print(f"[crew] Running fallback query: '{fq}'", flush=True)
            hits = search_tavily(fq)
            print(f"[crew] Fallback query '{fq}' returned {len(hits)} whitelisted hits", flush=True)
            for hit in hits[:2]:
                url = hit.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    raw_tavily_results.append(hit)

    whitelisted_hits = [r for r in raw_tavily_results if r.get("url") and _is_allowed_domain(r["url"])]

    if whitelisted_hits:
        print(
            f"[crew] {len(whitelisted_hits)} whitelisted hits before reachability check: "
            f"{[h['url'] for h in whitelisted_hits]}",
            flush=True,
        )
        urls_to_verify = [r["url"] for r in whitelisted_hits]
        reachability_map = verify_urls_in_parallel(urls_to_verify)

        for hit in whitelisted_hits:
            u = hit["url"]
            if reachability_map.get(u, False):
                verified_search_urls.add(u)
                verified_web_evidence.append({
                    "title": hit.get("title", "Legal Source"),
                    "url": u,
                    "snippet": hit.get("content", "")
                })
            else:
                print(f"[crew] Pre-search discarded UNREACHABLE url: {u}", flush=True)

    if verified_web_evidence:
        web_evidence_parts = []
        for idx, item in enumerate(verified_web_evidence, start=1):
            web_evidence_parts.append(
                f"[Web Source {idx}]\nTitle: {item['title']}\nURL: {item['url']}\nSnippet: {item['snippet']}"
            )
        web_evidence_block = "\n\n".join(web_evidence_parts)
    else:
        web_evidence_block = "No authoritative legal sources were located on bailii.org, legislation.gov.uk, gov.uk or scotcourts.gov.uk."

    verified_evidence_text = (
        f"--- VERIFIED OFFICIAL LEGAL SEARCH RESULTS (BAILII/GOV.UK/LEGISLATION.GOV.UK) ---\n"
        f"{web_evidence_block}\n\n"
        f"--- VAULT EVIDENCE (CLIENT PDFs & DOCUMENTS) ---\n"
        f"{vault_evidence[:2000]}"
    )

    print(f"[crew] Pre-search complete. Verified URLs passed to LLM: {list(verified_search_urls)}", flush=True)
    print(f"[crew] Total verified web evidence items: {len(verified_web_evidence)}", flush=True)

    legal_strategist = Agent(
        role="Lexis Senior Civil Litigation Strategist",
        goal=(
            "Synthesize verified legal evidence and internal context into a highly structured, "
            "strategic legal memo for the Managing Partner."
        ),
        backstory=(
            "You are an expert civil litigation strategist and Senior Partner. You take provided evidence, "
            "identify key liability vulnerabilities, analyze defense arguments, determine settlement triggers, "
            "and format all information into the required schema."
        ),
        llm=llm,
        tools=[],
        verbose=True,
        max_iter=3,
    )

    strategy_generation_task = Task(
        description=(
            "SYSTEM ROLE: You are a Senior Civil Litigation Strategist.\n\n"
            f"CASE CONTEXT:\n{case_context[:1200]}\n\n"
            f"VERIFIED EVIDENCE BLOCK:\n{verified_evidence_text}\n\n"
            f"{rejection_block}\n\n"
            "INSTRUCTIONS & EVIDENCE GUIDELINES (STRICT ZERO-HALLUCINATION):\n"
            "1. Synthesize the provided VERIFIED EVIDENCE BLOCK into a comprehensive, strategic legal memo for the Managing Partner.\n"
            "2. Fill the `evidence_log` with all relevant findings from the VERIFIED EVIDENCE BLOCK above.\n"
            "3. For Web Search items: copy the EXACT whitelisted URL (bailii.org, legislation.gov.uk, gov.uk, scotcourts.gov.uk) starting with https:// from the snippet into `web_url` and leave `document_name` as null.\n"
            "4. For internal Vault documents (Incident Report, Maintenance Log, Policy Document, Email Chain, Teams Conversation, Timeline Summary, etc.): set `document_name` to the exact title (e.g. 'Incident Report', 'Maintenance Log FLT-04') and set `web_url` strictly to null. ABSOLUTELY DO NOT generate, hallucinate, or attach external HTTP/HTTPS URLs to internal Vault documents or generic terms like 'email chain' or 'timeline'.\n"
            "5. Provide thorough, complete, professional legal reasoning for `liability_summary`, `settlement_trigger`, `barriers_to_defense`, and `next_tactical_move`. Write full, complete paragraphs — never cut off mid-sentence."
        ),
        expected_output=(
            "A structured JSON object matching the ResearchOutput schema with full, un-truncated legal analysis and strictly validated source citations."
        ),
        agent=legal_strategist,
        output_pydantic=ResearchOutput,
    )

    crew = Crew(
        agents=[legal_strategist],
        tasks=[strategy_generation_task],
        process=Process.sequential,
        verbose=True,
    )

    max_retries = 3
    retry_delay = 65

    result = None
    for attempt in range(max_retries):
        try:
            result = crew.kickoff()
            break
        except Exception as e:
            if "RateLimitError" in str(e) or "429" in str(e):
                if attempt < max_retries - 1:
                    print(f"\n[crew] Rate limit hit (429). Waiting {retry_delay}s before attempt {attempt + 2}/{max_retries}...", flush=True)
                    time.sleep(retry_delay)
                    continue
            raise

    output_obj = None
    raw_dict = None

    if hasattr(result, 'pydantic') and result.pydantic:
        output_obj = result.pydantic
    else:
        raw_text = str(result).strip()
        parsed = _extract_json_from_raw_text(raw_text)
        if parsed:
            raw_dict = parsed
            output_obj = _normalize_to_research_output(parsed)
        elif raw_text and raw_text.count("{") > raw_text.count("}"):
            print(
                "[crew] Output appears to be truncated JSON (unbalanced braces) and "
                "could not be parsed at all. Check max_tokens ceiling in get_crew_llm().",
                flush=True,
            )

    if output_obj:
        verified_evidence = []
        for item in output_obj.evidence_log:
            url = (item.web_url or "").strip()
            doc = (item.document_name or "").strip()
            if url and url.startswith("http"):
                if _is_allowed_domain(url):
                    verified_evidence.append(item)
                else:
                    print(f"[crew] Discarding non-whitelisted domain URL: '{item.title}' (URL: {url})", flush=True)
            elif doc:
                verified_evidence.append(item)

        output_obj.evidence_log = verified_evidence
        print(f"[crew] Evidence log filtered: {len(verified_evidence)} items kept", flush=True)

        valid_sources = set(verified_search_urls)
        for item in output_obj.evidence_log:
            doc = (item.document_name or "").strip()
            if doc:
                clean_ref = _clean_vault_reference(doc)
                if clean_ref:
                    valid_sources.add(clean_ref)

        output_obj.source_index = list(valid_sources)
        print(f"[crew] Programmatic source_index set: {output_obj.source_index}", flush=True)

    if output_obj:
        memo_text = _format_research_output(output_obj)
        ai_reasoning = output_obj.liability_summary or ""
    elif raw_dict:
        raw_evidence = raw_dict.get("evidence_log", [])
        verified_raw_evidence = []
        for item in raw_evidence:
            if isinstance(item, dict):
                item_url = str(item.get("source_url", "")).strip()
                item_title = str(item.get("title", "")).lower()
                if item_url.startswith("http"):
                    if _is_allowed_domain(item_url) and item_url in verified_search_urls:
                        verified_raw_evidence.append(item)
                    else:
                        print(f"[crew] Discarding raw_dict evidence: '{item.get('title')}' (URL: {item_url})", flush=True)
                elif item_url:
                    cleaned = _clean_vault_reference(item_url).lower()
                    is_known_vault = any(keyword in cleaned or keyword in item_title for keyword in vault_doc_keywords)
                    if is_known_vault:
                        verified_raw_evidence.append(item)
                    else:
                        print(f"[crew] Discarding raw_dict evidence (not in vault): '{item.get('title')}' (source: {item_url})", flush=True)
        raw_dict["evidence_log"] = verified_raw_evidence
        raw_dict["source_index"] = list(verified_search_urls)
        print(f"[crew] raw_dict filtered: {len(verified_raw_evidence)} evidence items, source_index: {raw_dict['source_index']}", flush=True)

        memo_text = _format_raw_research_dict(raw_dict)

        source_lines = ["\n### Source Index"]
        for url in raw_dict["source_index"]:
            url_clean = url.strip()
            if url_clean.startswith("http"):
                if _is_allowed_domain(url_clean):
                    source_lines.append(f"- [{url_clean}]({url_clean})")
                else:
                    print(f"[crew] raw_dict source_index: blocked non-whitelisted URL: {url_clean}", flush=True)
            else:
                clean_name = _clean_vault_reference(url_clean)
                if clean_name:
                    source_lines.append(f"- 📄 {clean_name}")
        memo_text += "\n".join(source_lines)

        raw_summary = raw_dict.get("liability_summary", "")
        if isinstance(raw_summary, dict):
            ai_reasoning = _flatten_liability_summary(raw_summary)
        else:
            ai_reasoning = str(raw_summary)
    else:
        # Fallback: raw LLM output could not be parsed into structured JSON.
        # Sanitize to strip any hallucinated URLs outside the whitelist.
        memo_text = _strip_non_whitelisted_urls(str(result))
        ai_reasoning = ""

    for marker in _CONTAMINATION_MARKERS:
        memo_text = re.sub(re.escape(marker), "", memo_text, flags=re.IGNORECASE)

    # Defense-in-depth: final sweep to catch any non-whitelisted URL that
    # slipped through structured filtering (e.g. hallucinated in free-text
    # fields like liability_summary or settlement_trigger).
    memo_text = _strip_non_whitelisted_urls(memo_text)

    return memo_text.strip(), ai_reasoning