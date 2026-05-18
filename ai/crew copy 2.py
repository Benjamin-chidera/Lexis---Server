from pydantic import BaseModel, Field, field_validator
from typing import List
from crewai import Agent, Task, Crew, Process
from .model_providers import get_crew_llm

# Centralized LLM for all agents (Granite 2B)
llm = get_crew_llm()


# ---------------------------------------------------------------------------
# Pydantic output model — forces the LLM to return structured validated data.
# This is what guarantees every evidence item has a source_url field.
# ---------------------------------------------------------------------------
class EvidenceItem(BaseModel):
    title: str = Field(description="Short title of the finding")
    summary: str = Field(description="Complete, detailed summary of relevance to the case")
    source_url: str = Field(
        description=(
            "The FULL, complete URL starting with https://. "
            "NEVER truncate. NEVER use partial URLs. "
            "Example: 'https://www.ftc.gov/full/path/to/ruling'. "
            "If no URL is available, use the exact court docket identifier."
        )
    )
    evidence_type: str = Field(description="One of: URL, PDF Record, Court Docket, Media/Image")

    @field_validator('source_url')
    @classmethod
    def url_must_be_complete(cls, v: str) -> str:
        # Strip whitespace to catch hidden truncation
        v = v.strip()

        # Flag it in the UI if the URL looks truncated
        if v and not v.startswith('http') and 'No.' not in v and 'Docket' not in v:
            return f"[INCOMPLETE URL — check source] {v}"

        return v


class ResearchOutput(BaseModel):
    liability_summary: str = Field(description="Summary of the opponent's legal vulnerabilities")
    evidence_log: List[EvidenceItem] = Field(description="All evidence items found, each with a source URL")
    source_index: List[str] = Field(description="A complete list of ALL relevant URLs found during the search")


def _get_search_tools():
    """Safely initializes and returns the search tool array."""
    try:
        from crewai_tools import TavilySearchResults
        return [TavilySearchResults()]
    except Exception:
        try:
            from langchain_tavily import TavilySearch
            from crewai.tools import tool as crewai_tool
            
            # Increased to 5 to get more sources per search query
            _tavily = TavilySearch(max_results=5)

            @crewai_tool("Tavily Web Search")
            def tavily_search(query: str) -> str:
                """Search the web for legal information, precedents, and fines using Tavily."""
                return str(_tavily.invoke(query))

            return [tavily_search]
        except Exception:
            return []


def run_chat_crew(context: str, vault_content: str, user_query: str) -> str:
    """
    Runs a CrewAI crew to answer the user's question about the case.
    Prioritizes internal vault docs, then searches the web if missing.
    """
    search_tools = _get_search_tools()

    lexis_agent = Agent(
        role="Lexis AI Legal Assistant",
        goal="Answer legal questions using provided documents. Force absolute grounding with source tags.",
        backstory=(
            "You are an elite legal analyst. Your internal memory is unreliable. You must rely "
            "exclusively on the provided text records or live search engine tool responses. "
            "Every stated fact must immediately display its source indicator."
        ),
        llm=llm,
        tools=search_tools,
        verbose=False,
    )

    answer_task = Task(
        description=(
            "1. Read the Case Documents below. Check if they answer the Attorney Request.\n"
            "2. If documents are insufficient, use Tavily Web Search to locate missing details.\n\n"
            "Case Context:\n{context}\n\n"
            "Case Documents:\n{vault_content}\n\n"
            "Attorney Request: {user_query}"
        ),
        expected_output=(
            "A precise legal response. Every conclusion must have an inline citation. "
            "Format: [Source: filename.pdf, Page X] or [Source: URL] for web findings. "
            "If no source exists, explicitly state '[No documentary evidence found]'."
        ),
        agent=lexis_agent,
    )

    crew = Crew(
        agents=[lexis_agent],
        tasks=[answer_task],
        process=Process.sequential,
        verbose=False, 
    )

    # Injecting parameters cleanly via inputs dictionary to avoid prompt format breaking
    result = crew.kickoff(inputs={
        "context": context,
        "vault_content": vault_content,
        "user_query": user_query
    })
    return str(result)


def run_research_crew(case_context: str) -> str:
    """
    Runs an autonomous background research crew that conducts adversarial legal research,
    hunting for precedents, regulatory fines, and court rulings.
    """
    search_tools = _get_search_tools()

    researcher = Agent(
        role="Adversarial Legal Intelligence Analyst",
        goal="Hunt for actionable legal leverage (regulatory fines, court records, liabilities) against the opponent.",
        backstory=(
            "You are a hostile corporate intelligence investigator. You specialize in tracking down "
            "where massive entities have violated compliance regulations, lost class actions, or signed settlements. "
            "You do not generalize; you extract specific links and clear litigation metrics."
        ),
        llm=llm,
        tools=search_tools,
        verbose=True,       # Shows tool calls so we can debug if it skips search
        max_iter=5,         # Prevents infinite loops or single-step deadlocks
    )

    research_task = Task(
        description=(
            f"Conduct an intense, adversarial legal research hunt regarding the following case:\n"
            f"{case_context[:800]}\n\n"
            "Use your tools to locate the following evidence types:\n"
            "1. Official PDF Filings (court orders, regulatory decisions, whitepapers).\n"
            "2. Government Records (SEC, FTC, GDPR, or state-level legal databases).\n"
            "3. Case Law and Dockets (Pacer identifiers or official court URLs).\n"
            "4. Visual/Documentary Evidence (links to maps, charts, or images relevant to the context).\n\n"
            "You must return a deep list of sources. Find as many relevant sites as possible (up to 15) to build a full Source Index."
        ),
        expected_output=(
            "A Strategic Legal Memo in clean Markdown.\n\n"
            "Must include:\n"
            "- Liability Summary\n"
            "- Categorized Evidence Log (with inline source links)\n"
            "- Full Source Index (list of all URLs searched)\n\n"
            "CRITICAL RULES:\n"
            "1. NEVER truncate a URL. Always provide the complete URL starting with https://. "
            "Example: 'https://www.sec.gov/full/path/to/filing' NOT 'https://www.sec.gov...'\n"
            "2. Every finding MUST have a complete inline citation: [Source](full_url).\n"
            "3. ANTI-HALLUCINATION: Never write a legal case without appending its direct [Source: URL/Docket]."
        ),
        agent=researcher,
        markdown=True,
        # Force structured Pydantic output so every item must include a source URL
        output_pydantic=ResearchOutput,
    )

    crew = Crew(
        agents=[researcher], 
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff(inputs={"case_profile": case_context[:800]})

    # Convert structured Pydantic output to a formatted Markdown string
    if hasattr(result, 'pydantic') and result.pydantic:
        output = result.pydantic
        lines = []
        lines.append("## Strategic Legal Memo")
        lines.append(f"\n### Liability Summary\n{output.liability_summary}")
        lines.append("\n### Evidence Log")
        for item in output.evidence_log:
            lines.append(f"\n#### [{item.evidence_type}] {item.title}")
            lines.append(f"{item.summary}")
            lines.append(f"[Source]({item.source_url})")

        lines.append("\n### Full Source Index (For Further Reading)")
        for url in output.source_index:
            lines.append(f"- {url}")

        return "\n".join(lines)

    # Fallback if pydantic parsing didn't fire
    return str(result) 