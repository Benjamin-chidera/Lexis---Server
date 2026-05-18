from pydantic import BaseModel, Field, field_validator
from typing import List
from crewai import Agent, Task, Crew, Process
from .model_providers import get_crew_llm

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
            "If no URL is available, use the exact court docket identifier."
        )
    )
    evidence_type: str = Field(description="One of: URL, PDF Record, Court Docket, Media/Image")

    @field_validator('source_url')
    @classmethod
    def url_must_be_complete(cls, v: str) -> str:
        # Strip whitespace to avoid hidden truncation
        v = v.strip()

        # If it looks like a truncated URL (no http and not a docket number), flag it
        if v and not v.startswith('http') and 'No.' not in v and 'Docket' not in v:
            # Prefix a warning so it's visible in the UI
            return f"[INCOMPLETE URL — check source] {v}"

        return v


class ResearchOutput(BaseModel):
    # The full structured output of the research task
    liability_summary: str = Field(description="High-level summary of the opponent's legal vulnerabilities")
    evidence_log: List[EvidenceItem] = Field(description="List of all evidence items found, each with a source URL")
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
            "You are Lexis AI, an expert legal research assistant. "
            "You have been given extracted text from case documents and web pages. "
            "You answer questions clearly and always cite the source of each fact."
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
            "If you used the web search, cite the URL."
        ),
        expected_output=(
            "A well-reasoned answer to the user's legal question with inline "
            "citations pointing back to the source documents."
        ),
        agent=lexis_agent,
    )

    crew = Crew(
        agents=[lexis_agent],
        tasks=[answer_task],
        process=Process.sequential,
        verbose=False, 
    )

    result = crew.kickoff()
    return str(result)


def run_research_crew(case_context: str) -> str:
    """
    Runs a background research crew that conducts adversarial legal research,
    hunting for precedents, regulatory fines, and court rulings.

    Every finding MUST include an inline citation (URL or docket number).
    Includes a full Source Index of all sites searched for further reading.
    Returns a Strategic Legal Memo formatted in Markdown.
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
            _tavily = TavilySearch(max_results=5)  # 5 results per query for a richer source index

            @crewai_tool("Tavily Web Search")
            def tavily_search(query: str) -> str:
                """Search the web for legal information using Tavily."""
                return str(_tavily.invoke(query))

            search_tools = [tavily_search]
        except Exception:
            pass

    researcher = Agent(
        role="Adversarial Legal Intelligence Analyst",
        goal=(
            "Hunt for multi-dimensional legal leverage: find real regulatory fines, "
            "official court records, PDF filings, and any relevant photographic or documentary evidence "
            "that can be used to defend or counter the case described. "
            "You must provide a direct path (URL, PDF link, or Docket) for every piece of evidence."
        ),
        backstory=(
            "You are a battle-hardened legal investigator who specializes in digital evidence. "
            "You don't just read the news; you hunt for the raw documents, the original PDF filings, "
            "the official government records, and the photographic evidence that others miss. "
            "You provide the exact links to the source material so that the attorney can "
            "download the records immediately."
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
            "A highly detailed Strategic Legal Memo formatted in strict Markdown. "
            "The report must contain a categorized 'Evidence Log' AND a 'Full Source Index' section.\n\n"
            "CRITICAL RULES:\n"
            "1. Every single legal finding mentioned MUST have an inline hyperlink citation.\n"
            "2. NEVER truncate a URL. Always provide the complete URL starting with https://. "
            "For example: 'https://www.reuters.com/full/article/path' not 'https://www.reuters...'\n"
            "3. Categorize the Evidence Log by type: [URLs], [PDF Records], [Court Dockets], [Media/Images].\n"
            "4. The 'Full Source Index' section at the end must list EVERY URL you found relevant during your search.\n"
            "5. Never synthesize facts without appending its corresponding reference link.\n"
            "6. If data is missing for a category, mark it as [Awaiting Document Discovery]."
        ),
        agent=researcher,
        markdown=True,
        # Force the LLM to return a validated Pydantic object.
        # This prevents it from writing vague summaries without URLs.
        output_pydantic=ResearchOutput,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    # If pydantic output succeeded, convert it to a well-formatted Markdown string.
    # This lets the rest of the app treat it as a plain string (no breaking changes).
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

    # Fallback: return raw string if pydantic parsing didn't fire
    return str(result)
 