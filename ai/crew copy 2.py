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
        if v and not v.startswith('http') and 'No.' not
<truncated 5729 bytes>
vidence Log (with inline source links)\n"
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
