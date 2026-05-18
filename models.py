from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    corporate_email: str = Field(unique=True, index=True)
    name: str = Field(default="")
    password_hash: Optional[str] = Field(default=None)
    role: str = Field(default="employee")  # "admin" or "employee"
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Case(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    # The user who owns this case (null for cases created before auth was added)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")

    # The user's text notes / context about this case
    context: str = Field(default="")

    # SQLite doesn't support arrays, so we store urls as a JSON string
    urls_json: str = Field(default="[]")

    # A JSON string of a dictionary mapping urls to their generated Markdown summaries
    # Example: '{"https://example.com": "# Example\\n\\nThis is an example site..."}'
    url_summaries_json: str = Field(default="{}")

    # Same for PDF file paths on disk
    pdf_paths_json: str = Field(default="[]")

    # Paths to uploaded images
    image_paths_json: str = Field(default="[]")

    # Tracks what stage the AI processing is in
    status: str = Field(default="pending")

    # Auto-set to the current time when a case is created
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CaseMessage(SQLModel, table=True):
    """Stores every chat message for a case so they persist across sessions."""
    id: Optional[int] = Field(default=None, primary_key=True)
    case_id: int = Field(foreign_key="case.id")

    # Either "user" or "ai"
    role: str

    content: str

    # Optional citation stored as a JSON string
    # Example: '{"filename": "contract.pdf", "exhibit": "Exhibit A", "page": 3}'
    citation_json: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.utcnow)


class Alert(SQLModel, table=True):
    """
    AI-generated research alerts that appear on the alerts page.
    Created by the background research task when it finds relevant info.
    """
    id: Optional[int] = Field(default=None, primary_key=True)

    # Which case triggered this alert (null = general system alert)
    case_id: Optional[int] = Field(default=None, foreign_key="case.id")

    title: str

    # A short paragraph summary of what was found
    summary: str

    # urgent | strategic | routine
    severity: str = Field(default="routine")

    # unread | read | archived
    status: str = Field(default="unread")

    created_at: datetime = Field(default_factory=datetime.utcnow)
