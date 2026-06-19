import os
from dotenv import load_dotenv
from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy import text as sqlalchemy_text
from typing import Generator

# Load environment variables
load_dotenv()

# SQLite file configuration (Commented out)
DATABASE_URL = "sqlite:///./legal_assistant.db"
# engine = create_engine(
#     DATABASE_URL,
#     connect_args={"check_same_thread": False},
# )

# Neon PostgreSQL configuration
# DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in environment variables")

# pool_pre_ping: test connections before use so stale ones (e.g. after a
# long LLM call where Neon closed the idle connection) are recycled silently.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)





def create_tables():
    # Creates all tables that are defined as SQLModel table=True models
    SQLModel.metadata.create_all(engine)

    # Add columns to existing alert tables that were created before
    # these fields were introduced. PostgreSQL's IF NOT EXISTS makes this
    # safe to run on every startup — it's a no-op when the column already exists.
    if engine.dialect.name != "sqlite":
        with engine.connect() as conn:
            conn.execute(
                sqlalchemy_text(
                    "ALTER TABLE alert ADD COLUMN IF NOT EXISTS ai_reasoning TEXT"
                )
            )
            conn.execute(
                sqlalchemy_text(
                    "ALTER TABLE alert ADD COLUMN IF NOT EXISTS review_status VARCHAR NOT NULL DEFAULT 'pending'"
                )
            )
            conn.execute(
                sqlalchemy_text(
                    "ALTER TABLE alert ADD COLUMN IF NOT EXISTS rejection_reason TEXT"
                )
            )
            conn.execute(
                sqlalchemy_text(
                    "ALTER TABLE \"case\" ADD COLUMN IF NOT EXISTS case_result_status VARCHAR NOT NULL DEFAULT 'pending'"
                )
            )
            conn.execute(
                sqlalchemy_text(
                    "ALTER TABLE \"case\" ADD COLUMN IF NOT EXISTS case_result_reason TEXT"
                )
            )
            conn.commit()


def get_session() -> Generator[Session, None, None]:
    # Used as a FastAPI dependency to inject a DB session into routes
    with Session(engine) as session:
        yield session
