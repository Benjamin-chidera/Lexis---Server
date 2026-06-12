import os
import sys
from dotenv import load_dotenv
from sqlmodel import create_engine, Session, SQLModel, select
from sqlalchemy.orm import make_transient

# Add server directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import User, Case, CaseMessage, Alert


def sync():
    # Load environment variables
    load_dotenv()
    postgres_url = os.getenv("DATABASE_URL")
    if not postgres_url:
        print("Error: DATABASE_URL not found in environment variables.")
        return

    sqlite_url = "sqlite:///./legal_assistant.db"

    print(f"Connecting to Postgres...")
    print(f"Connecting to SQLite: {sqlite_url}")

    pg_engine = create_engine(postgres_url)
    sqlite_engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})

    # Create tables in SQLite if they don't exist
    SQLModel.metadata.create_all(sqlite_engine)

    # Order models to satisfy foreign key dependencies during copy
    models = [User, Case, CaseMessage, Alert]

    with Session(pg_engine) as pg_session, Session(sqlite_engine) as sl_session:
        for model in models:
            # Clear existing data in SQLite to avoid primary key conflicts
            existing_sl_items = sl_session.exec(select(model)).all()
            if existing_sl_items:
                print(f"Clearing existing {model.__name__} records in SQLite...")
                for item in existing_sl_items:
                    sl_session.delete(item)
                sl_session.commit()

            # Fetch all records from Postgres
            items = pg_session.exec(select(model)).all()
            print(f"Found {len(items)} records for {model.__name__} in Postgres.")

            # Copy to SQLite
            for item in items:
                # Make the item transient so it can be added to the SQLite session
                make_transient(item)
                sl_session.add(item)
            sl_session.commit()
            print(f"Successfully copied {len(items)} records for {model.__name__}.")


if __name__ == "__main__":
    sync()


# uv run scratch/sync_db.py
