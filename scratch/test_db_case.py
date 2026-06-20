import os
import sys
import json
from dotenv import load_dotenv

# Ensure server/ is on python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from sqlmodel import Session, select
from database import engine, create_tables
from models import Case

print("Creating tables if they don't exist...")
create_tables()

print("Testing database insertion of Case...")
try:
    with Session(engine) as session:
        new_case = Case(
            context="Test case insertion",
            urls_json=json.dumps(["https://example.com"]),
            pdf_paths_json=json.dumps([]),
            image_paths_json=json.dumps([]),
            status="pending",
            user_id=1,  # Set to an existing user id (discoverbenix@gmail.com has id 1)
        )
        session.add(new_case)
        session.commit()
        session.refresh(new_case)
        print("Success! Case inserted with ID:", new_case.id)
        
        # Now clean up / delete it
        session.delete(new_case)
        session.commit()
        print("Success! Case deleted successfully.")
except Exception as e:
    print("Database insertion failed:", e)
