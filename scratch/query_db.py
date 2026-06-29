import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from sqlmodel import Session, select
from database import engine
from models import Case

with Session(engine) as session:
    statement = select(Case)
    results = session.exec(statement).all()
    print("Number of cases in database:", len(results))
    print("Database URL:", os.getenv("DATABASE_URL"))
    for case in results:
        print(f"ID: {case.id} | User ID: {case.user_id} | Status: {case.status} | Context: {case.context[:40] if case.context else 'None'}...")
