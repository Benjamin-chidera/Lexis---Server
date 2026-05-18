from database import engine
from sqlmodel import Session, select
from models import CaseMessage, Case
import json

with Session(engine) as session:
    # Get the latest case that has messages
    cases = session.exec(select(Case).order_by(Case.id.desc())).all()
    for case in cases:
        messages = session.exec(select(CaseMessage).where(CaseMessage.case_id == case.id)).all()
        if messages:
            print(f"Found case {case.id} with {len(messages)} messages.")
            chat_history = [{"role": msg.role, "content": msg.content} for msg in messages]
            context = case.context
            
            pdfs_raw = getattr(case, 'pdf_paths_json', '[]') or '[]'
            pdf_paths = json.loads(pdfs_raw)
            
            urls_raw = getattr(case, 'urls_json', '[]') or '[]'
            urls = json.loads(urls_raw)
             
            imgs_raw = getattr(case, 'image_paths_json', '[]') or '[]'
            image_paths = json.loads(imgs_raw)

            from ai.graph import chat_graph
            
            initial_state = {
                "case_id": case.id,
                "context": context,
                "pdf_paths": pdf_paths,
                "urls": urls,
                "image_paths": image_paths,
                "chat_history": chat_history,
                "current_query": "What is the summary of the case?",
                "query_route": "",
                "vault_chunks": [],
                "web_results": "",
                "response": "",
                "citation": None,
            }
            
            print("Invoking graph...")
            final_state = chat_graph.invoke(initial_state)
            print("Response:", final_state.get("response"))
            break
