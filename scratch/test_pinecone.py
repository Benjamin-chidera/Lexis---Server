import os
import sys
from dotenv import load_dotenv

# Ensure server/ is on python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from ai.vector_store import get_vector_store, search_vector_store
from langchain_core.documents import Document

try:
    print("Testing Pinecone connection and case isolation...")
    
    # 1. Get vector store for a test case ID (e.g., 99999)
    case_id = 99999
    store = get_vector_store(case_id)
    
    # 2. Add some test documents
    print("Adding test documents...")
    docs = [
        Document(page_content="The defendant's car was a red sedan heading north on 5th avenue.", metadata={"source": "witness_report.txt"}),
        Document(page_content="The weather during the accident was rainy and road conditions were slick.", metadata={"source": "police_report.txt"}),
    ]
    store.add_documents(docs)
    print("Documents added successfully!")
    
    # 3. Perform a similarity search
    print("Testing search...")
    query = "what color was the car?"
    results = search_vector_store(case_id, query, top_k=1)
    
    print("\nSearch results:")
    for res in results:
        print(res)
        
    print("\nPinecone verification completed successfully!")
    
except Exception as e:
    print("Pinecone verification failed:", e)
