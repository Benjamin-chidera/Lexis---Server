import os
import sys

# Ensure server/ is on python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from ai.model_providers import get_crew_llm, get_embeddings, get_chat_model

try:
    print("Testing get_embeddings()...")
    embeder = get_embeddings()
    res = embeder.embed_query("hello world")
    print("Embeddings success! Vector length:", len(res))
except Exception as e:
    print("Embeddings failed:", e)

try:
    print("\nTesting get_chat_model()...")
    chat = get_chat_model()
    res = chat.invoke("hello, tell me your model name in one word")
    print("Chat success! Response:", res)
except Exception as e:
    print("Chat failed:", e)

try:
    print("\nTesting get_crew_llm()...")
    llm = get_crew_llm()
    res = llm.call(messages=[{"role": "user", "content": "hi"}])
    print("Crew LLM success! Response:", res)
except Exception as e:
    print("Crew LLM failed:", e)
  