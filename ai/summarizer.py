"""
summarizer.py

Provides AI-driven summarization for vault items.
Used by the frontend to display beautifully formatted details when a user clicks on a URL.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from .ingestion import extract_text_from_url
from .model_providers import get_chat_model

SYSTEM_PROMPT = """You are an expert legal and technical analyst. 
Your job is to read raw text extracted from a webpage and write a beautiful, 
comprehensive Markdown summary of it.

Rules:
1. Use a large H1 header (#) for the main title of the page.
2. Use H2 (##) and H3 (###) headers to break down the content into logical sections.
3. Extract and present the most important information, key facts, and features.
4. Use bullet points for readability.
5. If it is a company or product page, explain what it does and its main value proposition.
6. Make it look extremely clean, structured, and easy to read. Do not output raw HTML, only Markdown.
7. Do not include your reasoning process (e.g. <think> blocks), just the final Markdown.
"""

def generate_url_summary(url: str, raw_text: str = None) -> str:
    """
    Asks the LLM to format the raw text of a URL into a beautiful Markdown description.
    If raw_text is not provided, it will be fetched on demand.
    """
    if raw_text is None:
        raw_text = extract_text_from_url(url)
    
    if raw_text.startswith("[Error"):
        return f"# Error\n\nCould not fetch content from {url}.\n\n**Details:** {raw_text}"
        
    if not raw_text.strip():
        return f"# Empty Page\n\nNo readable text could be extracted from {url}."

    try:
        llm = get_chat_model(temperature=0)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"URL: {url}\n\nRAW TEXT EXTRACTED:\n{raw_text}"),
        ]

        response = llm.invoke(messages)
        content = response.content.strip()

        print("\n\n content : ", content)   
        
        # Sometimes DeepSeek models include <think>...</think> blocks in the output despite instructions.
        # We should strip them out so the UI looks clean.
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        return content

    except Exception as error:
        return f"# Summary Failed\n\nAn error occurred while analyzing the page.\n\n**Details:** {str(error)}"
