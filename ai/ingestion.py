import os
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()   

from langchain_community.document_loaders import PyPDFLoader

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Reads a PDF from disk and returns all its text, page by page,
    using LangChain's PyPDFLoader.
    """
    if not os.path.exists(pdf_path):
        return f"[PDF file not found: {pdf_path}]"

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        pages_text = []
        for doc in documents:
            page_number = doc.metadata.get('page', 0) + 1
            if doc.page_content.strip():
                pages_text.append(f"--- Page {page_number} ---\n{doc.page_content.strip()}")

        if not pages_text:
            return "[No readable text found in this PDF]"

        return "\n\n".join(pages_text)

    except Exception as error:
        return f"[Error reading PDF: {str(error)}]"


def extract_text_from_url(url: str) -> str:
    """
    Fetches a web page using LangChain's TavilyExtract tool with a 
    standard requests fallback if Tavily fails or returns empty content.
    """
    # 1. Try TavilyExtract first (higher quality cleaning)
    try:
        from langchain_tavily import TavilyExtract
        tool = TavilyExtract()
        results = tool.invoke({"urls": [url]})
        
        import json
        text = ""
        if isinstance(results, str):
            try:
                parsed = json.loads(results)
                if isinstance(parsed, list) and len(parsed) > 0:
                    text = parsed[0].get("raw_content", "") or parsed[0].get("content", "")
            except json.JSONDecodeError:
                text = results
        elif isinstance(results, list) and len(results) > 0:
            text = results[0].get("raw_content", "") or results[0].get("content", "")

        # If Tavily returned meaningful content, return it
        if text.strip() and "No Extracted Results" not in text and "Tavily Dashboard" not in text:
            return text[:5000]
            
    except Exception as error:
        print(f"[ingestion] TavilyExtract failed for {url}: {str(error)}")

    # 2. Fallback to standard requests + basic regex cleaning
    try:
        import requests
        import re
        
        print(f"[ingestion] Using fallback scraper for {url}")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        raw_html = response.text
        # Basic cleaning
        raw_html = re.sub(r"<script[^>]*>.*?</script>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
        raw_html = re.sub(r"<style[^>]*>.*?</style>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
        plain_text = re.sub(r"<[^>]+>", " ", raw_html)
        plain_text = re.sub(r"\s+", " ", plain_text).strip()

        return plain_text[:5000]

    except Exception as fallback_error:
        return f"[Error fetching URL: {str(fallback_error)}]"


def extract_all_vault_content(pdf_paths: List[str], urls: List[str]) -> Dict:
    """
    Extracts text from every PDF and URL in a case vault.
    """
    extracted_pdfs = []
    for path in pdf_paths:
        filename = os.path.basename(path)
        content = extract_text_from_pdf(path)
        extracted_pdfs.append({
            "filename": filename,
            "path": path,
            "content": content,
        })

    extracted_urls = []
    for url in urls:
        content = extract_text_from_url(url)
        extracted_urls.append({
            "url": url,
            "content": content,
        })

    return {
        "pdfs": extracted_pdfs,
        "urls": extracted_urls,
    }
