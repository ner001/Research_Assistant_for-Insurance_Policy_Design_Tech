import os
from pypdf import PdfReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document


def extract_text_with_pypdf2(pdf_path: str) -> str:
    """Extract text from PDF using pypdf"""
    reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    print(f"Extracted {len(text)} characters")
    return text
