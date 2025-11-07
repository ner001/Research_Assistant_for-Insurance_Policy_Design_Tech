import os
from pypdf import PdfReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    return text

def chunk_text(text: str, chunk_size=1000, chunk_overlap=200):
    """Chunk the text into smaller segments for summarization."""
    document = Document(text=text)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents([document])
    return [node.text for node in nodes]