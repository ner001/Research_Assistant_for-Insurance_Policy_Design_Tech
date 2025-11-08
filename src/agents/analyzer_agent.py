from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from datetime import datetime
from mcp import Context

  
class DocumentAnalyzerAgent:
    def __init__(self, pdf_path: str = None, vectorstore=None, metadatas=None, context: Context = None):
        self.vectorstore = vectorstore
        self.metadatas = metadatas or []
        self.pdf_path = pdf_path
        self.llm = Ollama(model="llama3:8b", temperature=0.3)
        self.context = Context() if context is None else context

    def load_vectorstore(self, vectorstore, metadatas=None):
        self.vectorstore = vectorstore
        if metadatas:
            self.metadatas = metadatas

    def query_vectorstore(self, query: str, k: int = 3):
        if not self.vectorstore:
            return []
        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def answer_query(self, query: str) -> str:
        """Answer a specific query using the most relevant chunk and LLM for better output"""
        if not self.vectorstore:
            return "Please process a document first."
        if not query.strip():
            return "Query cannot be empty." 

        print(f"🔍 Querying vector store for: {query}")
        
        # Get top relevant chunk
        response = self.query_vectorstore(query, k=1)
        if not response:
            return "No relevant document chunks found."
        top_doc = response[0]

        # write selected evidence into context for other agents
        if self.context:
            self.context.set("last_query", query)
            self.context.set("last_top_chunk", {
                "summary": top_doc.metadata.get("summary"),
                "original_text": top_doc.metadata.get("original_text"),
                "source": top_doc.metadata.get("source")
            })
        # Now you can access metadata
        original_text = top_doc.metadata.get("original_text", "")
        summary = top_doc.metadata.get("summary", "")
        keywords = top_doc.metadata.get("keywords", "")
        questions = top_doc.metadata.get("questions", "")

        # Create the output prompt

        prompt = f"""
        You are a highly knowledgeable insurance AI assistant with expertise in life, health, and property insurance policies, procedures, and regulations. 
        Your task is to answer the user's query accurately and comprehensively using the document content provided.

        Below you have:

        - **Document Content:** Full text of the relevant insurance document chunk.
        - **Summary:** A concise overview of the chunk key points.
        - **Keywords:** Important insurance terms and concepts from the chunk.
        - **Generated Questions:** Potential questions that this chunk answers.

        Use all of this information to understand the context fully. If the user's query can be answered using the provided information, give a detailed, structured, and professional answer. 
        Include relevant details such as policy conditions, coverage, procedures, and exceptions where applicable. 

        If the information is insufficient to answer the query, clearly state that the answer cannot be determined from the document.

        **Document Content:**
        {original_text}

        **Summary:**
        {summary}
        **Keywords:**
        {keywords}
        **Generated Questions:**
        {questions}

        **User Query:**
        {query}

        **Instructions for Answering:**
        - Prioritize clarity, precision, and relevance to insurance context.
        - Use terminology appropriate for insurance professionals.
        - Provide examples if necessary to illustrate key points.
        - Do not make assumptions beyond the information provided.
        - If multiple interpretations are possible, outline them clearly.

        Provide the answer below:
        """
        response = self.llm.invoke(prompt)
        return response
