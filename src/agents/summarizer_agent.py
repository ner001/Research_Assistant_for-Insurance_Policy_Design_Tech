from langchain_community.llms import Ollama
from mcp import Context

class SummarizerAgent:
    def __init__(self, vectorstore=None, metadatas=None, context=None):
        self.vectorstore = vectorstore
        self.metadatas = metadatas or []
        self.context = context
        # init LLM as you use elsewhere
        self.llm = Ollama(model="llama3:8b", temperature=0.3)

    def generate_summary(self):
        # Prefer context-provided chunk summaries (faster / consistent)
        chunk_summaries = None
        if self.context:
            chunk_summaries = self.context.get("chunk_summaries")

        if not chunk_summaries:
            # fallback: extract from metadatas
            chunk_summaries = [md.get("summary") for md in self.metadatas if isinstance(md, dict) and md.get("summary")]

        # combine/populate prompt for LLM
        combined = "\n\n".join(s for s in chunk_summaries if s)
        prompt = f"""
            You are an advanced AI assistant specialized in document comprehension and summarization. 
            Your task is to read and deeply understand a set of short summaries (each corresponding to a small chunk of the same document). 
            Then, synthesize a **comprehensive, faithful, and well-structured global summary** that accurately reflects the content and intent of the entire document.

            Each chunk summary captures local information. Your goal is to combine these insights to recover the full meaning of the document, 
            eliminating redundancy and ensuring logical flow. Identify recurring themes, major sections, and any cause-effect or procedural relationships.

            Instructions:
            1. **Do not hallucinate** or add information not supported by the chunks.
            2. If specific information is missing, clearly write “not specified”.
            3. **Focus** on:
            - The main purpose and context of the document.
            - Key topics, decisions, or findings.
            - Important entities, dates, or quantitative details.
            - Relationships between concepts or sections.
            4. Write in **a clear, professional tone**, suitable for an executive or research summary.
            5. Length: around **8 well-structured sentences** (or 1 concise paragraph).

            Chunk Summaries:
            {combined}

            Now, write the **final global summary** that captures the full meaning of the document.
            """
        response = self.llm.invoke(prompt)
        return response