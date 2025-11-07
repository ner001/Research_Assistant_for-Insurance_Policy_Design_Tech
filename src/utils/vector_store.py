from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStoreManager:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = None

    def create_vector_store(self, texts, metadatas):
        """Create a vector store from the provided texts and metadata."""
        self.vectorstore = FAISS.from_texts(texts=texts, embedding=self.embeddings, metadatas=metadatas)

    def similarity_search(self, query, k=5):
        """Perform a similarity search in the vector store."""
        if self.vectorstore is None:
            raise ValueError("Vector store has not been created yet.")
        return self.vectorstore.similarity_search(query, k=k)

    def get_vector_store(self):
        """Return the current vector store."""
        return self.vectorstore