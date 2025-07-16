from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Tuple

# Initialize the embedding model (uses OPENAI_API_KEY from environment)
embeddings = OpenAIEmbeddings()

class VectorStore:
    def __init__(self):
        self.store = None

    def ingest_documents(self, docs: List[str]):
        """
        Ingest a list of documents (strings) and create a FAISS vector store.
        """
        langchain_docs = [Document(page_content=doc) for doc in docs]
        self.store = FAISS.from_documents(langchain_docs, embeddings)

    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Perform a similarity search and return top-k documents with scores.
        """
        if not self.store:
            raise ValueError("No documents ingested yet.")
        results = self.store.similarity_search_with_score(query, k=k)
        return [(doc.page_content, score) for doc, score in results] 