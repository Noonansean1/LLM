# vectordb/azure_search.py
import os, time, json, requests
from typing import Iterable, Dict, Any, List
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
from azure.search.documents.models import VectorizedQuery

load_dotenv()

from vectordb.base import VectorStore

# Azure AI Search
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = "customer-service-rag-index"
VECTOR_FIELD = os.getenv("AZURE_SEARCH_VECTOR_FIELD", "contentVector")

class AzureSearchStore(VectorStore):
    """
    Assumes an index with fields:
      - id (key, Edm.String)
      - content (Edm.String)
      - vector (Collection(Single)) with appropriate dimensions
      - (optional) metadata fields
    """
    def __init__(self, index_name: str):
        self.endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.key = os.getenv("AZURE_SEARCH_KEY")
        if not (self.endpoint and self.key):
            raise RuntimeError("Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY")
        self.index_name = index_name
        self.client = SearchClient(self.endpoint, index_name, AzureKeyCredential(self.key))

    def upsert(self, items: Iterable[Dict[str, Any]]) -> None:
        # 'merge_or_upload' works well for idempotent updates
        self.client.merge_or_upload_documents(list(items))

    def search(self, query_vector, k):

        url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/search?api-version=2024-07-01"
        headers = {"Content-Type": "application/json", "api-key": SEARCH_KEY}
        payload = {
            "top": k,
            "vectorQueries": [
                {
                    "kind": "vector",
                    "vector": query_vector,
                    "fields": VECTOR_FIELD,
                    "k": k
                }
            ]
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload))
        if resp.status_code >= 400:
            try:
                print("❌ Search error:", resp.json())
            except Exception:
                print("❌ Search error:", resp.text)
            resp.raise_for_status()
        return resp.json().get("value", [])
