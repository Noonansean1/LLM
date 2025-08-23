# agent_connect_loop_high.py
import os, time, json, requests
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.ai.projects import AIProjectClient
from azure.identity import AzureCliCredential
from azure.ai.agents.models import ListSortOrder

load_dotenv()

# Azure OpenAI (embeddings)
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
EMBED_MODEL = os.getenv("AZURE_EMBED_MODEL", "text-embedding-3-small")
AOAI_API_VERSION = "2024-02-15-preview"

# Azure AI Search
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = "customer-service-rag-index"
VECTOR_FIELD = os.getenv("AZURE_SEARCH_VECTOR_FIELD", "contentVector")

# Foundry Agent
FOUNDRY_PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT") \
    or "https://adm-sn-all-resource.services.ai.azure.com/api/projects/adm-sn-all"
AGENT_ID = os.getenv("FOUNDRY_AGENT_ID") or "asst_8yIHi2J290Ko7YyhuItaJL91"

# Retrieval behavior
TOP_K = int(os.getenv("TOP_K", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.80"))

# ---------- Clients ----------
aoai = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_KEY,
    api_version=AOAI_API_VERSION,
)

project = AIProjectClient(
    credential=AzureCliCredential(),
    endpoint=FOUNDRY_PROJECT_ENDPOINT,
)

# ---------- Helpers ----------
def embed(text: str):
    resp = aoai.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def search_vectors(query_vector):
    url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/search?api-version=2024-07-01"
    headers = {"Content-Type": "application/json", "api-key": SEARCH_KEY}
    payload = {
        "top": TOP_K,
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": query_vector,
                "fields": VECTOR_FIELD,
                "k": TOP_K
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

from pathlib import Path

def read_local(filename: str) -> str:
    base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    p = base / filename
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    return p.read_text(encoding="utf-8", errors="ignore")
