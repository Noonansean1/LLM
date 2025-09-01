# agent_connect_loop_high.py
import ast
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


def load_pairs_from_text(raw: str):
    """
    Accepts text that is:
      - JSON: [["Title","Content"], ...]  OR [{"title":"...","content":"..."}, ...]
      - Python literal: [("Title","Content"), ...]  (possibly assigned: FAQ_ENTRIES = [ ... ])
    Returns: List[Tuple[str, str]]
    """
    raw = raw.strip()

    # 1) Try JSON first
    try:
        data = json.loads(raw)
    except Exception:
        # 2) Try to extract the list literal (handles 'VAR = [ ... ]' too)
        m = re.search(r'\[[\s\S]*\]', raw)
        if not m:
            raise ValueError("Could not find a JSON or Python list in the file.")
        data = ast.literal_eval(m.group(0))

    # Normalize to list of (title, content)
    pairs = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            t, c = item
        elif isinstance(item, dict) and "title" in item and "content" in item:
            t, c = item["title"], item["content"]
        else:
            raise ValueError("Each item must be (title, content) or {'title','content'}.")

        t = str(t).strip()
        c = str(c).strip()
        if c:  # skip empty content
            pairs.append((t, c))

    if not pairs:
        raise ValueError("Parsed zero valid (title, content) pairs.")
    return pairs