import os, json, requests, textwrap
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# ---- Config from env ----
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
EMBED_MODEL = os.getenv("AZURE_EMBED_MODEL", "text-embedding-3-small")
AOAI_API_VERSION = "2024-02-15-preview"

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
VECTOR_FIELD = os.getenv("AZURE_SEARCH_VECTOR_FIELD", "contentVector")

TOP_K = int(os.getenv("TOP_K", "3"))
THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
API_VERSION = "2024-07-01"

missing = [k for k,v in [
  ("AZURE_OPENAI_ENDPOINT", AOAI_ENDPOINT),
  ("AZURE_OPENAI_API_KEY/AZURE_OPENAI_KEY", AOAI_KEY),
  ("AZURE_SEARCH_ENDPOINT", SEARCH_ENDPOINT),
  ("AZURE_SEARCH_KEY", SEARCH_KEY),
  ("AZURE_SEARCH_INDEX_NAME", INDEX_NAME)
] if not v]
if missing:
    raise SystemExit(f"Missing env vars: {', '.join(missing)}")

# ---- Clients ----
aoai = AzureOpenAI(azure_endpoint=AOAI_ENDPOINT, api_key=AOAI_KEY, api_version=AOAI_API_VERSION)
headers = {"api-key": SEARCH_KEY, "Content-Type": "application/json"}

def embed(text: str):
    return aoai.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def vector_search(qvec):
    url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/search?api-version={API_VERSION}"
    payload = {
        "top": TOP_K,
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": qvec,
                "fields": VECTOR_FIELD,
                "k": TOP_K
            }
        ]
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise SystemExit(f"Search error {r.status_code}: {detail}")
    return r.json().get("value", [])


if __name__ == "__main__":
    user_query = input("You: ").strip()
    if not user_query:
        raise SystemExit("Empty query.")

    qvec = embed(user_query)
    results = vector_search(qvec)

    if not results:
        print("No hits.")
        raise SystemExit(0)

    top = results[0]["@search.score"]
    print(f"\n🔎 Top similarity: {top:.3f}")
    used, skipped = [], []
    for r in results:
        row = {
            "score": r["@search.score"],
            "id": r.get("id") or r.get("key") or "",
            "title": r.get("title") or "",
            "content": (r.get("content") or "").replace("\n", " ")
        }
        (used if row["score"] >= THRESHOLD else skipped).append(row)

    print("\n✅ Will USE (>= threshold):")
    if used:
        for x in used:
            print(f"- {x['score']:.3f} [{x['title'] or x['id']}] {textwrap.shorten(x['content'], 150)}")
    else:
        print("- (none)")

    print("\n🚫 Will SKIP (< threshold):")
    if skipped:
        for x in skipped:
            print(f"- {x['score']:.3f} [{x['title'] or x['id']}] {textwrap.shorten(x['content'], 150)}")
    else:
        print("- (none)")
