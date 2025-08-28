import os, json, uuid, requests
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# --- Env ---
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY      = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME      = "customer-service-rag-index"

AOAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
EMBED_MODEL     = os.getenv("AZURE_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMS      = 1536 if EMBED_MODEL.endswith("small") else 3072  # adjust if your deployment name differs

assert SEARCH_ENDPOINT and SEARCH_KEY, "Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY in .env"
assert AOAI_ENDPOINT and AOAI_KEY,     "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env"

API_VERSION = "2024-07-01"
HEADERS = {"Content-Type": "application/json", "api-key": SEARCH_KEY}

# --- Azure OpenAI client (for embeddings) ---
aoai = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_KEY,
    api_version="2024-02-15-preview",
)

def delete_index_if_exists():
    url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version={API_VERSION}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code == 200:
        print(f"Deleting existing index: {INDEX_NAME}")
        d = requests.delete(url, headers=HEADERS, timeout=20)
        d.raise_for_status()

def create_index():
    print(f"Creating index: {INDEX_NAME} (vector dims={EMBED_DIMS})")
    url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version={API_VERSION}"

    schema = {
        "name": INDEX_NAME,
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True},
            {"name": "title", "type": "Edm.String", "searchable": True, "retrievable": True},
            {"name": "content", "type": "Edm.String", "searchable": True, "retrievable": True},

            # ✅ Vector field (GA schema for 2024-07-01)
            {
                "name": "contentVector",
                "type": "Collection(Edm.Single)",
                "searchable": True,                 # must be true for vector search
                "retrievable": True,                # set False if you don't want vectors returned
                "dimensions": EMBED_DIMS,           # 1536 for text-embedding-3-small; 3072 for -large
                "vectorSearchProfile": "vprofile"   # refers to a profile defined below
            }
        ],

        # ✅ Index-level vector config (algorithms + profiles)
        "vectorSearch": {
            "algorithms": [
                {"name": "hnsw", "kind": "hnsw"},
                # You could also add {"name": "exhaustive", "kind": "exhaustiveKnn"}
            ],
            "profiles": [
                {"name": "vprofile", "algorithm": "hnsw"}
            ]
        }
    }

    r = requests.put(url, headers=HEADERS, data=json.dumps(schema))
    if r.status_code not in (200, 201):
        print("❌ Failed to create index")
        try:
            print("Response:", r.json())
        except Exception:
            print("Response (raw):", r.text)
        r.raise_for_status()
    print("✅ Index created.")


def embed(text: str):
    return aoai.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def upload_docs(docs):
    url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/index?api-version={API_VERSION}"
    payload = {"value": [
        {
            "@search.action": "mergeOrUpload",
            "id": d["id"],
            "title": d["title"],
            "content": d["content"],
            "contentVector": d["vector"]
        } for d in docs
    ]}
    r = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    print("Docs uploaded.")

if __name__ == "__main__":
    # 1) (Re)create the index
    delete_index_if_exists()
    create_index()

    # 2) Dummy data
    Customer_service_data = [
        ("Refund Policy", "Customers can request a refund within 30 days of purchase with proof of receipt."),
        ("Shipping Times", "Standard shipping takes 5–7 business days. Express shipping takes 2–3 business days."),
        ("Password Reset", "Customers can reset their password by clicking 'Forgot Password' on the login page."),
        ("Business Hours", "Customer service is available Monday to Friday, 9 AM to 6 PM EST."),
        ("Warranty", "All electronics are covered under a 1-year limited warranty."),
        ("456", "Policy number 456 relates to a cancel for any reason ticket for Railgrid.")
    ]

    # 3) Embed and prepare docs
    docs = []
    for title, content in Customer_service_data:
        vec = embed(content)
        docs.append({"id": str(uuid.uuid4()), "title": title, "content": content, "vector": vec})

    # 4) Upload
    upload_docs(docs)

    print(f"\n✅ Done. Index '{INDEX_NAME}' now contains {len(docs)} vectorized docs.")