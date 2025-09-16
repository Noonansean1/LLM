import os
import json
import uuid
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# --- Env ---
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY      = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME      = "customer-service-rag-index"

AOAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_KEY        = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
EMBED_MODEL     = os.getenv("AZURE_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMS      = 1536 if EMBED_MODEL.endswith("small") else 3072

API_VERSION = "2024-07-01"
HEADERS = {"Content-Type": "application/json", "api-key": SEARCH_KEY}

# --- Azure OpenAI client ---
aoai = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_KEY,
    api_version="2024-02-15-preview",
)

# --- Embedding function ---
def embed(text: str):
    return aoai.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

# --- Upload to Azure Search ---
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
    print(f"✅ Uploaded {len(docs)} docs to index.")

# --- Read CSV and prepare docs ---
def prepare_docs_from_csv(csv_path: str):
    df = pd.read_csv(csv_path, sep=None, engine='python')  # auto-detects separator
    docs = []

    for _, row in df.iterrows():
        # Combine key columns into a single text field
        content_fields = [
            f"ID: {row.get('hs_object_id', '')}",
            f"Pipeline: {row.get('hs_pipeline', '')}",
            f"Stage: {row.get('hs_pipeline_stage', '')}",
            f"Received Date: {row.get('received_date', '')}",
            f"Resolved Date: {row.get('resolved_date', '')}",
            f"Customer Country: {row.get('customer_cor', '')}",
            f"Actions to Resolve: {row.get('actions_to_resolve', '')}",
            f"Value of Redress: {row.get('value_of_redress', '')}",
            f"Complaint RCA: {row.get('complaint_rca', '')}",
            f"Complaint Reason: {row.get('complaint_reason', '')}",
            f"Product: {row.get('hs_product_name', '')}"
        ]
        content = " | ".join([c for c in content_fields if c])

        doc = {
            "id": str(row.get('hs_object_id')),
            "title": f"Complaint {row.get('hs_object_id', '')}",
            "content": content,
            "vector": embed(content)
        }
        docs.append(doc)
    
    return docs

# --- Main ---
if __name__ == "__main__":
    csv_file = "/Users/sean.noonan/development/aiFoundry/ai_agents/customer_service/customer_complaints.csv"  # <-- your CSV file path
    docs = prepare_docs_from_csv(csv_file)
    upload_docs(docs)
    print(f"\n✅ Done. Index '{INDEX_NAME}' now contains {len(docs)} vectorized docs.")
