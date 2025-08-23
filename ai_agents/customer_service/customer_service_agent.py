# agent_connect_loop_high.py
import os, time, json, requests
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.ai.projects import AIProjectClient
from azure.identity import AzureCliCredential
from azure.ai.agents.models import ListSortOrder

# ---------- Config / Env ----------
load_dotenv()  # loads .env into environment

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

# ---------- Chat loop ----------
thread = project.agents.threads.create()
print(f"🧵 Started new thread: {thread.id}")

system_prompt =(
    "You are a helpful, concise customer service assistant. "
)

print("💬 Chat started. Type 'exit' to quit.\n")


while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("👋 Exiting chat.")
        break
    if not user_input:
        print("⚠️  Please enter a message.")
        continue

    try:
        # 1) Embed the query
        q_vec = embed(user_input)

        # 2) Vector search in AI Search
        results = search_vectors(q_vec)

        # 3) Build context only if top score >= threshold
        context_text = ""
        if results:
            top_score = results[0]["@search.score"]
            print(f"🔎 Top match score: {top_score:.3f}")
            if top_score >= SIMILARITY_THRESHOLD:
                context_text = "\n".join(
                    r.get("content", "") for r in results if r["@search.score"] >= SIMILARITY_THRESHOLD
                )

        # 4) Augment user input (only when relevant)
        if context_text:
            print("📎 Using retrieved context from vector store")
            augmented = (
                f"User question:\n{user_input}\n\n"
                f"Relevant docs (only include if directly helpful to the user):\n{context_text}"
            )
        else:
            print("ℹ️ No relevant context found (using plain user input)")
            augmented = user_input

        # 5) Add in the company policies
        RULES_TEXT = read_local("cs_policy.txt")

        if RULES_TEXT:
            augmented += "\n\nCompany rules (verbatim, follow strictly):\n" + RULES_TEXT    

        # 5) Send to agent
        project.agents.messages.create(thread_id=thread.id, role="user", content=augmented)

        run = project.agents.runs.create_and_process(thread_id=thread.id,
            agent_id=AGENT_ID,
            instructions=system_prompt   # or: override_instructions=system_prompt
            )
        print("⏳ Waiting for agent response...")
         
        while run.status not in ("completed", "failed"):
            time.sleep(1)
            run = project.agents.runs.create_and_process(
            thread_id=thread.id,
            agent_id=AGENT_ID,
            instructions=system_prompt   # or: override_instructions=system_prompt
            )
 

        if run.status == "failed":
            print(f"❌ Run failed: {run.last_error}")
            continue

        # 6) Print the latest assistant reply
        messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
        for m in messages:
            if m.text_messages:
                print(f"{m.role}: {m.text_messages[-1].text.value}")

    except Exception as e:
        print(f"🚨 Error: {e}")
        continue
