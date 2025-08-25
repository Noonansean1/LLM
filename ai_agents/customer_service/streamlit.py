# streamlit_cs_agent.py
# Streamlit UI for a Customer Service Agent that augments Azure AI Agent responses
# with context retrieved from Azure AI Search via vector similarity.
#
# SAFE SECRETS: Prefer environment variables or .streamlit/secrets.toml (do not commit .env to git!)
#
# ──────────────────────────────────────────────────────────────────────────────
# Requirements (install):
#   pip install streamlit python-dotenv requests openai azure-identity azure-ai-projects
#
# Run:
#   streamlit run streamlit_cs_agent.py
#
# .streamlit/secrets.toml example (optional):
# [default]
# AZURE_OPENAI_ENDPOINT = "https://<your-aoai>.openai.azure.com"
# AZURE_OPENAI_API_KEY = "<key>"
# AZURE_EMBED_MODEL = "text-embedding-3-small"  # or your deployment name
# AZURE_SEARCH_ENDPOINT = "https://<your-search>.search.windows.net"
# AZURE_SEARCH_KEY = "<admin-or-query-key>"
# AZURE_SEARCH_VECTOR_FIELD = "contentVector"
# FOUNDRY_PROJECT_ENDPOINT = "https://<your-project-endpoint>"
# FOUNDRY_AGENT_ID = "asst_xxx"
# TOP_K = 3
# SIMILARITY_THRESHOLD = 0.80

import os
import time
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

# Azure OpenAI (embeddings)
from openai import AzureOpenAI

# Azure AI Foundry Agents (Projects)
from azure.ai.projects import AIProjectClient
from azure.identity import AzureCliCredential, DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder

# ──────────────────────────────────────────────────────────────────────────────
# Page setup
st.set_page_config(page_title="CS Agent (RAG + Azure)", page_icon="💬", layout="wide")
st.title("💬 Customer Service Agent · RAG + Azure")
st.caption("Embeds your question → searches Azure AI Search → augments Azure Agent reply")

# ──────────────────────────────────────────────────────────────────────────────
# Config loading (env or Streamlit secrets)
load_dotenv()  # load from local .env if present (DON'T COMMIT IT)

# Helper to read from st.secrets first, then env
get = lambda k, d=None: (
    (st.secrets.get("default", {}).get(k) if isinstance(st.secrets, dict) and "default" in st.secrets else st.secrets.get(k, None))
    if hasattr(st, "secrets") else None
) or os.getenv(k, d)

AOAI_ENDPOINT = get("AZURE_OPENAI_ENDPOINT")
AOAI_KEY = get("AZURE_OPENAI_API_KEY") or get("AZURE_OPENAI_KEY")
EMBED_MODEL = get("AZURE_EMBED_MODEL", "text-embedding-3-small")
AOAI_API_VERSION = get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

SEARCH_ENDPOINT = get("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = get("AZURE_SEARCH_KEY")
INDEX_NAME = get("AZURE_SEARCH_INDEX", "customer-service-rag-index")
VECTOR_FIELD = get("AZURE_SEARCH_VECTOR_FIELD", "contentVector")

FOUNDRY_PROJECT_ENDPOINT = get("FOUNDRY_PROJECT_ENDPOINT") or "https://adm-sn-all-resource.services.ai.azure.com/api/projects/adm-sn-all"
AGENT_ID = get("FOUNDRY_AGENT_ID") or "asst_8yIHi2J290Ko7YyhuItaJL91"

TOP_K = int(float(get("TOP_K", 3)))
SIMILARITY_THRESHOLD = float(get("SIMILARITY_THRESHOLD", 0.80))

system_prompt =(
    "You are a helpful, concise customer service assistant for the embedded insurance provider Companjon."
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: settings & status
with st.sidebar:
    st.subheader("Settings")
    TOP_K = st.number_input("Top K", min_value=1, max_value=10, value=int(TOP_K), step=1)
    SIMILARITY_THRESHOLD = st.slider("Similarity threshold", 0.0, 1.0, float(SIMILARITY_THRESHOLD), 0.01)

    st.markdown("---")
    st.caption("Connection status")

# ──────────────────────────────────────────────────────────────────────────────
# Initialize clients lazily and store in session state
if "aoai_client" not in st.session_state:
    st.session_state.aoai_client = None
if "project_client" not in st.session_state:
    st.session_state.project_client = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of {role: "user"|"assistant", content: str}


def get_credential():
    """Prefer Azure CLI cred locally; fallback to DefaultAzureCredential."""
    try:
        return AzureCliCredential()
    except Exception:
        return DefaultAzureCredential(exclude_interactive_browser_credential=False)


@st.cache_resource(show_spinner=False)
def init_aoai_client(endpoint: str, key: str, api_version: str):
    if not endpoint or not key:
        return None
    return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=api_version)


@st.cache_resource(show_spinner=False)
def init_project_client(project_endpoint: str):
    if not project_endpoint:
        return None
    cred = get_credential()
    return AIProjectClient(credential=cred, endpoint=project_endpoint)


# Create clients
st.session_state.aoai_client = init_aoai_client(AOAI_ENDPOINT, AOAI_KEY, AOAI_API_VERSION)
st.session_state.project_client = init_project_client(FOUNDRY_PROJECT_ENDPOINT)

with st.sidebar:
    st.write(
        f"🔷 AOAI: {'✅' if st.session_state.aoai_client else '❌'}  ·  "
        f"Search: {'✅' if (SEARCH_ENDPOINT and SEARCH_KEY) else '❌'}  ·  "
        f"Agents: {'✅' if st.session_state.project_client else '❌'}"
    )
    st.caption(f"Index: `{INDEX_NAME}` · Vector field: `{VECTOR_FIELD}`")


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions

def embed(text: str):
    client = st.session_state.aoai_client
    if not client:
        raise RuntimeError("Azure OpenAI client not configured")
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def search_vectors(query_vector):
    if not (SEARCH_ENDPOINT and SEARCH_KEY):
        return []
    url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/search?api-version=2024-07-01"
    headers = {"Content-Type": "application/json", "api-key": SEARCH_KEY}
    payload = {
        "top": TOP_K,
        "vectorQueries": [
            {"kind": "vector", "vector": query_vector, "fields": VECTOR_FIELD, "k": TOP_K}
        ],
        # You could add a filter here if you want to scope by product/locale/etc.
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Search error: {err}")
    data = r.json()
    return data.get("value", [])


def ensure_thread():
    if st.session_state.thread_id:
        return st.session_state.thread_id
    client = st.session_state.project_client
    if not client:
        raise RuntimeError("Azure AI Project client not configured")
    thread = client.agents.threads.create()
    st.session_state.thread_id = thread.id
    return thread.id

def read_local(filename: str) -> str:
    base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    p = base / filename
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    return p.read_text(encoding="utf-8", errors="ignore")


def send_to_agent(user_text: str):
    """Embed → search → optionally augment → send to agent → return (assistant_reply, retrieved_docs)."""
    q_vec = embed(user_text)
    results = search_vectors(q_vec)

    # Build context if top score passes threshold
    context_text = ""
    retrieved = []  # for UI: list of dicts with content and score
    if results:
        top_score = results[0].get("@search.score", 0.0)
        if top_score >= SIMILARITY_THRESHOLD:
            for r in results:
                score = float(r.get("@search.score", 0.0))
                if score >= SIMILARITY_THRESHOLD:
                    retrieved.append({
                        "score": score,
                        "content": r.get("content", ""),
                        "id": r.get("id") or r.get("@search.documentId") or "",
                    })
            context_text = "\n".join([d["content"] for d in retrieved])

    if context_text:
        augmented = (
            f"User question:\n{user_text}\n\n"
            f"Relevant docs (only include if directly helpful to the user):\n{context_text}"
        )
    else:
        augmented = user_text

    # Add in the company policies
    RULES_TEXT = read_local("cs_policy.txt")

    if RULES_TEXT:
        augmented += "\n\nCompany rules (verbatim, follow strictly):\n" + RULES_TEXT            

    # Send to agent
    thread_id = ensure_thread()
    proj = st.session_state.project_client
    proj.agents.messages.create(thread_id=thread_id, role="user", content=augmented)

    run = proj.agents.runs.create_and_process(thread_id=thread_id, agent_id=AGENT_ID, instructions=system_prompt)
    # Poll until complete or failed
    while run.status not in ("completed", "failed"):
        time.sleep(1)
        run = proj.agents.runs.get(thread_id=thread_id, run_id=run.id)

    if run.status == "failed":
        raise RuntimeError(str(run.last_error))

    # Fetch last assistant message
    msgs = proj.agents.messages.list(thread_id=thread_id, order=ListSortOrder.ASCENDING)
    assistant_reply = None
    for m in msgs:
        if m.role == "assistant" and m.text_messages:
            assistant_reply = m.text_messages[-1].text.value
    return assistant_reply or "(No reply)", retrieved


# ──────────────────────────────────────────────────────────────────────────────
# Chat UI
chat_container = st.container()
retrieval_container = st.container()

# Existing history
with chat_container:
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

# Chat input
prompt = st.chat_input("Ask a customer question… e.g., 'How long is express shipping?'")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking with RAG…"):
            try:
                reply, retrieved_docs = send_to_agent(prompt)
                st.session_state.history.append({"role": "assistant", "content": reply})
                st.markdown(reply)

                # Show retrieval if any
                if retrieved_docs:
                    with st.expander("📎 Retrieved context", expanded=False):
                        for i, d in enumerate(sorted(retrieved_docs, key=lambda x: -x["score"])):
                            st.markdown(f"**Doc {i+1} · score={d['score']:.3f}**\n\n{d['content']}")
                else:
                    retrieval_container.info("No relevant context found (used plain question)")

            except Exception as e:
                st.error(f"Error: {e}")

# Utilities row
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🧹 Clear chat"):
        st.session_state.history = []
        st.session_state.thread_id = None
        st.rerun()
with col2:
    st.write("\u200b")
with col3:
    st.caption("Tip: manage secrets via environment or `.streamlit/secrets.toml`.")

# Footer
st.markdown("---")
st.caption(
    "This app embeds your query using Azure OpenAI, searches Azure AI Search for context, "
    "and sends an augmented prompt to your Azure Agent thread."
)
