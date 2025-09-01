# app.py
import json
import os
import ast
import re
import streamlit as st
from agents.customer_service_agent import QnAAgent
from agents.ingest_agent import IngestAgent
from vectordb.azure_search import AzureSearchStore
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

# ==== CONFIG ====
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "my-index")
AGENT_ID = os.getenv("AGENT_ID")
SIM_THRESH = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

# Bring your own embedding function (sync wrapper)
# Here we assume you have a utility `embed(text:str)->list[float]`
from utils.helpers_func import embed,load_pairs_from_text

st.set_page_config(page_title="Azure Foundry Agent", page_icon="🧩", layout="centered")
st.title("🧩 Azure Foundry Agent — Q&A + Ingest")

# Vector store
store = AzureSearchStore(index_name=INDEX_NAME)

# Agents
qna = QnAAgent(name="qna", vector_store=store, embed_fn=embed,similarity_threshold=SIM_THRESH, agent_id=AGENT_ID, verbose=True)

def embed_batch(texts):
    return [embed(t) for t in texts]

ingest = IngestAgent(embed_fn=embed_batch, index_name=os.getenv("AZURE_SEARCH_INDEX"))

with st.sidebar:
    st.subheader("Settings")
    system_prompt = st.text_area("System prompt", value="You are a helpful, concise customer service agent acting as a co-worker.")

st.header("Ask a question")
q = st.text_input("Your question")
if st.button("Ask"):
    if not q.strip():
        st.warning("Type a question.")
    else:
        with st.spinner("Thinking..."):
            answer = qna.execute(question=q, system_prompt=system_prompt)
        st.success("Answer:")
        st.write(answer)

st.markdown("---")


st.header("Upload FAQ tuples file (single file)")
file = st.file_uploader(
    "Upload a file containing a list of (title, content) pairs (Python or JSON).",
    type=["txt", "json", "py"],
    accept_multiple_files=False
)
recreate = st.checkbox("Recreate index (delete & create)", value=False)

if st.button("Ingest"):
    if not file:
        st.warning("Please select a file.")
    else:
        raw = file.read().decode("utf-8", errors="ignore")
        try:
            pairs = load_pairs_from_text(raw)
        except Exception as e:
            st.error(f"Could not parse file: {e}")
            st.stop()

        # Preview so you can confirm it's not line-splitting
        st.write(f"Parsed {len(pairs)} pairs.")
        st.caption(f"Example: {pairs[0][0]} → {pairs[0][1][:120]}...")

        with st.spinner("Embedding & upserting..."):
            res = ingest.execute_pairs(pairs, recreate=recreate, create_if_missing=not recreate)

        st.success(f"Ingested {res['ingested']} pairs into '{ingest.index_name}'.")