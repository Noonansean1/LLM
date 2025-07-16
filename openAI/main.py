import streamlit as st
from prompt_chain import run_prompt_chain
from retrieval import VectorStore
from mlops import log_experiment

# Sample documents for vector search
documents = [
    "Mars is the fourth planet from the Sun and is known as the Red Planet.",
    "Olympus Mons is the largest volcano in the solar system, located on Mars.",
    "Mars once had flowing water and a thick atmosphere.",
    "Jupiter is the largest planet in the solar system.",
    "Venus is the hottest planet in our solar system."
]

# Initialize vector store and ingest documents
vector_store = VectorStore()
vector_store.ingest_documents(documents)

st.title("AI Foundry Demo: LLM & Vector Search")

st.header("Prompt Engineering & Orchestration")
prompt = st.text_area("Enter your prompt for the LLM:", "What are three interesting facts about Mars?")
if st.button("Run LLM Prompt"):
    log_experiment("LLM Prompt Run", {"prompt": prompt})
    response = run_prompt_chain(prompt)
    st.subheader("LLM Response:")
    st.write(response)

st.header("Vector Search & Retrieval")
query = st.text_input("Enter your search query:", "Mars volcano")
if st.button("Run Vector Search"):
    log_experiment("Vector Search Run", {"query": query})
    results = vector_store.similarity_search(query)
    st.subheader("Top Similar Documents:")
    for doc, score in results:
        st.write(f"Score: {score:.2f}")
        st.write(doc)
        st.markdown("---")

st.info("Experiment tracking and logging are enabled. See logs for details.\n\nThis demo is modular and ready for cross-functional collaboration.") 