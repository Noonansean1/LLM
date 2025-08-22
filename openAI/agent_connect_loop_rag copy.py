from azure.ai.projects import AIProjectClient
from azure.identity import AzureCliCredential
from azure.ai.agents.models import ListSortOrder
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import time

# === Azure AI Agent Project Setup ===
project = AIProjectClient(
    credential=AzureCliCredential(),
    endpoint="https://adm-sn-4418-resource.services.ai.azure.com/api/projects/adm-sn-4418"
)

agent_id = "asst_tn0Tmtqh1Tx8qWsPnzNhhc5S"

# === Azure Cognitive Search Setup ===
search_endpoint = "https://<your-search-service>.search.windows.net"
search_index_name = "<your-index-name>"
search_key = "<your-admin-or-query-key>"

search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=search_index_name,
    credential=AzureKeyCredential(search_key)
)

# === Retrieve matching chunks from Azure Search ===
def retrieve_context_from_docs(query, top_k=3):
    results = search_client.search(
        search_text=query,
        top=top_k,
        include_total_count=True
    )
    context_chunks = []
    for result in results:
        if 'content' in result:
            context_chunks.append(result['content'])  # assumes your index field is 'content'
    return "\n\n".join(context_chunks)

# === Start agent chat ===
thread = project.agents.threads.create()
print(f"🧵 Started new thread: {thread.id}")
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
        # === Step 1: Retrieve context from search ===
        retrieved_context = retrieve_context_from_docs(user_input)

        # === Step 2: Build prompt with context ===
        augmented_prompt = f"""You are a helpful assistant. Use the following context to answer the question:

{retrieved_context}

Question: {user_input}
"""

        # === Step 3: Send message to the agent thread ===
        message = project.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content=augmented_prompt
        )

        # === Step 4: Run the agent and wait for response ===
        run = project.agents.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent_id
        )

        print("⏳ Waiting for agent response...")
        while run.status not in ("completed", "failed"):
            time.sleep(1)
            run = project.agents.runs.get(thread_id=thread.id, run_id=run.id)

        if run.status == "failed":
            print(f"❌ Run failed: {run.last_error}")
            continue

        # === Step 5: Print agent response ===
        messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
        for message in messages:
            if message.role == "assistant" and message.text_messages:
                print(f"{message.role}: {message.text_messages[-1].text.value}")

    except Exception as e:
        print(f"🚨 Error: {e}")
        continue

