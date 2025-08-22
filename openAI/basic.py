# pip install openai
from openai import AzureOpenAI
import os, time

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-05-01-preview",  # or the version enabled on your resource
)

VECTOR_STORE_ID = "<your_vector_store_id>"  # copy from Assistant vector stores
MODEL = "gpt-4o-mini"                       # or the model you deployed

# 1) Create an assistant that can use file search over your vector store
assistant = client.beta.assistants.create(
    model=MODEL,
    name="RAG Demo",
    tools=[{"type": "file_search"}],
    tool_resources={
        "file_search": {"vector_store_ids": [VECTOR_STORE_ID]}
    },
)

# 2) Start a thread with a user question
thread = client.beta.threads.create()
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="What is RAG and how do we use it here?"
)

# 3) Run the assistant (it will retrieve from the vector store)
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# 4) Poll until complete, then read messages
while True:
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    if run.status in ("completed", "failed", "cancelled", "expired"):
        break
    time.sleep(0.8)

msgs = client.beta.threads.messages.list(thread_id=thread.id)
for m in reversed(msgs.data):  # newest last
    if m.role == "assistant":
        print(m.content[0].text.value)
        break
