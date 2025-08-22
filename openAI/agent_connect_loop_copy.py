from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder
import time
from azure.identity import AzureCliCredential

project = AIProjectClient(
    credential=AzureCliCredential(),
    endpoint="https://adm-sn-all-resource.services.ai.azure.com/api/projects/adm-sn-all"
)

# Set your agent ID
agent_id = "asst_8yIHi2J290Ko7YyhuItaJL91"

# Start new thread
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
        # Send message to thread
        message = project.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_input
        )

        # Start and process run
        run = project.agents.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent_id
        )
        print("⏳ Waiting for agent response...")
        # Wait until run completes or fails
        while run.status not in ("completed", "failed"):
            time.sleep(1)
            run = project.agents.runs.get(thread_id=thread.id, run_id=run.id)

        if run.status == "failed":
            print(f"❌ Run failed: {run.last_error}")
            continue

        # Get messages
        messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)

        # Print only agent responses
        for message in messages:
            if message.text_messages:
                print(f"{message.role}: {message.text_messages[-1].text.value}")



    except Exception as e:
        print(f"🚨 Error: {e}")
        continue

