# agent_connect_loop_high.py
import os, time, json, requests
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.ai.projects import AIProjectClient
from azure.identity import AzureCliCredential
from azure.ai.agents.models import ListSortOrder
from utils.helpers_func import *

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
