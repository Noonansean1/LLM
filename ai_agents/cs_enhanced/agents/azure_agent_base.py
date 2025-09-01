# agents/azure_agent_base.py
import os
from loguru import logger
from azure.identity import AzureCliCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import ListSortOrder
from agents.agent_base import AgentBase
from dotenv import load_dotenv

load_dotenv()

class AzureAgentBase(AgentBase):
    def __init__(self, name: str, agent_id: str | None = None, **kw):
        super().__init__(name=name, **kw)
        self.endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
        self.project_name = os.getenv("AZURE_AI_PROJECT_NAME")
        self.agent_id = agent_id or os.getenv("AGENT_ID")
        if not (self.endpoint and self.project_name and self.agent_id):
            raise RuntimeError("Set AZURE_SEARCH_ENDPOINT, AZURE_AI_PROJECT_NAME, AGENT_ID")

        cred = AzureCliCredential()  # requires `az login`
        self.client = AIProjectClient(endpoint=self.endpoint, project_name=self.project_name, credential=cred)
        self.thread_id = None

    def new_thread(self):
        thread = self.client.agents.threads.create()
        self.thread_id = thread.id
        self.log_in(f"Started thread: {self.thread_id}")
        return self.thread_id

    def ensure_thread(self):
        return self.thread_id or self.new_thread()

    def send_user_message(self, content: str):
        tid = self.ensure_thread()
        self.client.agents.messages.create(thread_id=tid, role="user", content=content)
        self.log_in(f"User message added to thread {tid}")

    def run_once(self, instructions: str | None = None):
        tid = self.ensure_thread()
        run = self.client.agents.runs.create_and_process(
            thread_id=tid, agent_id=self.agent_id, instructions=instructions
        )
        if getattr(run, "status", None) == "failed":
            raise RuntimeError(f"Run failed: {run.last_error}")
        return run

    def fetch_last_assistant_reply(self) -> str:
        if not self.thread_id:
            return ""
        msgs = self.client.agents.messages.list(thread_id=self.thread_id, order=ListSortOrder.ASCENDING)
        for m in reversed(list(msgs)):
            if m.role == "assistant" and getattr(m, "text_messages", None):
                return m.text_messages[-1].text.value
        return ""
