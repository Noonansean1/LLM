# agents/qna_agent.py
from agents.azure_agent_base import AzureAgentBase
from vectordb.base import VectorStore
from pathlib import Path

class QnAAgent(AzureAgentBase):
    def __init__(self, name, vector_store: VectorStore | None, embed_fn, similarity_threshold=0.75, **kw):
        super().__init__(name=name, **kw)
        self.vs = vector_store
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold

    def read_local(self, filename: str) -> str:
        base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        p = base.parent / "data" / filename
        
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")
        return p.read_text(encoding="utf-8", errors="ignore")    

    def build_augmented(self, question: str) -> str:
        if not self.vs:
            return question

        q_vec = self.embed_fn(question)
        results = self.vs.search(q_vec, 3)
        if not results:
            return question

        top = results[0].get("@search.score", 0.0)
        if top < self.similarity_threshold:
            print("This happened")
            return question

        context = "\n\n".join(str(r.get("content", "")) for r in results if r.get("@search.score", 0) >= self.similarity_threshold)

        # 5) Add in the company policies
        RULES_TEXT = self.read_local("cs_policy.txt")

        if RULES_TEXT:
            context += "\n\nCompany rules (verbatim, follow strictly):\n" + RULES_TEXT    

        return (
            f"User question:\n{question}\n\n"
            f"Relevant docs (only include if directly helpful to the user):\n{context}"
        )

    def execute(self, question: str, system_prompt: str | None = None) -> str:
        content = self.build_augmented(question)
        self.send_user_message(content)
        self.run_once(instructions=system_prompt)
        return self.fetch_last_assistant_reply()
