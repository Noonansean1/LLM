# agents/simple_rest_ingest_agent.py
import os, json, uuid, requests
from typing import Iterable, List, Dict, Tuple, Union, Optional

Pair = Tuple[str, str]
Doc  = Dict[str, object]

class IngestAgent:
    """
    Minimal REST-based ingest agent for Azure Cognitive Search.
    Mirrors your old pattern but as a class.

    Env (used if args not passed):
      AZURE_SEARCH_ENDPOINT=https://<name>.search.windows.net
      AZURE_SEARCH_API_KEY=<ADMIN KEY>
      AZURE_SEARCH_INDEX=demo-rag-index
      AZURE_SEARCH_API_VERSION=2024-07-01
      AZURE_SEARCH_VECTOR_FIELD=contentVector
      AZURE_EMBED_DIM=1536
    """

    def __init__(
        self,
        embed_fn,                                # (List[str]) -> List[List[float]]  OR  (str) -> List[float]
        *,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        api_version: Optional[str] = None,
        vector_field: Optional[str] = None,
        embed_dims: Optional[int] = None,
        timeout: int = 60,
    ):
        self.embed_fn    = embed_fn
        self.endpoint    = (endpoint or os.getenv("AZURE_SEARCH_ENDPOINT") or "").rstrip("/")
        self.api_key     = api_key  or os.getenv("AZURE_SEARCH_API_KEY") or os.getenv("AZURE_SEARCH_KEY")
        self.index_name  = index_name or os.getenv("AZURE_SEARCH_INDEX") or os.getenv("AZURE_SEARCH_INDEX_NAME") or "demo-rag-index"
        self.api_version = api_version or os.getenv("AZURE_SEARCH_API_VERSION") or "2024-07-01"
        self.vector_field = vector_field or os.getenv("AZURE_SEARCH_VECTOR_FIELD") or "contentVector"
        self.embed_dims   = int(embed_dims or os.getenv("AZURE_EMBED_DIM") or 1536)
        self.timeout = timeout

        if not self.endpoint or not self.api_key:
            raise RuntimeError("Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY (admin key).")

        self._idx_url  = f"{self.endpoint}/indexes/{self.index_name}?api-version={self.api_version}"
        self._docs_url = f"{self.endpoint}/indexes/{self.index_name}/docs/index?api-version={self.api_version}"
        self._headers  = {"Content-Type": "application/json", "api-key": self.api_key}

    # ---------- Index ----------
    def delete_index_if_exists(self) -> bool:
        r = requests.get(self._idx_url, headers=self._headers, timeout=self.timeout)
        if r.status_code == 200:
            d = requests.delete(self._idx_url, headers=self._headers, timeout=self.timeout)
            d.raise_for_status()
            return True
        if r.status_code == 404:
            return False
        try:
            raise RuntimeError(f"Index probe failed: {r.status_code} {r.json()}")
        except Exception:
            raise RuntimeError(f"Index probe failed: {r.status_code} {r.text}")

    def create_index(self) -> None:
        schema = {
            "name": self.index_name,
            "fields": [
                {"name": "id", "type": "Edm.String", "key": True},
                {"name": "title", "type": "Edm.String", "searchable": True, "retrievable": True},
                {"name": "content", "type": "Edm.String", "searchable": True, "retrievable": True},
                {
                    "name": self.vector_field,
                    "type": "Collection(Edm.Single)",
                    "searchable": True,          # must be True for vector search
                    "retrievable": True,
                    "dimensions": self.embed_dims,
                    "vectorSearchProfile": "vprofile"
                }
            ],
            "vectorSearch": {
                "algorithms": [{ "name": "hnsw", "kind": "hnsw" }],
                "profiles":   [{ "name": "vprofile", "algorithm": "hnsw" }]
            }
        }
        r = requests.put(self._idx_url, headers=self._headers, data=json.dumps(schema), timeout=self.timeout)
        if r.status_code not in (200, 201):
            try:
                raise RuntimeError(f"Create index failed: {r.status_code} {r.json()}")
            except Exception:
                raise RuntimeError(f"Create index failed: {r.status_code} {r.text}")

    def ensure_index(self) -> None:
        r = requests.get(self._idx_url, headers=self._headers, timeout=self.timeout)
        if r.status_code == 404:
            self.create_index()
        elif r.status_code != 200:
            try:
                raise RuntimeError(f"Index probe failed: {r.status_code} {r.json()}")
            except Exception:
                raise RuntimeError(f"Index probe failed: {r.status_code} {r.text}")

    def recreate_index(self) -> None:
        self.delete_index_if_exists()
        self.create_index()

    # ---------- Embedding ----------
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            vecs = self.embed_fn(texts)       # batch style
        except TypeError:
            vecs = [self.embed_fn(t) for t in texts]  # single-text style

        if not vecs:
            return []
        if len(vecs[0]) != self.embed_dims:
            raise ValueError(f"Embedding dim mismatch: got {len(vecs[0])}, expected {self.embed_dims}")
        return vecs

    # ---------- Upload ----------
    def _upload_docs(self, docs: List[Doc]) -> None:
        payload = {"value": [
            {
                "@search.action": "mergeOrUpload",
                "id": d["id"],
                "title": d["title"],
                "content": d["content"],
                self.vector_field: d["vector"]
            } for d in docs
        ]}
        r = requests.post(self._docs_url, headers=self._headers, data=json.dumps(payload), timeout=self.timeout)
        if r.status_code >= 300:
            try:
                raise RuntimeError(f"Upload failed: {r.status_code} {r.json()}")
            except Exception:
                raise RuntimeError(f"Upload failed: {r.status_code} {r.text}")

    # ---------- Public: execute ----------
    def execute_pairs(self, pairs: Iterable[Pair], *, recreate: bool = False, create_if_missing: bool = True) -> Dict[str, int]:
        """Ingest (title, content) pairs; embeds content and uploads."""
        items: List[Doc] = [{"id": str(uuid.uuid4()), "title": t, "content": c} for (t, c) in pairs]

        if recreate:
            self.recreate_index()
        elif create_if_missing:
            self.ensure_index()

        texts = [d["content"] for d in items]
        vecs  = self._embed_batch(texts)
        docs  = [{**d, "vector": v} for d, v in zip(items, vecs)]

        self._upload_docs(docs)
        return {"ingested": len(docs)}

    def execute_docs(self, docs: Iterable[Doc], *, recreate: bool = False, create_if_missing: bool = True) -> Dict[str, int]:
        """
        Ingest dict docs that already have embeddings:
        each doc must have keys: id, title, content, vector
        (kept for full compatibility with your old upload_docs path)
        """
        docs = list(docs)
        if recreate:
            self.recreate_index()
        elif create_if_missing:
            self.ensure_index()
        self._upload_docs(docs)
        return {"ingested": len(docs)}
