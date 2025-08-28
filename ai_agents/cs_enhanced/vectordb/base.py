# vectordb/base.py
from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any, List

class VectorStore(ABC):
    @abstractmethod
    def upsert(self, items: Iterable[Dict[str, Any]]) -> None:
        """Upsert list of {'id': str, 'content': str, 'vector': list[float], ...}"""
        pass

    @abstractmethod
    def search(self, vector: list[float], k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k results with '@search.score' & 'content'."""
        pass
