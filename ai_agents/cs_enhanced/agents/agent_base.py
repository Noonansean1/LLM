# agents/agent_base.py
from abc import ABC, abstractmethod
from loguru import logger


class AgentBase(ABC):
    def __init__(self, name: str, max_retries: int = 2, verbose: bool = True):
        self.name = name
        self.max_retries = max_retries
        self.verbose = verbose

    @abstractmethod
    def execute(self, *args, **kwargs):
        """Implement in subclasses."""
        pass

    def log_in(self, msg: str):
        if self.verbose:
            logger.info(f"[{self.name}] {msg}")

