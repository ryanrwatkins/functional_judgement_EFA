"""
Wrapper clients for different LLMs, stubbed to simulate numeric responses.
"""
import random
from typing import Any, Dict


class BaseClient:
    """Base class for LLM clients."""

    def __init__(self, model_name: str, seed: int = None) -> None:
        self.model_name = model_name
        if seed is not None:
            random.seed(seed)

    def chat(self, prompt: str) -> Any:
        """
        Simulate a chat completion by returning a random integer 1-5.
        """
        return random.randint(1, 5)


class OpenAIClient(BaseClient):
    """Simulated OpenAI client."""
    pass


class ClaudeClient(BaseClient):
    """Simulated Claude client."""
    pass


class LocalLlamaClient(BaseClient):
    """Simulated local LLaMA client."""
    pass
