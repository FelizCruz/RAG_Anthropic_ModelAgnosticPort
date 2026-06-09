from typing import Protocol


class ModelRequestError(RuntimeError):
    """Raised when a model provider cannot complete a request."""


class ChatModel(Protocol):
    def generate(self, *, instructions: str, prompt: str) -> str:
        """Generate text from instructions and a user prompt."""
