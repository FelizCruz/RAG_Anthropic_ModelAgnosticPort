from collections.abc import Sequence

from rag_learn.providers.base import ChatModel, ModelRequestError


class FallbackChatModel:
    def __init__(self, models: Sequence[ChatModel]) -> None:
        if not models:
            raise ValueError("At least one chat model is required")
        self._models = models

    def generate(self, *, instructions: str, prompt: str) -> str:
        failures = []

        for model in self._models:
            try:
                return model.generate(instructions=instructions, prompt=prompt)
            except ModelRequestError as error:
                failures.append(str(error))

        raise ModelRequestError("All model providers failed:\n" + "\n".join(failures))
