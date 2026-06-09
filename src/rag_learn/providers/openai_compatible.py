import re

from openai import OpenAI, OpenAIError

from rag_learn.config import ModelConfig
from rag_learn.providers.base import ModelRequestError


class OpenAICompatibleChatModel:
    def __init__(self, config: ModelConfig) -> None:
        self._name = config.name
        self._model = config.model
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def generate(self, *, instructions: str, prompt: str) -> str:
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt},
                ],
            )
        except OpenAIError as error:
            raise ModelRequestError(f"{self._name} failed: {error}") from error

        content = completion.choices[0].message.content or ""
        return re.sub(r"<thought>.*?</thought>", "", content, flags=re.DOTALL).strip()
