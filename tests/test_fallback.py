import pytest

from rag_learn.providers import FallbackChatModel, ModelRequestError


class StubModel:
    def __init__(self, result: str | Exception) -> None:
        self._result = result

    def generate(self, *, instructions: str, prompt: str) -> str:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def test_uses_next_model_when_primary_fails() -> None:
    model = FallbackChatModel(
        [StubModel(ModelRequestError("rate limited")), StubModel("fallback answer")]
    )

    assert model.generate(instructions="Be clear", prompt="What is RAG?") == (
        "fallback answer"
    )


def test_reports_failure_when_all_models_fail() -> None:
    model = FallbackChatModel(
        [StubModel(ModelRequestError("first")), StubModel(ModelRequestError("second"))]
    )

    with pytest.raises(ModelRequestError, match="All model providers failed"):
        model.generate(instructions="Be clear", prompt="What is RAG?")
