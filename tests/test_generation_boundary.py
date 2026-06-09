from rag_learn.providers import ChatModel
from rag_learn.config import ModelConfig
from rag_learn.providers import OpenAICompatibleChatModel


class FakeChatModel:
    def generate(self, *, instructions: str, prompt: str) -> str:
        return f"{instructions} | {prompt}"


def ask_rag_question(model: ChatModel, question: str) -> str:
    return model.generate(
        instructions="Answer using retrieved context.",
        prompt=question,
    )


def test_application_can_use_provider_neutral_chat_model() -> None:
    answer = ask_rag_question(FakeChatModel(), "What is RAG?")

    assert answer == "Answer using retrieved context. | What is RAG?"


def test_openai_compatible_adapter_hides_thought_block(monkeypatch) -> None:
    model = OpenAICompatibleChatModel(
        ModelConfig(name="test", api_key="key", base_url="https://example.com", model="m")
    )

    class Message:
        content = "<thought>private reasoning</thought>Final answer"

    class Choice:
        message = Message()

    class Completion:
        choices = [Choice()]

    monkeypatch.setattr(
        model._client.chat.completions, "create", lambda **kwargs: Completion()
    )

    assert model.generate(instructions="Be clear", prompt="Question") == "Final answer"
