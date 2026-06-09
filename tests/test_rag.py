from rag_learn.providers import ChatModel
from rag_learn.rag import RagAssistant
from rag_learn.retrieval import Document


DOCUMENT = Document(
    id="shipping",
    title="Shipping policy",
    text="Standard shipping takes five days.",
)


class FakeEmbeddingModel:
    def __init__(self, similarity: float) -> None:
        self._similarity = similarity

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0], [self._similarity, 1.0 - self._similarity]]


class RecordingChatModel:
    def __init__(self) -> None:
        self.prompt = ""

    def generate(self, *, instructions: str, prompt: str) -> str:
        self.prompt = prompt
        return "Five days."


def test_rag_passes_relevant_document_to_chat_model() -> None:
    chat_model = RecordingChatModel()
    assistant = RagAssistant(
        [DOCUMENT], FakeEmbeddingModel(1.0), chat_model, minimum_score=0.5
    )

    answer = assistant.answer("How long is shipping?")

    assert answer.text == "Five days."
    assert answer.sources == [DOCUMENT]
    assert DOCUMENT.text in chat_model.prompt


def test_rag_does_not_call_chat_model_without_relevant_context() -> None:
    chat_model: ChatModel = RecordingChatModel()
    assistant = RagAssistant(
        [DOCUMENT], FakeEmbeddingModel(0.0), chat_model, minimum_score=0.5
    )

    answer = assistant.answer("Can I pay using cryptocurrency?")

    assert answer.sources == []
    assert "not have enough relevant information" in answer.text


def test_rag_citations_point_to_original_document() -> None:
    chunk = Document(
        id="handbook-paragraph-1",
        title="Customer handbook (paragraph 2)",
        text="Orders cannot be cancelled after shipping.",
        source_id="handbook",
        source_title="Customer handbook",
    )
    assistant = RagAssistant(
        [chunk], FakeEmbeddingModel(1.0), RecordingChatModel(), minimum_score=0.5
    )

    answer = assistant.answer("Can I cancel after shipping?")

    assert answer.sources == [chunk]
    assert answer.citations == ["Customer handbook"]
