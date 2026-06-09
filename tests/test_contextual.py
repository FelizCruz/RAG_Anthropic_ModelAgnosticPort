from rag_learn.contextual import contextualize_document, contextualize_documents
from rag_learn.retrieval import Document


class FakeChatModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, *, instructions: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return "This section covers Project Phoenix incident E42."


DOCUMENT = Document(
    id="section-1",
    title="Software Engineering",
    text="The patch reduced failures.",
    source_id="report",
    source_title="Annual Report",
)


def test_contextualization_prepends_context_and_preserves_source_metadata() -> None:
    contextualized = contextualize_document(DOCUMENT, "Full report", FakeChatModel())

    assert contextualized.text.startswith(
        "Context: This section covers Project Phoenix incident E42."
    )
    assert contextualized.text.endswith(DOCUMENT.text)
    assert contextualized.id == DOCUMENT.id
    assert contextualized.source_title == DOCUMENT.source_title


def test_contextualization_sends_source_and_chunk_to_model() -> None:
    model = FakeChatModel()

    contextualize_documents([DOCUMENT], "Full report text", model)

    assert "Full report text" in model.prompts[0]
    assert DOCUMENT.text in model.prompts[0]
