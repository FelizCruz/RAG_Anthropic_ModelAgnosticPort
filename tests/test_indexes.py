from rag_learn.indexes import BM25Index, VectorIndex
from rag_learn.retrieval import Document


DOCUMENTS = [
    Document(id="one", title="Error E42", text="Payment failed."),
    Document(id="two", title="Support", text="Contact us for help."),
]


class RecordingEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if len(texts) == 2:
            return [[1.0, 0.0], [0.0, 1.0]]
        return [[1.0, 0.0]]


def test_vector_index_embeds_documents_once_then_only_embeds_query() -> None:
    model = RecordingEmbeddingModel()
    index = VectorIndex(model)

    index.add_documents(DOCUMENTS)
    index.search("E42")
    index.search("payment error")

    assert len(model.calls) == 3
    assert len(model.calls[0]) == 2
    assert all(len(call) == 1 for call in model.calls[1:])


def test_bm25_index_reuses_added_documents() -> None:
    index = BM25Index()
    index.add_documents(DOCUMENTS)

    assert index.search("E42")[0][0].id == "one"
