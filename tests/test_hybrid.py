from rag_learn.hybrid import Retriever, reciprocal_rank_fusion
from rag_learn.retrieval import Document


FIRST = Document(id="first", title="First", text="")
SECOND = Document(id="second", title="Second", text="")
THIRD = Document(id="third", title="Third", text="")


def test_fusion_rewards_document_ranked_by_both_methods() -> None:
    lexical = [(FIRST, 10.0), (SECOND, 5.0)]
    semantic = [(THIRD, 0.9), (SECOND, 0.8)]

    results = reciprocal_rank_fusion([lexical, semantic])

    assert results[0][0] == SECOND


def test_fusion_ignores_incompatible_raw_scores() -> None:
    lexical = [(FIRST, 1000.0)]
    semantic = [(SECOND, 0.9)]

    results = reciprocal_rank_fusion([lexical, semantic])

    assert results[0][1] == results[1][1]


class FakeIndex:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.added: list[Document] = []

    def add_documents(self, documents: list[Document]) -> None:
        self.added.extend(documents)

    def search(self, query: str, *, limit: int = 3) -> list[tuple[Document, float]]:
        return self.results[:limit]


def test_retriever_indexes_documents_in_every_index() -> None:
    first_index = FakeIndex([])
    second_index = FakeIndex([])
    retriever = Retriever(first_index, second_index)

    retriever.add_documents([FIRST])

    assert first_index.added == [FIRST]
    assert second_index.added == [FIRST]


def test_retriever_fuses_multiple_indexes() -> None:
    retriever = Retriever(
        FakeIndex([(FIRST, 10.0), (SECOND, 5.0)]),
        FakeIndex([(THIRD, 0.9), (SECOND, 0.8)]),
    )

    assert retriever.search("query")[0][0] == SECOND
