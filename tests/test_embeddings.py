import pytest

from rag_learn.embeddings import cosine_similarity, semantic_search
from rag_learn.retrieval import Document


class FakeEmbeddingModel:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ]


def test_cosine_similarity_measures_vector_direction() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_semantic_search_ranks_closest_vector_first() -> None:
    documents = [
        Document(id="unrelated", title="Shipping", text="Delivery times"),
        Document(id="related", title="Support", text="Email us for help"),
    ]

    results = semantic_search("I need help", documents, FakeEmbeddingModel())

    assert results[0][0].id == "related"
