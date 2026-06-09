from collections import defaultdict

from rag_learn.indexes import SearchIndex
from rag_learn.retrieval import Document


def reciprocal_rank_fusion(
    rankings: list[list[tuple[Document, float]]],
    *,
    limit: int = 3,
    rank_constant: int = 60,
) -> list[tuple[Document, float]]:
    scores: dict[str, float] = defaultdict(float)
    documents: dict[str, Document] = {}

    for ranking in rankings:
        for rank, (document, _) in enumerate(ranking, start=1):
            documents[document.id] = document
            scores[document.id] += 1 / (rank_constant + rank)

    ordered_ids = sorted(scores, key=scores.get, reverse=True)
    return [(documents[id], scores[id]) for id in ordered_ids[:limit]]


class Retriever:
    def __init__(self, *indexes: SearchIndex) -> None:
        if not indexes:
            raise ValueError("At least one index is required")
        self._indexes = indexes

    def add_documents(self, documents: list[Document]) -> None:
        for index in self._indexes:
            index.add_documents(documents)

    def search(self, query: str, *, limit: int = 3) -> list[tuple[Document, float]]:
        rankings = [index.search(query, limit=limit * 5) for index in self._indexes]
        return reciprocal_rank_fusion(rankings, limit=limit)
