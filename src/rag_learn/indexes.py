from collections import Counter
from typing import Protocol

from rag_learn.bm25 import _tokens
from rag_learn.embeddings import EmbeddingModel, cosine_similarity
from rag_learn.retrieval import Document


class SearchIndex(Protocol):
    def add_documents(self, documents: list[Document]) -> None: ...

    def search(self, query: str, *, limit: int = 3) -> list[tuple[Document, float]]: ...


class VectorIndex:
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self._embedding_model = embedding_model
        self._documents: list[Document] = []
        self._vectors: list[list[float]] = []

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return

        texts = [f"{document.title}: {document.text}" for document in documents]
        vectors = self._embedding_model.embed(texts)
        self._documents.extend(documents)
        self._vectors.extend(vectors)

    def search(self, query: str, *, limit: int = 3) -> list[tuple[Document, float]]:
        if not self._documents:
            return []

        query_vector = self._embedding_model.embed([query])[0]
        results = [
            (document, cosine_similarity(query_vector, vector))
            for document, vector in zip(self._documents, self._vectors, strict=True)
        ]
        return sorted(results, key=lambda result: result[1], reverse=True)[:limit]

class BM25Index:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._documents: list[Document] = []
        self._document_terms: list[list[str]] = []
        self._k1 = k1
        self._b = b

    def add_documents(self, documents: list[Document]) -> None:
        self._documents.extend(documents)
        self._document_terms.extend(
            _tokens(f"{document.title} {document.text}") for document in documents
        )

    def search(self, query: str, *, limit: int = 3) -> list[tuple[Document, float]]:
        if not self._documents:
            return []

        import math

        average_length = sum(map(len, self._document_terms)) / len(self._document_terms)
        document_frequency = Counter(
            term for terms in self._document_terms for term in set(terms)
        )
        results = []

        for document, terms in zip(
            self._documents, self._document_terms, strict=True
        ):
            frequencies = Counter(terms)
            score = 0.0

            for term in _tokens(query):
                frequency = frequencies[term]
                if frequency == 0:
                    continue

                containing_documents = document_frequency[term]
                idf = math.log(
                    1
                    + (len(self._documents) - containing_documents + 0.5)
                    / (containing_documents + 0.5)
                )
                numerator = idf * frequency * (self._k1 + 1)
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * len(terms) / average_length
                )
                score += numerator / denominator

            if score > 0:
                results.append((document, score))

        return sorted(results, key=lambda result: result[1], reverse=True)[:limit]
