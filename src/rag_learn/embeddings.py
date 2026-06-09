import math
from typing import Protocol

from openai import OpenAI, OpenAIError

from rag_learn.providers.base import ModelRequestError
from rag_learn.retrieval import Document


class EmbeddingModel(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Convert texts into semantic vectors."""


class GoogleEmbeddingModel:
    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(
                model="gemini-embedding-001",
                input=texts,
            )
        except OpenAIError as error:
            raise ModelRequestError(f"Google embeddings failed: {error}") from error

        return [item.embedding for item in response.data]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))
    return dot_product / (left_length * right_length)


def semantic_search(
    query: str,
    documents: list[Document],
    embedding_model: EmbeddingModel,
    *,
    limit: int = 3,
) -> list[tuple[Document, float]]:
    texts = [query] + [f"{document.title}: {document.text}" for document in documents]
    query_vector, *document_vectors = embedding_model.embed(texts)

    results = [
        (document, cosine_similarity(query_vector, vector))
        for document, vector in zip(documents, document_vectors, strict=True)
    ]
    return sorted(results, key=lambda result: result[1], reverse=True)[:limit]
