import math
import re
from collections import Counter

from rag_learn.retrieval import Document


def bm25_search(
    query: str,
    documents: list[Document],
    *,
    limit: int = 3,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[Document, float]]:
    if not documents:
        return []

    query_terms = _tokens(query)
    document_terms = [
        _tokens(f"{document.title} {document.text}") for document in documents
    ]
    average_length = sum(map(len, document_terms)) / len(document_terms)
    document_frequency = Counter(
        term for terms in document_terms for term in set(terms)
    )
    results = []

    for document, terms in zip(documents, document_terms, strict=True):
        frequencies = Counter(terms)
        score = 0.0

        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue

            containing_documents = document_frequency[term]
            inverse_document_frequency = math.log(
                1
                + (len(documents) - containing_documents + 0.5)
                / (containing_documents + 0.5)
            )
            saturation = frequency * (k1 + 1)
            normalization = frequency + k1 * (
                1 - b + b * len(terms) / average_length
            )
            score += inverse_document_frequency * saturation / normalization

        if score > 0:
            results.append((document, score))

    return sorted(results, key=lambda result: result[1], reverse=True)[:limit]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
