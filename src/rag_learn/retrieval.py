import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source_id: str | None = None
    source_title: str | None = None


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: int


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def keyword_search(
    query: str, documents: list[Document], *, limit: int = 3
) -> list[SearchResult]:
    query_terms = tokenize(query)
    results = []

    for document in documents:
        document_terms = tokenize(f"{document.title} {document.text}")
        score = len(query_terms & document_terms)
        if score > 0:
            results.append(SearchResult(document=document, score=score))

    return sorted(results, key=lambda result: result.score, reverse=True)[:limit]
