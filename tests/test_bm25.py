from rag_learn.bm25 import bm25_search
from rag_learn.retrieval import Document


DOCUMENTS = [
    Document(id="general", title="Errors", text="An error occurred."),
    Document(id="specific", title="Error E42", text="E42 means payment failure."),
    Document(id="other", title="Error E17", text="E17 means login failure."),
]


def test_bm25_ranks_rare_matching_term_first() -> None:
    results = bm25_search("error E42", DOCUMENTS)

    assert results[0][0].id == "specific"
    assert results[0][1] > results[1][1]


def test_bm25_returns_empty_results_for_no_match() -> None:
    assert bm25_search("shipping", DOCUMENTS) == []


def test_bm25_returns_empty_results_for_empty_index() -> None:
    assert bm25_search("error", []) == []
