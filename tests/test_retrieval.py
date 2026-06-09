from rag_learn.retrieval import Document, keyword_search, tokenize


DOCUMENTS = [
    Document(id="refunds", title="Refund policy", text="Refunds are allowed for 30 days."),
    Document(id="shipping", title="Shipping policy", text="Shipping takes five days."),
]


def test_tokenize_normalizes_case_and_punctuation() -> None:
    assert tokenize("Refund, REFUND!") == {"refund"}


def test_keyword_search_ranks_document_with_more_matching_terms_first() -> None:
    results = keyword_search("refund policy days", DOCUMENTS)

    assert results[0].document.id == "refunds"
    assert results[0].score == 3


def test_keyword_search_omits_irrelevant_documents() -> None:
    assert keyword_search("password reset", DOCUMENTS) == []
