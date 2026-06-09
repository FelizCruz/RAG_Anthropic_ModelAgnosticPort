import pytest

from rag_learn.chunking import fixed_size_chunks, paragraph_chunks
from rag_learn.retrieval import Document


DOCUMENT = Document(
    id="guide",
    title="Guide",
    text="one two three four five six\n\nSecond paragraph.",
)


def test_fixed_size_chunks_overlap_at_boundaries() -> None:
    chunks = fixed_size_chunks(DOCUMENT, words_per_chunk=4, overlap_words=2)

    assert chunks[0].text == "one two three four"
    assert chunks[1].text == "three four five six"


def test_fixed_size_chunks_reject_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_words"):
        fixed_size_chunks(DOCUMENT, words_per_chunk=4, overlap_words=4)


def test_paragraph_chunks_preserve_paragraph_boundaries() -> None:
    chunks = paragraph_chunks(DOCUMENT)

    assert [chunk.text for chunk in chunks] == [
        "one two three four five six",
        "Second paragraph.",
    ]
    assert all(chunk.source_id == "guide" for chunk in chunks)
    assert all(chunk.source_title == "Guide" for chunk in chunks)
