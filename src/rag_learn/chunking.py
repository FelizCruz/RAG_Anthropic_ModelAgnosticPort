import re

from rag_learn.retrieval import Document


def fixed_size_chunks(
    document: Document, *, words_per_chunk: int, overlap_words: int = 0
) -> list[Document]:
    if words_per_chunk <= 0:
        raise ValueError("words_per_chunk must be positive")
    if overlap_words < 0 or overlap_words >= words_per_chunk:
        raise ValueError("overlap_words must be between 0 and words_per_chunk - 1")

    words = document.text.split()
    step = words_per_chunk - overlap_words
    chunks = []

    for index, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + words_per_chunk]
        chunks.append(
            Document(
                id=f"{document.id}-fixed-{index}",
                title=f"{document.title} (fixed chunk {index + 1})",
                text=" ".join(chunk_words),
                source_id=document.source_id or document.id,
                source_title=document.source_title or document.title,
            )
        )
        if start + words_per_chunk >= len(words):
            break

    return chunks


def paragraph_chunks(document: Document) -> list[Document]:
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", document.text)
        if paragraph.strip()
    ]
    return [
        Document(
            id=f"{document.id}-paragraph-{index}",
            title=f"{document.title} (paragraph {index + 1})",
            text=paragraph,
            source_id=document.source_id or document.id,
            source_title=document.source_title or document.title,
        )
        for index, paragraph in enumerate(paragraphs)
    ]
