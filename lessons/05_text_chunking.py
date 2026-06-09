from rag_learn.chunking import fixed_size_chunks, paragraph_chunks
from rag_learn.config import get_google_api_key
from rag_learn.embeddings import GoogleEmbeddingModel, semantic_search
from rag_learn.retrieval import Document


HANDBOOK = Document(
    id="handbook",
    title="Customer handbook",
    text="""Standard shipping takes five to seven business days. Express shipping takes two business days.

Customers may cancel an order before it has shipped. After shipping, an order can no longer be cancelled.

Technical support is available by email every weekday. Include your order number when contacting support.""",
)


def print_chunks(label: str, chunks: list[Document]) -> None:
    print(f"\n{label}:")
    for chunk in chunks:
        print(f"- {chunk.id}: {chunk.text}")


def main() -> None:
    fixed = fixed_size_chunks(HANDBOOK, words_per_chunk=14, overlap_words=4)
    paragraphs = paragraph_chunks(HANDBOOK)

    print_chunks("Fixed-size chunks with overlap", fixed)
    print_chunks("Paragraph-aware chunks", paragraphs)

    question = "Can I cancel an order after it ships?"
    model = GoogleEmbeddingModel(get_google_api_key())

    print(f"\nQuestion: {question}")
    print("\nBest fixed-size result:")
    document, score = semantic_search(question, fixed, model, limit=1)[0]
    print(f"- {score:.3f}: {document.text}")

    print("\nBest paragraph-aware result:")
    document, score = semantic_search(question, paragraphs, model, limit=1)[0]
    print(f"- {score:.3f}: {document.text}")


if __name__ == "__main__":
    main()
