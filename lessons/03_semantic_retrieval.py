from rag_learn.config import get_google_api_key
from rag_learn.embeddings import GoogleEmbeddingModel, semantic_search
from rag_learn.retrieval import Document, keyword_search


DOCUMENTS = [
    Document(
        id="refunds",
        title="Refund policy",
        text="Customers may request a refund within 30 days of purchase.",
    ),
    Document(
        id="shipping",
        title="Shipping policy",
        text="Standard shipping usually takes five to seven business days.",
    ),
    Document(
        id="support",
        title="Technical support",
        text="Technical support is available by email every weekday.",
    ),
]


def main() -> None:
    api_key = get_google_api_key()
    question = "My package has not arrived yet."

    print(f"Question: {question}\n")
    print("Keyword results:")
    for result in keyword_search(question, DOCUMENTS):
        print(f"- {result.document.title}: {result.score}")

    print("\nSemantic results:")
    for document, score in semantic_search(
        question, DOCUMENTS, GoogleEmbeddingModel(api_key)
    ):
        print(f"- {document.title}: {score:.3f}")


if __name__ == "__main__":
    main()
