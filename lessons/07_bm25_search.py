from rag_learn.bm25 import bm25_search
from rag_learn.retrieval import Document, keyword_search


DOCUMENTS = [
    Document(
        id="general-errors",
        title="General troubleshooting",
        text="When an error occurs, restart the device and try again.",
    ),
    Document(
        id="error-e42",
        title="Payment error E42",
        text="Error E42 means the payment authorization failed.",
    ),
    Document(
        id="error-e17",
        title="Login error E17",
        text="Error E17 means the account password is incorrect.",
    ),
]


def main() -> None:
    question = "What does error E42 mean?"
    print(f"Question: {question}\n")

    print("Basic keyword scores:")
    for result in keyword_search(question, DOCUMENTS):
        print(f"- {result.document.title}: {result.score}")

    print("\nBM25 scores:")
    for document, score in bm25_search(question, DOCUMENTS):
        print(f"- {document.title}: {score:.3f}")


if __name__ == "__main__":
    main()
