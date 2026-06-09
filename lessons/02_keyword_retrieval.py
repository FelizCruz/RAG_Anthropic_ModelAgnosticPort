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
    question = "How many days do I have to request a refund?"
    results = keyword_search(question, DOCUMENTS, limit=2)

    print(f"Question: {question}\n")
    for result in results:
        print(f"Score: {result.score}")
        print(f"Source: {result.document.title}")
        print(f"Text: {result.document.text}\n")


if __name__ == "__main__":
    main()
