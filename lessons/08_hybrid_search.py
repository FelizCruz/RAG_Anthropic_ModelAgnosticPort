from pathlib import Path

from rag_learn.config import get_google_api_key
from rag_learn.course_report import load_report_sections
from rag_learn.embeddings import GoogleEmbeddingModel
from rag_learn.hybrid import Retriever
from rag_learn.indexes import BM25Index, VectorIndex


def main() -> None:
    report_path = Path(__file__).parents[1] / "data" / "report.md"
    sections = load_report_sections(report_path)
    retriever = Retriever(
        BM25Index(),
        VectorIndex(GoogleEmbeddingModel(get_google_api_key())),
    )
    retriever.add_documents(sections)

    questions = [
        "What happened during incident INC-2023-Q4-011?",
        "Which project experienced an external dependency delay?",
    ]

    print(f"Indexed {len(sections)} report sections once.\n")
    for question in questions:
        print(f"Question: {question}")
        for document, score in retriever.search(question):
            print(f"- {document.title}: {score:.4f}")
        print()


if __name__ == "__main__":
    main()
