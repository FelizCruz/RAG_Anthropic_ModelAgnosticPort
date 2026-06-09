from pathlib import Path

from rag_learn.config import get_google_api_key, load_model_configs
from rag_learn.contextual import contextualize_documents
from rag_learn.course_report import load_report_sections
from rag_learn.embeddings import GoogleEmbeddingModel
from rag_learn.hybrid import Retriever
from rag_learn.indexes import BM25Index, VectorIndex
from rag_learn.providers import FallbackChatModel, OpenAICompatibleChatModel


def create_retriever(sections, embedding_model) -> Retriever:
    retriever = Retriever(BM25Index(), VectorIndex(embedding_model))
    retriever.add_documents(sections)
    return retriever


def print_results(label: str, retriever: Retriever, question: str) -> None:
    print(f"\n{label}:")
    for document, score in retriever.search(question):
        print(f"- {document.title}: {score:.4f}")


def main() -> None:
    report_path = Path(__file__).parents[1] / "data" / "report.md"
    source_text = report_path.read_text(encoding="utf-8")
    sections = load_report_sections(report_path)
    chat_model = FallbackChatModel(
        [OpenAICompatibleChatModel(config) for config in load_model_configs()]
    )
    contextualized = contextualize_documents(sections, source_text, chat_model)
    embedding_model = GoogleEmbeddingModel(get_google_api_key())

    regular_retriever = create_retriever(sections, embedding_model)
    contextual_retriever = create_retriever(contextualized, embedding_model)
    question = "Which section discusses how resource constraints affected clinical work?"

    print(f"Contextualized {len(contextualized)} report sections.")
    print_results("Regular hybrid retrieval", regular_retriever, question)
    print_results("Contextual hybrid retrieval", contextual_retriever, question)


if __name__ == "__main__":
    main()
