from rag_learn.config import get_google_api_key, load_model_configs
from rag_learn.embeddings import GoogleEmbeddingModel
from rag_learn.providers import FallbackChatModel, OpenAICompatibleChatModel
from rag_learn.rag import RagAssistant
from rag_learn.retrieval import Document


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
    chat_model = FallbackChatModel(
        [OpenAICompatibleChatModel(config) for config in load_model_configs()]
    )
    assistant = RagAssistant(
        DOCUMENTS,
        GoogleEmbeddingModel(get_google_api_key()),
        chat_model,
        minimum_score=0.60,
    )

    questions = [
        "How long does standard shipping take?",
        "What should I do when my package is late?",
        "Can I pay using cryptocurrency?",
    ]

    for question in questions:
        answer = assistant.answer(question)
        print(f"Question: {question}")
        print(f"Answer: {answer.text}")
        print(f"Sources: {', '.join(source.title for source in answer.sources) or 'None'}\n")


if __name__ == "__main__":
    main()
