from rag_learn.chunking import paragraph_chunks
from rag_learn.config import get_google_api_key, load_model_configs
from rag_learn.embeddings import GoogleEmbeddingModel
from rag_learn.providers import FallbackChatModel, OpenAICompatibleChatModel
from rag_learn.rag import RagAssistant
from rag_learn.retrieval import Document


HANDBOOK = Document(
    id="customer-handbook",
    title="Customer handbook",
    text="""Standard shipping takes five to seven business days. Express shipping takes two business days.

Customers may cancel an order before it has shipped. After shipping, an order can no longer be cancelled.

Technical support is available by email every weekday. Include your order number when contacting support.""",
)


def main() -> None:
    chunks = paragraph_chunks(HANDBOOK)
    chat_model = FallbackChatModel(
        [OpenAICompatibleChatModel(config) for config in load_model_configs()]
    )
    assistant = RagAssistant(
        chunks,
        GoogleEmbeddingModel(get_google_api_key()),
        chat_model,
        minimum_score=0.60,
        limit=2,
    )

    question = "Can I cancel an order after it ships?"
    answer = assistant.answer(question)

    print(f"Question: {question}")
    print(f"Answer: {answer.text}")
    print(f"Citations: {', '.join(answer.citations) or 'None'}")
    print("Retrieved chunks:")
    for chunk in answer.sources:
        print(f"- {chunk.id}: {chunk.text}")


if __name__ == "__main__":
    main()
