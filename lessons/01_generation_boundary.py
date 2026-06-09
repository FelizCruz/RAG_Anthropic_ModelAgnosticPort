from rag_learn.config import load_model_configs
from rag_learn.providers import (
    FallbackChatModel,
    ModelRequestError,
    OpenAICompatibleChatModel,
)


def main() -> None:
    model = FallbackChatModel(
        [OpenAICompatibleChatModel(config) for config in load_model_configs()]
    )

    try:
        answer = model.generate(
            instructions="Explain concepts clearly to a beginner in two sentences.",
            prompt="Why might a RAG answer be more reliable than an ordinary model answer?",
        )
    except ModelRequestError as error:
        print(error)
        return

    print(answer)


if __name__ == "__main__":
    main()
