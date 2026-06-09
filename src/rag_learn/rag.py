from dataclasses import dataclass

from rag_learn.embeddings import EmbeddingModel, semantic_search
from rag_learn.providers.base import ChatModel
from rag_learn.retrieval import Document


@dataclass(frozen=True)
class RagAnswer:
    text: str
    sources: list[Document]

    @property
    def citations(self) -> list[str]:
        return list(
            dict.fromkeys(source.source_title or source.title for source in self.sources)
        )


class RagAssistant:
    def __init__(
        self,
        documents: list[Document],
        embedding_model: EmbeddingModel,
        chat_model: ChatModel,
        *,
        minimum_score: float,
        limit: int = 2,
    ) -> None:
        self._documents = documents
        self._embedding_model = embedding_model
        self._chat_model = chat_model
        self._minimum_score = minimum_score
        self._limit = limit

    def answer(self, question: str) -> RagAnswer:
        results = semantic_search(
            question,
            self._documents,
            self._embedding_model,
            limit=self._limit,
        )
        relevant = [
            document for document, score in results if score >= self._minimum_score
        ]

        if not relevant:
            return RagAnswer(
                text="I do not have enough relevant information to answer that.",
                sources=[],
            )

        context = "\n\n".join(
            (
                f"Source: {document.source_title or document.title}\n"
                f"Excerpt: {document.text}"
            )
            for document in relevant
        )
        prompt = f"""Context:
{context}

Question: {question}"""
        answer = self._chat_model.generate(
            instructions=(
                "Answer using only the provided context. "
                "If the context does not contain the answer, say you do not know. "
                "Do not use outside knowledge."
            ),
            prompt=prompt,
        )
        return RagAnswer(text=answer, sources=relevant)
