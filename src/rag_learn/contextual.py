from rag_learn.providers.base import ChatModel
from rag_learn.retrieval import Document


def contextualize_documents(
    documents: list[Document],
    source_text: str,
    chat_model: ChatModel,
) -> list[Document]:
    return [
        contextualize_document(document, source_text, chat_model)
        for document in documents
    ]


def contextualize_document(
    document: Document,
    source_text: str,
    chat_model: ChatModel,
) -> Document:
    context = chat_model.generate(
        instructions=(
            "Write one short sentence that situates the provided chunk within "
            "the source document. Mention its subject and useful identifiers. "
            "Return only the situating sentence."
        ),
        prompt=f"""Source document:
{source_text}

Chunk title: {document.title}
Chunk:
{document.text}""",
    )
    return Document(
        id=document.id,
        title=document.title,
        text=f"Context: {context}\n\n{document.text}",
        source_id=document.source_id,
        source_title=document.source_title,
    )
