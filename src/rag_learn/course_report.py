import re
from pathlib import Path

from rag_learn.retrieval import Document


def load_report_sections(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"\n## ", text)
    documents = []

    for index, section in enumerate(sections):
        clean = section.strip()
        if not clean:
            continue

        first_line, _, body = clean.partition("\n")
        title = first_line.strip("# *")
        documents.append(
            Document(
                id=f"report-section-{index}",
                title=title,
                text=body.strip() or title,
                source_id="course-report",
                source_title="Annual Interdisciplinary Research Review",
            )
        )

    return documents
