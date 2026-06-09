from pathlib import Path

from rag_learn.course_report import load_report_sections


def test_course_report_is_chunked_by_markdown_section(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text("# Report\nIntro\n\n## Section One\nDetails", encoding="utf-8")

    sections = load_report_sections(report)

    assert [section.title for section in sections] == ["Report", "Section One"]
