from pathlib import Path

from app.rag.extract import discover_pdf_files, extract_pdf_pages, generate_ingestion_report

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "knowledge_base" / "raw"
EXPECTED_FILES = {
    "6eb0c35ba63b776487a025e5123b6b12.pdf",
    "a4ccf4c6d97c4f1f36a6d83f8c64213d.pdf",
    "c328da14bbb15fc4ad571dc33e7a4ab3.pdf",
    "de4758d5bfeffc456d7de97a801891b0.pdf",
}


def test_discover_pdf_files_lists_only_existing_sources() -> None:
    files = discover_pdf_files(RAW_DIR)
    assert {path.name for path in files} == EXPECTED_FILES


def test_extract_pdf_pages_preserves_page_numbers_and_text() -> None:
    pdf_path = RAW_DIR / "c328da14bbb15fc4ad571dc33e7a4ab3.pdf"
    records = extract_pdf_pages(pdf_path)

    assert records[0]["source_id"] == "code_on_wages_2019"
    assert records[0]["page_number"] == 1
    assert records[-1]["page_number"] == 29
    assert "CHAPTER I" in records[0]["text"]
    assert any("17." in record["text"] or "Section 17" in record["text"] for record in records)
    assert any("18." in record["text"] or "Section 18" in record["text"] for record in records)


def test_generate_ingestion_report_contains_all_files() -> None:
    report = generate_ingestion_report(RAW_DIR, ROOT / "knowledge_base" / "extracted")

    assert report["pdf_count"] == 4
    assert {item["filename"] for item in report["files"]} == EXPECTED_FILES
    assert all(item["pages"] >= 1 for item in report["files"])
    assert all(item["non_empty_pages"] >= 1 for item in report["files"])
