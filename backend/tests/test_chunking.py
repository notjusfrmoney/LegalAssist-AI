import json
import re
from pathlib import Path

from app.rag.chunk import chunk_document, chunk_extracted_documents, generate_chunk_report

ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_DIR = ROOT / "knowledge_base" / "extracted"


def test_chunk_document_preserves_metadata_and_legal_boundaries() -> None:
    text = (
        "Section 17. This section deals with wages.\n\n"
        "(1) The employer shall pay wages promptly.\n\n"
        "Section 18. This section deals with deductions.\n\n"
        "(1) Deductions shall be made only as permitted under this Code.\n\n"
        "Proviso: Provided that no deduction shall be made without written authorisation."
    )

    chunks = chunk_document(
        text,
        "sample_act",
        chunk_size=3500,
        overlap=400,
        page_number=7,
        document_title="Sample Act",
    )

    assert chunks
    assert all(chunk["source_id"] == "sample_act" for chunk in chunks)
    assert all(chunk["page_number"] == 7 for chunk in chunks)
    assert all(chunk["document_title"] == "Sample Act" for chunk in chunks)
    assert any("Section 17" in chunk["text"] for chunk in chunks)
    assert any("Section 18" in chunk["text"] for chunk in chunks)
    assert not any("Section 17" in chunk["text"] and "Section 18" in chunk["text"] for chunk in chunks)


def test_chunk_document_keeps_faq_question_and_answer_together() -> None:
    faq_path = EXTRACTED_DIR / "labour_codes_faq_january_2026.jsonl"
    faq_text = ""
    with faq_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("page_number") == 1:
                faq_text = payload["text"]
                break

    assert "What" in faq_text
    assert "The definition of" in faq_text

    chunks = chunk_document(faq_text, "faq_test", chunk_size=3500, overlap=400, page_number=1, document_title="FAQ")

    assert chunks
    assert any("What" in chunk["text"] and "The definition of" in chunk["text"] for chunk in chunks)


def test_chunk_extracted_documents_creates_processed_jsonl_with_metadata() -> None:
    output_path = ROOT / "knowledge_base" / "processed" / "test_chunks.jsonl"
    chunks = chunk_extracted_documents(EXTRACTED_DIR, output_path=output_path, chunk_size=3500, overlap=400)

    assert chunks
    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    assert lines
    assert all("source_id" in line for line in lines)
    assert all("page_number" in line for line in lines)
    assert all("text" in line for line in lines)

    assert any(chunk.get("source_filename") for chunk in chunks)
    assert any(chunk.get("source_url") for chunk in chunks)
    assert any(chunk.get("verification_status") for chunk in chunks)

    output_path.unlink(missing_ok=True)


def test_generate_chunk_report_reports_counts_and_oversized_sections() -> None:
    chunks = [
        {"chunk_id": "a-1", "source_id": "a", "page_number": 1, "text": "x" * 40, "token_count": 10},
        {"chunk_id": "a-2", "source_id": "a", "page_number": 2, "text": "y" * 4000, "token_count": 50},
    ]

    report = generate_chunk_report(chunks, chunk_size=3500, overlap=400)

    assert report["chunk_count"] == 2
    assert report["total_characters"] == 4040
    assert report["total_tokens_estimate"] == 60
    assert report["oversized_sections"]


def test_chunk_document_handles_long_document_sections_without_merging_numbered_sections() -> None:
    text = (
        "Section 17. This section defines wages.\n\n"
        "(1) The employer shall ensure compliance.\n\n"
        "(2) The worker shall be paid in accordance with the law.\n\n"
        "Section 18. This section covers deductions.\n\n"
        "(1) Deductions may be used only for lawful purposes.\n\n"
        "(2) Any excess deduction shall be refunded."
    )

    chunks = chunk_document(text, "code_on_wages_2019", chunk_size=3500, overlap=400, page_number=17)

    assert chunks
    assert not any("Section 17" in chunk["text"] and "Section 18" in chunk["text"] for chunk in chunks)
    assert sum("Section 17" in chunk["text"] for chunk in chunks) >= 1
    assert sum("Section 18" in chunk["text"] for chunk in chunks) >= 1


def test_primary_legislation_chunks_use_actual_numbered_section_metadata() -> None:
    output_path = ROOT / "knowledge_base" / "processed" / "section_boundary_test.jsonl"
    chunks = chunk_extracted_documents(EXTRACTED_DIR, output_path=output_path)
    statutory_chunks = [
        chunk for chunk in chunks if chunk.get("source_id") == "code_on_wages_2019"
    ]

    section_14 = [chunk for chunk in statutory_chunks if chunk.get("section_number") == "14"]
    section_17 = [chunk for chunk in statutory_chunks if chunk.get("section_number") == "17"]
    section_18 = [chunk for chunk in statutory_chunks if chunk.get("section_number") == "18"]

    assert section_14
    assert any("overtime rate" in chunk["text"].lower() for chunk in section_14)
    assert section_17
    assert any("monthly basis" in chunk["text"].lower() for chunk in section_17)
    assert section_18
    assert any("deductions from the wages" in chunk["text"].lower() for chunk in section_18)
    assert all(
        not (
            re.search(r"(?m)^\s*17\.\s+\(1\)", chunk["text"])
            and re.search(r"(?m)^\s*18\.\s+\(1\)", chunk["text"])
        )
        for chunk in statutory_chunks
    )
    assert all(chunk["page_number"] == int(chunk["page_range"].split()[-1]) for chunk in statutory_chunks)
    assert all(chunk.get("source_filename") for chunk in statutory_chunks)
    assert len({chunk["chunk_id"] for chunk in statutory_chunks}) == len(statutory_chunks)

    output_path.unlink(missing_ok=True)
