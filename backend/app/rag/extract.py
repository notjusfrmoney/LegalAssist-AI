from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import fitz

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPO_ROOT / "knowledge_base" / "raw"
DEFAULT_EXTRACTED_DIR = REPO_ROOT / "knowledge_base" / "extracted"

SOURCE_METADATA: dict[str, dict[str, Any]] = {
    "c328da14bbb15fc4ad571dc33e7a4ab3.pdf": {
        "source_id": "code_on_wages_2019",
        "document_title": "Code on Wages, 2019",
        "source_type": "primary_legislation",
        "publisher": "Government of India, Ministry of Labour and Employment",
        "source_url": "https://labour.gov.in/sites/default/files/the_code_on_wages_2019_no._29_of_2019.pdf",
        "jurisdiction": "India",
        "government_sphere": "central_and_state_code",
        "publication_date": "2019-08-08",
        "effective_status_note": "Primary legislation; applicability depends on the establishment, sphere and facts.",
        "verification_status": "needs_review",
    },
    "6eb0c35ba63b776487a025e5123b6b12.pdf": {
        "source_id": "code_on_wages_central_rules_2026",
        "document_title": "Code on Wages (Central) Rules, 2026",
        "source_type": "delegated_rules",
        "publisher": "Government of India, Ministry of Labour and Employment",
        "source_url": "https://www.labour.gov.in/static/uploads/2026/05/6eb0c35ba63b776487a025e5123b6b12.pdf",
        "jurisdiction": "India",
        "government_sphere": "central",
        "publication_date": "2026-05-08",
        "effective_status_note": "Central Rules for the Central sphere; not automatically applicable to every State-sphere establishment.",
        "verification_status": "needs_review",
    },
    "de4758d5bfeffc456d7de97a801891b0.pdf": {
        "source_id": "labour_codes_faq_january_2026",
        "document_title": "Ministry of Labour FAQs on Labour Codes, January 2026",
        "source_type": "official_faq",
        "publisher": "Government of India, Ministry of Labour and Employment",
        "source_url": "https://www.labour.gov.in/static/uploads/2026/01/de4758d5bfeffc456d7de97a801891b0.pdf",
        "jurisdiction": "India",
        "government_sphere": "central_and_state_code",
        "publication_date": "2026-01-01",
        "effective_status_note": "Official explanatory guidance; lower authority than the Act and Rules.",
        "verification_status": "needs_review",
    },
    "a4ccf4c6d97c4f1f36a6d83f8c64213d.pdf": {
        "source_id": "labour_codes_additional_faq_march_2026",
        "document_title": "Additional FAQs on Labour Codes, as on 16 March 2026",
        "source_type": "official_faq",
        "publisher": "Government of India, Ministry of Labour and Employment",
        "source_url": "https://www.labour.gov.in/static/uploads/2026/03/a4ccf4c6d97c4f1f36a6d83f8c64213d.pdf",
        "jurisdiction": "India",
        "government_sphere": "central_and_state_code",
        "publication_date": "2026-03-16",
        "effective_status_note": "Official explanatory guidance; lower authority than the Act and Rules.",
        "verification_status": "needs_review",
    },
}


def discover_pdf_files(raw_dir: str | Path | None = None) -> list[Path]:
    directory = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    if not directory.exists():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"],
        key=lambda value: value.name,
    )


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    pdf_file = Path(pdf_path)
    metadata = SOURCE_METADATA.get(pdf_file.name, {})
    source_id = metadata.get("source_id") or re.sub(r"[^a-z0-9]+", "_", pdf_file.stem.lower()).strip("_")
    document_title = metadata.get("document_title") or pdf_file.stem

    doc = fitz.open(str(pdf_file))
    records: list[dict[str, Any]] = []

    for page_index in range(doc.page_count):
        page = doc[page_index]
        raw_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES)
        clean_text = _normalize_text(raw_text or "")
        records.append(
            {
                "source_id": source_id,
                "document_title": document_title,
                "page_number": page_index + 1,
                "text": clean_text,
            }
        )

    return records


def write_extracted_pages(records: list[dict[str, Any]], extracted_dir: str | Path | None = None) -> Path:
    directory = Path(extracted_dir) if extracted_dir is not None else DEFAULT_EXTRACTED_DIR
    directory.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["source_id"], []).append(record)

    for source_id, items in grouped.items():
        file_path = directory / f"{source_id}.jsonl"
        with file_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    return directory


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def generate_ingestion_report(raw_dir: str | Path | None = None, extracted_dir: str | Path | None = None) -> dict[str, Any]:
    directory = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    extracted_path = Path(extracted_dir) if extracted_dir is not None else DEFAULT_EXTRACTED_DIR
    files: list[dict[str, Any]] = []

    for pdf_path in discover_pdf_files(directory):
        records = load_jsonl(extracted_path / f"{SOURCE_METADATA.get(pdf_path.name, {}).get('source_id') or re.sub(r'[^a-z0-9]+', '_', pdf_path.stem.lower()).strip('_')}.jsonl")
        if not records:
            records = extract_pdf_pages(pdf_path)
            write_extracted_pages(records, extracted_path)

        pages = len(records)
        non_empty_pages = sum(1 for record in records if record.get("text", "").strip())
        extracted_characters = sum(len(record.get("text", "")) for record in records)
        empty_pages = pages - non_empty_pages
        warnings: list[str] = []
        for record in records:
            text = record.get("text", "")
            normalized = text.strip()
            if not normalized or len(normalized) < 80:
                warnings.append(f"page {record['page_number']}: possibly image-only or sparse text; manual review recommended")

        files.append(
            {
                "filename": pdf_path.name,
                "pages": pages,
                "non_empty_pages": non_empty_pages,
                "extracted_characters": extracted_characters,
                "empty_pages": empty_pages,
                "warnings": warnings,
            }
        )

    return {"files": files, "pdf_count": len(files)}


def print_ingestion_report(report: dict[str, Any]) -> None:
    print("Ingestion report")
    print("-" * 96)
    print(f"{'Filename':<64} {'Pages':>6} {'Non-empty':>10} {'Chars':>12} {'Empty':>8} {'Warnings':<50}")
    for item in report["files"]:
        warnings = "; ".join(item["warnings"]) if item["warnings"] else "none"
        print(
            f"{item['filename']:<64} {item['pages']:>6} {item['non_empty_pages']:>10} {item['extracted_characters']:>12} {item['empty_pages']:>8} {warnings:<50}"
        )


def build_manifest(raw_dir: str | Path | None = None) -> dict[str, Any]:
    directory = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    pdf_files = discover_pdf_files(directory)
    documents = []
    missing_files = []
    expected = [
        "code_on_wages_2019.pdf",
        "code_on_wages_commencement_notification_2025.pdf",
        "code_on_wages_central_rules_2026.pdf",
        "labour_codes_faq_january_2026.pdf",
        "labour_codes_additional_faq_march_2026.pdf",
    ]

    for expected_name in expected:
        if not any(pdf_file.name == expected_name for pdf_file in pdf_files):
            missing_files.append(expected_name)

    for pdf_file in pdf_files:
        metadata = SOURCE_METADATA.get(pdf_file.name) or {
            "source_id": re.sub(r"[^a-z0-9]+", "_", pdf_file.stem.lower()).strip("_"),
            "document_title": pdf_file.stem,
            "source_type": "unknown",
            "publisher": "Unverified local source",
            "source_url": None,
            "jurisdiction": "unknown",
            "government_sphere": "unknown",
            "publication_date": None,
            "effective_status_note": "Not yet matched to an official source; local document needs provenance review.",
            "verification_status": "needs_review",
        }
        documents.append(
            {
                "source_id": metadata["source_id"],
                "filename": pdf_file.name,
                "document_title": metadata["document_title"],
                "publisher": metadata["publisher"],
                "source_url": metadata.get("source_url"),
                "source_type": metadata["source_type"],
                "jurisdiction": metadata["jurisdiction"],
                "government_sphere": metadata.get("government_sphere", "unknown"),
                "publication_date": metadata.get("publication_date"),
                "effective_status_note": metadata["effective_status_note"],
                "verification_status": metadata["verification_status"],
                "is_primary_legal_source": metadata["source_type"] == "primary_legislation",
                "file_status": "present",
            }
        )

    external_references = [
        {
            "source_id": "code_on_wages_commencement_notification_2025",
            "document_title": "Code on Wages commencement notification, 2025",
            "filename": "code_on_wages_commencement_notification_2025.pdf",
            "publisher": "Government of India, Ministry of Labour and Employment",
            "source_url": "https://labour.gov.in/sites/default/files/code_on_wages_commencement_notification_2025.pdf",
            "source_type": "commencement_notification",
            "jurisdiction": "India",
            "government_sphere": "central_and_state_code",
            "publication_date": "2025-01-01",
            "effective_status_note": "Official commencement notification; not locally ingested in the current source pack.",
            "verification_status": "external_reference",
            "is_primary_legal_source": False,
            "file_status": "not_locally_ingested",
        }
    ]

    return {
        "project": "LegalAssist AI",
        "source_directory": "knowledge_base/raw",
        "documents": documents,
        "external_references": external_references,
        "missing_files": missing_files,
    }
