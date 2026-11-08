from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.extract import (
    DEFAULT_EXTRACTED_DIR,
    DEFAULT_RAW_DIR,
    build_manifest,
    discover_pdf_files,
    extract_pdf_pages,
    generate_ingestion_report,
    print_ingestion_report,
    write_extracted_pages,
)
from app.rag.practical_guide import (
    DEFAULT_GUIDE_PATH,
    merge_practical_guide_manifest,
)


def write_checksums(raw_dir: Path, checksums_path: Path) -> None:
    checksum_map: dict[str, str] = {}
    for pdf_path in discover_pdf_files(raw_dir):
        digest = hashlib.sha256()
        with pdf_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        checksum_map[pdf_path.name] = digest.hexdigest()
    checksums_path.write_text(json.dumps(checksum_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    raw_dir = ROOT / "knowledge_base" / "raw"
    extracted_dir = ROOT / "knowledge_base" / "extracted"
    manifest_path = ROOT / "knowledge_base" / "source_manifest.json"
    checksums_path = ROOT / "knowledge_base" / "source_checksums.json"

    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = discover_pdf_files(raw_dir)
    if not pdf_files:
        print("No PDF files found in knowledge_base/raw/.")
        manifest = build_manifest(raw_dir)
        if DEFAULT_GUIDE_PATH.is_file():
            manifest = merge_practical_guide_manifest(manifest, DEFAULT_GUIDE_PATH)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        checksums_path.write_text("{}\n", encoding="utf-8")
        return

    for pdf_path in pdf_files:
        records = extract_pdf_pages(pdf_path)
        write_extracted_pages(records, extracted_dir)

    manifest = build_manifest(raw_dir)
    if DEFAULT_GUIDE_PATH.is_file():
        manifest = merge_practical_guide_manifest(manifest, DEFAULT_GUIDE_PATH)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_checksums(raw_dir, checksums_path)

    report = generate_ingestion_report(raw_dir, extracted_dir)
    print_ingestion_report(report)


if __name__ == "__main__":
    main()
