from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.build_index import DEFAULT_CHUNKS_PATH, DEFAULT_VECTOR_STORE_DIR, build_and_save_faiss_index
from app.rag.chunk import generate_chunk_report
from app.rag.practical_guide import (
    DEFAULT_GUIDE_PATH,
    GUIDE_SOURCE_ID,
    chunk_practical_guide,
    merge_practical_guide_manifest,
)

PROCESSED_DIR = ROOT / "knowledge_base" / "processed"
MANIFEST_PATH = ROOT / "knowledge_base" / "source_manifest.json"
REPORT_PATH = PROCESSED_DIR / "chunk_report.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def ingest_practical_guide() -> dict[str, Any]:
    if not DEFAULT_GUIDE_PATH.is_file():
        raise FileNotFoundError(f"Practical guide not found: {DEFAULT_GUIDE_PATH}")
    if not DEFAULT_CHUNKS_PATH.is_file():
        raise FileNotFoundError(
            f"Canonical processed chunks not found: {DEFAULT_CHUNKS_PATH}"
        )
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Source manifest not found: {MANIFEST_PATH}")

    original_chunks_bytes = DEFAULT_CHUNKS_PATH.read_bytes()
    existing_chunks = _read_jsonl(DEFAULT_CHUNKS_PATH)
    official_chunks = [
        chunk
        for chunk in existing_chunks
        if chunk.get("source_id") != GUIDE_SOURCE_ID
    ]
    guide_chunks = chunk_practical_guide(DEFAULT_GUIDE_PATH)
    merged_chunks = official_chunks + guide_chunks

    _write_jsonl(DEFAULT_CHUNKS_PATH, merged_chunks)
    try:
        index_info = build_and_save_faiss_index(
            DEFAULT_CHUNKS_PATH,
            output_dir=DEFAULT_VECTOR_STORE_DIR,
        )
    except Exception:
        DEFAULT_CHUNKS_PATH.write_bytes(original_chunks_bytes)
        raise

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_chunk_report(merged_chunks)
    source_counts = Counter(
        (chunk.get("source_id"), chunk.get("source_type", "official_or_legacy"))
        for chunk in merged_chunks
    )
    report["source_breakdown"] = [
        {
            "source_id": source_id,
            "source_type": source_type,
            "chunk_count": count,
        }
        for (source_id, source_type), count in sorted(source_counts.items())
    ]
    report["practical_guide"] = {
        "source_id": GUIDE_SOURCE_ID,
        "chunk_count": len(guide_chunks),
        "module_counts": dict(
            sorted(
                Counter(chunk.get("module_id") or "General" for chunk in guide_chunks).items()
            )
        ),
        "chunk_target_characters": 1600,
        "oversized_chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "characters": chunk["char_count"],
                "location": chunk["location"],
            }
            for chunk in guide_chunks
            if chunk["char_count"] > 1600
        ],
    }

    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    merge_practical_guide_manifest(manifest, DEFAULT_GUIDE_PATH)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "guide_chunk_count": len(guide_chunks),
        "total_chunk_count": len(merged_chunks),
        "index_count": index_info["vector_count"],
        "index_fingerprint": index_info["fingerprint"],
        "missing_companion_files": manifest.get("missing_companion_files", []),
        "report_path": str(REPORT_PATH),
    }


def main() -> None:
    result = ingest_practical_guide()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
