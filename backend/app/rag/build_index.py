"""FAISS index build and persistence utilities."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

from app.rag.embed import EMBEDDING_DIM, MODEL_NAME, generate_document_embeddings, load_embedding_model

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNKS_PATH = REPO_ROOT / "knowledge_base" / "processed" / "chunks.jsonl"
DEFAULT_VECTOR_STORE_DIR = REPO_ROOT / "knowledge_base" / "processed" / "vector_store"

SOURCE_AUTHORITY_BY_TYPE = {
    "primary_legislation": "statutory",
    "delegated_rules": "delegated_rules",
    "official_faq": "official_guidance",
    "practical_guidance": "non_authoritative",
}


def _load_chunks(chunks_path: str | Path = DEFAULT_CHUNKS_PATH) -> list[dict[str, Any]]:
    path = Path(chunks_path)
    if not path.exists():
        return []
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def _build_fingerprint(chunks: Sequence[dict[str, Any]], *, model_name: str, batch_size: int) -> str:
    payload = {
        "model_name": model_name,
        "batch_size": batch_size,
        "embedding_dim": EMBEDDING_DIM,
        "chunks": [
            {
                "chunk_id": chunk.get("chunk_id"),
                "source_id": chunk.get("source_id"),
                "text": chunk.get("text", ""),
                "source_type": chunk.get("source_type"),
                "authority_level": chunk.get("authority_level"),
                "verification_status": chunk.get("verification_status"),
                "module_id": chunk.get("module_id"),
                "section_heading": chunk.get("section_heading"),
                "claim_ids": chunk.get("claim_ids"),
                "unresolved_refs": chunk.get("unresolved_refs"),
            }
            for chunk in chunks
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_metadata_rows(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        text = str(chunk.get("text", "")).strip()
        section_number = chunk.get("section_number")
        source_type = chunk.get("source_type")
        record = {
            "vector_id": index,
            "chunk_id": chunk.get("chunk_id"),
            "source_id": chunk.get("source_id"),
            "source_filename": chunk.get("source_filename") or (chunk.get("source_id") + ".jsonl" if chunk.get("source_id") else None),
            "document_title": chunk.get("document_title"),
            "section_number": section_number,
            "page_number": chunk.get("page_number"),
            "original_pdf_page_number": chunk.get("page_number"),
            "source_url": chunk.get("source_url"),
            "source_type": source_type,
            "authority_level": chunk.get("authority_level")
            or SOURCE_AUTHORITY_BY_TYPE.get(source_type, "unverified"),
            "verification_status": chunk.get("verification_status"),
            "module_id": chunk.get("module_id"),
            "module_heading": chunk.get("module_heading"),
            "section_heading": chunk.get("section_heading"),
            "heading_path": chunk.get("heading_path"),
            "location": chunk.get("location"),
            "line_start": chunk.get("line_start"),
            "line_end": chunk.get("line_end"),
            "claim_ids": chunk.get("claim_ids", []),
            "unresolved_refs": chunk.get("unresolved_refs", []),
            "text": text,
        }
        rows.append(record)
    return rows


def build_and_save_faiss_index(
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    *,
    output_dir: str | Path = DEFAULT_VECTOR_STORE_DIR,
    model_name: str = MODEL_NAME,
    batch_size: int = 1,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Build the FAISS index from chunks and persist metadata for later retrieval."""
    chunks = _load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunk data found in {chunks_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    index_path = output_path / "index.faiss"
    metadata_path = output_path / "metadata.jsonl"
    info_path = output_path / "index_info.json"

    fingerprint = _build_fingerprint(chunks, model_name=model_name, batch_size=batch_size)
    if index_path.exists() and info_path.exists() and not force_rebuild:
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        if info.get("fingerprint") == fingerprint and info.get("model_name") == model_name and info.get("embedding_dim") == EMBEDDING_DIM:
            return load_existing_index(output_dir)

    model = load_embedding_model(model_name)
    embeddings = generate_document_embeddings(chunks, model=model, batch_size=batch_size)
    if embeddings.shape[0] != len(chunks):
        raise ValueError(f"Embedding count {embeddings.shape[0]} does not match chunk count {len(chunks)}")

    normalized = np.asarray(embeddings, dtype=np.float32).copy()
    faiss.normalize_L2(normalized)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(normalized)
    faiss.write_index(index, str(index_path))

    metadata_rows = _prepare_metadata_rows(chunks)
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    info = {
        "model_name": model_name,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_count": len(chunks),
        "vector_count": index.ntotal,
        "fingerprint": fingerprint,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunks_path": str(Path(chunks_path)),
        "output_dir": str(output_path),
        "top_k_default": 5,
        "min_similarity_default": 0.2,
        "authority_counts": {
            authority: sum(
                row["authority_level"] == authority for row in metadata_rows
            )
            for authority in sorted(
                {row["authority_level"] for row in metadata_rows}
            )
        },
    }
    with info_path.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return info


def load_existing_index(index_dir: str | Path = DEFAULT_VECTOR_STORE_DIR) -> dict[str, Any]:
    """Load the FAISS index, metadata rows and index metadata from disk."""
    index_path = Path(index_dir) / "index.faiss"
    metadata_path = Path(index_dir) / "metadata.jsonl"
    info_path = Path(index_dir) / "index_info.json"

    if not index_path.exists() or not metadata_path.exists() or not info_path.exists():
        raise FileNotFoundError(f"Vector store not found in {index_dir}")

    index = faiss.read_index(str(index_path))
    metadata_rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                metadata_rows.append(json.loads(line))

    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)

    return {"index": index, "metadata": metadata_rows, "info": info}


def load_index(index_dir: str | Path = DEFAULT_VECTOR_STORE_DIR) -> dict[str, Any]:
    """Compatibility wrapper for loading an existing vector store."""
    return load_existing_index(index_dir)


def build_index() -> dict[str, Any]:
    """Compatibility wrapper for index creation."""
    return build_and_save_faiss_index()
