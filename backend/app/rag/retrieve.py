"""Retrieval helpers for the local vector index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.rag.build_index import DEFAULT_VECTOR_STORE_DIR, build_and_save_faiss_index, load_existing_index
from app.rag.embed import MODEL_NAME, embed_query, load_embedding_model

_AUTHORITY_ORDER = {
    "statutory": 0,
    "delegated_rules": 1,
    "official_guidance": 2,
    "non_authoritative": 3,
}


def retrieve_top_k(
    query: str,
    *,
    chunks_path: str | Path = Path("knowledge_base/processed/chunks.jsonl"),
    index_dir: str | Path = DEFAULT_VECTOR_STORE_DIR,
    top_k: int = 5,
    min_similarity: float = 0.2,
    model_name: str = MODEL_NAME,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant chunks for a user query using cosine similarity."""
    if not query or not query.strip():
        return []

    try:
        loaded = load_existing_index(index_dir)
        index = loaded["index"]
        metadata = loaded["metadata"]
        info = loaded["info"]
    except FileNotFoundError:
        info = build_and_save_faiss_index(chunks_path=chunks_path, output_dir=index_dir, model_name=model_name)
        loaded = load_existing_index(index_dir)
        index = loaded["index"]
        metadata = loaded["metadata"]

    if index.ntotal == 0:
        return []

    model = load_embedding_model(model_name)
    query_vector = embed_query(query, model=model)
    faiss.normalize_L2(query_vector)
    candidate_count = min(max(top_k * 4, top_k), index.ntotal)
    distances, indices = index.search(query_vector, candidate_count)

    results: list[dict[str, Any]] = []
    for score, vector_id in zip(distances[0], indices[0]):
        if vector_id < 0:
            continue
        if float(score) < min_similarity:
            continue
        row = metadata[int(vector_id)]
        results.append(
            {
                "chunk_id": row.get("chunk_id"),
                "score": float(score),
                "text": row.get("text"),
                "source_id": row.get("source_id"),
                "source_filename": row.get("source_filename"),
                "document_title": row.get("document_title"),
                "section_number": row.get("section_number"),
                "page_number": row.get("page_number"),
                "original_pdf_page_number": row.get("original_pdf_page_number"),
                "source_url": row.get("source_url"),
                "source_type": row.get("source_type"),
                "authority_level": row.get("authority_level", "unverified"),
                "verification_status": row.get("verification_status"),
                "module_id": row.get("module_id"),
                "module_heading": row.get("module_heading"),
                "section_heading": row.get("section_heading"),
                "heading_path": row.get("heading_path"),
                "location": row.get("location"),
                "line_start": row.get("line_start"),
                "line_end": row.get("line_end"),
                "claim_ids": row.get("claim_ids", []),
                "unresolved_refs": row.get("unresolved_refs", []),
                "vector_id": int(vector_id),
                "index_info": info,
            }
        )

    results.sort(
        key=lambda item: (
            _AUTHORITY_ORDER.get(item["authority_level"], 4),
            -item["score"],
        )
    )
    return results[:top_k]


def retrieve(query: str, top_k: int = 5, min_similarity: float = 0.2) -> list[dict[str, Any]]:
    """Compatibility wrapper for simple retrieval calls."""
    return retrieve_top_k(query, top_k=top_k, min_similarity=min_similarity)
