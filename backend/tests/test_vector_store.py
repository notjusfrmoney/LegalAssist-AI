import json
import shutil
from pathlib import Path

import numpy as np

from app.rag.build_index import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_VECTOR_STORE_DIR,
    build_and_save_faiss_index,
    load_existing_index,
)
from app.rag.embed import EMBEDDING_DIM, embed_query, load_embedding_model
from app.rag.retrieve import retrieve_top_k


def _write_retrieval_outputs(
    results: dict[str, list[dict]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase5_practical_guide_retrieval_results.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_embedding_dimension_and_normalization() -> None:
    model = load_embedding_model()
    vector = embed_query("When should an employer pay wages?", model=model)

    assert vector.shape == (1, EMBEDDING_DIM)
    norms = np.linalg.norm(vector, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-4, atol=1e-4)


def test_faiss_index_count_matches_chunk_count(tmp_path: Path) -> None:
    vector_store_dir = tmp_path / "vector_store"
    info = build_and_save_faiss_index(
        chunks_path=DEFAULT_CHUNKS_PATH,
        output_dir=vector_store_dir,
        force_rebuild=True,
    )
    loaded = load_existing_index(vector_store_dir)

    assert info["vector_count"] == loaded["index"].ntotal
    assert info["chunk_count"] == loaded["index"].ntotal
    assert info["embedding_dim"] == EMBEDDING_DIM


def test_metadata_count_matches_vector_count() -> None:
    loaded = load_existing_index(DEFAULT_VECTOR_STORE_DIR)
    assert len(loaded["metadata"]) == loaded["index"].ntotal
    assert all(row.get("chunk_id") for row in loaded["metadata"])
    assert all(row.get("source_id") for row in loaded["metadata"])


def test_index_can_be_loaded_again() -> None:
    loaded = load_existing_index(DEFAULT_VECTOR_STORE_DIR)

    assert loaded["index"] is not None
    assert loaded["metadata"]
    assert loaded["info"]["embedding_dim"] == EMBEDDING_DIM
    assert loaded["info"]["model_name"] == "BAAI/bge-small-en-v1.5"


def test_retrieval_returns_relevant_metadata_for_realistic_queries(
    tmp_path: Path,
) -> None:
    vector_store_dir = tmp_path / "vector_store"
    shutil.copytree(DEFAULT_VECTOR_STORE_DIR, vector_store_dir)
    question_set = {
        "When should an employer pay monthly salary?": ["salary", "monthly", "pay"],
        "What deductions can an employer make from wages?": ["deduction", "wages", "recover"],
        "Is an employee entitled to overtime wages?": ["overtime", "wages", "employee"],
        "What happens to unpaid wages after resignation?": ["resignation", "final wages", "unpaid"],
    }

    results_by_question: dict[str, list[dict]] = {}
    for question, expected_keywords in question_set.items():
        matches = retrieve_top_k(
            question,
            top_k=3,
            min_similarity=0.1,
            index_dir=vector_store_dir,
        )
        assert matches, f"No retrieval hits for: {question}"
        results_by_question[question] = [
            {
                "chunk_id": item["chunk_id"],
                "score": round(float(item["score"]), 6),
                "source_id": item["source_id"],
                "document_title": item["document_title"],
                "section_number": item["section_number"],
                "page_number": item["page_number"],
                "source_url": item["source_url"],
                "verification_status": item["verification_status"],
                "text_preview": item["text"][:400],
            }
            for item in matches
        ]

        combined_text = "\n".join(item["text"] for item in matches).lower()
        assert any(keyword.lower() in combined_text for keyword in expected_keywords), (
            f"Query {question!r} did not return relevant legal text: {combined_text[:400]}"
        )

    _write_retrieval_outputs(results_by_question, tmp_path / "evaluation")
