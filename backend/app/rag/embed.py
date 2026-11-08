"""Sentence-transformer embedding utilities for legal chunk retrieval."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load and cache the configured text-embedding model."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
    return _MODEL_CACHE[model_name]


def _normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return values / norms


def generate_document_embeddings(
    chunks: Sequence[dict],
    *,
    model_name: str = MODEL_NAME,
    model: SentenceTransformer | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed legal chunks in batches and normalize them for cosine similarity."""
    texts = [str(chunk.get("text", "")).strip() for chunk in chunks if str(chunk.get("text", "")).strip()]
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    active_model = model or load_embedding_model(model_name)
    vectors = active_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    matrix = _normalize_embeddings(matrix)
    if matrix.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {matrix.shape[1]}")
    return matrix


def embed_query(
    query: str,
    *,
    model_name: str = MODEL_NAME,
    model: SentenceTransformer | None = None,
) -> np.ndarray:
    """Embed a single user query for cosine-similarity search."""
    if not query or not query.strip():
        return np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
    active_model = model or load_embedding_model(model_name)
    vector = active_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    vector = _normalize_embeddings(vector)
    if vector.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {vector.shape[1]}")
    return vector


def build_embedding(text: str) -> list[float]:
    """Compatibility wrapper for legacy callers."""
    model = load_embedding_model()
    vector = embed_query(text, model=model)[0]
    return [float(value) for value in vector]
