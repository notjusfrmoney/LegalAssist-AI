"""Placeholder retrieval service."""


class RetrievalService:
    def __init__(self) -> None:
        self.index_loaded = False

    def load_index(self) -> None:
        self.index_loaded = True

    def search(self, query: str, top_k: int = 5) -> list[dict[str, str | float]]:
        return [{"chunk_id": "placeholder", "score": 0.0, "text": query}][:top_k]
