from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.rag.build_index import DEFAULT_VECTOR_STORE_DIR, load_existing_index
from app.rag.chat_service import ChatServiceError, generate_chat_response
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["chat"])
settings = get_settings()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        return await generate_chat_response(request, settings)
    except ChatServiceError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error


@router.get("/sources")
async def sources() -> list[dict[str, str | None]]:
    try:
        loaded = load_existing_index(DEFAULT_VECTOR_STORE_DIR)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="The legal knowledge base is unavailable.") from error

    unique_sources: dict[str, dict[str, str | None]] = {}
    for row in loaded["metadata"]:
        source_id = row.get("source_id")
        if not source_id or source_id in unique_sources:
            continue
        unique_sources[source_id] = {
            "source_id": source_id,
            "document_title": row.get("document_title"),
            "source_type": row.get("source_type"),
            "jurisdiction": row.get("jurisdiction"),
            "source_url": row.get("source_url"),
            "verification_status": row.get("verification_status"),
        }
    return list(unique_sources.values())
