"""Run two real Groq-backed conversations and save sanitized response artifacts."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings
from app.rag.chat_service import ChatServiceError, generate_chat_response
from app.schemas.chat import ChatMessage, ChatProfile, ChatRequest

OUTPUT_DIR = ROOT / "evaluation" / "phase5_conversation_outputs"


async def _run_conversation(
    conversation_id: str,
    messages: list[ChatMessage],
    profile: ChatProfile | None = None,
) -> dict[str, Any]:
    request = ChatRequest(messages=messages, profile=profile)
    response = await generate_chat_response(request, get_settings())
    return {
        "conversation_id": conversation_id,
        "request": request.model_dump(exclude_none=True),
        "response": response.model_dump(),
    }


async def main() -> None:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured in backend/.env.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conversations = (
        (
            "monthly_salary_due",
            [
                ChatMessage(
                    role="user",
                    content="When should an employer pay monthly salary?",
                )
            ],
            ChatProfile(state="Maharashtra", employment_type="monthly salaried"),
        ),
        (
            "salary_due_follow_up",
            [
                ChatMessage(
                    role="user",
                    content="When should an employer pay monthly salary?",
                ),
                ChatMessage(
                    role="assistant",
                    content="The indexed Code on Wages passage states a payment timeline. Which payment was delayed?",
                ),
                ChatMessage(
                    role="user",
                    content="What if it is still unpaid?",
                ),
            ],
            None,
        ),
    )

    for conversation_id, messages, profile in conversations:
        try:
            result = await _run_conversation(conversation_id, messages, profile)
        except ChatServiceError as error:
            raise RuntimeError(
                f"Manual integration failed with service status {error.status_code}: {error.detail}"
            ) from error
        output_path = OUTPUT_DIR / f"{conversation_id}.json"
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{conversation_id}: saved {output_path}; "
            f"citations={len(result['response']['citations'])}; "
            f"answer_chars={len(result['response']['answer'])}"
        )


if __name__ == "__main__":
    asyncio.run(main())
