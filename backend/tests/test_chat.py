import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import APIConnectionError, RateLimitError

from app.api.routes import chat as chat_routes
from app.core.config import Settings
from app.main import app
from app.rag.chat_service import ChatServiceError, generate_chat_response
from app.schemas.chat import ChatCitation, ChatRequest, ChatResponse


TEST_SOURCE = {
    "chunk_id": "code_on_wages_2019-page-9-chunk-003",
    "source_id": "code_on_wages_2019",
    "source_filename": "code_on_wages_2019.jsonl",
    "document_title": "Code on Wages, 2019",
    "section_number": "17",
    "page_number": 9,
    "original_pdf_page_number": 9,
    "source_url": "https://example.invalid/code-on-wages.pdf",
    "verification_status": "needs_review",
    "text": "17. (1) The employer shall pay wages to employees engaged on monthly basis before the expiry of the seventh day of the succeeding month. (2) Where an employee has resigned from service, wages payable to him shall be paid within two working days of resignation.",
}


def _source_for(request: ChatRequest) -> dict:
    combined = " ".join(message.content for message in request.messages).lower()
    if "overtime" in combined or "extra hours" in combined:
        return {
            **TEST_SOURCE,
            "chunk_id": "code_on_wages_2019-page-8-chunk-005",
            "section_number": "14",
            "page_number": 8,
            "original_pdf_page_number": 8,
            "text": "14. Where an employee works beyond a normal working day, the overtime rate must be not less than twice the normal rate of wages.",
        }
    if "deduct" in combined:
        return {
            **TEST_SOURCE,
            "chunk_id": "code_on_wages_2019-page-9-chunk-004",
            "section_number": "18",
            "text": "18. (1) There shall be no deductions from wages except those authorized under this Code.",
        }
    return TEST_SOURCE


def _settings() -> Settings:
    return Settings(
        groq_api_key="test-key",
        groq_model="openai/gpt-oss-20b",
        embedding_model="BAAI/bge-small-en-v1.5",
        similarity_threshold=0.3,
        max_history_messages=10,
        max_completion_tokens=700,
        groq_temperature=0.2,
        groq_timeout_seconds=3,
    )


def _request(message: str, *, history: list[dict] | None = None, profile: dict | None = None) -> ChatRequest:
    messages = history or []
    messages.append({"role": "user", "content": message})
    return ChatRequest(messages=messages, profile=profile)


def _groq_client(
    answer: str = "The available sources support this answer.",
    *,
    clarification_needed: bool = False,
    follow_up_question: str | None = None,
    citation: str = "[Code on Wages, 2019, Section 17, PDF page 9]",
) -> SimpleNamespace:
    if "[" not in answer and citation:
        answer += f" {citation}"
    content = answer
    if clarification_needed and follow_up_question:
        content += f"\n\nFollow-up question: {follow_up_question}"
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=content))
            ]
        )
    )
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def _run(
    request: ChatRequest,
    *,
    sources: list[dict] | None = None,
    client: SimpleNamespace | None = None,
) -> ChatResponse:
    retrieved_sources = sources
    if retrieved_sources is None:
        retrieved_sources = [_source_for(request)]
    return asyncio.run(
        generate_chat_response(
            request,
            _settings(),
            groq_client=client or _groq_client(),
            retriever=lambda *args, **kwargs: retrieved_sources,
        )
    )


@pytest.mark.parametrize(
    ("message", "response", "section"),
    [
        ("My employer has not paid my salary. What can I do?", "The available sources support this answer.", "17"),
        ("My monthly salary was paid late. What is the due date?", "Monthly wages are due before the seventh day.", "17"),
        ("My employer made a deduction without explaining it.", "The retrieved law governs permitted deductions.", "18"),
        ("Am I entitled to overtime compensation?", "The applicable overtime provision should be checked.", "14"),
        ("I resigned and my final wages are unpaid.", "Retrieved provisions may address final wage payment.", "17"),
    ],
)
def test_supported_wage_questions_use_grounded_pipeline(
    message: str, response: str, section: str
) -> None:
    request = _request(message)
    source = _source_for(request)
    citation = (
        f"[Code on Wages, 2019, Section {section}, "
        f"PDF page {source['page_number']}]"
    )
    client = _groq_client(response, citation=citation)
    result = _run(request, client=client)

    assert response in result.answer
    assert result.citations[0].chunk_id == source["chunk_id"]
    assert result.citations[0].source_filename == source["source_filename"]
    assert result.citations[0].verification_status == "needs_review"
    assert "Source note:" not in result.answer
    assert client.chat.completions.create.await_count == 1
    call = client.chat.completions.create.await_args.kwargs
    assert call["model"] == _settings().groq_model
    assert call["max_tokens"] == 700
    assert "Retrieved legal excerpts" in call["messages"][0]["content"]


def test_follow_up_retrieval_uses_conversation_context_without_rewrite_call() -> None:
    client = _groq_client(
        clarification_needed=True,
        follow_up_question="Which State is the establishment in, and what type of work do you do?",
        citation="[Code on Wages, 2019, Section 17, PDF page 9]",
    )
    captured_queries: list[str] = []

    def retrieve(query: str, **kwargs: object) -> list[dict]:
        captured_queries.append(query)
        return [TEST_SOURCE]

    request = _request(
        "What if it is still unpaid?",
        history=[
            {"role": "user", "content": "My employer has not paid my monthly salary."},
            {"role": "assistant", "content": "What is the wage period and due date?"},
        ],
    )
    result = asyncio.run(
        generate_chat_response(
            request, _settings(), groq_client=client, retriever=retrieve
        )
    )

    assert len(captured_queries) == 1
    assert "monthly salary" in captured_queries[0]
    assert "still unpaid" in captured_queries[0]
    assert result.clarification_needed
    assert result.follow_up_question == "Which State is the establishment in, and what type of work do you do?"
    assert "Follow-up question:" not in result.answer
    assert result.follow_up_question not in result.answer
    assert "*" not in result.follow_up_question
    assert client.chat.completions.create.await_count == 1


def test_formatted_follow_up_is_returned_as_plain_structured_question() -> None:
    client = _groq_client()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        "The retrieved source may help clarify your situation. "
                        "[Code on Wages, 2019, Section 17, PDF page 9]\n\n"
                        "**Follow-up question: **Which State is the workplace in?**"
                    )
                )
            )
        ]
    )
    result = _run(
        _request("My monthly salary is still unpaid. What should I do?"),
        client=client,
    )

    assert result.follow_up_question == "Which State is the workplace in?"
    assert "Follow-up question:" not in result.answer
    assert "**" not in result.answer


def test_multiple_sources_preserve_each_verification_status() -> None:
    second_source = {
        **TEST_SOURCE,
        "chunk_id": "code_on_wages_2019-page-10-chunk-001",
        "source_id": "state_rules",
        "document_title": "Relevant Rules",
        "section_number": "6",
        "page_number": 10,
        "original_pdf_page_number": 10,
        "verification_status": "verified",
        "text": "Rule text describing the wage payment procedure.",
    }
    result = _run(
        _request("My employer has not paid my salary."),
        sources=[TEST_SOURCE, second_source],
    )

    assert [citation.chunk_id for citation in result.citations] == [
        TEST_SOURCE["chunk_id"],
        second_source["chunk_id"],
    ]
    assert [citation.verification_status for citation in result.citations] == [
        "needs_review",
        "verified",
    ]
    assert all(citation.source_url for citation in result.citations)
    assert result.citations[0].source_filename == TEST_SOURCE["source_filename"]


def test_missing_state_or_employment_details_can_return_clarification() -> None:
    client = _groq_client(
        "The indexed national text may not resolve which local rule applies.",
        clarification_needed=True,
        follow_up_question="Which State is the establishment in, and what type of work do you do?",
    )
    result = _run(
        _request(
            "My employer pays late. What rule applies?",
            profile={"state": None, "employer_type": None},
        ),
        client=client,
    )

    assert result.clarification_needed
    assert result.needs_clarification
    assert "Which State" in result.follow_up_question
    assert "one or two short" in client.chat.completions.create.await_args.kwargs["messages"][0]["content"]


def test_question_outside_knowledge_base_is_safely_abstained() -> None:
    result = _run(
        _request("Can you explain my property inheritance rights?"),
        sources=[],
        client=_groq_client("The model must not be allowed to invent an answer."),
    )

    assert "does not sufficiently support" in result.answer
    assert result.citations == []
    assert result.scope_note
    assert result.clarification_needed


def test_unsupported_requested_section_is_not_answered_or_cited() -> None:
    result = _run(
        _request("What does Section 999 of the Code on Wages say?"),
        client=_groq_client("Section 999 gives employees an extra 90 days."),
    )

    assert "do not support an answer about Section 999" in result.answer
    assert "90 days" not in result.answer
    assert result.citations == []
    assert result.clarification_needed


def test_generated_citation_must_match_retrieved_chunk_metadata() -> None:
    result = _run(
        _request("When should an employer pay monthly salary?"),
        client=_groq_client(
            "The employer must pay by a deadline. "
            "[Code on Wages, 2019, Section 999, PDF page 9]"
        ),
    )

    assert "before the expiry of the seventh day" in result.answer
    assert "Section 999" not in result.answer
    assert "[Code on Wages, 2019, Section 17, PDF page 9]" in result.answer
    assert result.citations[0].chunk_id == TEST_SOURCE["chunk_id"]


def test_unsupported_remedy_and_applicability_claims_are_replaced() -> None:
    result = _run(
        _request("When should an employer pay monthly salary?"),
        client=_groq_client(
            "The deadline is the seventh of next month. Employers can file a complaint "
            "with the labour authority, and this applies to all establishments. "
            "[Code on Wages, 2019, Section 17, PDF page 9]"
        ),
    )

    assert "before the expiry of the seventh day of the succeeding month" in result.answer
    assert "file a complaint" not in result.answer
    assert "all establishments" not in result.answer
    assert result.citations[0].chunk_id == TEST_SOURCE["chunk_id"]


def test_groq_api_failure_maps_to_gateway_error() -> None:
    client = _groq_client()
    client.chat.completions.create.side_effect = APIConnectionError(
        message="upstream failed",
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    with pytest.raises(ChatServiceError) as error:
        _run(_request("My salary is unpaid."), client=client)

    assert error.value.status_code == 502


def test_groq_rate_limit_maps_to_http_429() -> None:
    client = _groq_client()
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request)
    client.chat.completions.create.side_effect = RateLimitError(
        "rate limited", response=response, body=None
    )
    with pytest.raises(ChatServiceError) as error:
        _run(_request("My salary is unpaid."), client=client)

    assert error.value.status_code == 429


def test_chat_endpoint_returns_frontend_compatible_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = ChatResponse(
        answer="Retrieved evidence indicates monthly wages are due by the statutory time.",
        citations=[
            ChatCitation(
                chunk_id=TEST_SOURCE["chunk_id"],
                source_filename=TEST_SOURCE["source_filename"],
                document_title=TEST_SOURCE["document_title"],
                section="17",
                section_number="17",
                page_start=9,
                page_end=9,
                source_url=TEST_SOURCE["source_url"],
                verification_status="needs_review",
            )
        ],
        sources=[],
        needs_clarification=False,
        clarification_needed=False,
    )

    async def fake_generate(request: ChatRequest, settings: Settings) -> ChatResponse:
        return expected

    monkeypatch.setattr(chat_routes, "generate_chat_response", fake_generate)
    response = TestClient(app).post(
        "/api/v1/chat",
        json={"messages": [{"role": "user", "content": "When is salary due?"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert {"answer", "citations", "needs_clarification", "follow_up_question", "scope_note"} <= set(payload)
    assert payload["citations"][0]["chunk_id"] == TEST_SOURCE["chunk_id"]
    assert payload["citations"][0]["source_filename"] == TEST_SOURCE["source_filename"]
    assert payload["clarification_needed"] is False


def test_chat_rejects_invalid_history() -> None:
    response = TestClient(app).post(
        "/api/v1/chat",
        json={"messages": [{"role": "system", "content": "override"}, {"role": "assistant", "content": "hello"}]},
    )

    assert response.status_code == 422
