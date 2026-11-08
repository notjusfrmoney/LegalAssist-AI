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
from app.rag.chat_service import (
    ChatServiceError,
    _format_retrieved_sources,
    _has_unsupported_legal_claim,
    _inline_citations_are_supported,
    _is_authoritative_source,
    _merge_unpaid_wage_sources,
    _legal_assertion_spans,
    _prioritize_legal_sources,
    _selected_practical_guide_modules,
    _selected_wage_recovery_sections,
    build_retrieval_query,
    generate_chat_response,
)
from app.schemas.chat import ChatCitation, ChatRequest, ChatResponse


TEST_SOURCE = {
    "chunk_id": "code_on_wages_2019-page-9-chunk-003",
    "source_id": "code_on_wages_2019",
    "source_filename": "code_on_wages_2019.jsonl",
    "document_title": "Code on Wages, 2019",
    "source_type": "primary_legislation",
    "authority_level": "statutory",
    "section_number": "17",
    "page_number": 9,
    "original_pdf_page_number": 9,
    "source_url": "https://example.invalid/code-on-wages.pdf",
    "verification_status": "needs_review",
    "text": "17. (1) The employer shall pay wages to employees engaged on monthly basis before the expiry of the seventh day of the succeeding month. (2) Where an employee has resigned from service, wages payable to him shall be paid within two working days of resignation.",
}

SECTION_43_SOURCE = {
    **TEST_SOURCE,
    "chunk_id": "code_on_wages_2019-page-19-chunk-002",
    "section_number": "43",
    "page_number": 19,
    "original_pdf_page_number": 19,
    "text": (
        "43. Every employer shall pay all amounts required to be paid under this "
        "Code to every employee: Provided that where such employer fails to make "
        "such payment in accordance with this Code, then the proprietor of the "
        "establishment shall be responsible for such payment."
    ),
}

SECTION_45_SOURCE = {
    **TEST_SOURCE,
    "chunk_id": "code_on_wages_2019-page-20-chunk-002",
    "section_number": "45",
    "page_number": 20,
    "original_pdf_page_number": 20,
    "text": (
        "45. The appropriate Government may appoint authorities to hear and "
        "determine claims which arises under the provisions of this Code. Any "
        "application before the authority for a claim may be filed by the "
        "employee concerned. The application may be filed within a period of "
        "three years from the date on which the claim arises."
    ),
}

PRACTICAL_SOURCE = {
    "chunk_id": "employment_wage_practical_guide_claude-PRACTICAL-01-section-010-chunk-01",
    "source_id": "employment_wage_practical_guide_claude",
    "source_filename": "employment_wage_practical_guide_claude.md",
    "document_title": "Employment Wage Rights – Practical Action Guide",
    "source_type": "practical_guidance",
    "authority_level": "non_authoritative",
    "module_id": "PRACTICAL-01",
    "module_heading": "PRACTICAL-01: Delayed Monthly Salary",
    "section_heading": "D. Evidence to preserve",
    "heading_path": [
        "PART B – THE SIX PRACTICAL MODULES",
        "PRACTICAL-01: Delayed Monthly Salary",
        "D. Evidence to preserve",
    ],
    "location": "PART B > PRACTICAL-01: Delayed Monthly Salary > D. Evidence to preserve",
    "claim_ids": ["WAGE-016"],
    "unresolved_refs": ["UNRES-10"],
    "page_number": None,
    "original_pdf_page_number": None,
    "source_url": None,
    "verification_status": "pending_human_review",
    "text": "Keep payslips and bank statements together as practical evidence.",
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


def _salary_follow_up_request(message: str) -> ChatRequest:
    return ChatRequest(
        messages=[
            {"role": "assistant", "content": "Hi — tell me what happened."},
            {"role": "user", "content": "My last month salary is pending"},
            {
                "role": "assistant",
                "content": (
                    "Which month's salary is pending, and is your employer a "
                    "private company, government establishment, or another type "
                    "of organization?"
                ),
            },
            {
                "role": "user",
                "content": "October's salary and I work for private company",
            },
            {
                "role": "assistant",
                "content": (
                    "The Code on Wages, 2019 provides a monthly wage-payment "
                    "deadline [Code on Wages, 2019, Section 17, PDF page 9]."
                ),
            },
            {"role": "user", "content": message},
        ]
    )


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


def test_authority_order_places_law_before_faqs_and_practical_guidance() -> None:
    sources = [
        {**PRACTICAL_SOURCE},
        {**TEST_SOURCE, "source_type": "official_faq", "authority_level": "official_guidance"},
        {**TEST_SOURCE, "source_type": "delegated_rules", "authority_level": "delegated_rules"},
        {**TEST_SOURCE, "source_type": "primary_legislation", "authority_level": "statutory"},
    ]

    ordered = _prioritize_legal_sources(sources)

    assert [source["source_type"] for source in ordered] == [
        "primary_legislation",
        "delegated_rules",
        "official_faq",
        "practical_guidance",
    ]
    formatted = _format_retrieved_sources(ordered)
    assert "authority_level=non_authoritative" in formatted
    assert "unresolved_refs=UNRES-10" in formatted


def test_practical_guidance_citation_preserves_heading_and_review_metadata() -> None:
    answer = (
        "Consider keeping your recent payslips and bank statements together. "
        "[Employment Wage Rights – Practical Action Guide, PRACTICAL-01, "
        "D. Evidence to preserve]"
    )
    client = _groq_client(answer)
    result = _run(
        _request("How can I organize evidence for late salary?"),
        sources=[PRACTICAL_SOURCE],
        client=client,
    )

    assert result.answer == answer
    source = result.citations[0]
    assert source.source_type == "practical_guidance"
    assert source.authority_level == "non_authoritative"
    assert source.verification_status == "pending_human_review"
    assert source.module_id == "PRACTICAL-01"
    assert source.section_heading == "D. Evidence to preserve"
    assert source.page_start is None
    assert source.claim_ids == ["WAGE-016"]
    assert source.unresolved_refs == ["UNRES-10"]


def test_practical_guidance_with_normal_metadata_is_not_authoritative() -> None:
    assert not _is_authoritative_source(PRACTICAL_SOURCE)


def test_practical_guide_source_id_overrides_missing_source_type() -> None:
    source = {
        "source_id": "employment_wage_practical_guide_claude",
        "authority_level": "statutory",
    }

    assert not _is_authoritative_source(source)


def test_conflicting_practical_guidance_authority_metadata_fails_closed() -> None:
    source = {
        **PRACTICAL_SOURCE,
        "source_type": "primary_legislation",
        "authority_level": "statutory",
    }

    assert not _is_authoritative_source(source)


@pytest.mark.parametrize(
    "source",
    [
        {**TEST_SOURCE, "source_type": "primary_legislation", "authority_level": "statutory"},
        {
            **TEST_SOURCE,
            "source_type": "delegated_rules",
            "authority_level": "delegated_rules",
        },
    ],
)
def test_matching_official_source_authority_metadata_remains_supported(
    source: dict,
) -> None:
    assert _is_authoritative_source(source)


@pytest.mark.parametrize(
    "provenance_field",
    ["chunk_id", "document_title"],
)
def test_practical_guide_provenance_overrides_conflicting_statutory_metadata(
    provenance_field: str,
) -> None:
    source = {
        **TEST_SOURCE,
        "source_type": "primary_legislation",
        "authority_level": "statutory",
        "source_id": None,
        "source_filename": None,
        "source_file": None,
        "chunk_id": "employment_wage_practical_guide_claude-PRACTICAL-01-001",
        "document_title": "Employment Wage Rights – Practical Action Guide",
    }
    if provenance_field == "chunk_id":
        source["document_title"] = "Mislabelled legal source"
    else:
        source["chunk_id"] = "mislabelled-legal-chunk"

    assert not _is_authoritative_source(source)


def test_official_faq_authority_metadata_remains_supported() -> None:
    source = {
        **TEST_SOURCE,
        "source_type": "official_faq",
        "authority_level": "official_guidance",
    }

    assert _is_authoritative_source(source)


def test_practical_guidance_cannot_support_a_settled_legal_claim() -> None:
    unsupported_answer = (
        "The law requires employers to pay monthly wages by the seventh day. "
        "[Employment Wage Rights – Practical Action Guide, PRACTICAL-01, "
        "B2. Legal explanation]"
    )
    result = _run(
        _request("When is monthly salary legally due?"),
        sources=[PRACTICAL_SOURCE],
        client=_groq_client(unsupported_answer),
    )

    assert "does not sufficiently support" in result.answer
    assert "employers to pay monthly wages" not in result.answer
    assert result.citations == []
    assert not _inline_citations_are_supported(unsupported_answer, [PRACTICAL_SOURCE])


def test_legal_assertion_cited_only_to_practical_guidance_is_rejected() -> None:
    answer = (
        "The law requires employers to pay monthly wages by the seventh day "
        "[Employment Wage Rights – Practical Action Guide, PRACTICAL-01, "
        "D. Evidence to preserve]"
    )

    assert not _inline_citations_are_supported(answer, [PRACTICAL_SOURCE])


def test_unrelated_official_citation_elsewhere_does_not_support_guide_cited_claim() -> None:
    answer = (
        "The law requires employers to pay monthly wages by the seventh day "
        "[Employment Wage Rights – Practical Action Guide, PRACTICAL-01, "
        "D. Evidence to preserve]. "
        "The Code is titled Code on Wages "
        "[Code on Wages, 2019, Section 17, PDF page 9]."
    )

    assert not _inline_citations_are_supported(
        answer, [PRACTICAL_SOURCE, TEST_SOURCE]
    )


def test_legal_assertion_with_matching_official_citation_is_accepted() -> None:
    answer = (
        "Monthly wages must be paid before the seventh day "
        "[Code on Wages, 2019, Section 17, PDF page 9]"
    )

    assert _inline_citations_are_supported(answer, [TEST_SOURCE])


def test_overtime_claim_citing_unrelated_section_17_is_rejected() -> None:
    answer = (
        "Overtime is paid at twice the normal rate "
        "[Code on Wages, 2019, Section 17, PDF page 9]"
    )

    assert not _inline_citations_are_supported(answer, [TEST_SOURCE])


def test_overtime_claim_with_supporting_section_14_evidence_is_accepted() -> None:
    overtime_source = _source_for(_request("Is overtime paid at twice the rate?"))
    answer = (
        "Overtime is paid at twice the normal rate "
        "[Code on Wages, 2019, Section 14, PDF page 8]"
    )

    assert overtime_source["section_number"] == "14"
    assert _inline_citations_are_supported(answer, [overtime_source])


def test_wage_deadline_claim_citing_unrelated_section_14_is_rejected() -> None:
    unrelated_source = _source_for(_request("Is overtime paid at twice the rate?"))
    answer = (
        "Monthly wages must be paid before the seventh day "
        "[Code on Wages, 2019, Section 14, PDF page 8]"
    )

    assert not _inline_citations_are_supported(answer, [unrelated_source])


def test_deduction_claim_citing_unrelated_section_17_is_rejected() -> None:
    answer = (
        "Deductions from wages are permitted only when authorized by law "
        "[Code on Wages, 2019, Section 17, PDF page 9]"
    )

    assert not _inline_citations_are_supported(answer, [TEST_SOURCE])


def test_resignation_wage_claim_citing_unrelated_section_18_is_rejected() -> None:
    deductions_source = _source_for(_request("What deductions can be made from wages?"))
    answer = (
        "Wages after resignation are paid within two working days "
        "[Code on Wages, 2019, Section 18, PDF page 9]"
    )

    assert not _inline_citations_are_supported(answer, [deductions_source])


def test_two_legal_claims_with_only_one_supporting_citation_are_rejected() -> None:
    answer = (
        "Monthly wages must be paid before the seventh day and overtime is paid "
        "at twice the normal rate [Code on Wages, 2019, Section 17, PDF page 9]"
    )

    assert not _inline_citations_are_supported(answer, [TEST_SOURCE])


def test_employer_payment_claim_requires_section_43_support() -> None:
    section_43_answer = (
        "Every employer shall pay all amounts required under this Code "
        "[Code on Wages, 2019, Section 43, PDF page 19]"
    )
    section_17_answer = section_43_answer.replace("Section 43", "Section 17").replace(
        "PDF page 19", "PDF page 9"
    )

    assert _inline_citations_are_supported(section_43_answer, [SECTION_43_SOURCE])
    assert not _inline_citations_are_supported(section_17_answer, [TEST_SOURCE])


def test_claims_authority_assertion_requires_supporting_section_45_text() -> None:
    answer = (
        "The appropriate Government may appoint authorities to hear and determine "
        "claims which arise under the provisions of this Code "
        "[Code on Wages, 2019, Section 45, PDF page 20]"
    )
    wrong_section = answer.replace("Section 45", "Section 17").replace(
        "PDF page 20", "PDF page 9"
    )
    unsupported_text = {
        **SECTION_45_SOURCE,
        "text": "45. This section contains provisions about wages and employees.",
    }

    assert _inline_citations_are_supported(answer, [SECTION_45_SOURCE])
    assert not _inline_citations_are_supported(wrong_section, [TEST_SOURCE])
    assert not _inline_citations_are_supported(answer, [unsupported_text])


def test_distinct_legal_claims_in_one_sentence_each_need_matching_evidence() -> None:
    sentence = (
        "Every employer shall pay all amounts required under this Code, and the "
        "appropriate Government may appoint authorities to hear and determine "
        "claims arising under the provisions of this Code"
    )
    answer_with_both = (
        f"{sentence} [Code on Wages, 2019, Section 43, PDF page 19] "
        "[Code on Wages, 2019, Section 45, PDF page 20]"
    )
    answer_with_one = (
        f"{sentence} [Code on Wages, 2019, Section 43, PDF page 19]"
    )

    assert _inline_citations_are_supported(
        answer_with_both, [SECTION_43_SOURCE, SECTION_45_SOURCE]
    )
    assert not _inline_citations_are_supported(
        answer_with_one, [SECTION_43_SOURCE, SECTION_45_SOURCE]
    )


def test_supported_section_45_filing_phrase_passes_unsupported_claim_guard() -> None:
    answer = (
        "The employee concerned may file a wage claim under Section 45 "
        "[Code on Wages, 2019, Section 45, PDF page 20]"
    )

    assert _inline_citations_are_supported(answer, [SECTION_45_SOURCE])
    assert not _has_unsupported_legal_claim(answer, [SECTION_45_SOURCE])


def test_section_45_does_not_support_an_unstated_labour_authority_route() -> None:
    answer = (
        "Employees can contact the labour authority to file a wage claim "
        "[Code on Wages, 2019, Section 45, PDF page 20]"
    )

    assert not _inline_citations_are_supported(answer, [SECTION_45_SOURCE])


def test_claim_deadline_requires_authoritative_section_45_text() -> None:
    answer = (
        "An application under the Code must be filed within three years from "
        "when the claim arises [Code on Wages, 2019, Section 45, PDF page 20]"
    )
    practical_answer = (
        "An application under the Code must be filed within three years from "
        "when the claim arises [Employment Wage Rights – Practical Action Guide, "
        "PRACTICAL-01, D. Evidence to preserve]"
    )

    assert _inline_citations_are_supported(answer, [SECTION_45_SOURCE])
    assert not _inline_citations_are_supported(
        practical_answer, [PRACTICAL_SOURCE]
    )


def test_practical_guide_cannot_support_a_cautiously_worded_claim_procedure() -> None:
    answer = (
        "Consider filing a wage claim with the appointed authority "
        "[Employment Wage Rights – Practical Action Guide, PRACTICAL-01, "
        "D. Evidence to preserve]"
    )

    assert not _inline_citations_are_supported(answer, [PRACTICAL_SOURCE])
    assert _has_unsupported_legal_claim(answer, [PRACTICAL_SOURCE])


def test_practical_recommendation_can_be_supported_by_practical_guidance() -> None:
    answer = (
        "Consider keeping recent payslips and bank statements together "
        "[Employment Wage Rights – Practical Action Guide, PRACTICAL-01, "
        "D. Evidence to preserve]"
    )

    assert _inline_citations_are_supported(answer, [PRACTICAL_SOURCE])


def test_source_gap_about_authority_does_not_reject_supported_practical_advice() -> None:
    answer = (
        "The supplied materials do not name a labour authority or procedure for "
        "your workplace. Consider keeping payslips and bank statements together "
        "[Employment Wage Rights – Practical Action Guide, PRACTICAL-01, "
        "D. Evidence to preserve]"
    )
    result = _run(
        _request("My salary is unpaid."),
        sources=[PRACTICAL_SOURCE],
        client=_groq_client(answer),
    )

    assert "do not name a labour authority" in result.answer
    assert "keeping payslips and bank statements together" in result.answer
    assert result.citations[0].verification_status == "pending_human_review"


def test_unsupported_procedural_advice_remains_rejected() -> None:
    answer = (
        "Contact the labour authority to file a wage claim within 30 days. "
        "[Employment Wage Rights – Practical Action Guide, PRACTICAL-01, "
        "D. Evidence to preserve]"
    )
    result = _run(
        _request("My salary is unpaid."),
        sources=[PRACTICAL_SOURCE],
        client=_groq_client(answer),
    )

    assert "does not sufficiently support a reliable answer" in result.answer
    assert "Contact the labour authority" not in result.answer
    assert result.citations == []


def test_separate_practical_and_legal_claims_need_their_own_citations() -> None:
    answer = (
        "Keep payslips and bank statements together "
        "[Employment Wage Rights – Practical Action Guide, PRACTICAL-01, "
        "D. Evidence to preserve]. "
        "Monthly wages must be paid before the seventh day "
        "[Code on Wages, 2019, Section 17, PDF page 9]."
    )

    assert _inline_citations_are_supported(
        answer, [PRACTICAL_SOURCE, TEST_SOURCE]
    )


def test_declarative_wage_deadline_cited_only_to_practical_guidance_is_rejected() -> None:
    answer = (
        "The employer pays monthly wages by the seventh day of the succeeding month "
        "[Employment Wage Rights – Practical Action Guide, PRACTICAL-01, "
        "D. Evidence to preserve]"
    )

    assert not _inline_citations_are_supported(answer, [PRACTICAL_SOURCE])


def test_plain_declarative_overtime_obligation_cited_only_to_practical_guidance_is_rejected() -> None:
    answer = (
        "Overtime is paid at twice the normal rate "
        "[Employment Wage Rights – Practical Action Guide, PRACTICAL-01, "
        "D. Evidence to preserve]"
    )

    assert not _inline_citations_are_supported(answer, [PRACTICAL_SOURCE])


@pytest.mark.parametrize(
    "answer",
    [
        "Deductions from wages are permitted only when authorized by law",
        "Wages after resignation are paid within two working days",
    ],
)
def test_declarative_deduction_and_resignation_rules_are_legal_assertions(
    answer: str,
) -> None:
    cited_answer = (
        f"{answer} [Employment Wage Rights – Practical Action Guide, "
        "PRACTICAL-01, D. Evidence to preserve]"
    )

    assert not _inline_citations_are_supported(
        cited_answer, [PRACTICAL_SOURCE]
    )


def test_practical_evidence_preservation_recommendation_remains_supported() -> None:
    answer = (
        "Keep payslips and bank statements together "
        "[Employment Wage Rights – Practical Action Guide, PRACTICAL-01, "
        "D. Evidence to preserve]"
    )

    assert _inline_citations_are_supported(answer, [PRACTICAL_SOURCE])


def test_incidental_salary_mention_is_not_classified_as_a_legal_rule() -> None:
    assert not _legal_assertion_spans(
        "My employer deposited my salary yesterday."
    )


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
        "What can I do now?",
        history=[
            {"role": "assistant", "content": "Hi — tell me what happened."},
            {
                "role": "user",
                "content": "My employer has not paid my salary for two months.",
            },
            {"role": "assistant", "content": "The salary is still unpaid."},
            {"role": "user", "content": "My workplace is in Maharashtra."},
        ],
    )
    result = asyncio.run(
        generate_chat_response(
            request, _settings(), groq_client=client, retriever=retrieve
        )
    )

    assert len(captured_queries) == 3
    assert "two months" in captured_queries[0]
    assert "Maharashtra" in captured_queries[0]
    assert "What can I do now?" in captured_queries[0]
    assert "Hi — tell me what happened." not in captured_queries[0]
    assert result.clarification_needed
    assert result.follow_up_question == "Which State is the establishment in, and what type of work do you do?"
    assert "Follow-up question:" not in result.answer
    assert result.follow_up_question not in result.answer
    assert "*" not in result.follow_up_question
    assert client.chat.completions.create.await_count == 1
    generation_messages = client.chat.completions.create.await_args.kwargs["messages"]
    assert generation_messages[-1] == {
        "role": "user",
        "content": "What can I do now?",
    }
    assert generation_messages[1]["content"] == "Hi — tell me what happened."
    assert generation_messages[2]["content"] == (
        "My employer has not paid my salary for two months."
    )
    assert generation_messages[3]["content"] == "The salary is still unpaid."
    assert generation_messages[4]["content"] == "My workplace is in Maharashtra."


@pytest.mark.parametrize(
    "follow_up",
    [
        "What can I do now?",
        "What should I do?",
        "What are my next steps?",
        "How can I recover it?",
        "How do I proceed?",
        "What if my employer refuses?",
        "What happens if they don't respond?",
        "Can I take action?",
    ],
)
def test_context_dependent_follow_ups_retain_the_latest_wage_topic(
    follow_up: str,
) -> None:
    request = _request(
        follow_up,
        history=[
            {"role": "assistant", "content": "Static welcome message."},
            {
                "role": "user",
                "content": "My salary has been unpaid for two months.",
            },
        ],
    )

    query = build_retrieval_query(request, max_history_messages=10)

    assert "unpaid for two months" in query
    assert follow_up in query
    assert "Static welcome message" not in query


def test_salary_clarification_preserves_topic_and_generation_history() -> None:
    clarification = (
        "Which month's salary is unpaid, and do you work for a private company, "
        "a government department, or another type of employer?"
    )
    current_question = "October's salary and I work for private company"
    request = ChatRequest(
        messages=[
            {"role": "assistant", "content": "Static welcome message."},
            {"role": "user", "content": "My last month salary is pending"},
            {
                "role": "assistant",
                "content": (
                    "I need one detail.\n\nFollow-up question: " + clarification
                ),
            },
            {"role": "user", "content": current_question},
        ]
    )

    retrieval_query = build_retrieval_query(request, max_history_messages=10)

    assert "My last month salary is pending" in retrieval_query
    assert current_question in retrieval_query
    assert "Static welcome message" not in retrieval_query
    assert _selected_wage_recovery_sections(retrieval_query) == {"17"}

    client = _groq_client()
    result = _run(request, sources=[TEST_SOURCE, PRACTICAL_SOURCE], client=client)
    completion_messages = client.chat.completions.create.await_args.kwargs["messages"]

    assert result.citations
    assert completion_messages[-1] == {
        "role": "user",
        "content": current_question,
    }
    assert [message["role"] for message in completion_messages[1:]] == [
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert completion_messages[3]["content"].endswith(clarification)


@pytest.mark.parametrize(
    ("follow_up", "expected_modules", "expected_sections"),
    [
        ("what i can do", {"PRACTICAL-01"}, {"17"}),
        ("What should I do now?", {"PRACTICAL-01"}, {"17"}),
        ("How can I recover my salary?", {"PRACTICAL-01"}, {"17"}),
        ("How should I approach HR?", {"PRACTICAL-01"}, {"17"}),
        (
            "What should I do if they don't respond?",
            {"PRACTICAL-01", "PRACTICAL-05"},
            {"17"},
        ),
        (
            "Where do I file a wage claim?",
            {"PRACTICAL-01"},
            {"17", "45"},
        ),
    ],
)
def test_fifth_turn_salary_follow_up_retrieval_intent(
    follow_up: str,
    expected_modules: set[str],
    expected_sections: set[str],
) -> None:
    request = _salary_follow_up_request(follow_up)
    query = build_retrieval_query(request, max_history_messages=10)

    assert "My last month salary is pending" in query
    assert "October's salary and I work for private company" in query
    assert follow_up in query
    assert _selected_practical_guide_modules(query) == expected_modules
    assert _selected_wage_recovery_sections(query) == expected_sections
    if "45" not in expected_sections:
        assert "45" not in _selected_wage_recovery_sections(query)


def test_practical_next_step_follow_up_returns_cited_guide_sources() -> None:
    practical_contact_source = {
        **PRACTICAL_SOURCE,
        "chunk_id": "guide-PRACTICAL-01-written-request",
        "section_heading": "C. What should the employee do? (chronological)",
        "heading_path": [
            "PART B – THE SIX PRACTICAL MODULES",
            "PRACTICAL-01: Delayed Monthly Salary",
            "C. What should the employee do? (chronological)",
        ],
        "text": (
            "Raise the missing salary in writing with HR or payroll, stating "
            "what is missing, when it was due, and asking for a specific date."
        ),
    }
    calls: list[str] = []

    def retrieve(query: str, **kwargs: object) -> list[dict]:
        calls.append(query)
        if query.startswith("PRACTICAL-01"):
            return [PRACTICAL_SOURCE, practical_contact_source]
        return [TEST_SOURCE]

    answer = (
        "Keep payslips and bank statements together as practical evidence "
        "[Employment Wage Rights – Practical Action Guide, PRACTICAL-01, "
        "D. Evidence to preserve]. Send HR or payroll a concise written request "
        "identifying the unpaid month and asking for a specific payment date "
        "[Employment Wage Rights – Practical Action Guide, PRACTICAL-01, "
        "C. What should the employee do? (chronological)]."
    )
    client = _groq_client(answer=answer, citation="")
    request = _salary_follow_up_request("what i can do")

    response = asyncio.run(
        generate_chat_response(
            request,
            _settings(),
            groq_client=client,
            retriever=retrieve,
        )
    )

    assert any(query.startswith("PRACTICAL-01") for query in calls)
    assert response.answer == answer
    practical_citations = [
        citation for citation in response.citations
        if citation.source_type == "practical_guidance"
    ]
    assert {citation.module_id for citation in practical_citations} == {"PRACTICAL-01"}
    assert all(
        citation.authority_level == "non_authoritative"
        and citation.verification_status == "pending_human_review"
        for citation in practical_citations
    )
    assert client.chat.completions.create.await_args.kwargs["messages"][-1] == {
        "role": "user",
        "content": "what i can do",
    }


def test_follow_up_does_not_reuse_an_older_unrelated_topic() -> None:
    for unrelated_message in (
        "What are my rights as a tenant?",
        "I am looking for a new apartment.",
    ):
        request = _request(
            "What can I do now?",
            history=[
                {"role": "user", "content": "My salary is unpaid for two months."},
                {"role": "assistant", "content": "What happened?"},
                {"role": "user", "content": unrelated_message},
            ],
        )

        assert build_retrieval_query(request, max_history_messages=10) == (
            "What can I do now?"
        )


@pytest.mark.parametrize(
    ("query", "expected_modules"),
    [
        (
            "My salary is unpaid for two months and I am still working.",
            {"PRACTICAL-01"},
        ),
        ("My salary is unpaid after I resigned.", {"PRACTICAL-02"}),
        (
            "My wages remain unpaid after my employment was terminated.",
            {"PRACTICAL-06"},
        ),
        (
            "My employer is not responding to my written request about unpaid salary.",
            {"PRACTICAL-01", "PRACTICAL-05"},
        ),
        (
            "Previous wage issue: salary was unpaid. Current follow-up: "
            "What happens if they don't respond?",
            {"PRACTICAL-01", "PRACTICAL-05"},
        ),
        (
            "My salary is unpaid; I have not resigned.",
            {"PRACTICAL-01"},
        ),
        ("My salary is unpaid.", {"PRACTICAL-01"}),
    ],
)
def test_unpaid_wage_guide_retrieval_is_scenario_aware(
    query: str, expected_modules: set[str]
) -> None:
    guide_sources = [
        {
            **PRACTICAL_SOURCE,
            "chunk_id": f"guide-{module_id.lower()}",
            "module_id": module_id,
            "module_heading": module_id,
            "score": 0.9,
        }
        for module_id in (
            "PRACTICAL-01",
            "PRACTICAL-02",
            "PRACTICAL-05",
            "PRACTICAL-06",
        )
    ]
    guide_sources.extend(
        [
            {
                **PRACTICAL_SOURCE,
                "chunk_id": "guide-p01-legal-explanation",
                "section_heading": "B2. Legal explanation",
            },
            {
                **PRACTICAL_SOURCE,
                "chunk_id": "guide-p05-formal-routes",
                "module_id": "PRACTICAL-05",
                "section_heading": "C. What should the employee do? (chronological)",
            },
        ]
    )

    merged = _merge_unpaid_wage_sources(
        [guide_sources], query=query, top_k=10
    )

    assert {source["module_id"] for source in merged} == expected_modules
    assert all(
        source["authority_level"] == "non_authoritative"
        and source["verification_status"] == "pending_human_review"
        for source in merged
    )


@pytest.mark.parametrize(
    ("query", "expected_sections"),
    [
        ("When should monthly salary be paid?", {"17"}),
        ("Who is responsible for paying wages?", {"43"}),
        ("How do I file a wage claim with the appointed authority?", {"45"}),
        (
            "How do I escalate my unpaid wage issue to the Labour Department?",
            {"17", "45"},
        ),
        (
            "My salary is unpaid. Who is responsible and how can I file a wage claim?",
            {"17", "43", "45"},
        ),
        (
            "Previous wage issue: salary unpaid. What happens if they don't respond?",
            {"17"},
        ),
        ("My salary is unpaid. What should I do?", {"17"}),
        ("my boss did not gave me 2 month salary what to do", {"17"}),
        ("My salary is unpaid. What are my next steps?", {"17"}),
        ("How can I recover my unpaid salary?", {"17"}),
        (
            "My salary is unpaid. How do I file a wage claim with the appointed authority?",
            {"17", "45"},
        ),
    ],
)
def test_wage_retrieval_sections_follow_question_intent(
    query: str, expected_sections: set[str]
) -> None:
    assert _selected_wage_recovery_sections(query) == expected_sections


def test_contextual_follow_up_selects_section_45_only_for_explicit_claim_intent() -> None:
    request = _request(
        "Where do I file a wage claim?",
        history=[
            {"role": "user", "content": "My salary is unpaid for two months."},
            {"role": "assistant", "content": "What would you like to know?"},
        ],
    )
    query = build_retrieval_query(request, max_history_messages=10)

    assert "My salary is unpaid for two months." in query
    assert "Where do I file a wage claim?" in query
    assert _selected_wage_recovery_sections(query) == {"17", "45"}


@pytest.mark.parametrize(
    ("query", "source", "answer"),
    [
        (
            "When should monthly salary be paid?",
            TEST_SOURCE,
            "Monthly wages must be paid before the seventh day "
            "[Code on Wages, 2019, Section 17, PDF page 9]",
        ),
        (
            "Who is responsible for paying wages?",
            SECTION_43_SOURCE,
            "Every employer shall pay all amounts required under this Code "
            "[Code on Wages, 2019, Section 43, PDF page 19]",
        ),
        (
            "How do I file a wage claim with the appointed authority?",
            SECTION_45_SOURCE,
            "Section 45 lets employees submit a wage claim to an appointed "
            "authority [Code on Wages, 2019, Section 45, PDF page 20]",
        ),
    ],
)
def test_generation_retrieves_only_statutory_sections_matching_intent(
    query: str, source: dict, answer: str
) -> None:
    captured_queries: list[str] = []

    def retrieve(retrieval_query: str, **kwargs: object) -> list[dict]:
        captured_queries.append(retrieval_query)
        if "Section" in retrieval_query:
            return [source]
        return []

    client = _groq_client(answer)
    result = asyncio.run(
        generate_chat_response(
            _request(query), _settings(), groq_client=client, retriever=retrieve
        )
    )

    assert [citation.section_number for citation in result.citations] == [
        source["section_number"]
    ]
    assert result.citations[0].verification_status == "needs_review"
    assert len(captured_queries) == 2
    assert captured_queries[0] == query
    assert f"Section {source['section_number']}" in captured_queries[1]


def test_unrelated_section_17_text_is_not_accepted_for_wage_retrieval() -> None:
    unrelated = {
        **TEST_SOURCE,
        "text": "17. This section defines terms used in the Code.",
    }

    assert _merge_unpaid_wage_sources(
        [[unrelated]], query="When should monthly salary be paid?", top_k=5
    ) == []


def test_unpaid_wage_retrieval_merges_relevant_official_and_practical_sources() -> None:
    section_43 = {**SECTION_43_SOURCE, "score": 0.77}
    section_45 = {**SECTION_45_SOURCE, "score": 0.83}
    practical_2 = {
        **PRACTICAL_SOURCE,
        "chunk_id": "employment_wage_practical_guide_claude-PRACTICAL-01-section-011-chunk-01",
        "section_heading": "E. Written request",
        "score": 0.81,
    }
    irrelevant_section = {
        **TEST_SOURCE,
        "chunk_id": "code_on_wages_2019-page-4-chunk-002",
        "section_number": "2",
        "text": "Definitions used in this Code.",
        "score": 0.91,
    }
    captured_queries: list[str] = []

    def retrieve(query: str, **kwargs: object) -> list[dict]:
        captured_queries.append(query)
        if "Section 17" in query:
            return [TEST_SOURCE]
        if "Section 43" in query:
            return [section_43]
        if "Section 45" in query:
            return [section_45]
        return [
            irrelevant_section,
            PRACTICAL_SOURCE,
            practical_2,
            {**PRACTICAL_SOURCE, "score": 0.79},
        ]

    client = _groq_client()
    request = _request(
        "My salary is unpaid for the last two months. Who is responsible for payment, and what wage claim can I file to recover it?"
    )
    result = asyncio.run(
        generate_chat_response(
            request, _settings(), groq_client=client, retriever=retrieve
        )
    )

    assert len(captured_queries) == 5
    assert captured_queries[0] == request.messages[-1].content
    assert any(
        query.startswith("PRACTICAL-01 delayed monthly salary")
        for query in captured_queries
    )
    citations = {citation.chunk_id: citation for citation in result.citations}
    assert set(citations) == {
        TEST_SOURCE["chunk_id"],
        section_43["chunk_id"],
        section_45["chunk_id"],
        PRACTICAL_SOURCE["chunk_id"],
        practical_2["chunk_id"],
    }
    assert citations[TEST_SOURCE["chunk_id"]].verification_status == "needs_review"
    assert citations[PRACTICAL_SOURCE["chunk_id"]].verification_status == (
        "pending_human_review"
    )
    system_context = client.chat.completions.create.await_args.kwargs["messages"][0][
        "content"
    ]
    assert "section=43" in system_context
    assert "section=45" in system_context
    assert "pending_human_review" in system_context
    assert "Definitions used in this Code" not in system_context
    assert client.chat.completions.create.await_args.kwargs["messages"][-1][
        "content"
    ] == request.messages[-1].content


@pytest.mark.parametrize(
    "message",
    [
        "my salary is unpaid for last 2 months",
        "My salary hasn't been paid for two months.",
        "my boss did not gave me 2 month salary what to do",
    ],
)
def test_short_and_informal_unpaid_salary_queries_retrieve_legal_sources(
    message: str,
) -> None:
    section_43 = SECTION_43_SOURCE
    section_45 = SECTION_45_SOURCE
    captured_queries: list[str] = []

    def retrieve(query: str, **kwargs: object) -> list[dict]:
        captured_queries.append(query)
        if "Section 17" in query:
            return [TEST_SOURCE]
        if "Section 43" in query:
            return [section_43]
        if "Section 45" in query:
            return [section_45]
        return [
            {
                **PRACTICAL_SOURCE,
                "chunk_id": "employment_wage_practical_guide_claude-PRACTICAL-01-section-011-chunk-01",
                "score": 0.8,
            }
        ]

    client = _groq_client()
    request = _request(message)
    result = asyncio.run(
        generate_chat_response(
            request, _settings(), groq_client=client, retriever=retrieve
        )
    )

    assert len(captured_queries) == 3
    assert captured_queries[0] == message
    assert {
        citation.section_number for citation in result.citations
        if citation.source_type == "primary_legislation"
    } == {"17"}
    assert any(
        citation.source_type == "practical_guidance"
        and citation.verification_status == "pending_human_review"
        for citation in result.citations
    )
    assert client.chat.completions.create.await_args.kwargs["messages"][-1][
        "content"
    ] == message


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


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"messages": []},
        {"messages": [{"role": "assistant", "content": "No user message"}]},
    ],
)
def test_chat_returns_validation_error_for_missing_or_incomplete_history(
    payload: dict,
) -> None:
    response = TestClient(app).post("/api/v1/chat", json=payload)

    assert response.status_code == 422
    assert isinstance(response.json().get("detail"), list)


def test_chat_endpoint_returns_controlled_error_for_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_generate(request: ChatRequest, settings: Settings) -> ChatResponse:
        raise RuntimeError("internal detail that must not reach the user")

    monkeypatch.setattr(chat_routes, "generate_chat_response", fail_generate)
    response = TestClient(app).post(
        "/api/v1/chat",
        json={"messages": [{"role": "user", "content": "My salary is pending"}]},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "The assistant could not complete the request. Please try again."
    }
