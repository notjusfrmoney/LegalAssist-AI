from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class ChatProfile(BaseModel):
    state: str | None = Field(default=None, max_length=100)
    industry: str | None = Field(default=None, max_length=200)
    employer_type: str | None = Field(default=None, max_length=200)
    employment_type: str | None = Field(default=None, max_length=200)
    wage_period: str | None = Field(default=None, max_length=100)


class ChatCitation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str | None = None
    source_id: str | None = None
    source_filename: str | None = None
    document_title: str | None = None
    source_type: str | None = None
    authority_level: str | None = None
    section: str | None = None
    section_number: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    module_id: str | None = None
    module_heading: str | None = None
    section_heading: str | None = None
    heading_path: list[str] | None = None
    location: str | None = None
    claim_ids: list[str] = Field(default_factory=list)
    unresolved_refs: list[str] = Field(default_factory=list)
    source_url: str | None = None
    verification_status: str | None = None
    excerpt: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[ChatCitation] = Field(default_factory=list)
    sources: list[ChatCitation] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_needed: bool = False
    follow_up_question: str | None = None
    scope_note: str | None = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=50)
    profile: ChatProfile | None = None

    @model_validator(mode="after")
    def require_user_message(self) -> ChatRequest:
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("At least one user message is required.")
        if self.messages[-1].role != "user":
            raise ValueError("The final conversation message must be from the user.")
        return self
