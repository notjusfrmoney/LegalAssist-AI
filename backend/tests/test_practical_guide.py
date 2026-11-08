from __future__ import annotations

import re

import numpy as np

from app.rag import retrieve as retrieve_module
from app.rag.build_index import _prepare_metadata_rows
from app.rag.practical_guide import (
    DEFAULT_GUIDE_PATH,
    GUIDE_AUTHORITY_LEVEL,
    GUIDE_SOURCE_ID,
    GUIDE_SOURCE_TYPE,
    GUIDE_VERIFICATION_STATUS,
    chunk_practical_guide,
    practical_guide_manifest_entry,
)


def test_parent_and_child_sections_are_chunked_without_duplicate_content(tmp_path) -> None:
    guide = tmp_path / "guide.md"
    guide.write_text(
        "# Guide\n"
        "## REF-01: Applicability\n"
        "Parent-only applicability guidance.\n"
        "### Questions\n"
        "Child-only question guidance.\n"
        "#### State\n"
        "Nested state guidance.\n",
        encoding="utf-8",
    )

    chunks = chunk_practical_guide(guide)
    combined_text = "\n".join(chunk["text"] for chunk in chunks)

    for phrase in (
        "Parent-only applicability guidance.",
        "Child-only question guidance.",
        "Nested state guidance.",
    ):
        assert combined_text.count(phrase) == 1
    parent_chunk = next(
        chunk for chunk in chunks if "Parent-only applicability guidance." in chunk["text"]
    )
    assert parent_chunk["module_id"] == "REF-01"
    assert parent_chunk["line_start"] == 3
    assert parent_chunk["line_end"] == 3


def test_every_guide_chunk_has_a_module_and_all_eleven_modules_are_represented() -> None:
    chunks = chunk_practical_guide()
    expected_modules = {
        *(f"REF-{number:02}" for number in range(1, 6)),
        *(f"PRACTICAL-{number:02}" for number in range(1, 7)),
    }

    assert all(
        chunk["module_id"] == "GENERAL"
        or re.fullmatch(r"(?:REF|PRACTICAL)-\d{2}", chunk["module_id"])
        for chunk in chunks
    )
    assert expected_modules <= {chunk["module_id"] for chunk in chunks}


def test_guide_chunks_preserve_all_claim_and_unresolved_reference_ids() -> None:
    markdown = DEFAULT_GUIDE_PATH.read_text(encoding="utf-8")
    chunks = chunk_practical_guide()
    source_claim_ids = set(re.findall(r"\b[A-Z][A-Z0-9]+-\d{3,}\b", markdown))
    source_unresolved_refs = {
        value.upper()
        for value in re.findall(r"\bUNRES(?:-[A-Za-z0-9]+)?\b", markdown, re.IGNORECASE)
    }
    chunk_claim_ids = {
        claim_id for chunk in chunks for claim_id in chunk["claim_ids"]
    }
    chunk_unresolved_refs = {
        ref for chunk in chunks for ref in chunk["unresolved_refs"]
    }

    assert source_claim_ids <= chunk_claim_ids
    assert source_unresolved_refs <= chunk_unresolved_refs


def test_guide_chunks_preserve_modules_headings_claims_and_unresolved_references() -> None:
    chunks = chunk_practical_guide()
    expected_modules = {
        *(f"REF-{number:02}" for number in range(1, 6)),
        *(f"PRACTICAL-{number:02}" for number in range(1, 7)),
    }

    assert chunks
    assert expected_modules <= {chunk["module_id"] for chunk in chunks}
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all(chunk["source_id"] == GUIDE_SOURCE_ID for chunk in chunks)
    assert all(chunk["source_filename"] == DEFAULT_GUIDE_PATH.name for chunk in chunks)
    assert all(chunk["source_type"] == GUIDE_SOURCE_TYPE for chunk in chunks)
    assert all(chunk["authority_level"] == GUIDE_AUTHORITY_LEVEL for chunk in chunks)
    assert all(chunk["verification_status"] == GUIDE_VERIFICATION_STATUS for chunk in chunks)
    assert all(chunk["is_primary_legal_source"] is False for chunk in chunks)
    assert all(chunk["page_number"] is None for chunk in chunks)
    assert all(chunk["heading_path"] and chunk["location"] for chunk in chunks)
    assert any("WAGE-011" in chunk["claim_ids"] for chunk in chunks)
    assert any("UNRES-10" in chunk["unresolved_refs"] for chunk in chunks)
    assert max(chunk["char_count"] for chunk in chunks) <= 1600


def test_markdown_tables_keep_headers_and_all_rows_across_chunks() -> None:
    markdown = DEFAULT_GUIDE_PATH.read_text(encoding="utf-8")
    chunks = chunk_practical_guide()
    chunk_texts = [chunk["text"] for chunk in chunks]

    tables: list[list[str]] = []
    current: list[str] = []
    for line in markdown.splitlines():
        if line.lstrip().startswith("|"):
            current.append(line)
        elif current:
            if len(current) > 2:
                tables.append(current)
            current = []
    if len(current) > 2:
        tables.append(current)

    assert tables
    for table in tables:
        header, separator, *rows = table
        matching_chunks = [
            text
            for text in chunk_texts
            if header in text and separator in text
        ]
        assert matching_chunks, f"Table header not retained: {header}"
        combined = "\n".join(matching_chunks)
        assert all(row in combined for row in rows)


def test_manifest_records_missing_companion_files_without_fabricating_them() -> None:
    entry, missing = practical_guide_manifest_entry()

    assert entry["source_type"] == "practical_guidance"
    assert entry["authority_level"] == "non_authoritative"
    assert entry["verification_status"] == "pending_human_review"
    assert entry["is_primary_legal_source"] is False
    assert missing == [
        "claude_legal_source_register.md",
        "claude_legal_risk_review.md",
        "claude_unresolved_issues.md",
        "claude_research_summary.md",
    ]
    assert all(item["file_status"] == "not_present" for item in entry["companion_files"])


def test_manifest_merge_preserves_existing_official_source_metadata() -> None:
    official = {
        "source_id": "code_on_wages_2019",
        "source_type": "primary_legislation",
        "verification_status": "needs_review",
        "source_url": "https://example.invalid/act.pdf",
    }
    external_reference = {
        "source_id": "external_notification",
        "file_status": "not_locally_ingested",
        "source_url": "https://example.invalid/notification.pdf",
    }
    manifest = {
        "documents": [official],
        "external_references": [external_reference],
        "missing_files": ["original-notification.pdf"],
    }

    from app.rag.practical_guide import merge_practical_guide_manifest

    result = merge_practical_guide_manifest(manifest)

    assert result["documents"][0] is official
    assert result["external_references"] == [external_reference]
    assert result["missing_files"] == ["original-notification.pdf"]
    assert next(
        document
        for document in result["documents"]
        if document["source_id"] == GUIDE_SOURCE_ID
    )["authority_level"] == "non_authoritative"


def test_vector_metadata_classifies_guide_without_relabelling_official_sources() -> None:
    rows = _prepare_metadata_rows(
        [
            {
                "chunk_id": "act-chunk",
                "source_id": "code_on_wages_2019",
                "source_type": "primary_legislation",
                "verification_status": "needs_review",
                "text": "statutory excerpt",
            },
            {
                "chunk_id": "guide-chunk",
                "source_id": GUIDE_SOURCE_ID,
                "source_type": "practical_guidance",
                "authority_level": "non_authoritative",
                "verification_status": "pending_human_review",
                "module_id": "REF-01",
                "section_heading": "Applicability triage",
                "text": "practical guidance",
            },
        ]
    )

    assert rows[0]["source_type"] == "primary_legislation"
    assert rows[0]["authority_level"] == "statutory"
    assert rows[0]["verification_status"] == "needs_review"
    assert rows[1]["source_type"] == "practical_guidance"
    assert rows[1]["authority_level"] == "non_authoritative"
    assert rows[1]["verification_status"] == "pending_human_review"
    assert rows[1]["module_id"] == "REF-01"


def test_retrieval_returns_practical_source_metadata(monkeypatch) -> None:
    practical = {
        "chunk_id": "guide-PRACTICAL-01-001",
        "source_id": GUIDE_SOURCE_ID,
        "source_filename": DEFAULT_GUIDE_PATH.name,
        "document_title": "Employment Wage Rights Practical Guide",
        "source_type": "practical_guidance",
        "authority_level": "non_authoritative",
        "verification_status": "pending_human_review",
        "module_id": "PRACTICAL-01",
        "section_heading": "D. Evidence to preserve",
        "heading_path": ["PRACTICAL-01: Delayed salary", "D. Evidence to preserve"],
        "location": "PRACTICAL-01: Delayed salary > D. Evidence to preserve",
        "claim_ids": ["WAGE-016"],
        "unresolved_refs": ["UNRES-10"],
        "text": "Keep payslips and bank statements together.",
    }

    class FakeIndex:
        ntotal = 1

        def search(self, vector, count):
            return np.array([[0.88]], dtype=np.float32), np.array([[0]], dtype=np.int64)

    monkeypatch.setattr(
        retrieve_module,
        "load_existing_index",
        lambda _path: {"index": FakeIndex(), "metadata": [practical], "info": {}},
    )
    monkeypatch.setattr(retrieve_module, "load_embedding_model", lambda _name: object())
    monkeypatch.setattr(
        retrieve_module,
        "embed_query",
        lambda _query, model: np.ones((1, 384), dtype=np.float32),
    )

    results = retrieve_module.retrieve_top_k("How can I organize wage evidence?")

    assert len(results) == 1
    for field in (
        "source_type",
        "authority_level",
        "verification_status",
        "module_id",
        "section_heading",
        "location",
        "claim_ids",
        "unresolved_refs",
    ):
        assert results[0][field] == practical[field]


def test_retrieval_ranks_official_sources_above_practical_guidance(monkeypatch) -> None:
    rows = [
        {"chunk_id": "guide", "source_type": "practical_guidance", "authority_level": "non_authoritative"},
        {"chunk_id": "faq", "source_type": "official_faq", "authority_level": "official_guidance"},
        {"chunk_id": "act", "source_type": "primary_legislation", "authority_level": "statutory"},
        {"chunk_id": "rules", "source_type": "delegated_rules", "authority_level": "delegated_rules"},
    ]

    class FakeIndex:
        ntotal = len(rows)

        def search(self, vector, count):
            scores = np.array([[0.98, 0.96, 0.92, 0.90]], dtype=np.float32)[:, :count]
            indices = np.array([[0, 1, 2, 3]], dtype=np.int64)[:, :count]
            return scores, indices

    monkeypatch.setattr(
        retrieve_module,
        "load_existing_index",
        lambda _path: {"index": FakeIndex(), "metadata": rows, "info": {}},
    )
    monkeypatch.setattr(retrieve_module, "load_embedding_model", lambda _name: object())
    monkeypatch.setattr(
        retrieve_module,
        "embed_query",
        lambda _query, model: np.ones((1, 384), dtype=np.float32),
    )

    results = retrieve_module.retrieve_top_k("wage question", top_k=4)

    assert [row["chunk_id"] for row in results] == ["act", "rules", "faq", "guide"]
