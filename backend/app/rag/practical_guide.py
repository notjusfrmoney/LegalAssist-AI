"""Ingestion helpers for the supplementary Claude practical-guidance document."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

GUIDE_SOURCE_ID = "employment_wage_practical_guide_claude"
GUIDE_FILENAME = "employment_wage_practical_guide_claude.md"
GUIDE_TITLE = "Employment Wage Rights – Practical Action Guide for Employees in India"
GUIDE_SOURCE_TYPE = "practical_guidance"
GUIDE_AUTHORITY_LEVEL = "non_authoritative"
GUIDE_VERIFICATION_STATUS = "pending_human_review"
GUIDE_COMPANION_FILENAMES = (
    "claude_legal_source_register.md",
    "claude_legal_risk_review.md",
    "claude_unresolved_issues.md",
    "claude_research_summary.md",
)
DEFAULT_GUIDE_PATH = (
    Path(__file__).resolve().parents[3]
    / "knowledge_base"
    / "raw"
    / GUIDE_FILENAME
)
GUIDE_CHUNK_TARGET = 1600

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MODULE_RE = re.compile(r"\b((?:REF|PRACTICAL)-\d{2})\b", re.IGNORECASE)
_CLAIM_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d{3,}\b")
_UNRESOLVED_RE = re.compile(r"\bUNRES(?:-[A-Za-z0-9]+)?\b", re.IGNORECASE)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


def _heading_sections(markdown: str) -> list[dict[str, Any]]:
    lines = markdown.splitlines()
    headings: list[tuple[int, int, str]] = []
    for line_index, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match:
            headings.append((line_index, len(match.group(1)), match.group(2).strip()))

    sections: list[dict[str, Any]] = []
    parents: list[tuple[int, str]] = []
    for heading_index, (line_index, level, title) in enumerate(headings):
        while parents and parents[-1][0] >= level:
            parents.pop()
        parents.append((level, title))
        end_index = (
            headings[heading_index + 1][0]
            if heading_index + 1 < len(headings)
            else len(lines)
        )

        heading_path = [name for _, name in parents]
        module_match = next(
            (
                match
                for name in reversed(heading_path)
                if (match := _MODULE_RE.search(name))
            ),
            None,
        )
        section_lines = lines[line_index + 1 : end_index]
        nonblank_line_indices = [
            index for index, line in enumerate(section_lines, start=line_index + 1)
            if line.strip()
        ]
        body = "\n".join(section_lines).strip()
        sections.append(
            {
                "level": level,
                "heading": title,
                "heading_path": heading_path,
                "module_id": module_match.group(1).upper() if module_match else "GENERAL",
                "line_start": nonblank_line_indices[0] + 1 if nonblank_line_indices else line_index + 2,
                "line_end": nonblank_line_indices[-1] + 1 if nonblank_line_indices else line_index + 1,
                "body": body,
            }
        )
    return sections


def _split_table_rows(table: str, *, max_chars: int) -> list[str]:
    rows = [line for line in table.splitlines() if line.strip()]
    if len(rows) <= 2:
        return [table]
    header, separator, *body_rows = rows
    result: list[str] = []
    current: list[str] = []
    for row in body_rows:
        candidate = "\n".join([header, separator, *current, row])
        if current and len(candidate) > max_chars:
            result.append("\n".join([header, separator, *current]))
            current = [row]
        else:
            current.append(row)
    if current:
        result.append("\n".join([header, separator, *current]))
    return result


def _split_long_paragraph(paragraph: str, *, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+(?=(?:[A-Z0-9*#(\[]|[-*+]\s))", paragraph)
    if len(sentences) == 1:
        return [paragraph]
    result: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            result.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        result.append(current)
    return result


def _content_atoms(body: str, *, max_chars: int) -> list[str]:
    lines = body.splitlines()
    atoms: list[str] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue

        if lines[index].lstrip().startswith("```"):
            block = [lines[index]]
            index += 1
            while index < len(lines):
                block.append(lines[index])
                is_end = lines[index].lstrip().startswith("```")
                index += 1
                if is_end:
                    break
            atoms.append("\n".join(block))
            continue

        if lines[index].lstrip().startswith("|"):
            table = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table.append(lines[index])
                index += 1
            atoms.extend(_split_table_rows("\n".join(table), max_chars=max_chars))
            continue

        if _LIST_ITEM_RE.match(lines[index]):
            item = [lines[index]]
            index += 1
            while index < len(lines) and lines[index].strip():
                if _LIST_ITEM_RE.match(lines[index]) or lines[index].lstrip().startswith("|"):
                    break
                item.append(lines[index])
                index += 1
            atoms.extend(_split_long_paragraph("\n".join(item), max_chars=max_chars))
            continue

        paragraph = [lines[index]]
        index += 1
        while index < len(lines) and lines[index].strip():
            if (
                lines[index].lstrip().startswith("|")
                or lines[index].lstrip().startswith("```")
                or _LIST_ITEM_RE.match(lines[index])
            ):
                break
            paragraph.append(lines[index])
            index += 1
        atoms.extend(_split_long_paragraph("\n".join(paragraph), max_chars=max_chars))
    return atoms


def _module_heading(heading_path: list[str], module_id: str | None) -> str | None:
    if not module_id:
        return None
    return next(
        (heading for heading in heading_path if module_id.casefold() in heading.casefold()),
        None,
    )


def chunk_practical_guide(
    guide_path: str | Path = DEFAULT_GUIDE_PATH,
    *,
    chunk_target: int = GUIDE_CHUNK_TARGET,
) -> list[dict[str, Any]]:
    path = Path(guide_path)
    if not path.is_file():
        raise FileNotFoundError(f"Practical guide not found: {path}")
    markdown = path.read_text(encoding="utf-8")
    title_match = re.search(r"(?m)^#\s+(.+?)\s*$", markdown)
    document_title = title_match.group(1).strip() if title_match else GUIDE_TITLE
    chunks: list[dict[str, Any]] = []
    for section_index, section in enumerate(_heading_sections(markdown), start=1):
        module_id = section["module_id"]
        if not section["body"]:
            continue
        module_title = _module_heading(section["heading_path"], module_id)
        heading_path = section["heading_path"]
        heading_location = " > ".join(heading_path)
        section_heading = section["heading"]
        context = (
            f"Document: {document_title}\n"
            f"Module: {module_id}"
            + (f" — {module_title}" if module_title and module_title != section_heading else "")
            + f"\nHeading: {heading_location}\n\n"
        )
        body = section["body"]
        atoms = _content_atoms(body, max_chars=max(400, chunk_target - len(context)))
        if not atoms:
            atoms = [""]

        groups: list[list[str]] = []
        current: list[str] = []
        current_size = len(context)
        for atom in atoms:
            atom_size = len(atom) + (2 if current else 0)
            if current and current_size + atom_size > chunk_target:
                groups.append(current)
                current = []
                current_size = len(context)
                atom_size = len(atom)
            current.append(atom)
            current_size += atom_size
        if current:
            groups.append(current)

        for chunk_index, group in enumerate(groups, start=1):
            text = context + "\n\n".join(atom for atom in group if atom).strip()
            module_slug = module_id
            chunk_id = (
                f"{GUIDE_SOURCE_ID}-{module_slug}-section-{section_index:03d}"
                f"-chunk-{chunk_index:02d}"
            )
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_id": GUIDE_SOURCE_ID,
                    "source_filename": path.name,
                    "document_title": document_title,
                    "source_type": GUIDE_SOURCE_TYPE,
                    "authority_level": GUIDE_AUTHORITY_LEVEL,
                    "verification_status": GUIDE_VERIFICATION_STATUS,
                    "is_primary_legal_source": False,
                    "jurisdiction": "India",
                    "government_sphere": "central_and_state_code",
                    "source_url": None,
                    "page_number": None,
                    "page_range": None,
                    "module_id": module_id,
                    "module_heading": module_title,
                    "section_heading": section_heading,
                    "heading_path": heading_path,
                    "location": heading_location,
                    "line_start": section["line_start"],
                    "line_end": section["line_end"],
                    "claim_ids": sorted(set(_CLAIM_ID_RE.findall(text))),
                    "unresolved_refs": sorted(
                        {value.upper() for value in _UNRESOLVED_RE.findall(text)}
                    ),
                    "text": text,
                    "char_count": len(text),
                    "token_count": max(1, len(text.split())),
                }
            )
    return chunks


def practical_guide_manifest_entry(
    guide_path: str | Path = DEFAULT_GUIDE_PATH,
) -> tuple[dict[str, Any], list[str]]:
    path = Path(guide_path)
    content = path.read_text(encoding="utf-8")
    title_match = re.search(r"(?m)^#\s+(.+?)\s*$", content)
    title = title_match.group(1).strip() if title_match else GUIDE_TITLE
    companion_names = list(GUIDE_COMPANION_FILENAMES)
    declared_names = re.search(
        r"(?ms)^companion_files:\s*\n((?:\s+-\s+[^\n]+\n?)+)",
        content,
    )
    if declared_names:
        parsed_names = re.findall(r"^\s+-\s+([A-Za-z0-9_.-]+)", declared_names.group(1), re.M)
        if parsed_names:
            companion_names = parsed_names
    missing = [
        name for name in companion_names if not (path.parent / name).is_file()
    ]
    entry = {
        "source_id": GUIDE_SOURCE_ID,
        "filename": path.name,
        "document_title": title,
        "publisher": "Claude-generated independent practical guide",
        "source_url": None,
        "source_type": GUIDE_SOURCE_TYPE,
        "authority_level": GUIDE_AUTHORITY_LEVEL,
        "jurisdiction": "India",
        "government_sphere": "central_and_state_code",
        "verification_status": GUIDE_VERIFICATION_STATUS,
        "review_status": "pending_human_review",
        "is_primary_legal_source": False,
        "file_status": "present",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "companion_files": [
            {
                "filename": name,
                "file_status": "not_present" if name in missing else "present",
            }
            for name in companion_names
        ],
        "limitations": [
            "The guide is supplementary practical guidance, not an authoritative legal source.",
            "Its cited claim register, risk review, unresolved-issues register, and research summary are unavailable; referenced claims cannot be independently checked from those companion files.",
            "UNRES references identify unresolved issues and must not be presented as settled law.",
        ],
    }
    return entry, missing


def merge_practical_guide_manifest(
    manifest: dict[str, Any],
    guide_path: str | Path = DEFAULT_GUIDE_PATH,
) -> dict[str, Any]:
    entry, missing_companions = practical_guide_manifest_entry(guide_path)
    documents = manifest.setdefault("documents", [])
    documents[:] = [
        document
        for document in documents
        if document.get("source_id") != GUIDE_SOURCE_ID
    ]
    documents.append(entry)
    manifest["missing_companion_files"] = missing_companions
    return manifest
