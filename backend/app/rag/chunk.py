"""Legal-aware chunking utilities for extracted source text."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from app.rag.extract import DEFAULT_EXTRACTED_DIR, SOURCE_METADATA

DEFAULT_CHUNK_SIZE = 3500
DEFAULT_CHUNK_OVERLAP = 400


def _normalize_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_statutory_sections(
    text: str, active_section: str | None = None
) -> tuple[list[tuple[str | None, str]], str | None]:
    normalized = _normalize_text(text)
    if not normalized:
        return [], active_section

    heading_pattern = re.compile(
        r"(?m)^\s*(\d{1,3})\.\s+(?=(?:\(\d+\)|[A-Z]))"
    )
    matches = list(heading_pattern.finditer(normalized))
    if not matches:
        return [(active_section, normalized)], active_section

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        prefix = normalized[: matches[0].start()].strip()
        if prefix:
            sections.append((active_section, prefix))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        section_text = normalized[match.start() : end].strip()
        section_number = match.group(1)
        if section_text:
            sections.append((section_number, section_text))
        active_section = section_number

    return sections, active_section


def _faq_number_from_text(text: str) -> str | None:
    match = re.search(r"(?m)^\s*(\d+)\.\s+", text)
    return match.group(1) if match else None


def _split_by_logical_blocks(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    boundary_pattern = re.compile(
        r"(?m)^(?:\s*(?:Section|Sec\.)\s+[IVXLC0-9]+(?:\.\s*|\s+|$))",
        re.IGNORECASE,
    )
    matches = list(boundary_pattern.finditer(normalized))
    if not matches:
        return [normalized]

    starts = [match.start() for match in matches]
    if starts[0] != 0:
        starts = [0] + starts
    boundaries = starts + [len(normalized)]
    chunks: list[str] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment = normalized[start:end].strip()
        if segment:
            chunks.append(segment)
    return chunks


def _split_long_segment(segment: str, *, chunk_size: int, overlap: int) -> list[str]:
    segment = _normalize_text(segment)
    if not segment:
        return []
    if len(segment) <= chunk_size:
        return [segment]

    boundaries: list[int] = []
    for match in re.finditer(
        r"(?im)^(?:\s*(?:\([a-z]\)|\([ivx]+\)|\([0-9]+\)|\d+\.|Proviso\s*:|Explanation\s*:|Exception\s*:|Clause\s+\(?[A-Za-z0-9]+\)?\s*:|Sub-section\s*\([^)]*\)\s*:|Section\s+[IVXLC0-9]+(?:\.\s*|\s+)|Chapter\s+[IVXLC0-9]+(?:\.\s*|\s+)))",
        segment,
    ):
        if match.start() > 0:
            boundaries.append(match.start())

    if boundaries:
        pieces = []
        start = 0
        for boundary in boundaries:
            if boundary - start > 80:
                candidate = segment[start:boundary].strip()
                if candidate:
                    pieces.append(candidate)
                start = boundary
        tail = segment[start:].strip()
        if tail:
            pieces.append(tail)
        if pieces:
            return pieces

    pieces: list[str] = []
    start = 0
    while start < len(segment):
        end = min(start + chunk_size, len(segment))
        piece = segment[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(segment):
            break
        start = max(start + chunk_size - overlap, end - overlap)

    return pieces


def _estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def _finalize_chunk(
    chunk_text: str,
    source_id: str,
    *,
    chunk_index: int,
    page_number: int | None = None,
    document_title: str | None = None,
    metadata: dict[str, Any] | None = None,
    section_number: str | None = None,
) -> dict[str, Any]:
    clean_text = chunk_text.strip()
    chunk = {
        "chunk_id": f"{source_id}-page-{page_number or 0}-chunk-{chunk_index}",
        "source_id": source_id,
        "document_title": document_title,
        "page_number": page_number,
        "page_range": f"page {page_number}" if page_number is not None else None,
        "section_number": section_number,
        "text": clean_text,
        "char_count": len(clean_text),
        "token_count": _estimate_tokens(clean_text),
    }
    if metadata:
        chunk.update(metadata)
    return chunk


def chunk_document(
    text: str,
    source_id: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    page_number: int | None = None,
    document_title: str | None = None,
    metadata: dict[str, Any] | None = None,
    section_number: str | None = None,
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        chunk_size = DEFAULT_CHUNK_SIZE
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = min(500, max(300, chunk_size // 7))

    blocks = _split_by_logical_blocks(text)
    if not blocks:
        return []

    chunks: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks, start=1):
        pieces = _split_long_segment(block, chunk_size=chunk_size, overlap=overlap)
        for piece_index, piece in enumerate(pieces, start=1):
            chunks.append(
                _finalize_chunk(
                    piece,
                    source_id,
                    chunk_index=(block_index * 100) + piece_index,
                    page_number=page_number,
                    document_title=document_title,
                    metadata=metadata,
                    section_number=section_number,
                )
            )

    return chunks


def load_extracted_pages(extracted_dir: str | Path = DEFAULT_EXTRACTED_DIR) -> list[dict[str, Any]]:
    directory = Path(extracted_dir)
    if not directory.exists():
        return []

    records: list[dict[str, Any]] = []
    for file_path in sorted(directory.glob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record.setdefault("source_filename", file_path.name)
                records.append(record)
    return records


def chunk_extracted_documents(
    extracted_dir: str | Path = DEFAULT_EXTRACTED_DIR,
    *,
    output_path: str | Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    output_file = Path(output_path) if output_path is not None else Path(extracted_dir).resolve().parent / "processed" / "chunks.jsonl"
    chunks: list[dict[str, Any]] = []
    source_metadata_lookup = {
        value.get("source_id"): value for value in SOURCE_METADATA.values() if value.get("source_id")
    }

    active_section_by_source: dict[str, str | None] = {}
    for record in load_extracted_pages(extracted_dir):
        source_id = record.get("source_id")
        if not source_id:
            continue
        metadata = source_metadata_lookup.get(source_id, {})
        page_metadata = {
                "source_filename": record.get("source_filename") or metadata.get("source_filename") or metadata.get("filename"),
                "source_url": metadata.get("source_url"),
                "source_type": metadata.get("source_type"),
                "publication_date": metadata.get("publication_date"),
                "verification_status": metadata.get("verification_status"),
                "publisher": metadata.get("publisher"),
                "jurisdiction": metadata.get("jurisdiction"),
        }
        page_text = record.get("text", "")
        page_chunks: list[dict[str, Any]] = []
        if metadata.get("source_type") == "primary_legislation":
            sections, active_section_by_source[source_id] = _split_statutory_sections(
                page_text, active_section_by_source.get(source_id)
            )
            for section_number, section_text in sections:
                page_chunks.extend(
                    chunk_document(
                        section_text,
                        source_id,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        page_number=record.get("page_number"),
                        document_title=record.get("document_title"),
                        metadata=page_metadata,
                        section_number=section_number,
                    )
                )
        else:
            page_chunks.extend(
                chunk_document(
                    page_text,
                    source_id,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    page_number=record.get("page_number"),
                    document_title=record.get("document_title"),
                    metadata=page_metadata,
                )
            )
        for page_chunk_index, page_chunk in enumerate(page_chunks, start=1):
            page_chunk["chunk_id"] = (
                f"{source_id}-page-{record.get('page_number') or 0}-chunk-{page_chunk_index:03d}"
            )
        chunks.extend(page_chunks)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for item in chunks:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    return chunks


def generate_chunk_report(chunks: Iterable[dict[str, Any]], *, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> dict[str, Any]:
    chunk_list = list(chunks)
    total_chars = sum(len(item.get("text", "")) for item in chunk_list)
    total_tokens = sum(int(item.get("token_count", 0)) for item in chunk_list)
    max_chars = max((len(item.get("text", "")) for item in chunk_list), default=0)
    oversized_sections = []
    for item in chunk_list:
        if len(item.get("text", "")) > chunk_size:
            oversized_sections.append(
                {
                    "chunk_id": item.get("chunk_id"),
                    "source_id": item.get("source_id"),
                    "page_number": item.get("page_number"),
                    "char_count": len(item.get("text", "")),
                }
            )

    return {
        "chunk_count": len(chunk_list),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "total_characters": total_chars,
        "total_tokens_estimate": total_tokens,
        "average_chunk_characters": round(total_chars / len(chunk_list), 2) if chunk_list else 0,
        "max_chunk_characters": max_chars,
        "oversized_sections": oversized_sections,
    }


def print_chunk_report(report: dict[str, Any]) -> None:
    print("Chunk report")
    print("-" * 88)
    print(f"Chunks: {report['chunk_count']}")
    print(f"Total characters: {report['total_characters']}")
    print(f"Total tokens (approx.): {report['total_tokens_estimate']}")
    print(f"Average chunk characters: {report['average_chunk_characters']}")
    print(f"Max chunk characters: {report['max_chunk_characters']}")
    if report["oversized_sections"]:
        print("Oversized sections:")
        for item in report["oversized_sections"]:
            print(f"  - {item['chunk_id']} @ page {item['page_number']}: {item['char_count']} chars")
    else:
        print("Oversized sections: none")


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    output_dir = root / "knowledge_base" / "processed"
    chunks = chunk_extracted_documents(root / "knowledge_base" / "extracted", output_path=output_dir / "chunks.jsonl")
    report = generate_chunk_report(chunks, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
    (output_dir / "chunk_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_chunk_report(report)


if __name__ == "__main__":
    main()
