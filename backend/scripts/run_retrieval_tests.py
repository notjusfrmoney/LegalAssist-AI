"""Run targeted retrieval evaluation against expected Code on Wages provisions."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.rag.retrieve import retrieve_top_k

CHUNKS_PATH = ROOT / "knowledge_base" / "processed" / "chunks.jsonl"
OUTPUT_PATH = ROOT / "evaluation" / "phase4_retrieval_results.json"
EXPECTED_SOURCE_ID = "code_on_wages_2019"

EVALUATION_CASES = (
    {
        "query": "When should an employer pay monthly salary?",
        "expected_section": "17",
        "patterns": (
            (
                "Section 17(1): monthly wages due before the seventh day",
                re.compile(
                    r"17\.\s*\(1\).*?monthly basis.*?before the expiry of the seventh day of the succeeding month",
                    re.IGNORECASE | re.DOTALL,
                ),
            ),
        ),
    },
    {
        "query": "What deductions can an employer make from wages?",
        "expected_section": "18",
        "patterns": (
            (
                "Section 18(1): deductions only when authorised by the Code",
                re.compile(
                    r"18\.\s*\(1\).*?there shall be no deductions from the wages.*?except those as are authorised under this Code",
                    re.IGNORECASE | re.DOTALL,
                ),
            ),
            (
                "Section 18(3): total deductions capped at fifty per cent",
                re.compile(
                    r"\(3\)\s+Notwithstanding anything contained in this Code.*?total amount of deductions.*?under sub-section\s*\(2\).*?shall not exceed fifty per cent",
                    re.IGNORECASE | re.DOTALL,
                ),
            ),
            (
                "Section 18(5): employer responsibility for un-deposited deductions",
                re.compile(
                    r"\(5\)\s+Where any deduction is made.*?under this section.*?such employee shall not be held responsible",
                    re.IGNORECASE | re.DOTALL,
                ),
            ),
        ),
    },
    {
        "query": "Is an employee entitled to overtime wages?",
        "expected_section": "14",
        "patterns": (
            (
                "Section 14: overtime rate at least twice the normal rate",
                re.compile(
                    r"14\.\s+Where an employee.*?overtime rate.*?twice the normal rate of wages",
                    re.IGNORECASE | re.DOTALL,
                ),
            ),
        ),
    },
    {
        "query": "What happens to unpaid wages after resignation?",
        "expected_section": "17(2)",
        "patterns": (
            (
                "Section 17(2): wages due within two working days after resignation",
                re.compile(
                    r"17\.\s*\(1\).*?\(2\).*?resigned from service.*?wages payable to him shall be paid within two working days",
                    re.IGNORECASE | re.DOTALL,
                ),
            ),
        ),
    },
)


def _read_chunks() -> list[dict[str, Any]]:
    with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _section_evidence(text: str, patterns: tuple[tuple[str, re.Pattern[str]], ...]) -> dict[str, str] | None:
    normalized_text = " ".join(text.split())
    for label, pattern in patterns:
        match = pattern.search(normalized_text)
        if match:
            return {"basis": label, "excerpt": match.group(0)}
    return None


def _reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank is not None else 0.0


def _evaluate_case(case: dict[str, Any], corpus: list[dict[str, Any]]) -> dict[str, Any]:
    matches_in_corpus = [
        chunk
        for chunk in corpus
        if chunk.get("source_id") == EXPECTED_SOURCE_ID
        and _section_evidence(str(chunk.get("text", "")), case["patterns"])
    ]
    retrieved = retrieve_top_k(case["query"], top_k=5, min_similarity=0.0)
    if len(retrieved) != 5:
        raise RuntimeError(
            f"Expected five retrieval results for {case['query']!r}, received {len(retrieved)}"
        )

    evaluated_results: list[dict[str, Any]] = []
    first_match_rank: int | None = None
    for rank, item in enumerate(retrieved, start=1):
        evidence = None
        if item.get("source_id") == EXPECTED_SOURCE_ID:
            evidence = _section_evidence(str(item.get("text", "")), case["patterns"])
        if evidence and first_match_rank is None:
            first_match_rank = rank
        evaluated_results.append(
            {
                "rank": rank,
                "chunk_id": item.get("chunk_id"),
                "document_title": item.get("document_title"),
                "section_number": item.get("section_number"),
                "page_number": item.get("page_number"),
                "original_pdf_page_number": item.get(
                    "original_pdf_page_number", item.get("page_number")
                ),
                "similarity_score": round(float(item["score"]), 6),
                "source_id": item.get("source_id"),
                "source_filename": item.get("source_filename"),
                "source_url": item.get("source_url"),
                "verification_status": item.get("verification_status"),
                "expected_section_match": evidence is not None,
                "section_match_evidence": evidence,
                "text": item.get("text", ""),
            }
        )

    return {
        "query": case["query"],
        "expected_document_title": "Code on Wages, 2019",
        "expected_source_id": EXPECTED_SOURCE_ID,
        "expected_legal_section": case["expected_section"],
        "expected_section_found_in_corpus": bool(matches_in_corpus),
        "matching_corpus_chunks": [
            {
                "chunk_id": chunk.get("chunk_id"),
                "page_number": chunk.get("page_number"),
                "section_evidence": _section_evidence(
                    str(chunk.get("text", "")), case["patterns"]
                ),
            }
            for chunk in matches_in_corpus
        ],
        "expected_section_found_in_top5": first_match_rank is not None,
        "first_expected_section_rank": first_match_rank,
        "results": evaluated_results,
    }


def main() -> None:
    corpus = _read_chunks()
    evaluations = [_evaluate_case(case, corpus) for case in EVALUATION_CASES]
    query_count = len(evaluations)

    metrics = {
        f"hit_at_{k}": round(
            sum(
                item["first_expected_section_rank"] is not None
                and item["first_expected_section_rank"] <= k
                for item in evaluations
            )
            / query_count,
            4,
        )
        for k in (1, 3, 5)
    }
    metrics["mean_reciprocal_rank"] = round(
        sum(_reciprocal_rank(item["first_expected_section_rank"]) for item in evaluations)
        / query_count,
        4,
    )

    report = {
        "evaluation": {
            "retriever": "BAAI/bge-small-en-v1.5 + FAISS IndexFlatIP",
            "top_k": 5,
            "min_similarity": 0.0,
            "query_count": query_count,
            "relevance_rule": (
                "A hit requires retrieved Code on Wages text containing the expected "
                "section's substantive provision. Section metadata or keyword overlap alone "
                "does not count; Section 18 continuation subsections are matched by their "
                "provision text."
            ),
            "metrics_basis": "First substantive expected-section match per query",
        },
        "metrics": metrics,
        "queries": evaluations,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Saved evaluation: {OUTPUT_PATH}")
    print(
        "Metrics: "
        + ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    )
    for item in evaluations:
        rank = item["first_expected_section_rank"]
        print(
            f"{item['expected_legal_section']}: "
            f"{'hit at rank ' + str(rank) if rank is not None else 'miss in top 5'} "
            f"({item['query']})"
        )
        for result in item["results"]:
            print(
                f"  {result['rank']}. {result['chunk_id']} | "
                f"{result['document_title']} | section={result['section_number']} | "
                f"page={result['original_pdf_page_number']} | "
                f"score={result['similarity_score']:.6f} | "
                f"expected-section-match={result['expected_section_match']}"
            )


if __name__ == "__main__":
    main()
