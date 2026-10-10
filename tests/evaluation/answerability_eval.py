from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ragforge.app.generation.service import answer_query
from ragforge.core.config import Settings


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "eval" / "lease_eval.json"
OUTPUT = ROOT / "results_answerability_eval.json"

ABSTENTION_PATTERNS = [
    r"\bnot available in the provided documents\b",
    r"\bnot (?:specified|stated|provided|mentioned) in the (?:provided )?documents\b",
    r"\bthe document(?:s)? (?:does|do) not specify\b",
    r"\bthe document(?:s)? (?:does|do) not mention\b",
    r"\binsufficient information\b",
    r"\bcannot determine\b",
    r"\bcan't determine\b",
    r"\bcannot be determined\b",
    r"\bnot enough information\b",
    r"\bnot provided in the (?:lease|document)\b",
    r"\bunable to determine\b",
    r"\bi cannot provide information\b",
    r"\bi don't have enough information\b",
    r"\bi cannot provide a reliable answer because the retrieved context\b",
]


def normalize(text: str) -> str:
    """Ignore case, whitespace, and punctuation for evidence matching."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def is_abstention(answer: str) -> bool:
    """Heuristic only: inspect wording, not semantic correctness."""
    return any(
        re.search(pattern, answer, flags=re.IGNORECASE)
        for pattern in ABSTENTION_PATTERNS
    )


def main() -> None:
    dataset = json.loads(DATASET.read_text())
    questions = dataset["questions"]
    settings = Settings()

    records = []

    print(f"Evaluating {len(questions)} questions")
    print("This calls the configured Ollama model for each question.\n")

    for index, question in enumerate(questions, start=1):
        qid = question["id"]
        query = question["question"]
        answerable = question["answerable"]

        print(f"[{index}/{len(questions)}] {qid}: {query}")

        started = time.perf_counter()

        try:
            result = answer_query(
                settings=settings,
                query=query,
                top_k=5,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

            answer = result.answer
            sources = [
                {
                    "citation": f"[Source {i}]",
                    "page": str(source.metadata.get("page_number", "unknown")),
                    "section": str(source.metadata.get("section_number", "unknown")),
                    "text": source.text,
                }
                for i, source in enumerate(result.sources, start=1)
            ]

            abstained = is_abstention(answer)
            evidence = question.get("evidence", [])

            matched_evidence = [
                phrase
                for phrase in evidence
                if any(
                    normalize(phrase) in normalize(source["text"])
                    for source in sources
                )
            ]

            all_evidence_found = (
                bool(evidence) and len(matched_evidence) == len(evidence)
            )

            if answerable and abstained:
                outcome = "false_abstention"
            elif answerable and all_evidence_found:
                outcome = "supported_answer_candidate"
            elif answerable:
                outcome = "answer_without_all_gold_evidence"
            elif abstained:
                outcome = "correct_abstention"
            else:
                outcome = "unsupported_answer_candidate"

            record = {
                "id": qid,
                "question": query,
                "expected_answerable": answerable,
                "expected": question.get("expected"),
                "tags": question.get("tags", []),
                "outcome": outcome,
                "abstention_heuristic": abstained,
                "evidence_count": len(evidence),
                "matched_evidence": matched_evidence,
                "all_gold_evidence_retrieved": all_evidence_found,
                "latency_ms": elapsed_ms,
                "answer": answer,
                "sources": sources,
            }

            print(f"  Outcome: {outcome}")
            print(f"  Abstention detected: {abstained}")
            print(
                f"  Gold evidence retrieved: "
                f"{len(matched_evidence)}/{len(evidence)}"
            )
            print(f"  Answer: {answer[:250].replace(chr(10), ' ')}")
            print()

        except Exception as exc:
            record = {
                "id": qid,
                "question": query,
                "expected_answerable": answerable,
                "outcome": "evaluation_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  ERROR: {type(exc).__name__}: {exc}\n")

        records.append(record)

    counts: dict[str, int] = {}
    for record in records:
        outcome = record["outcome"]
        counts[outcome] = counts.get(outcome, 0) + 1

    report = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "total_questions": len(questions),
        "completed_questions": sum(
            record["outcome"] != "evaluation_error" for record in records
        ),
        "outcome_counts": counts,
        "records": records,
        "limitations": [
            "Abstention detection uses wording patterns and can misclassify answers.",
            "Retrieved gold evidence does not prove the generated answer is correct.",
            "Review unsupported-answer candidates and false-abstention cases manually.",
        ],
    }

    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Questions: {len(questions)}")
    for outcome, count in sorted(counts.items()):
        print(f"{outcome}: {count}")
    print(f"\nDetailed answers and sources saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
