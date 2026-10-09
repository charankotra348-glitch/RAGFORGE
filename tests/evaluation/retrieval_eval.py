"""Compare retrieval strategies on a hand-labeled question set.

Strategies (all retrieve from the live Qdrant index, no LLM involved):

    dense          Qdrant vector search only
    bm25           BM25 keyword search only
    hybrid         dense + BM25 merged with Reciprocal Rank Fusion
    hybrid+rerank  hybrid candidates re-scored by the cross-encoder

Requires Qdrant running with the document already indexed:

    uv run python -m ragforge.vector.reindex data/test.pdf
    uv run python tests/evaluation/retrieval_eval.py
    uv run python tests/evaluation/retrieval_eval.py --verbose
    uv run python tests/evaluation/retrieval_eval.py --json results.json
    uv run python tests/evaluation/retrieval_eval.py --pool 20   # wider candidate pool
    uv run python tests/evaluation/retrieval_eval.py --pool 27 --trace q04   # ranked chunks for one question
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document

from ragforge.app.retrieval.bm25 import BM25Retriever
from ragforge.app.retrieval.fusion import reciprocal_rank_fusion
from ragforge.app.retrieval.reranker import rerank
from ragforge.app.retrieval.types import SearchResult
from ragforge.app.retrieval.vector import get_qdrant, search_vectors
from ragforge.app.retrieval.vector_store import get_indexed_chunks
from ragforge.core.config import get_settings
from ragforge.evaluation.metrics import (
    coverage_at_k,
    hit_at_k,
    missing_evidence,
    mrr,
    ndcg_at_k,
    normalize,
    relevance_flags,
)
from ragforge.ingestion.embedder import embed_chunks

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "data" / "eval" / "lease_eval.json"
DEFAULT_POOL = 10  # candidates fetched per retriever (override with --pool)
HIT_KS = (1, 5)
COVERAGE_KS = (3, 5, 8)
NDCG_K = 5
VERBOSE_K = 5

Strategy = Callable[[str], list[SearchResult]]


def build_strategies(chunks: list[Document], pool: int) -> dict[str, Strategy]:
    settings = get_settings()
    client = get_qdrant(settings)
    bm25 = BM25Retriever(chunks=chunks)

    def dense(query: str) -> list[SearchResult]:
        vector = embed_chunks(settings, [Document(page_content=query)])[0]
        return search_vectors(client, settings, vector, top_k=pool)

    def bm25_only(query: str) -> list[SearchResult]:
        return bm25.search(query, top_k=pool)

    def hybrid(query: str) -> list[SearchResult]:
        return reciprocal_rank_fusion(
            vector_results=dense(query),
            bm25_results=bm25_only(query),
            top_k=pool,
        )

    def hybrid_rerank(query: str) -> list[SearchResult]:
        return rerank(query, hybrid(query), top_k=pool)

    return {
        "dense": dense,
        "bm25": bm25_only,
        "hybrid": hybrid,
        "hybrid+rerank": hybrid_rerank,
    }


def check_corpus(chunks: list[Document], questions: list[dict]) -> list[str]:
    """Return warnings about the index and the labels (never silently ignored)."""
    warnings: list[str] = []

    counts = Counter(normalize(chunk.page_content) for chunk in chunks)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    if duplicates:
        warnings.append(
            f"{duplicates} duplicate chunk(s) in the index. Duplicates waste "
            "top-K slots and inflate metrics; remove the stale document copy."
        )

    haystacks = [normalize(chunk.page_content) for chunk in chunks]
    for question in questions:
        for snippet in question["evidence"]:
            if not any(normalize(snippet) in haystack for haystack in haystacks):
                warnings.append(
                    f"{question['id']}: evidence not found in ANY indexed chunk "
                    f"(label typo, or the chunker split it): {snippet!r}"
                )
    return warnings


def evaluate(
    strategies: dict[str, Strategy],
    chunks: list[Document],
    questions: list[dict],
) -> dict:
    answerable = [q for q in questions if q["answerable"]]
    corpus = [normalize(chunk.page_content) for chunk in chunks]

    # Warm up models so latency numbers exclude one-time loading.
    for strategy in strategies.values():
        strategy(questions[0]["question"])

    report: dict = {"strategies": {}, "no_answer_scores": {}}

    for name, strategy in strategies.items():
        per_question: dict[str, dict] = {}
        latencies: list[float] = []

        for question in questions:
            started = time.perf_counter()
            results = strategy(question["question"])
            latencies.append((time.perf_counter() - started) * 1000)

            if not question["answerable"]:
                if name == "hybrid+rerank" and results:
                    report["no_answer_scores"][question["id"]] = results[0].rerank_score
                continue

            texts = [result.text for result in results]
            evidence = question["evidence"]
            flags = relevance_flags(texts, evidence)
            total_relevant = sum(
                any(normalize(snippet) in haystack for snippet in evidence)
                for haystack in corpus
            )
            first_rank = next((i for i, f in enumerate(flags, 1) if f), None)

            entry = {
                "tags": question.get("tags", []),
                "first_relevant_rank": first_rank,
                "mrr": mrr(flags),
                f"ndcg@{NDCG_K}": ndcg_at_k(flags, NDCG_K, total_relevant),
                "missing_at_verbose_k": missing_evidence(texts, evidence, VERBOSE_K),
                "top_score": (
                    results[0].rerank_score if name == "hybrid+rerank" and results else None
                ),
            }
            for k in HIT_KS:
                entry[f"hit@{k}"] = hit_at_k(flags, k)
            for k in COVERAGE_KS:
                entry[f"coverage@{k}"] = coverage_at_k(texts, evidence, k)
            per_question[question["id"]] = entry

        def mean(key: str) -> float:
            return statistics.fmean(item[key] for item in per_question.values())

        summary = {f"hit@{k}": mean(f"hit@{k}") for k in HIT_KS}
        summary.update({f"coverage@{k}": mean(f"coverage@{k}") for k in COVERAGE_KS})
        summary["mrr"] = mean("mrr")
        summary[f"ndcg@{NDCG_K}"] = mean(f"ndcg@{NDCG_K}")
        summary["latency_ms_mean"] = statistics.fmean(latencies)

        report["strategies"][name] = {"summary": summary, "per_question": per_question}

    report["answerable_questions"] = len(answerable)
    report["rerank_top1_answerable"] = [
        entry["top_score"]
        for entry in report["strategies"]["hybrid+rerank"]["per_question"].values()
        if entry["top_score"] is not None
    ]
    return report


def print_report(report: dict, verbose: bool, pool: int) -> None:
    columns = (
        [f"hit@{k}" for k in HIT_KS]
        + [f"coverage@{k}" for k in COVERAGE_KS]
        + ["mrr", f"ndcg@{NDCG_K}", "latency_ms_mean"]
    )
    labels = (
        [f"Hit@{k}" for k in HIT_KS]
        + [f"Cov@{k}" for k in COVERAGE_KS]
        + ["MRR", f"nDCG@{NDCG_K}", "ms/query"]
    )

    print(f"\nRetrieval evaluation over {report['answerable_questions']} answerable questions (candidate pool = {pool})")
    print("Hit@K = relevant chunk in top K. Cov@K = share of required evidence retrieved in top K.\n")
    print(f"{'strategy':<15}" + "".join(f"{label:>10}" for label in labels))
    for name, data in report["strategies"].items():
        row = f"{name:<15}"
        for column in columns:
            value = data["summary"][column]
            row += f"{value:>10.0f}" if column == "latency_ms_mean" else f"{value:>10.2f}"
        print(row)

    print_tag_breakdown(report)

    if verbose:
        print(f"\nPer question (top {VERBOSE_K}): first relevant rank, and evidence still missing")
        for name, data in report["strategies"].items():
            print(f"\n[{name}]")
            for qid, entry in data["per_question"].items():
                rank = entry["first_relevant_rank"] or "-"
                missing = entry["missing_at_verbose_k"]
                note = "complete" if not missing else f"MISSING {len(missing)}: " + " | ".join(missing)
                print(f"  {qid}  first_rank={rank!s:<3} {note}")

    answerable_scores = report["rerank_top1_answerable"]
    no_answer = report["no_answer_scores"]
    if answerable_scores and no_answer:
        print("\nTop-1 rerank score (could back an 'insufficient evidence' threshold):")
        print(f"  answerable questions:   min {min(answerable_scores):.2f}, max {max(answerable_scores):.2f}")
        print("  unanswerable questions: " + ", ".join(f"{qid}={score:.2f}" for qid, score in no_answer.items()))
        print(f"  Only {len(no_answer)} unanswerable questions: treat this as a hint, not a calibrated threshold.")


def print_tag_breakdown(report: dict) -> None:
    """Mean Cov@5 per question tag, so you can see which kinds of question fail."""
    strategies = report["strategies"]
    tag_counts: Counter[str] = Counter()
    first = next(iter(strategies.values()))["per_question"]
    for entry in first.values():
        tag_counts.update(entry["tags"])
    if not tag_counts:
        return

    key = "coverage@5"
    print("\nMean Cov@5 by question type (n = questions with that tag)")
    print(f"{'tag':<16}{'n':>4}" + "".join(f"{name:>15}" for name in strategies))
    for tag, count in sorted(tag_counts.items()):
        if tag == "unanswerable":
            continue
        row = f"{tag:<16}{count:>4}"
        for data in strategies.values():
            values = [e[key] for e in data["per_question"].values() if tag in e["tags"]]
            row += f"{statistics.fmean(values):>15.2f}"
        print(row)


def print_trace(strategies: dict[str, Strategy], questions: list[dict], ids: list[str]) -> None:
    """Show every ranked chunk for the given questions, marking required evidence."""
    by_id = {question["id"]: question for question in questions}
    for qid in ids:
        if qid not in by_id:
            raise SystemExit(f"Unknown question id {qid!r}. Known: {', '.join(by_id)}")
        question = by_id[qid]
        evidence = question["evidence"]
        print(f"\n=== TRACE {qid}: {question['question']}")
        for number, snippet in enumerate(evidence, start=1):
            print(f"  evidence[{number}]: {snippet}")

        for name, strategy in strategies.items():
            print(f"\n  [{name}]")
            for rank, result in enumerate(strategy(question["question"]), start=1):
                haystack = normalize(result.text)
                found = [n for n, snippet in enumerate(evidence, 1) if normalize(snippet) in haystack]
                scores = []
                if result.vector_score is not None:
                    scores.append(f"vec={result.vector_score:.3f}")
                if result.bm25_score is not None:
                    scores.append(f"bm25={result.bm25_score:.2f}")
                if name in ("hybrid", "hybrid+rerank"):
                    scores.append(f"rrf={result.rrf_score:.4f}")
                if result.rerank_score is not None:
                    scores.append(f"rerank={result.rerank_score:.2f}")
                page = result.metadata.get("page_number")
                section = result.metadata.get("section_number")
                mark = f"EVIDENCE{found}" if found else ""
                preview = " ".join(result.text.split())[:60]
                print(f"    {rank:>2}  p{page} s{section!s:<4} {' '.join(scores):<46} {mark:<13} {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pool", type=int, default=DEFAULT_POOL, help="Candidates fetched per retriever before fusion/reranking.")
    parser.add_argument("--json", type=Path, help="Also write full results to this JSON file.")
    parser.add_argument("--trace", nargs="+", metavar="QID", help="Print the full ranked chunk list for these question ids (e.g. q04).")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    questions = dataset["questions"]

    settings = get_settings()
    chunks = get_indexed_chunks(get_qdrant(settings), settings)
    if not chunks:
        raise SystemExit("The Qdrant collection is empty. Index a document first.")
    print(f"Loaded {len(chunks)} indexed chunks from collection {settings.qdrant_collection!r}")

    warnings = check_corpus(chunks, questions)
    for warning in warnings:
        print(f"WARNING: {warning}")

    strategies = build_strategies(chunks, args.pool)
    report = evaluate(strategies, chunks, questions)
    print_report(report, args.verbose, args.pool)

    if args.trace:
        print_trace(strategies, questions, args.trace)

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
