"""Answer-level evaluation of the production RAG path (no LLM judge).

Runs every eval question through answer_query(), the same function the
/query endpoint calls: hybrid retrieval, generation, then the grounding
verifier (unless --verifier off). Each answer is scored with deterministic
checks from ragforge.evaluation.answer_checks: fact recall, completeness,
refusals, citation validity and citation support.

When the verifier is on, the model's original answer (the "candidate") is
scored too, so every verifier block is classified:

    blocked_wrong          answerable; candidate was wrong or had no facts (good block)
    blocked_correct        answerable; candidate was complete (lost a correct answer)
    blocked_partial        answerable; candidate had some facts (lost a partial answer)
    blocked_hallucination  unanswerable; candidate made a claim (good block)
    blocked_refusal        unanswerable; candidate was already a refusal (no change)

Requires Qdrant (with the document indexed) and Ollama running:

    uv run python tests/evaluation/answer_eval.py
    uv run python tests/evaluation/answer_eval.py --verifier off
    uv run python tests/evaluation/answer_eval.py --prompt clean-v2
    uv run python tests/evaluation/answer_eval.py --num-ctx 4096
    uv run python tests/evaluation/answer_eval.py --model llama3.1:8b
    uv run python tests/evaluation/answer_eval.py --only q04 q05 q06 --verbose
    uv run python tests/evaluation/answer_eval.py --json data/eval/runs/clean-3b-verifier.json

No services needed:

    uv run python tests/evaluation/answer_eval.py --compare A.json B.json
    uv run python tests/evaluation/answer_eval.py --rescore A.json --json A-rescored.json

--rescore re-scores the saved answers with the current checkers and labels,
so a scorer fix does not require re-running the model. Saved runs store,
for each retrieved chunk, whether it contains the question's evidence (not
the chunk text), so citation support can be re-scored too.

Change one variable per run. With 28 answerable questions, one question is
about 0.036 of fact recall, so small differences are within noise.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ragforge.app.generation.llm import generate_answer_with_stats
from ragforge.app.generation.prompts import SYSTEM_PROMPTS
from ragforge.app.generation.service import answer_query, get_hybrid_searcher
from ragforge.core.config import get_settings
from ragforge.evaluation.answer_checks import score_answer
from ragforge.evaluation.metrics import relevance_flags

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "eval" / "lease_eval.json"
DEFAULT_TOP_K = 5  # same default as answer_query()
CONFIG_KEYS = ("model", "prompt", "num_ctx", "top_k", "verifier")


def git_revision() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{rev}{'+dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_value(report: dict, key: str):
    """Config value with defaults for runs saved before a key existed."""
    defaults = {"verifier": "off"}  # runs before the verifier flag called generation directly
    return report["config"].get(key, defaults.get(key))


def score_row(question: dict, row: dict) -> None:
    """(Re)compute row["score"], row["candidate_score"] and row["verifier_effect"]."""
    has_evidence = [src.get("contains_evidence", False) for src in row.get("sources", [])]
    abstained = row.get("abstained")
    score = score_answer(question, row["answer"], source_has_evidence=has_evidence, abstained=abstained)
    row["score"] = score.to_dict()

    row["candidate_score"] = None
    row["verifier_effect"] = None
    if row.get("abstention_reason") != "failed_verification":
        return

    candidate = score_answer(question, row["candidate_answer"], source_has_evidence=has_evidence)
    row["candidate_score"] = candidate.to_dict()
    if question["answerable"]:
        if candidate.contradicted or candidate.fact_recall == 0.0:
            row["verifier_effect"] = "blocked_wrong"
        elif candidate.complete:
            row["verifier_effect"] = "blocked_correct"
        else:
            row["verifier_effect"] = "blocked_partial"
    else:
        row["verifier_effect"] = "blocked_refusal" if candidate.correct_refusal else "blocked_hallucination"


def run(args: argparse.Namespace) -> dict:
    settings = get_settings()
    model = args.model or settings.ollama_model
    system_prompt = SYSTEM_PROMPTS[args.prompt]
    verify = args.verifier == "on"

    questions = json.loads(args.dataset.read_text(encoding="utf-8"))["questions"]
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {q["id"] for q in questions}
        if unknown:
            raise SystemExit(f"Unknown question ids: {sorted(unknown)}")
        questions = [q for q in questions if q["id"] in wanted]

    if not args.no_warmup:
        # Load the model before timing so question 1 does not absorb the load time.
        searcher = get_hybrid_searcher(settings)
        warm_results = searcher.search(questions[0]["question"], top_k=args.top_k)
        generate_answer_with_stats(
            settings, "warm-up", warm_results, model=model, system_prompt=system_prompt, num_ctx=args.num_ctx
        )

    rows = []
    for question in questions:
        result = answer_query(
            settings,
            question["question"],
            top_k=args.top_k,
            verify=verify,
            model=model,
            system_prompt=system_prompt,
            num_ctx=args.num_ctx,
        )
        evidence = question.get("evidence") or []
        texts = [source.text for source in result.sources]
        has_evidence = relevance_flags(texts, evidence) if evidence else [False] * len(texts)
        stats = result.generation_stats or {}
        timings = result.timings_ms
        row = {
            "id": question["id"],
            "question": question["question"],
            "tags": question.get("tags", []),
            "expected": question["expected"],
            "answer": result.answer,
            "candidate_answer": result.candidate_answer,
            "abstained": result.abstained,
            "abstention_reason": result.abstention_reason,
            "verification": result.verification,
            "retrieval_ms": timings.get("retrieval"),
            "generation_ms": timings.get("generation"),
            "verification_ms": timings.get("verification"),
            "prompt_tokens": stats.get("prompt_eval_count"),
            "output_tokens": stats.get("eval_count"),
            "sources": [
                {
                    "page": source.metadata.get("page_number"),
                    "section": source.metadata.get("section_number"),
                    "rerank_score": source.rerank_score,
                    "contains_evidence": flag,
                }
                for source, flag in zip(result.sources, has_evidence)
            ],
        }
        score_row(question, row)
        rows.append(row)
        if not args.quiet:
            s = row["score"]
            mark = f"recall={s['fact_recall']:.2f}" if s["answerable"] else f"refused={s['correct_refusal']}"
            effect = f"  [{row['verifier_effect']}]" if row["verifier_effect"] else ""
            print(f"  {question['id']}  {mark}  {timings.get('generation', 0):7.0f} ms{effect}", flush=True)

    return {
        "config": {
            "model": model,
            "prompt": args.prompt,
            "system_prompt": system_prompt,
            "num_ctx": args.num_ctx,
            "top_k": args.top_k,
            "verifier": args.verifier,
            "verifier_model": settings.ollama_model if verify else None,
            "dataset": str(args.dataset.relative_to(ROOT) if args.dataset.is_relative_to(ROOT) else args.dataset),
            "only": args.only,
            "git": git_revision(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "summary": summarize(rows),
        "questions": rows,
    }


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["score"]["answerable"]]
    unanswerable = [r for r in rows if not r["score"]["answerable"]]

    def pct(values: list) -> dict:
        values = [v for v in values if v is not None]
        if not values:
            return {"mean": None, "p50": None, "max": None}
        return {
            "mean": round(statistics.mean(values), 1),
            "p50": round(statistics.median(values), 1),
            "max": round(max(values), 1),
        }

    def count(items: list[dict], key: str) -> int:
        return sum(bool(r["score"].get(key)) for r in items)

    with_valid_cites = [r for r in answerable if r["score"]["citation_supported"] is not None]
    by_tag: dict[str, list[float]] = defaultdict(list)
    for r in answerable:
        for tag in r["tags"]:
            by_tag[tag].append(r["score"]["fact_recall"])

    prompt_tokens = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens") is not None]
    output_tokens = [r["output_tokens"] for r in rows if r.get("output_tokens") is not None]
    verified = [r for r in rows if r.get("verification")]

    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "mean_fact_recall": round(statistics.mean(r["score"]["fact_recall"] for r in answerable), 3)
        if answerable
        else None,
        "complete": count(answerable, "complete"),
        "contradicted": count(answerable, "contradicted"),
        "false_refusals": count(answerable, "false_refusal"),
        "hedged": count(answerable, "hedged"),
        "correct_refusals": count(unanswerable, "correct_refusal"),
        "citations_valid": count(rows, "citations_valid"),
        "answerable_without_citation": sum(not r["score"]["citations"] for r in answerable),
        "citation_supported": count(with_valid_cites, "citation_supported"),
        "n_with_valid_citation": len(with_valid_cites),
        "abstention_reasons": dict(Counter(r.get("abstention_reason") for r in rows if r.get("abstention_reason"))),
        "verifier_checked": len(verified),
        "verifier_status": dict(Counter(r["verification"]["status"] for r in verified)),
        "verifier_hit_token_limit": sum(bool(r["verification"].get("hit_token_limit")) for r in verified),
        "verifier_output_tokens_mean": round(statistics.mean(
            r["verification"]["output_tokens"] for r in verified if r["verification"].get("output_tokens") is not None
        ), 1) if any(r["verification"].get("output_tokens") is not None for r in verified) else None,
        "verifier_effects": dict(Counter(r["verifier_effect"] for r in rows if r.get("verifier_effect"))),
        "fact_recall_by_tag": {
            tag: {"n": len(v), "mean": round(statistics.mean(v), 3)} for tag, v in sorted(by_tag.items())
        },
        "retrieval_ms": pct([r.get("retrieval_ms") for r in rows]),
        "generation_ms": pct([r.get("generation_ms") for r in rows]),
        "verification_ms": pct([r.get("verification_ms") for r in rows]),
        "prompt_tokens_max": max(prompt_tokens) if prompt_tokens else None,
        "output_tokens_mean": round(statistics.mean(output_tokens), 1) if output_tokens else None,
    }


def print_report(report: dict, verbose: bool) -> None:
    c, s = report["config"], report["summary"]
    na, nu = s["n_answerable"], s["n_unanswerable"]
    ctx = c["num_ctx"] if c["num_ctx"] is not None else "ollama default"
    print()
    print(
        f"Answer eval  model={c['model']}  prompt={c['prompt']}  verifier={config_value(report, 'verifier')}  "
        f"num_ctx={ctx}  top_k={c['top_k']}  git={c['git']}"
    )
    if c.get("rescored_at"):
        print(f"(re-scored {c['rescored_at']} with the current checkers; answers unchanged)")
    print()
    if na:
        print(f"answerable (n={na})")
        print(f"  mean fact recall      {s['mean_fact_recall']:.3f}")
        print(f"  complete answers      {s['complete']}/{na}")
        print(f"  contradicted (wrong)  {s['contradicted']}/{na}")
        print(f"  false refusals        {s['false_refusals']}/{na}")
        print(f"  answered + hedged     {s['hedged']}/{na}  (answer also says something is not in the sources)")
    if nu:
        print(f"unanswerable (n={nu})")
        print(f"  correct refusals      {s['correct_refusals']}/{nu}  (answer leads with the refusal)")
    if s.get("abstention_reasons"):
        print(f"abstention reasons      {s['abstention_reasons']}")
    if s.get("verifier_checked"):
        print(f"verifier (checked {s['verifier_checked']} answers)")
        print(f"  status                {s['verifier_status']}")
        print(f"  hit token limit       {s.get('verifier_hit_token_limit')}/{s['verifier_checked']}  "
              f"(mean output tokens {s.get('verifier_output_tokens_mean')})")
        print(f"  effect of blocks      {s['verifier_effects'] or 'none'}")
    print(f"citations (n={na + nu} answers)")
    print(f"  all in range          {s['citations_valid']}/{na + nu}")
    if na:
        print(f"  answerable, no cite   {s['answerable_without_citation']}/{na}")
        print(f"  cited chunk has evid. {s['citation_supported']}/{s['n_with_valid_citation']}")
    print("latency ms (mean / p50 / max)")
    for key in ("retrieval", "generation", "verification"):
        v = s.get(f"{key}_ms") or {}
        if v.get("mean") is not None:
            print(f"  {key:<21} {v['mean']} / {v['p50']} / {v['max']}")
    print(f"  output tokens (mean)  {s.get('output_tokens_mean')}")
    print(f"  max prompt tokens     {s['prompt_tokens_max']}  (Ollama can undercount when it reuses a cached prefix)")

    if s["fact_recall_by_tag"]:
        print()
        print("fact recall by tag")
        for tag, v in s["fact_recall_by_tag"].items():
            print(f"  {tag:<14} n={v['n']:<3} {v['mean']:.3f}")

    print()
    print(f"{'id':<5} {'recall':>6} {'flag':<10} {'cites':<10} {'support':>7}  missing / verifier")
    for row in report["questions"]:
        sc = row["score"]
        if sc["answerable"]:
            recall = f"{sc['fact_recall']:.2f}"
            flag = (
                "WRONG" if sc.get("contradicted")
                else "refused" if sc.get("false_refusal")
                else "hedged" if sc.get("hedged")
                else ""
            )
        else:
            recall = "ok" if sc["correct_refusal"] else "MISS"
            flag = "" if sc["correct_refusal"] else ("late-hedge" if sc["refused"] else "answered")
        cites = ",".join(map(str, sc["citations"])) or "-"
        if sc["out_of_range_citations"]:
            cites += "!"
        support = {True: "yes", False: "no", None: "-"}[sc["citation_supported"]]
        detail = "; ".join(group[0] for group in sc["missing_facts"])
        if sc.get("contradiction"):
            detail = f"says: {sc['contradiction']!r}"
        if row.get("verifier_effect"):
            status = (row.get("verification") or {}).get("status")
            detail = f"[{row['verifier_effect']}, {status}] {detail}".strip()
        print(f"{row['id']:<5} {recall:>6} {flag:<10} {cites:<10} {support:>7}  {detail}")

    if verbose:
        for row in report["questions"]:
            print()
            print(f"--- {row['id']}: {row['question']}")
            print(f"expected:  {row['expected']}")
            if row.get("candidate_answer") and row["candidate_answer"] != row["answer"]:
                print(f"candidate: {row['candidate_answer']}")
                print(f"verifier:  {row.get('verification')}")
            print(f"answer:    {row['answer']}")
            print(f"sources:   {[(src['section'], src['page']) for src in row['sources']]}")


def rescore(path: Path, dataset: Path) -> dict:
    """Re-score saved answers with the current checkers and labels."""
    report = json.loads(path.read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in json.loads(dataset.read_text(encoding="utf-8"))["questions"]}
    for row in report["questions"]:
        question = questions[row["id"]]
        sources = row.get("sources", [])
        old_support = row.get("score", {}).get("citation_supported")
        row.setdefault("candidate_answer", row["answer"])
        score_row(question, row)
        if not (sources and all("contains_evidence" in src for src in sources)):
            # Older runs did not store evidence flags: keep the citation
            # support computed at run time.
            row["score"]["citation_supported"] = old_support
        row["expected"] = question["expected"]
    report["summary"] = summarize(report["questions"])
    report["config"]["rescored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return report


def compare(path_a: Path, path_b: Path) -> None:
    a = json.loads(path_a.read_text(encoding="utf-8"))
    b = json.loads(path_b.read_text(encoding="utf-8"))

    def label(report: dict) -> str:
        return "/".join(f"{k}={config_value(report, k)}" for k in CONFIG_KEYS)

    print(f"A = {path_a.name}: {label(a)}")
    print(f"B = {path_b.name}: {label(b)}")
    changed_keys = [k for k in CONFIG_KEYS if config_value(a, k) != config_value(b, k)]
    if len(changed_keys) != 1:
        print(f"WARNING: runs differ in {changed_keys or 'nothing'}; a clean comparison changes exactly one variable.")

    rows_a = {r["id"]: r["score"] for r in a["questions"]}
    rows_b = {r["id"]: r["score"] for r in b["questions"]}
    common = sorted(set(rows_a) & set(rows_b))

    def value(score: dict) -> float:
        return score["fact_recall"] if score["answerable"] else float(score["correct_refusal"])

    better = [q for q in common if value(rows_b[q]) > value(rows_a[q])]
    worse = [q for q in common if value(rows_b[q]) < value(rows_a[q])]
    print()
    keys = ("mean_fact_recall", "complete", "contradicted", "false_refusals", "hedged",
            "correct_refusals", "answerable_without_citation", "citation_supported",
            "abstention_reasons", "verifier_effects")
    for key in keys:
        print(f"  {key:<28} A={a['summary'].get(key)}  B={b['summary'].get(key)}")
    for key in ("generation_ms", "verification_ms"):
        print(f"  {key + ' mean':<28} A={(a['summary'].get(key) or {}).get('mean')}  B={(b['summary'].get(key) or {}).get('mean')}")
    print(f"  {'output tokens mean':<28} A={a['summary'].get('output_tokens_mean')}  B={b['summary'].get('output_tokens_mean')}")
    print()
    print(f"questions where B is better ({len(better)}): {' '.join(better) or '-'}")
    print(f"questions where B is worse  ({len(worse)}): {' '.join(worse) or '-'}")
    print(f"unchanged: {len(common) - len(better) - len(worse)} of {len(common)}")
    for qid in better + worse:
        print(f"  {qid}: {value(rows_a[qid]):.2f} -> {value(rows_b[qid]):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", help="Ollama generation model (default: OLLAMA_MODEL). The verifier always uses OLLAMA_MODEL.")
    parser.add_argument("--num-ctx", type=int, help="Ollama context window (default: Ollama's own default).")
    parser.add_argument("--prompt", choices=sorted(SYSTEM_PROMPTS), default="clean-v2")
    parser.add_argument("--verifier", choices=("on", "off"), default="off", help="Grounding verifier (production default: off).")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Chunks passed to the model.")
    parser.add_argument("--only", nargs="+", metavar="QID", help="Run only these question ids.")
    parser.add_argument("--verbose", action="store_true", help="Print every answer with expected text and sources.")
    parser.add_argument("--json", type=Path, help="Write the full run (config, summary, answers) to this file.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip the untimed model warm-up call.")
    parser.add_argument("--quiet", action="store_true", help="Do not print progress per question.")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A.json", "B.json"), help="Compare two saved runs.")
    parser.add_argument("--rescore", type=Path, metavar="RUN.json", help="Re-score a saved run with the current checkers.")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    if args.rescore:
        report = rescore(args.rescore, args.dataset)
    else:
        if args.top_k <= 0:
            parser.error("--top-k must be greater than 0")
        report = run(args)
    print_report(report, args.verbose)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
