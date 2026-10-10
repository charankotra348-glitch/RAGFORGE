"""Offline smoke test for tests/evaluation/answer_eval.py.

answer_query() and the warm-up are stubbed, so this checks the runner's
plumbing, scoring of verifier blocks and report format, not model quality.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ragforge.app.generation.grounding import SAFE_ABSTENTION
from ragforge.app.generation.service import RAGResponse
from ragforge.app.retrieval.types import SearchResult

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "evaluation" / "answer_eval.py"
REFUSAL = "The information is not available in the provided documents."


def load_module():
    spec = importlib.util.spec_from_file_location("answer_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_fake_answer_query(questions: dict, candidates: dict | None = None, blocked: set | None = None):
    """Perfect answers by default; ``candidates`` overrides a question's model answer,
    ``blocked`` lists question ids the verifier rejects."""
    candidates = candidates or {}
    blocked = blocked or set()

    def fake_answer_query(settings, query, top_k=5, *, verify=True, model=None, system_prompt=None, num_ctx=None):
        q = questions[query]
        evidence = q.get("evidence") or ["unrelated clause"]
        sources = [SearchResult(text=" ".join(evidence), metadata={"page_number": 1, "section_number": "1"})]
        default = f"{q['expected']} [Source 1]" if q["answerable"] else REFUSAL
        candidate = candidates.get(q["id"], default)
        response = RAGResponse(
            query=query,
            answer=candidate,
            sources=sources,
            candidate_answer=candidate,
            timings_ms={"retrieval": 1.0, "generation": 2.0},
            generation_stats={"prompt_eval_count": 100, "eval_count": 10},
        )
        if candidate == REFUSAL:
            response.abstained, response.abstention_reason = True, "model_abstained"
        elif verify:
            supported = q["id"] not in blocked
            response.verification = {
                "supported": supported,
                "status": "supported" if supported else "unsupported",
                "reason": None,
                "latency_ms": 3.0,
            }
            response.timings_ms["verification"] = 3.0
            if not supported:
                response.answer = SAFE_ABSTENTION
                response.abstained, response.abstention_reason = True, "failed_verification"
        return response

    return fake_answer_query


@pytest.fixture()
def answer_eval(monkeypatch):
    module = load_module()
    questions = {q["question"]: q for q in json.loads(module.DEFAULT_DATASET.read_text(encoding="utf-8"))["questions"]}
    module._questions = questions
    monkeypatch.setattr(module, "answer_query", make_fake_answer_query(questions))
    return module


def run_main(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["answer_eval.py", "--no-warmup", "--quiet", "--model", "stub", *argv])
    module.main()


def test_runner_scores_perfect_stub_answers(answer_eval, tmp_path, monkeypatch, capsys) -> None:
    out = tmp_path / "run.json"
    run_main(answer_eval, monkeypatch, "--verifier", "on", "--json", str(out))

    report = json.loads(out.read_text(encoding="utf-8"))
    s = report["summary"]
    assert report["config"]["verifier"] == "on"
    assert s["n_answerable"] == 28 and s["n_unanswerable"] == 7
    assert s["mean_fact_recall"] == 1.0 and s["complete"] == 28
    assert s["correct_refusals"] == 7 and s["false_refusals"] == 0
    assert s["citations_valid"] == 35 and s["citation_supported"] == 28
    assert s["verifier_checked"] == 28 and s["verifier_effects"] == {}
    assert "mean fact recall" in capsys.readouterr().out


def test_verifier_blocks_are_classified(answer_eval, tmp_path, monkeypatch) -> None:
    by_id = {q["id"]: q for q in answer_eval._questions.values()}
    candidates = {
        "q01": f"{by_id['q01']['expected']} [Source 1]",  # correct, will be blocked
        "q06": "The total is $850 [Source 1].",  # wrong, will be blocked
        "q35": "The quiet hours are 9:00 A.M. to 8:00 P.M. [Source 1].",  # hallucination, will be blocked
    }
    monkeypatch.setattr(
        answer_eval,
        "answer_query",
        make_fake_answer_query(answer_eval._questions, candidates, blocked={"q01", "q06", "q35"}),
    )
    out = tmp_path / "run.json"
    run_main(answer_eval, monkeypatch, "--verifier", "on", "--only", "q01", "q06", "q35", "--json", str(out))

    report = json.loads(out.read_text(encoding="utf-8"))
    effects = {row["id"]: row["verifier_effect"] for row in report["questions"]}
    assert effects == {"q01": "blocked_correct", "q06": "blocked_wrong", "q35": "blocked_hallucination"}
    rows = {row["id"]: row for row in report["questions"]}
    # The safe abstention is a refusal through the structured flag, not its wording.
    assert rows["q35"]["score"]["correct_refusal"] is True
    assert rows["q01"]["score"]["false_refusal"] is True
    assert report["summary"]["abstention_reasons"] == {"failed_verification": 3}


def test_compare_lists_changed_questions(answer_eval, tmp_path, monkeypatch, capsys) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    run_main(answer_eval, monkeypatch, "--json", str(a), "--only", "q01", "q06")
    report = json.loads(a.read_text(encoding="utf-8"))
    report["config"]["prompt"] = "clean-v2"
    report["questions"][1]["score"]["fact_recall"] = 0.0
    b.write_text(json.dumps(report), encoding="utf-8")

    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["answer_eval.py", "--compare", str(a), str(b)])
    answer_eval.main()
    out = capsys.readouterr().out
    assert "B is worse  (1): q06" in out
    assert "WARNING" not in out


def test_compare_treats_runs_without_verifier_key_as_verifier_off(answer_eval, tmp_path, monkeypatch, capsys) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    run_main(answer_eval, monkeypatch, "--verifier", "on", "--json", str(b), "--only", "q01")
    old = json.loads(b.read_text(encoding="utf-8"))
    del old["config"]["verifier"]
    a.write_text(json.dumps(old), encoding="utf-8")

    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["answer_eval.py", "--compare", str(a), str(b)])
    answer_eval.main()
    out = capsys.readouterr().out
    assert "verifier=off" in out and "verifier=on" in out
    assert "WARNING" not in out


def test_rescore_recomputes_from_saved_answers(answer_eval, tmp_path, monkeypatch) -> None:
    run, rescored = tmp_path / "run.json", tmp_path / "rescored.json"
    run_main(answer_eval, monkeypatch, "--json", str(run), "--only", "q06", "q35")
    report = json.loads(run.read_text(encoding="utf-8"))
    assert all("contains_evidence" in src for row in report["questions"] for src in row["sources"])
    assert "system_prompt" in report["config"]

    # Simulate a stored answer that the old checker scored differently.
    report["questions"][0]["answer"] = REFUSAL
    report["questions"][0]["score"]["fact_recall"] = 1.0
    run.write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["answer_eval.py", "--rescore", str(run), "--json", str(rescored)])
    answer_eval.main()
    new = json.loads(rescored.read_text(encoding="utf-8"))
    q06 = new["questions"][0]["score"]
    assert q06["fact_recall"] == 0.0 and q06["false_refusal"] is True
    assert new["config"]["rescored_at"]
    assert new["questions"][0]["answer"] == report["questions"][0]["answer"]


def test_rescore_handles_runs_saved_before_the_verifier(answer_eval, tmp_path, monkeypatch) -> None:
    old_run = ROOT / "data" / "eval" / "runs" / "clean-3b.json"
    if not old_run.exists():
        pytest.skip("saved run not present")
    out = tmp_path / "rescored.json"
    monkeypatch.setattr(sys, "argv", ["answer_eval.py", "--rescore", str(old_run), "--json", str(out)])
    answer_eval.main()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["n_answerable"] == 28
    assert report["summary"]["verifier_effects"] == {}


def test_verifier_defaults_to_off(answer_eval, tmp_path, monkeypatch) -> None:
    out = tmp_path / "run.json"
    run_main(answer_eval, monkeypatch, "--only", "q01", "--json", str(out))
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["config"]["verifier"] == "off"
    assert report["summary"]["verifier_checked"] == 0
