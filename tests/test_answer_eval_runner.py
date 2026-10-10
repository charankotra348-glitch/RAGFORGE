"""Offline smoke test for tests/evaluation/answer_eval.py.

Retrieval and Ollama are stubbed, so this checks the runner's plumbing and
report format, not model quality.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ragforge.app.retrieval.types import SearchResult

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "evaluation" / "answer_eval.py"


@pytest.fixture()
def answer_eval(monkeypatch):
    spec = importlib.util.spec_from_file_location("answer_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    questions = {q["question"]: q for q in json.loads(module.DEFAULT_DATASET.read_text(encoding="utf-8"))["questions"]}

    class FakeSearcher:
        def search(self, query, top_k=5):
            q = questions.get(query)
            evidence = (q or {}).get("evidence") or ["unrelated clause"]
            return [SearchResult(text=" ".join(evidence), metadata={"page_number": 1, "section_number": "1"})]

    def fake_generate(settings, query, results, **kwargs):
        q = questions.get(query)
        if q is None or not q["answerable"]:
            return "The information is not available in the provided documents.", {"prompt_eval_count": 100, "eval_count": 10}
        return f"{q['expected']} [Source 1]", {"prompt_eval_count": 100, "eval_count": 10}

    monkeypatch.setattr(module, "get_hybrid_searcher", lambda settings: FakeSearcher())
    monkeypatch.setattr(module, "generate_answer_with_stats", fake_generate)
    return module


def test_runner_scores_perfect_stub_answers(answer_eval, tmp_path, monkeypatch, capsys) -> None:
    out = tmp_path / "run.json"
    monkeypatch.setattr(sys, "argv", ["answer_eval.py", "--model", "stub", "--json", str(out), "--quiet"])
    answer_eval.main()

    report = json.loads(out.read_text(encoding="utf-8"))
    s = report["summary"]
    assert s["n_answerable"] == 28 and s["n_unanswerable"] == 7
    assert s["mean_fact_recall"] == 1.0 and s["complete"] == 28
    assert s["correct_refusals"] == 7 and s["false_refusals"] == 0
    assert s["citations_valid"] == 35 and s["citation_supported"] == 28
    assert "mean fact recall" in capsys.readouterr().out


def test_compare_lists_changed_questions(answer_eval, tmp_path, monkeypatch, capsys) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    monkeypatch.setattr(sys, "argv", ["answer_eval.py", "--model", "stub", "--json", str(a), "--quiet", "--only", "q01", "q06"])
    answer_eval.main()
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


def test_rescore_recomputes_from_saved_answers(answer_eval, tmp_path, monkeypatch, capsys) -> None:
    run, rescored = tmp_path / "run.json", tmp_path / "rescored.json"
    monkeypatch.setattr(sys, "argv", ["answer_eval.py", "--model", "stub", "--json", str(run), "--quiet", "--only", "q06", "q35"])
    answer_eval.main()
    report = json.loads(run.read_text(encoding="utf-8"))
    assert all("contains_evidence" in src for row in report["questions"] for src in row["sources"])
    assert "system_prompt" in report["config"]

    # Simulate a stored answer that the old checker scored differently.
    report["questions"][0]["answer"] = "The information is not available in the provided documents."
    report["questions"][0]["score"]["fact_recall"] = 1.0
    run.write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["answer_eval.py", "--rescore", str(run), "--json", str(rescored)])
    answer_eval.main()
    new = json.loads(rescored.read_text(encoding="utf-8"))
    q06 = new["questions"][0]["score"]
    assert q06["fact_recall"] == 0.0 and q06["false_refusal"] is True
    assert new["config"]["rescored_at"]
    assert new["questions"][0]["answer"] == report["questions"][0]["answer"]
