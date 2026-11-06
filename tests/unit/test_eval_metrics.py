import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from src.eval import metrics as m

# ---------- retrieval ----------

RANKED = m.paper_ranking(["a", "a", "b", "c", "b", "d"])  # chunk hits -> papers


def test_paper_ranking_dedupes_keeping_best_rank():
    assert RANKED == ["a", "b", "c", "d"]


def test_hit_recall_precision():
    assert m.hit_at_k(RANKED, ["c"], 2) == 0.0 and m.hit_at_k(RANKED, ["c"], 3) == 1.0
    assert m.recall_at_k(RANKED, ["b", "z"], 5) == 0.5
    assert m.precision_at_k(RANKED, ["a", "b"], 4) == 0.5
    assert m.recall_at_k(RANKED, [], 5) == 0.0


def test_mrr_and_ndcg():
    assert m.reciprocal_rank(RANKED, ["c"]) == pytest.approx(1 / 3)
    assert m.reciprocal_rank(RANKED, ["z"]) == 0.0
    assert m.ndcg_at_k(["a", "b"], ["a"], 5) == 1.0
    assert m.ndcg_at_k(["x", "a"], ["a"], 5) == pytest.approx(1 / 1.584962500721156)
    assert m.ndcg_at_k(["x"], ["a"], 5) == 0.0


# ---------- answers ----------


def test_fact_recall_accepts_alternatives():
    groups = [["91%", "91 percent"], ["360"], ["six"]]
    assert m.fact_recall("It reached 91 percent over 360 trials.", groups) == pytest.approx(2 / 3)
    assert m.missing_facts("It reached 91 percent over 360 trials.", groups) == ["six"]
    assert m.fact_recall("anything", []) is None


def test_forbidden_and_citations():
    assert m.forbidden_hits("Contact jane.doe@example.com", ["jane.doe@example.com", "sk-"]) == ["jane.doe@example.com"]
    assert m.citations("A [2] B [1][2] C [9]", 3) == [2, 1]
    assert m.citation_precision(["a", "b"], ["a"]) == 0.5
    assert m.citation_precision([], ["a"]) is None


@pytest.mark.parametrize(
    "answer",
    [
        "I searched the indexed papers but couldn't find passages that answer this question.",
        "The passages do not contain information about protein folding.",
        "I can only answer questions about the indexed research papers.",
        "Sorry, I can't help with that request.",
    ],
)
def test_refusals_are_detected(answer):
    assert m.is_refusal(answer, citation_count=0)


def test_cited_answers_are_never_refusals():
    assert not m.is_refusal("The passages do not contain X, but [1] shows Y.", citation_count=1)
    assert not m.is_refusal("HALO reaches 91% [1].", citation_count=1)


def test_percentile_and_avg():
    assert m.percentile([1, 2, 3, 4, 10], 50) == 3 and m.percentile([1, 2, 3, 4, 10], 95) == 10
    assert m.percentile([], 50) is None
    assert m.avg([1.0, None, 3.0]) == 2.0 and m.avg([None]) is None


# ---------- runner logic (against a mocked API) ----------


def load_runner():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = {"number": 1, "arxiv_id": "2611.06908", "title": "HALO", "section_titles": ["Abstract"], "text": "91%"}


def mock_client(answer_payload: dict, search_ids=("2611.06908",)) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/hybrid-search"):
            return httpx.Response(200, json={"hits": [{"arxiv_id": a} for a in search_ids]})
        return httpx.Response(200, json=answer_payload)

    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))


CASE = {
    "id": "gd-x",
    "category": "numeric_fact",
    "question": "HALO success rate?",
    "expected": {
        "behavior": "answer",
        "route": "search",
        "relevant_arxiv_ids": ["2611.06908"],
        "must_include": [["91%"], ["360"]],
        "must_not_include": [],
        "guardrail_input": "none",
        "min_citations": 1,
    },
    "reference_answer": "91% over 360 trials.",
}


def test_runner_scores_a_good_answer():
    run = load_runner()
    payload = {
        "answer": "HALO reaches 91% over 360 trials [1].",
        "sources": [SOURCE],
        "route": "search",
        "steps": [{"node": "route"}, {"node": "retrieve"}, {"node": "grade"}, {"node": "generate"}],
        "llm_calls": 3,
        "guardrail": {"input": {"action": "none"}, "output": {"action": "none", "grounding": 0.98, "relevance": 1.0}},
    }
    r = run.evaluate_case(mock_client(payload), CASE, "agent", judge_model=None)

    assert r["passed"] and r["behavior_ok"] and r["route_ok"] and r["guardrail_ok"]
    assert r["retrieval"]["hit@1"] == 1.0 and r["retrieval"]["mrr"] == 1.0
    assert r["fact_recall"] == 1.0 and r["citation_precision"] == 1.0 and r["grounding"] == 0.98


def test_runner_flags_a_false_refusal():
    run = load_runner()
    payload = {"answer": "The passages do not contain this.", "sources": [SOURCE], "route": "search", "steps": []}
    r = run.evaluate_case(mock_client(payload), CASE, "agent", judge_model=None)
    assert not r["passed"] and r["refused"] and r["missing_facts"] == ["91%", "360"]


def test_gate_compares_in_the_right_direction():
    run = load_runner()
    thresholds = {"fact_recall": 0.8, "false_refusal_rate": 0.1}
    assert run.check_gate({"fact_recall": 0.9, "false_refusal_rate": 0.05}, thresholds) == []
    assert len(run.check_gate({"fact_recall": 0.7, "false_refusal_rate": 0.2}, thresholds)) == 2


def test_report_renders(tmp_path):
    run = load_runner()
    results = [run.evaluate_case(mock_client({"answer": "91% over 360 [1]", "sources": [SOURCE]}), CASE, "rag", None)]
    summary = run.summarize(results, "rag")
    md = run.markdown_report(
        {"mode": "rag", "base_url": "x", "dataset_version": "1", "timestamp": "t", "judge": None}, summary, results, []
    )
    assert "| gd-x | numeric_fact | yes |" in md
    json.dumps(summary)
