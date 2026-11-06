"""Offline checks on data/golden_dataset.json: catches a broken edit before an evaluation run.
(The evaluation itself needs the running API: uv run python scripts/run_eval.py)"""

import json
import re
from pathlib import Path

import pytest

DATASET = json.loads((Path(__file__).resolve().parents[2] / "data" / "golden_dataset.json").read_text(encoding="utf-8"))
GOLDENS = DATASET["goldens"]

CATEGORIES = {
    "factual_single",
    "numeric_fact",
    "full_text_section",
    "multi_paper",
    "paraphrase_vague",
    "keyword_exact",
    "semantic",
    "filtered",
    "unanswerable_in_scope",
    "off_topic",
    "prompt_injection",
    "pii",
}
BEHAVIORS = {"answer", "refuse_not_found", "direct", "blocked"}
ARXIV_ID = re.compile(r"^\d{4}\.\d{5}$")


def test_has_25_goldens_with_unique_ids():
    ids = [g["id"] for g in GOLDENS]
    assert len(GOLDENS) == 25
    assert len(set(ids)) == len(ids)
    assert ids == [f"gd-{i:03d}" for i in range(1, 26)]


def test_thresholds_cover_the_core_metrics():
    assert {
        "retrieval_hit_at_5",
        "retrieval_mrr",
        "fact_recall",
        "deepeval_correctness",
        "deepeval_faithfulness",
        "deepeval_answer_relevancy",
        "deepeval_contextual_recall",
        "behavior_accuracy",
        "citation_precision",
        "false_refusal_rate",
    } <= set(DATASET["thresholds"])


def test_covers_every_behavior_the_app_must_handle():
    assert {g["category"] for g in GOLDENS} == CATEGORIES
    assert {g["expected"]["behavior"] for g in GOLDENS} == BEHAVIORS


@pytest.mark.parametrize("golden", GOLDENS, ids=[g["id"] for g in GOLDENS])
def test_golden_is_well_formed(golden):
    exp = golden["expected"]
    assert golden["category"] in CATEGORIES
    assert golden["question"].strip() and golden["reference_answer"].strip()
    assert exp["behavior"] in BEHAVIORS
    assert exp["route"] in {"search", "direct", "blocked"}
    assert exp["guardrail_input"] in {"none", "blocked", "masked"}
    assert all(ARXIV_ID.match(a) for a in exp["relevant_arxiv_ids"])
    assert all(isinstance(group, list) and group for group in exp["must_include"])

    if exp["behavior"] == "answer":
        # Answerable questions need retrieval ground truth, checkable facts and at least one citation.
        assert exp["relevant_arxiv_ids"] and exp["must_include"] and exp["min_citations"] >= 1
        assert exp["min_citations"] <= len(exp["relevant_arxiv_ids"])
    else:
        assert exp["relevant_arxiv_ids"] == [] and exp["min_citations"] == 0

    for opt in golden.get("request", {}):
        assert opt in {"mode", "top_k", "categories", "date_from", "date_to"}


def test_facts_are_consistent_with_reference_answers():
    """Every required fact should appear in the reference answer (otherwise the golden contradicts itself)."""
    for g in GOLDENS:
        ref = g["reference_answer"].lower()
        for group in g["expected"]["must_include"]:
            assert any(alt.lower() in ref for alt in group), f"{g['id']}: {group} not in reference answer"
