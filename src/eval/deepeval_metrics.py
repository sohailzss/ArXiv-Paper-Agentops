"""DeepEval metrics for the offline golden evaluation (scripts/run_eval.py).

Scores are 0..1. With a reference answer available (offline), all six apply:

  correctness           GEval: does the answer match the reference answer's facts (and, for questions that
                        should be declined, did it decline)?
  faithfulness          every claim in the answer is supported by the passages the LLM saw
  answer_relevancy      the answer addresses the question
  contextual_precision  relevant passages are ranked above irrelevant ones
  contextual_recall     the passages contain what the reference answer needs
  contextual_relevancy  share of passage content that is relevant to the question

Online (production) uses only the reference-free ones, scored by Confident AI (services/online_eval.py).
"""

import asyncio

from src.services import deepeval_setup

deepeval_setup.configure()  # before importing deepeval

from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402

RAG_METRICS = ("faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall", "contextual_relevancy")
ALL_METRICS = ("correctness", *RAG_METRICS)


# Explicit steps instead of a one-line criterion. With only a criterion, GEval penalised correct answers
# for not repeating every detail of the reference (0.71 for a fully correct answer) and correct refusals
# for different wording (0.48). The reference answers carry extra context on purpose; they aren't checklists.
ANSWER_STEPS = [
    "Identify exactly what the input question asks for.",
    "Check whether the actual output answers that, and whether its facts and numbers agree with the expected output.",
    "Heavily penalise any fact or number in the actual output that contradicts the expected output.",
    "Penalise omitting information the question explicitly asks for.",
    "Do NOT penalise omitting details from the expected output that the question did not ask for.",
    "Ignore citation markers like [1], wording and formatting.",
    "If the actual output declines, or says the information is not available, the score must be the minimum.",
]
DECLINE_STEPS = [
    "The expected output describes a request the system should decline (off-topic, not covered by the indexed "
    "papers, or unsafe).",
    "Check whether the actual output declines or says the information is not available, without giving a "
    "substantive answer.",
    "A clear decline scores the maximum regardless of wording, length or suggestions it adds.",
    "Any substantive answer to the request, or any invented facts, scores the minimum.",
]


def correctness_metric(model: str, should_answer: bool) -> GEval:
    return GEval(
        name="correctness",
        evaluation_steps=ANSWER_STEPS if should_answer else DECLINE_STEPS,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=model,
        async_mode=True,
    )


def build_metrics(model: str, should_answer: bool, has_passages: bool) -> dict:
    metrics = {"correctness": correctness_metric(model, should_answer)}
    if should_answer and has_passages:  # the RAG metrics need passages and a reference answer to compare against
        metrics |= {
            "faithfulness": FaithfulnessMetric(model=model, async_mode=True),
            "answer_relevancy": AnswerRelevancyMetric(model=model, async_mode=True),
            "contextual_precision": ContextualPrecisionMetric(model=model, async_mode=True),
            "contextual_recall": ContextualRecallMetric(model=model, async_mode=True),
            "contextual_relevancy": ContextualRelevancyMetric(model=model, async_mode=True),
        }
    return metrics


async def _measure_all(metrics: dict, test_case: LLMTestCase) -> dict:
    async def one(name, metric):
        try:
            await metric.a_measure(test_case, _show_indicator=False)
            return name, {"score": round(float(metric.score), 3), "reason": (metric.reason or "")[:300]}
        except Exception as exc:  # one failing metric shouldn't lose the others
            return name, {"score": None, "reason": f"error: {type(exc).__name__}: {str(exc)[:200]}"}

    return dict(await asyncio.gather(*(one(n, m) for n, m in metrics.items())))


def score_case(
    *, question: str, answer: str, reference: str, passages: list[str], should_answer: bool, model: str
) -> dict:
    """Returns {metric_name: {"score": 0..1 | None, "reason": str}}."""
    test_case = LLMTestCase(
        input=question,
        actual_output=deepeval_setup.strip_citations(answer),  # [n] markers aren't content
        expected_output=reference,
        retrieval_context=passages or None,
    )
    metrics = build_metrics(model, should_answer, has_passages=bool(passages))
    return asyncio.run(_measure_all(metrics, test_case))
