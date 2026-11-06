import os
import random
import threading
from datetime import UTC, datetime

import pytest

from src.config import OnlineEvalSettings, RAGSettings, Settings
from src.schemas.rag import AskRequest
from src.schemas.search import ChunkHit, HybridSearchResponse
from src.services.llm.client import LLMResult
from src.services.online_eval import OnlineEvaluator
from src.services.rag import RAGService
from src.services.tracing import Tracer

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")


class RecordingTracer(Tracer):
    enabled = True

    def __init__(self):
        self.scores: list[tuple] = []
        self.flushed = 0

    def score(self, trace_id, name, value, comment=None):
        self.scores.append((trace_id, name, value, comment))

    def flush(self):
        self.flushed += 1


class FixedScores(OnlineEvaluator):
    """Skips the real judge: returns canned metric results."""

    def __init__(self, *args, results=None, error=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = results or [("faithfulness", 1.0, "all supported"), ("answer_relevancy", 0.9, "on topic")]
        self.error = error
        self.calls = []

    def score(self, question, answer, passages):
        self.calls.append((question, answer, passages))
        if self.error:
            raise self.error
        return self.results


def run(ev: OnlineEvaluator, **kwargs) -> bool:
    scheduled = ev.submit(**{"trace_id": "t1", "question": "q", "answer": "a [1]", "passages": ["p"], **kwargs})
    ev._pool.shutdown(wait=True)  # wait for the background job in tests
    return scheduled


# ---------- scheduling ----------


def test_scores_land_on_the_langfuse_trace():
    tracer = RecordingTracer()
    ev = FixedScores(tracer, "gpt-4.1", sample_rate=1.0)

    assert run(ev)
    assert tracer.scores == [
        ("t1", "deepeval_faithfulness", 1.0, "all supported"),
        ("t1", "deepeval_answer_relevancy", 0.9, "on topic"),
    ]
    assert tracer.flushed == 1 and ev._pending == 0


def test_sampling_rate_is_respected():
    ev = FixedScores(
        RecordingTracer(), "m", sample_rate=0.2, max_pending=10_000, rng=random.Random(0)
    )  # cap tested separately
    scheduled = sum(ev.submit(trace_id="t", question="q", answer="a", passages=["p"]) for _ in range(1000))
    ev._pool.shutdown(wait=True)
    assert 150 < scheduled < 250


@pytest.mark.parametrize("missing", [{"trace_id": None}, {"passages": []}, {"answer": ""}])
def test_nothing_to_score_is_skipped(missing):
    ev = FixedScores(RecordingTracer(), "m", sample_rate=1.0)
    assert run(ev, **missing) is False and ev.calls == []


def test_backlog_is_capped():
    gate = threading.Event()

    class Slow(FixedScores):
        def score(self, *a):
            gate.wait(5)
            return []

    ev = Slow(RecordingTracer(), "m", sample_rate=1.0, max_pending=2, workers=1)
    results = [ev.submit(trace_id="t", question="q", answer="a", passages=["p"]) for _ in range(5)]
    gate.set()
    ev._pool.shutdown(wait=True)
    assert results == [True, True, False, False, False] and ev._pending == 0


def test_failures_never_escape_and_free_the_slot():
    tracer = RecordingTracer()
    ev = FixedScores(tracer, "m", sample_rate=1.0, error=RuntimeError("judge down"))
    assert run(ev)
    assert tracer.scores == [] and ev._pending == 0


def test_metrics_without_a_score_are_not_sent():
    tracer = RecordingTracer()
    ev = FixedScores(tracer, "m", sample_rate=1.0, results=[("faithfulness", None, "error: timeout")])
    run(ev)
    assert tracer.scores == []


# ---------- configuration ----------


def test_off_when_disabled_without_langfuse_or_without_openai_key():
    on = RecordingTracer()
    assert (
        OnlineEvaluator.from_settings(Settings(openai_api_key="sk", online_eval=OnlineEvalSettings(enabled=False)), on)
        is None
    )
    assert OnlineEvaluator.from_settings(Settings(openai_api_key="sk"), Tracer()) is None  # Langfuse off
    assert OnlineEvaluator.from_settings(Settings(openai_api_key=""), on) is None


def test_from_settings_uses_configured_judge_and_metrics():
    settings = Settings(
        openai_api_key="sk",
        online_eval=OnlineEvalSettings(sample_rate=0.5, judge_model="gpt-4.1", metrics=["faithfulness", "bogus"]),
    )
    ev = OnlineEvaluator.from_settings(settings, RecordingTracer())
    assert ev.judge_model == "gpt-4.1" and ev.metrics == ("faithfulness",) and ev.sample_rate == 0.5
    ev.shutdown()


def test_score_uses_real_deepeval_metrics_with_the_judge_model(monkeypatch):
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric

    seen = []

    async def fake_measure(self, test_case, _show_indicator=True, **kwargs):
        seen.append((type(self).__name__, self.model.get_model_name(), test_case.retrieval_context))
        self.score, self.reason = 0.75, "ok"
        return 0.75

    for cls in (FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric):
        monkeypatch.setattr(cls, "a_measure", fake_measure)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    ev = OnlineEvaluator(RecordingTracer(), "gpt-4.1")
    results = ev.score("q", "a [1]", ["passage"])

    assert [(n, s) for n, s, _ in results] == [
        ("faithfulness", 0.75),
        ("answer_relevancy", 0.75),
        ("contextual_relevancy", 0.75),
    ]
    assert {name for name, _, _ in seen} == {"FaithfulnessMetric", "AnswerRelevancyMetric", "ContextualRelevancyMetric"}
    assert all(model == "gpt-4.1" and ctx == ["passage"] for _, model, ctx in seen)
    ev.shutdown()


# ---------- which answers RAG sends ----------


class Recorder:
    def __init__(self):
        self.calls = []

    def submit(self, **kwargs):
        self.calls.append(kwargs)
        return True


def hit(n):
    return ChunkHit(
        chunk_id=f"p{n}_0",
        arxiv_id=f"p{n}",
        chunk_index=0,
        section_titles=["Abstract"],
        text=f"Finding {n}.",
        title=f"Paper {n}",
        authors=["A"],
        categories=["cs.AI"],
        published_date=datetime(2026, 11, 1, tzinfo=UTC),
        pdf_url="x",
        score=0.03,
    )


class FakeSearch:
    class chunks:  # noqa: N801
        name = "c"

        @staticmethod
        def count():
            return 1

    class embedder:  # noqa: N801
        model_id = "openai:emb:4"

    def __init__(self, hits):
        self.hits = hits

    def search(self, query, **kw):
        return HybridSearchResponse(
            query=query, mode=kw["mode"], embedding_model="e", total=len(self.hits), hits=self.hits
        )


class FakeLLM:
    model_id, model, provider = "openai:gpt-test", "gpt-test", "openai"

    def __init__(self, text):
        self.text = text

    def generate(self, system, messages):
        return LLMResult(text=self.text, model="m")

    def stream(self, system, messages):
        yield self.text


class TraceIdTracer(Tracer):
    enabled = True

    def start_trace(self, name, **fields):
        from src.services.tracing import Trace

        t = Trace()
        t.trace_id = "trace-42"
        return t


def rag(text, hits=None):
    recorder = Recorder()
    service = RAGService(
        FakeSearch([hit(1)] if hits is None else hits),
        FakeLLM(text),
        Settings(rag=RAGSettings(top_k=1)),
        tracer=TraceIdTracer(),
        online_eval=recorder,
    )
    return service, recorder


def test_grounded_answers_are_queued_with_trace_id_and_exact_passages():
    service, recorder = rag("Finding one [1].")
    service.answer(AskRequest(query="q"))
    assert recorder.calls == [
        {
            "trace_id": "trace-42",
            "question": "q",
            "answer": "Finding one [1].",
            "passages": ['[1] "Paper 1" (arXiv:p1), section: Abstract\nFinding 1.'],
        }
    ]


def test_streamed_answers_are_queued_after_completion():
    service, recorder = rag("Finding one [1].")
    list(service.stream(AskRequest(query="q")))
    assert len(recorder.calls) == 1


@pytest.mark.parametrize(("text", "hits"), [("The passages do not contain this.", None), ("Finding one [1].", [])])
def test_answers_with_nothing_to_judge_are_skipped(text, hits):
    service, recorder = rag(text, hits)
    service.answer(AskRequest(query="q"))
    assert recorder.calls == []


def test_citation_markers_are_stripped_before_scoring(monkeypatch):
    from deepeval.metrics import AnswerRelevancyMetric

    from src.services.deepeval_setup import strip_citations

    assert strip_citations("42 physicists created them [1][2].") == "42 physicists created them."
    seen = []

    async def fake_measure(self, test_case, _show_indicator=True, **kwargs):
        seen.append(test_case.actual_output)
        self.score, self.reason = 1.0, "ok"

    monkeypatch.setattr(AnswerRelevancyMetric, "a_measure", fake_measure)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    ev = OnlineEvaluator(RecordingTracer(), "gpt-4.1", metrics=("answer_relevancy",))
    ev.score("How many?", "42 physicists [1].", ["p"])
    assert seen == ["42 physicists."]
    ev.shutdown()
