"""Online evaluation of live answers with DeepEval; scores are attached to the Langfuse trace.

A sample of finished answers (ONLINE_EVAL__SAMPLE_RATE, default 20%) is scored in a background thread
with DeepEval's reference-free metrics, since production has no reference answers:

  faithfulness          are the answer's claims supported by the passages the LLM saw?
  answer_relevancy      does the answer address the question?
  contextual_relevancy  were the retrieved passages about the question?

Each score lands on the request's Langfuse trace as `deepeval_<metric>` (0..1, with DeepEval's reason
as the comment), next to the prompt, answer, tokens and cost. Langfuse can then chart them over time
and filter low-scoring answers.

- No added latency: users get their answer first; scoring happens afterwards in a small thread pool.
- Bounded: at most ONLINE_EVAL__MAX_PENDING jobs wait; beyond that, samples are dropped (and logged),
  so a traffic burst can't pile up unbounded judge calls.
- Only answers grounded in passages are scored; refusals, direct replies, guardrail blocks and cache
  hits have nothing for these metrics to judge.
- The judge is ONLINE_EVAL__JUDGE_MODEL (default gpt-4o: gpt-4o-mini misjudged answers, gpt-4.1 hung
  inside DeepEval's faithfulness/relevancy calls). Needs OPENAI_API_KEY. Telemetry is opted out.
"""

import asyncio
import logging
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor

from src.config import Settings
from src.services import deepeval_setup
from src.services.tracing import Tracer

logger = logging.getLogger(__name__)

SUPPORTED_METRICS = ("faithfulness", "answer_relevancy", "contextual_relevancy")


class OnlineEvaluator:
    def __init__(
        self,
        tracer: Tracer,
        judge_model: str,
        metrics: tuple[str, ...] = SUPPORTED_METRICS,
        sample_rate: float = 0.2,
        max_pending: int = 20,
        workers: int = 2,
        rng: random.Random | None = None,
    ):
        self.tracer = tracer
        self.judge_model = judge_model
        self.metrics = metrics
        self.sample_rate = sample_rate
        self.max_pending = max_pending
        self._rng = rng or random.Random()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="online-eval")
        self._pending = 0
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings, tracer: Tracer) -> "OnlineEvaluator | None":
        cfg = settings.online_eval
        if not cfg.enabled:
            logger.info("Online evals off (disabled)")
            return None
        if not tracer.enabled:
            logger.info("Online evals off (scores are stored in Langfuse, which isn't configured)")
            return None
        if not settings.openai_api_key:
            logger.info("Online evals off (the DeepEval judge needs OPENAI_API_KEY)")
            return None
        deepeval_setup.configure()  # telemetry off, fail-fast judge timeouts; before deepeval is imported
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)  # DeepEval's judge reads it from env
        unknown = set(cfg.metrics) - set(SUPPORTED_METRICS)
        if unknown:
            logger.warning("Ignoring unsupported online metrics: %s", sorted(unknown))
        metrics = tuple(m for m in cfg.metrics if m in SUPPORTED_METRICS)
        logger.info(
            "Online evals on (DeepEval -> Langfuse scores; %s; judge %s; sample %.0f%%)",
            ", ".join(metrics),
            cfg.judge_model,
            cfg.sample_rate * 100,
        )
        return cls(tracer, cfg.judge_model, metrics, cfg.sample_rate, cfg.max_pending)

    # ---------- public ----------

    def submit(self, *, trace_id: str | None, question: str, answer: str, passages: list[str]) -> bool:
        """Maybe schedule scoring for one finished answer. Returns True if it was scheduled. Never raises."""
        if not trace_id or not passages or not answer:
            return False
        if self._rng.random() >= self.sample_rate:
            return False
        with self._lock:
            if self._pending >= self.max_pending:
                logger.warning("Online eval sample dropped: %d jobs already pending", self._pending)
                return False
            self._pending += 1
        try:
            self._pool.submit(self._run, trace_id, question, answer, passages)
        except RuntimeError:  # pool already shut down
            with self._lock:
                self._pending -= 1
            return False
        return True

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---------- worker ----------

    def _run(self, trace_id: str, question: str, answer: str, passages: list[str]) -> None:
        try:
            for name, score, reason in self.score(question, answer, passages):
                if score is not None:
                    self.tracer.score(trace_id, f"deepeval_{name}", score, reason)
            self.tracer.flush()
        except Exception as exc:  # a failed evaluation must never affect the app
            logger.warning("Online eval failed for trace %s: %s", trace_id, exc)
        finally:
            with self._lock:
                self._pending -= 1

    def score(self, question: str, answer: str, passages: list[str]) -> list[tuple[str, float | None, str]]:
        """Run the DeepEval metrics. Returns [(metric, score 0..1 or None, reason)]."""
        from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        factories = {
            "faithfulness": FaithfulnessMetric,
            "answer_relevancy": AnswerRelevancyMetric,
            "contextual_relevancy": ContextualRelevancyMetric,
        }
        test_case = LLMTestCase(
            input=question, actual_output=deepeval_setup.strip_citations(answer), retrieval_context=passages
        )
        metrics = {name: factories[name](model=self.judge_model, async_mode=True) for name in self.metrics}

        async def one(name, metric):
            try:
                await metric.a_measure(test_case, _show_indicator=False)
                return name, float(metric.score), metric.reason or ""
            except Exception as exc:
                logger.warning("DeepEval %s failed: %s", name, exc)
                return name, None, f"error: {exc}"

        async def run_all():
            return await asyncio.gather(*(one(n, m) for n, m in metrics.items()))

        # Each worker thread gets its own event loop, separate from the API's.
        return list(asyncio.run(run_all()))
