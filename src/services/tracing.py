"""Langfuse tracing behind a tiny interface, with a no-op fallback.

One trace per question:  rag.ask
                           ├─ cache        (hit or miss)
                           ├─ retrieval    (retriever: query, mode, top_k -> chunk ids)
                           └─ llm          (generation: prompt, answer, model, tokens, time to first token)

Observations are created with explicit parents (not "current span" context), because the streaming
endpoint runs its generator across threads, where OpenTelemetry's implicit context would break.
Tracing never breaks a request: every Langfuse call is wrapped, and failures are logged once.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from src.config import Settings

logger = logging.getLogger(__name__)


class Observation:
    """No-op observation; LangfuseObservation overrides these."""

    def update(self, **fields: Any) -> None:
        pass

    def end(self, **fields: Any) -> None:
        pass


class Trace(Observation):
    trace_id: str | None = None

    def span(self, name: str, as_type: str = "span", **fields: Any) -> Observation:
        return Observation()

    def generation(self, name: str, **fields: Any) -> Observation:
        return Observation()


class Tracer:
    enabled = False

    def start_trace(self, name: str, **fields: Any) -> Trace:
        return Trace()

    def score(self, trace_id: str | None, name: str, value: float, comment: str | None = None) -> None:
        """Attach a numeric score (e.g. an online eval metric) to a trace. No-op without Langfuse."""

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - depends on network/SDK state
        logger.warning("Langfuse call failed (ignored): %s", exc)
        return None


class LangfuseObservation(Observation):
    def __init__(self, observation):
        self._obs = observation

    def update(self, **fields: Any) -> None:
        if self._obs is not None:
            _safe(self._obs.update, **fields)

    def end(self, **fields: Any) -> None:
        if self._obs is None:
            return
        if fields:
            _safe(self._obs.update, **fields)
        _safe(self._obs.end)


class LangfuseTrace(Trace, LangfuseObservation):
    def __init__(self, root):
        LangfuseObservation.__init__(self, root)
        self.trace_id = getattr(root, "trace_id", None)

    def span(self, name: str, as_type: str = "span", **fields: Any) -> Observation:
        if self._obs is None:
            return Observation()
        return LangfuseObservation(_safe(self._obs.start_observation, name=name, as_type=as_type, **fields))

    def generation(self, name: str, **fields: Any) -> Observation:
        return self.span(name, as_type="generation", **fields)


class LangfuseTracer(Tracer):
    enabled = True

    def __init__(self, client, environment: str):
        self.client = client
        self.environment = environment

    def start_trace(self, name: str, tags: list[str] | None = None, **fields: Any) -> Trace:
        from langfuse import propagate_attributes
        from opentelemetry import context as otel_context

        def start():
            # Start the root in an EMPTY OpenTelemetry context, so it's a true root with a new trace id.
            # Otherwise it inherits the active span (Logfire's FastAPI request span) as its parent, which
            # Langfuse never receives, and every trace points at a missing parent. (Passing a
            # trace_context with only a trace_id isn't enough: the SDK then invents a placeholder parent.)
            # Attach/detach happen in this one synchronous block, so it's safe in streaming generators too.
            token = otel_context.attach(otel_context.Context())
            try:
                # propagate_attributes names the trace and tags it; the root is created inside it.
                with propagate_attributes(trace_name=name, tags=tags or [], environment=self.environment):
                    return self.client.start_observation(name=name, as_type="chain", **fields)
            finally:
                otel_context.detach(token)

        trace = LangfuseTrace(_safe(start))
        _link_to_request_span(trace.trace_id)
        return trace

    def score(self, trace_id: str | None, name: str, value: float, comment: str | None = None) -> None:
        if trace_id:
            _safe(
                self.client.create_score,
                name=name,
                value=float(value),
                trace_id=trace_id,
                data_type="NUMERIC",
                comment=(comment or "")[:1000] or None,
                environment=self.environment,
            )

    def flush(self) -> None:
        _safe(self.client.flush)

    def shutdown(self) -> None:
        _safe(self.client.shutdown)


def build_tracer(settings: Settings) -> Tracer:
    if not settings.langfuse_enabled:
        return Tracer()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        logger.info("Langfuse keys not set; tracing disabled")
        return Tracer()
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider

    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        environment=settings.environment,
        release=settings.app_version,
        # Own provider: Langfuse spans stay out of Logfire (which owns the global provider), and
        # Logfire's HTTP/SQL spans never get exported to Langfuse.
        tracer_provider=TracerProvider(),
    )
    return LangfuseTracer(client, settings.environment)


def _link_to_request_span(trace_id: str | None) -> None:
    """Tag the current (Logfire) request span with the Langfuse trace id, so you can go from a slow
    request in Logfire to its prompts in Langfuse. No-op when no span is active."""
    if not trace_id:
        return
    try:
        from opentelemetry import trace as otel_trace

        span = otel_trace.get_current_span()
        if span.is_recording():
            span.set_attribute("langfuse.trace_id", trace_id)
    except Exception:  # pragma: no cover
        pass


def now() -> datetime:
    return datetime.now(UTC)


def usage_details(usage: dict) -> dict[str, int] | None:
    """Map provider usage (OpenAI prompt/completion_tokens, Bedrock input/outputTokens) to Langfuse's keys."""
    if not usage:
        return None
    mapped = {
        "input": usage.get("prompt_tokens", usage.get("inputTokens")),
        "output": usage.get("completion_tokens", usage.get("outputTokens")),
        "total": usage.get("total_tokens", usage.get("totalTokens")),
    }
    return {k: int(v) for k, v in mapped.items() if v is not None} or None
