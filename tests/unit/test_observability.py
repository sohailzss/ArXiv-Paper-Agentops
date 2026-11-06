from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.config import LogfireSettings, Settings
from src.services import tracing
from src.services.observability import _send_mode, setup_logfire
from src.services.tracing import LangfuseTracer


def test_logfire_off_without_token_or_when_disabled():
    assert setup_logfire(Settings(logfire=LogfireSettings(enabled=True, token=""))) is False
    assert setup_logfire(Settings(logfire=LogfireSettings(enabled=False, token="pylf_x"))) is False


def test_send_mode_parsing():
    assert _send_mode("true") is True and _send_mode("False") is False
    assert _send_mode("if-token-present") == "if-token-present" and _send_mode("weird") == "if-token-present"


class FakeObservation:
    def __init__(self, trace_id):
        self.trace_id = trace_id

    def start_observation(self, **kw):
        return FakeObservation(self.trace_id)

    def update(self, **kw):
        pass

    def end(self):
        pass


class FakeLangfuse:
    """Records whether an OpenTelemetry span was active when the root observation was created."""

    def __init__(self):
        self.calls = []
        self.active_span_at_start = []

    def start_observation(self, **kw):
        from opentelemetry import trace as otel_trace

        self.calls.append(kw)
        self.active_span_at_start.append(otel_trace.get_current_span().get_span_context().is_valid)
        return FakeObservation("f" * 32)


def in_request_span(fn):
    provider = TracerProvider()
    with provider.get_tracer("test").start_as_current_span("GET /api/v1/ask"):
        return fn()


def test_langfuse_root_never_inherits_the_request_span():
    client = FakeLangfuse()
    trace = in_request_span(lambda: LangfuseTracer(client, "test").start_trace("rag.ask", input={"query": "q"}))

    # Created in an empty context: no parent, so Langfuse sees a true root (not a dangling child).
    assert client.active_span_at_start == [False]
    call = client.calls[0]
    assert call["name"] == "rag.ask" and call["as_type"] == "chain" and call["input"] == {"query": "q"}
    assert "trace_context" not in call
    assert trace.trace_id == "f" * 32


def test_request_context_is_restored_after_starting_a_trace():
    from opentelemetry import trace as otel_trace

    def check():
        before = otel_trace.get_current_span()
        LangfuseTracer(FakeLangfuse(), "test").start_trace("rag.ask")
        return otel_trace.get_current_span() is before

    assert in_request_span(check)


def test_request_span_gets_langfuse_trace_id(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("GET /api/v1/ask"):
        LangfuseTracer(FakeLangfuse(), "test").start_trace("rag.ask")

    (span,) = exporter.get_finished_spans()
    assert span.attributes["langfuse.trace_id"] == "f" * 32


def test_link_is_noop_without_active_span():
    tracing._link_to_request_span("abc")  # must not raise
    tracing._link_to_request_span(None)
