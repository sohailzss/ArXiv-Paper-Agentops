import json

from fastapi.testclient import TestClient

from src.dependencies import get_rag
from src.main import app
from src.schemas.rag import AskResponse, Source
from src.services.llm.client import LLMError

SOURCE = Source(
    number=1,
    arxiv_id="2611.06908",
    title="Perceptive Humanoid Motion from Simulation",
    section_titles=["Abstract"],
    chunk_id="2611.06908_0",
    url="https://arxiv.org/abs/2611.06908",
    pdf_url="https://arxiv.org/pdf/2611.06908",
    text="We learn humanoid skills in simulation.",
)


class FakeRAG:
    def __init__(self, error: Exception | None = None, stream_error: Exception | None = None):
        self.error, self.stream_error = error, stream_error
        self.requests = []

    def answer(self, request):
        if self.error:
            raise self.error
        self.requests.append(request)
        return AskResponse(
            query=request.query,
            answer="They train in simulation [1].",
            sources=[SOURCE],
            cited=[1],
            llm_model="openai:gpt-test",
            embedding_model="openai:emb:4",
            retrieval_ms=12,
            generation_ms=340,
        )

    def stream(self, request):
        yield {"type": "sources", "sources": [SOURCE.model_dump()], "embedding_model": "e", "retrieval_ms": 1}
        yield {"type": "token", "text": "Line one\nline two [1]"}
        if self.stream_error:
            raise self.stream_error
        yield {"type": "done", "cited": [1], "llm_model": "openai:gpt-test", "retrieval_ms": 1, "generation_ms": 2}


def with_rag(rag):
    app.dependency_overrides[get_rag] = lambda: rag
    return TestClient(app)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def teardown_function():
    app.dependency_overrides.clear()


def test_ask_returns_answer_and_sources():
    rag = FakeRAG()
    response = with_rag(rag).post("/api/v1/ask", json={"query": "How do robots learn to walk?", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "They train in simulation [1]." and body["cited"] == [1]
    assert body["sources"][0]["url"] == "https://arxiv.org/abs/2611.06908"
    assert rag.requests[0].top_k == 3 and rag.requests[0].mode == "hybrid"


def test_ask_validation():
    client = with_rag(FakeRAG())
    assert client.post("/api/v1/ask", json={"query": ""}).status_code == 422
    assert client.post("/api/v1/ask", json={"query": "x", "top_k": 50}).status_code == 422


def test_ask_llm_failure_is_503():
    response = with_rag(FakeRAG(error=LLMError("quota"))).post("/api/v1/ask", json={"query": "q"})
    assert response.status_code == 503 and "quota" in response.json()["detail"]


def test_stream_sends_sse_events_with_json_data():
    response = with_rag(FakeRAG()).post("/api/v1/ask/stream", json={"query": "q"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["sources", "token", "done"]
    assert events[1][1]["text"] == "Line one\nline two [1]"  # newline survives inside JSON


def test_stream_error_becomes_error_event():
    response = with_rag(FakeRAG(stream_error=LLMError("connection reset"))).post(
        "/api/v1/ask/stream", json={"query": "q"}
    )

    events = parse_sse(response.text)
    assert events[-1][0] == "error" and "connection reset" in events[-1][1]["detail"]


def test_unavailable_when_not_configured():
    app.state.rag = None
    app.state.rag_error = "LLM_PROVIDER=bedrock but BEDROCK_MODEL_ID is empty"
    response = TestClient(app).post("/api/v1/ask", json={"query": "q"})
    assert response.status_code == 503 and "BEDROCK_MODEL_ID" in response.json()["detail"]
