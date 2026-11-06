import json

from fastapi.testclient import TestClient

from src.dependencies import get_agent
from src.main import app
from src.schemas.agent import AgentResponse, AgentStep
from src.services.llm.client import LLMError

RESPONSE = AgentResponse(
    query="q",
    answer="Answer [1].",
    route="search",
    search_queries=["q", "better q"],
    steps=[AgentStep(node="route", detail={"route": "search"}), AgentStep(node="generate")],
    sources=[],
    cited=[1],
    llm_model="openai:gpt-test",
    embedding_model="openai:emb:4",
    llm_calls=4,
    total_ms=1200,
)


class FakeAgent:
    def __init__(self, error=None):
        self.error = error

    def run(self, request):
        if self.error:
            raise self.error
        return RESPONSE

    def stream(self, request):
        yield {"type": "step", "node": "route", "detail": {"route": "search"}}
        if self.error:
            raise self.error
        yield {"type": "token", "text": "Answer [1]."}
        yield {"type": "done", "cited": [1]}


def client(agent):
    app.dependency_overrides[get_agent] = lambda: agent
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_agent_ask_returns_steps_and_queries():
    body = client(FakeAgent()).post("/api/v1/agent/ask", json={"query": "q"}).json()
    assert body["search_queries"] == ["q", "better q"]
    assert [s["node"] for s in body["steps"]] == ["route", "generate"]


def test_agent_stream_and_errors():
    text = client(FakeAgent()).post("/api/v1/agent/stream", json={"query": "q"}).text
    assert [line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("event:")] == [
        "step",
        "token",
        "done",
    ]

    text = client(FakeAgent(error=LLMError("rate limited"))).post("/api/v1/agent/stream", json={"query": "q"}).text
    last = [line for line in text.splitlines() if line.startswith("data:")][-1]
    assert "rate limited" in json.loads(last[5:])["detail"]

    assert client(FakeAgent(error=LLMError("down"))).post("/api/v1/agent/ask", json={"query": "q"}).status_code == 503
