from datetime import UTC, date, datetime

import pytest
from fastapi.testclient import TestClient
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError

from src.dependencies import get_opensearch
from src.main import app
from src.schemas.search import SearchHit, SearchResponse


class FakeOpenSearch:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[dict] = []

    def search_papers(self, query, **kwargs):
        if self.fail:
            raise OpenSearchConnectionError("N/A", "connection refused", None)
        self.calls.append({"query": query, **kwargs})
        hit = SearchHit(
            arxiv_id="2611.06868",
            title="On Turn-Level Best-of-N Sampling for Scaling LLM Agents",
            authors=["A. Author"],
            abstract="We study agents.",
            categories=["cs.AI"],
            published_date=datetime(2026, 11, 4, tzinfo=UTC),
            pdf_url="https://arxiv.org/pdf/2611.06868",
            score=7.5,
            highlights={"title": ["Scaling LLM <mark>Agents</mark>"]},
        )
        return SearchResponse(query=query, total=1, size=kwargs["size"], offset=kwargs["offset"], hits=[hit])


@pytest.fixture
def fake():
    fake = FakeOpenSearch()
    app.dependency_overrides[get_opensearch] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


def test_post_search_passes_filters_and_returns_hits(fake):
    response = TestClient(app).post(
        "/api/v1/search",
        json={"query": "llm agents", "size": 5, "categories": ["cs.AI"], "date_from": "2026-11-01"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["hits"][0]["arxiv_id"] == "2611.06868"
    assert body["hits"][0]["highlights"]["title"] == ["Scaling LLM <mark>Agents</mark>"]
    assert fake.calls[0] == {
        "query": "llm agents",
        "size": 5,
        "offset": 0,
        "categories": ["cs.AI"],
        "date_from": date(2026, 11, 1),
        "date_to": None,
    }


def test_get_search_for_browser(fake):
    response = TestClient(app).get("/api/v1/search", params={"q": "agents", "category": ["cs.AI", "cs.LG"]})

    assert response.status_code == 200
    assert fake.calls[0]["categories"] == ["cs.AI", "cs.LG"]


def test_rejects_bad_input(fake):
    client = TestClient(app)
    assert client.post("/api/v1/search", json={"query": "x", "size": 500}).status_code == 422
    assert (
        client.post(
            "/api/v1/search", json={"query": "x", "date_from": "2026-11-05", "date_to": "2026-11-01"}
        ).status_code
        == 422
    )
    assert (
        client.get("/api/v1/search", params={"q": "x", "date_from": "2026-11-05", "date_to": "2026-11-01"}).status_code
        == 422
    )
    assert fake.calls == []


def test_returns_503_when_opensearch_down():
    app.dependency_overrides[get_opensearch] = lambda: FakeOpenSearch(fail=True)
    try:
        response = TestClient(app).post("/api/v1/search", json={"query": "agents"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
