from datetime import UTC, datetime

from fastapi.testclient import TestClient

from src.dependencies import get_hybrid_search
from src.main import app
from src.schemas.search import ChunkHit, HybridSearchResponse
from src.services.embeddings.client import EmbeddingError


class FakeService:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[dict] = []

    def search(self, query, **kwargs):
        if self.error:
            raise self.error
        self.calls.append({"query": query, **kwargs})
        hit = ChunkHit(
            chunk_id="2611.06868_3",
            arxiv_id="2611.06868",
            chunk_index=3,
            section_titles=["4 Results"],
            text="Best-of-N at the turn level improves agents.",
            title="On Turn-Level Best-of-N Sampling for Scaling LLM Agents",
            authors=["A. Author"],
            categories=["cs.AI"],
            published_date=datetime(2026, 11, 4, tzinfo=UTC),
            pdf_url="https://arxiv.org/pdf/2611.06868",
            score=0.032,
            bm25_rank=1,
            vector_rank=2,
        )
        return HybridSearchResponse(query=query, mode=kwargs["mode"], embedding_model="openai:x:4", total=1, hits=[hit])


def call(service, json):
    app.dependency_overrides[get_hybrid_search] = lambda: service
    try:
        return TestClient(app).post("/api/v1/hybrid-search", json=json)
    finally:
        app.dependency_overrides.clear()


def test_hybrid_search_defaults_to_hybrid_mode():
    service = FakeService()
    response = call(service, {"query": "best-of-n agents"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "hybrid"
    assert body["hits"][0]["bm25_rank"] == 1 and body["hits"][0]["vector_rank"] == 2
    assert service.calls[0]["mode"] == "hybrid" and service.calls[0]["size"] == 10


def test_validates_input():
    assert call(FakeService(), {"query": ""}).status_code == 422
    assert call(FakeService(), {"query": "x", "mode": "magic"}).status_code == 422


def test_embedding_failure_is_503():
    response = call(FakeService(error=EmbeddingError("quota exceeded")), {"query": "agents"})
    assert response.status_code == 503
    assert "quota exceeded" in response.json()["detail"]


def test_unavailable_when_not_initialised():
    app.state.hybrid_search = None
    app.state.hybrid_search_error = "No API key set for embeddings provider 'openai'"
    response = TestClient(app).post("/api/v1/hybrid-search", json={"query": "agents"})
    assert response.status_code == 503
    assert "No API key" in response.json()["detail"]
