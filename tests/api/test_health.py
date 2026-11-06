from fastapi.testclient import TestClient

from src.dependencies import get_database, get_opensearch
from src.main import app


class FakeDB:
    def ping(self):
        return True


class FakeOpenSearch:
    def __init__(self, healthy=True):
        self.healthy = healthy

    def health(self):
        if not self.healthy:
            raise ConnectionError("connection refused")
        return {"status": "green"}


def test_health_ok():
    app.dependency_overrides[get_database] = lambda: FakeDB()
    app.dependency_overrides[get_opensearch] = lambda: FakeOpenSearch()
    try:
        body = TestClient(app).get("/api/v1/health").json()
    finally:
        app.dependency_overrides.clear()
    assert body["status"] == "ok"
    assert body["services"]["opensearch"]["message"] == "cluster status: green"


def test_health_degraded_when_opensearch_down():
    app.dependency_overrides[get_database] = lambda: FakeDB()
    app.dependency_overrides[get_opensearch] = lambda: FakeOpenSearch(healthy=False)
    try:
        body = TestClient(app).get("/api/v1/health").json()
    finally:
        app.dependency_overrides.clear()
    assert body["status"] == "degraded"
    assert body["services"]["opensearch"]["status"] == "unhealthy"


def test_ping():
    assert TestClient(app).get("/api/v1/ping").json() == {"message": "pong"}
