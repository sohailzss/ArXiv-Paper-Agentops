import json

import httpx
import pytest

from src.config import EmbeddingsSettings, Settings
from src.services.embeddings import client as embeddings_module
from src.services.embeddings.client import (
    EmbeddingError,
    JinaEmbeddings,
    OpenAIEmbeddings,
    get_embeddings_client,
)

DIMS = 4


def make_settings(provider: str = "openai", **overrides) -> Settings:
    return Settings(
        openai_api_key="sk-test",
        jina_api_key="jina-test",
        embeddings=EmbeddingsSettings(provider=provider, dimensions=DIMS, batch_size=2, max_retries=2, **overrides),
    )


def fake_api(requests: list, statuses: list[int] | None = None):
    """Mock transport: records request bodies, answers with one vector per input."""
    statuses = list(statuses or [])

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append({"url": str(request.url), "auth": request.headers["Authorization"], "body": body})
        if statuses:
            status = statuses.pop(0)
            if status != 200:
                return httpx.Response(status, text="error")
        # Return out of order to check the client sorts by index.
        data = [{"index": i, "embedding": [float(i)] * DIMS} for i in range(len(body["input"]))][::-1]
        return httpx.Response(200, json={"data": data})

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(embeddings_module.time, "sleep", lambda s: None)


def test_openai_payload_batches_and_order():
    requests: list = []
    client = OpenAIEmbeddings(make_settings(), http_client=fake_api(requests))

    vectors = client.embed_documents(["a", "b", "c"])

    assert vectors == [[0.0] * DIMS, [1.0] * DIMS, [0.0] * DIMS]  # batches of 2, sorted by index
    assert len(requests) == 2
    assert requests[0]["url"] == "https://api.openai.com/v1/embeddings"
    assert requests[0]["auth"] == "Bearer sk-test"
    assert requests[0]["body"] == {"model": "text-embedding-3-small", "input": ["a", "b"], "dimensions": DIMS}
    assert client.model_id == f"openai:text-embedding-3-small:{DIMS}"


def test_jina_uses_query_and_passage_tasks():
    requests: list = []
    client = JinaEmbeddings(make_settings("jina"), http_client=fake_api(requests))

    client.embed_documents(["doc"])
    client.embed_query("question")

    assert requests[0]["url"] == "https://api.jina.ai/v1/embeddings"
    assert requests[0]["auth"] == "Bearer jina-test"
    assert requests[0]["body"]["task"] == "retrieval.passage"
    assert requests[1]["body"]["task"] == "retrieval.query"
    assert requests[1]["body"]["model"] == "jina-embeddings-v3"
    assert client.model_id == f"jina:jina-embeddings-v3:{DIMS}"


def test_retries_rate_limits_then_succeeds():
    requests: list = []
    client = OpenAIEmbeddings(make_settings(), http_client=fake_api(requests, statuses=[429, 503, 200]))

    assert client.embed_query("q") == [0.0] * DIMS
    assert len(requests) == 3


def test_gives_up_after_max_retries():
    client = OpenAIEmbeddings(make_settings(), http_client=fake_api([], statuses=[429, 429, 429]))
    with pytest.raises(EmbeddingError, match="after retries"):
        client.embed_query("q")


def test_does_not_retry_client_errors():
    requests: list = []
    client = OpenAIEmbeddings(make_settings(), http_client=fake_api(requests, statuses=[401]))
    with pytest.raises(EmbeddingError, match="401"):
        client.embed_query("q")
    assert len(requests) == 1


def test_wrong_dimensions_are_rejected():
    def handler(request):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

    client = OpenAIEmbeddings(make_settings(), http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(EmbeddingError, match="unexpected shape"):
        client.embed_query("q")


def test_factory_picks_provider_from_settings():
    assert isinstance(get_embeddings_client(make_settings("openai")), OpenAIEmbeddings)
    assert isinstance(get_embeddings_client(make_settings("JINA")), JinaEmbeddings)
    with pytest.raises(EmbeddingError, match="Unknown"):
        get_embeddings_client(make_settings("cohere"))


def test_missing_key_fails_fast():
    settings = Settings(openai_api_key="", embeddings=EmbeddingsSettings(provider="openai"))
    with pytest.raises(EmbeddingError, match="No API key"):
        get_embeddings_client(settings)
