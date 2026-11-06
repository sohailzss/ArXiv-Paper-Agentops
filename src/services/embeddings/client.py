"""Embedding providers behind one interface. Pick one with EMBEDDINGS__PROVIDER=openai|jina.

Both are plain HTTP calls (httpx), so there's no extra SDK to install in the API or Airflow images.
Documents and queries are embedded separately because Jina uses different task adapters for each
(OpenAI ignores the distinction, which is fine).
"""

import logging
import time
from abc import ABC, abstractmethod

import httpx

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class EmbeddingError(Exception):
    pass


class EmbeddingsClient(ABC):
    provider: str
    model: str

    def __init__(self, settings: Settings, api_key: str, http_client: httpx.Client | None = None):
        if not api_key:
            raise EmbeddingError(f"No API key set for embeddings provider '{self.provider}'")
        self.settings = settings.embeddings
        self.dimensions = self.settings.dimensions
        self.http = http_client or httpx.Client(timeout=self.settings.timeout_seconds)
        self.http.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def model_id(self) -> str:
        """Identifies the vector space. Vectors from different model_ids must never be mixed."""
        return f"{self.provider}:{self.model}:{self.dimensions}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        size = self.settings.batch_size
        for start in range(0, len(texts), size):
            vectors.extend(self._embed(texts[start : start + size], is_query=False))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], is_query=True)[0]

    def _embed(self, texts: list[str], is_query: bool) -> list[list[float]]:
        if not texts:
            return []
        payload = self._payload(texts, is_query)
        response = self._post_with_retries(payload)
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in data]
        if len(vectors) != len(texts) or any(len(v) != self.dimensions for v in vectors):
            raise EmbeddingError(f"{self.provider} returned unexpected shape for {len(texts)} inputs")
        return vectors

    def _post_with_retries(self, payload: dict) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                response = self.http.post(self.url, json=payload)
            except httpx.TransportError as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    return response
                if response.status_code not in RETRYABLE_STATUS:
                    raise EmbeddingError(
                        f"{self.provider} embeddings failed ({response.status_code}): {response.text[:300]}"
                    )
                last_error = EmbeddingError(f"{self.provider} returned {response.status_code}")
            if attempt < self.settings.max_retries:
                delay = 2**attempt
                logger.warning(
                    "%s embeddings attempt %d failed (%s), retrying in %ds",
                    self.provider,
                    attempt + 1,
                    last_error,
                    delay,
                )
                time.sleep(delay)
        raise EmbeddingError(f"{self.provider} embeddings failed after retries: {last_error}")

    @property
    @abstractmethod
    def url(self) -> str: ...

    @abstractmethod
    def _payload(self, texts: list[str], is_query: bool) -> dict: ...

    def close(self) -> None:
        self.http.close()


class OpenAIEmbeddings(EmbeddingsClient):
    provider = "openai"
    url = "https://api.openai.com/v1/embeddings"

    def __init__(self, settings: Settings, http_client: httpx.Client | None = None):
        self.model = settings.embeddings.openai_model
        super().__init__(settings, settings.openai_api_key, http_client)

    def _payload(self, texts: list[str], is_query: bool) -> dict:
        return {"model": self.model, "input": texts, "dimensions": self.dimensions}


class JinaEmbeddings(EmbeddingsClient):
    provider = "jina"
    url = "https://api.jina.ai/v1/embeddings"

    def __init__(self, settings: Settings, http_client: httpx.Client | None = None):
        self.model = settings.embeddings.jina_model
        super().__init__(settings, settings.jina_api_key, http_client)

    def _payload(self, texts: list[str], is_query: bool) -> dict:
        return {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
            "task": "retrieval.query" if is_query else "retrieval.passage",
            "normalized": True,
        }


PROVIDERS: dict[str, type[EmbeddingsClient]] = {"openai": OpenAIEmbeddings, "jina": JinaEmbeddings}


def get_embeddings_client(
    settings: Settings | None = None, http_client: httpx.Client | None = None
) -> EmbeddingsClient:
    settings = settings or get_settings()
    provider = settings.embeddings.provider.lower()
    if provider not in PROVIDERS:
        raise EmbeddingError(f"Unknown EMBEDDINGS__PROVIDER '{provider}'. Use one of: {', '.join(PROVIDERS)}")
    return PROVIDERS[provider](settings, http_client)
