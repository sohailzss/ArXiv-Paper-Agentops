"""Redis cache for RAG answers.

Key = hash of (normalised question, retrieval options, LLM model, embedding model, corpus size).
Including the corpus size (chunk count) means that when the DAG indexes new papers, old answers
simply stop matching: no explicit invalidation, and Airflow never needs to talk to Redis.

The cache is strictly best-effort: if Redis is down or slow, requests skip it instead of failing.
"""

import hashlib
import json
import logging
import re

from src.config import RedisSettings

logger = logging.getLogger(__name__)

KEY_PREFIX = "rag:answer:v1:"
_SPACES = re.compile(r"\s+")


def normalise_query(query: str) -> str:
    """'  What is RAG? ' and 'what is rag' should share an answer."""
    return _SPACES.sub(" ", query).strip().rstrip("?!. ").lower()


def cache_key(
    query: str,
    top_k: int,
    mode: str,
    categories: list[str] | None,
    date_from: str | None,
    date_to: str | None,
    llm_model: str,
    embedding_model: str,
    corpus_version: int,
) -> str:
    parts = {
        "q": normalise_query(query),
        "k": top_k,
        "mode": mode,
        "cats": sorted(categories or []),
        "from": date_from,
        "to": date_to,
        "llm": llm_model,
        "emb": embedding_model,
        "corpus": corpus_version,
    }
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    return KEY_PREFIX + digest


class RAGCache:
    def __init__(self, client, ttl_seconds: int):
        self.client = client
        self.ttl = ttl_seconds

    @classmethod
    def from_settings(cls, settings: RedisSettings) -> "RAGCache | None":
        if not settings.enabled:
            return None
        import redis

        client = redis.Redis.from_url(
            settings.url, decode_responses=True, socket_timeout=0.5, socket_connect_timeout=0.5
        )
        return cls(client, settings.ttl_seconds)

    def get(self, key: str) -> dict | None:
        try:
            raw = self.client.get(key)
        except Exception as exc:  # redis.RedisError, timeouts: never fail the request
            logger.warning("Cache read skipped: %s", exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def set(self, key: str, value: dict) -> None:
        try:
            self.client.set(key, json.dumps(value), ex=self.ttl)
        except Exception as exc:
            logger.warning("Cache write skipped: %s", exc)

    def ping(self) -> bool:
        return bool(self.client.ping())

    def stats(self) -> dict:
        info = self.client.info("stats")
        return {"hits": info.get("keyspace_hits", 0), "misses": info.get("keyspace_misses", 0)}

    def clear(self) -> int:
        """Delete all cached answers (not other keys). Returns how many were removed."""
        removed = 0
        for key in self.client.scan_iter(match=KEY_PREFIX + "*", count=500):
            removed += self.client.delete(key)
        return removed

    def close(self) -> None:
        self.client.close()
