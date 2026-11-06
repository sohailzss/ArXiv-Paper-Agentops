"""Check every external service before you start a phase.

Run from the project root:  uv run python scripts/test_connections.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.db.database import Database  # noqa: E402
from src.services.opensearch.client import OpenSearchClient  # noqa: E402

settings = get_settings()
results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn() or "ok"
        results.append((name, True, str(detail)))
    except Exception as exc:
        results.append((name, False, str(exc).splitlines()[0][:150]))


def neon() -> str:
    db = Database(settings)
    try:
        db.ping()
        db.create_tables()
        return "connected, tables ready"
    finally:
        db.close()


def opensearch() -> str:
    return f"cluster status: {OpenSearchClient(settings).health()['status']}"


def arxiv() -> str:
    r = httpx.get(settings.arxiv.base_url, params={"search_query": "cat:cs.AI", "max_results": 1}, timeout=30)
    r.raise_for_status()
    return "reachable"


def jina() -> str:
    if settings.embeddings.provider != "jina":
        return f"skipped (EMBEDDINGS__PROVIDER={settings.embeddings.provider})"
    if not settings.jina_api_key:
        raise RuntimeError("EMBEDDINGS__PROVIDER=jina but JINA_API_KEY is empty")
    r = httpx.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {settings.jina_api_key}"},
        json={
            "model": settings.embeddings.jina_model,
            "input": ["hello"],
            "dimensions": settings.embeddings.dimensions,
        },
        timeout=30,
    )
    if r.status_code == 403 and "BALANCE" in r.text:
        raise RuntimeError("key has no tokens left (top up at jina.ai/api-dashboard/key-manager)")
    r.raise_for_status()
    return "key works"


def embeddings() -> str:
    from src.services.embeddings.client import get_embeddings_client

    client = get_embeddings_client(settings)
    try:
        vector = client.embed_query("hello")
    finally:
        client.close()
    return f"{client.model_id} -> {len(vector)} dims"


def openai() -> str:
    if not settings.openai_api_key:
        return "skipped (no OPENAI_API_KEY yet, needed from Phase 5)"
    r = httpx.get(
        "https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {settings.openai_api_key}"}, timeout=30
    )
    r.raise_for_status()
    return "key works"


check("Neon Postgres", neon)
check("OpenSearch (docker)", opensearch)
check("arXiv API", arxiv)
check("Jina AI", jina)
check("OpenAI", openai)
check("Embeddings (active)", embeddings)

width = max(len(name) for name, _, _ in results)
for name, ok, detail in results:
    print(f"{'OK  ' if ok else 'FAIL'}  {name:<{width}}  {detail}")

sys.exit(0 if all(ok for _, ok, _ in results) else 1)
