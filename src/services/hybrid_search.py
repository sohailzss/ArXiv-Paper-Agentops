"""Hybrid search over chunks: BM25 + kNN, fused with Reciprocal Rank Fusion.

RRF only uses ranks, not raw scores, so it needs no tuning to combine BM25 scores (unbounded)
with cosine similarities (0..1): score(doc) = sum over lists of 1 / (k + rank).
Fusion is done here in Python rather than in an OpenSearch search pipeline, so it's easy to
test, works on any OpenSearch version, and the API can report each hit's rank in both lists.
"""

from datetime import date

from opentelemetry import trace

from src.schemas.search import ChunkHit, HybridSearchResponse, SearchMode
from src.services.embeddings.client import EmbeddingsClient
from src.services.opensearch.chunk_index import ChunkIndex
from src.services.opensearch.query_builder import build_filters

RRF_K = 60  # standard constant from the original RRF paper; dampens the effect of top ranks
CANDIDATES = 50  # hits fetched from each list before fusing

# Plain OpenTelemetry API: these spans go to Logfire when it's configured, and are no-ops otherwise.
_tracer = trace.get_tracer(__name__)


def rrf_fuse(ranked_lists: dict[str, list[str]], k: int = RRF_K) -> list[tuple[str, float, dict[str, int]]]:
    """ranked_lists: {"bm25": [id, ...], "vector": [id, ...]} best first.
    Returns [(id, score, {"bm25": rank, "vector": rank})] best first. Ranks are 1-based."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for list_name, ids in ranked_lists.items():
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(doc_id, {})[list_name] = rank
    # Ties broken by best single rank, then id, so results are deterministic.
    return sorted(
        ((doc_id, score, ranks[doc_id]) for doc_id, score in scores.items()),
        key=lambda item: (-item[1], min(item[2].values()), item[0]),
    )


class HybridSearchService:
    def __init__(self, chunk_index: ChunkIndex, embedder: EmbeddingsClient):
        self.chunks = chunk_index
        self.embedder = embedder

    def search(
        self,
        query: str,
        mode: SearchMode = "hybrid",
        size: int = 10,
        categories: list[str] | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> HybridSearchResponse:
        filters = build_filters(categories, date_from, date_to)
        candidates = max(CANDIDATES, size)
        sources: dict[str, dict] = {}
        ranked: dict[str, list[str]] = {}

        with _tracer.start_as_current_span("hybrid_search", attributes={"search.mode": mode, "search.size": size}):
            if mode in ("hybrid", "bm25"):
                with _tracer.start_as_current_span(
                    "opensearch.bm25", attributes={"opensearch.index": getattr(self.chunks, "name", "")}
                ):
                    hits = self.chunks.bm25_search(query, candidates, filters)
                ranked["bm25"] = [h["_id"] for h in hits]
                sources.update({h["_id"]: h["_source"] for h in hits})
            if mode in ("hybrid", "vector"):
                with _tracer.start_as_current_span(
                    "embed_query", attributes={"embedding.model": self.embedder.model_id}
                ):
                    vector = self.embedder.embed_query(query)
                with _tracer.start_as_current_span(
                    "opensearch.knn", attributes={"opensearch.index": getattr(self.chunks, "name", "")}
                ):
                    hits = self.chunks.vector_search(vector, candidates, filters)
                ranked["vector"] = [h["_id"] for h in hits]
                sources.update({h["_id"]: h["_source"] for h in hits})

        fused = rrf_fuse(ranked)[:size]
        hits = [
            ChunkHit(**sources[chunk_id], score=score, bm25_rank=ranks.get("bm25"), vector_rank=ranks.get("vector"))
            for chunk_id, score, ranks in fused
        ]
        return HybridSearchResponse(
            query=query, mode=mode, embedding_model=self.embedder.model_id, total=len(hits), hits=hits
        )
