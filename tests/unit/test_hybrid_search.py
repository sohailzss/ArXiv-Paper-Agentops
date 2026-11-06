from datetime import date

from src.services.hybrid_search import HybridSearchService, rrf_fuse
from src.services.opensearch.query_builder import build_chunk_vector_query


def source(chunk_id: str) -> dict:
    arxiv_id, index = chunk_id.split("_")
    return {
        "chunk_id": chunk_id,
        "arxiv_id": arxiv_id,
        "chunk_index": int(index),
        "section_titles": ["Intro"],
        "text": f"text of {chunk_id}",
        "title": "A paper",
        "authors": ["A. Author"],
        "categories": ["cs.AI"],
        "published_date": "2026-11-01T00:00:00+00:00",
        "pdf_url": "https://arxiv.org/pdf/x",
    }


class FakeChunkIndex:
    def __init__(self, bm25: list[str], vector: list[str]):
        self.bm25, self.vector = bm25, vector
        self.calls: list[tuple] = []

    def bm25_search(self, query, size, filters):
        self.calls.append(("bm25", query, size, filters))
        return [{"_id": i, "_source": source(i)} for i in self.bm25]

    def vector_search(self, vector, size, filters):
        self.calls.append(("vector", vector, size, filters))
        return [{"_id": i, "_source": source(i)} for i in self.vector]


class FakeEmbedder:
    model_id = "openai:test:4"

    def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]


def test_rrf_rewards_documents_found_by_both():
    fused = rrf_fuse({"bm25": ["a", "b", "c"], "vector": ["c", "d", "a"]}, k=60)
    ids = [doc_id for doc_id, _, _ in fused]

    assert ids[:2] == ["a", "c"]  # in both lists
    assert dict((d, r) for d, _, r in fused)["a"] == {"bm25": 1, "vector": 3}
    assert abs(fused[0][1] - (1 / 61 + 1 / 63)) < 1e-12


def test_rrf_single_list_keeps_order():
    assert [d for d, _, _ in rrf_fuse({"bm25": ["x", "y", "z"]})] == ["x", "y", "z"]


def test_rrf_is_deterministic_on_ties():
    fused = rrf_fuse({"bm25": ["a", "b"], "vector": ["b", "a"]})
    assert [d for d, _, _ in fused] == ["a", "b"]


def test_hybrid_mode_queries_both_and_fuses():
    index = FakeChunkIndex(bm25=["p1_0", "p2_1"], vector=["p2_1", "p3_0"])
    service = HybridSearchService(index, FakeEmbedder())

    result = service.search("agents", size=2, categories=["cs.AI"], date_from=date(2026, 11, 1))

    assert [h.chunk_id for h in result.hits] == ["p2_1", "p1_0"]
    assert result.hits[0].bm25_rank == 2 and result.hits[0].vector_rank == 1
    assert result.embedding_model == "openai:test:4"
    filters = index.calls[0][3]
    assert {"terms": {"categories": ["cs.AI"]}} in filters
    assert index.calls[1][1] == [0.1, 0.2, 0.3, 0.4]


def test_bm25_mode_skips_embedding():
    class NoEmbed(FakeEmbedder):
        def embed_query(self, text):
            raise AssertionError("should not embed in bm25 mode")

    index = FakeChunkIndex(bm25=["p1_0"], vector=["p9_9"])
    result = HybridSearchService(index, NoEmbed()).search("agents", mode="bm25")

    assert [h.chunk_id for h in result.hits] == ["p1_0"]
    assert result.hits[0].vector_rank is None
    assert [c[0] for c in index.calls] == ["bm25"]


def test_vector_query_puts_filters_inside_knn():
    body = build_chunk_vector_query([0.1], 5, [{"terms": {"categories": ["cs.AI"]}}])
    knn = body["query"]["knn"]["embedding"]
    assert knn["k"] == 5
    assert knn["filter"] == {"bool": {"filter": [{"terms": {"categories": ["cs.AI"]}}]}}
    assert "filter" not in build_chunk_vector_query([0.1], 5, [])["query"]["knn"]["embedding"]
