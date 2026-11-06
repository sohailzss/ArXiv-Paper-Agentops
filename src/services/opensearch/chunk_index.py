"""The chunk index: one document per chunk, with BM25 text and a kNN vector.

Each embedding provider gets its own index (arxiv-papers-chunks-openai-1024-v1,
arxiv-papers-chunks-jina-1024-v1), and the index records the exact model in _meta.
Switching EMBEDDINGS__PROVIDER therefore never mixes vectors from different models:
the other provider's index is simply used (or built by scripts/index_papers.py).
"""

import logging

from opensearchpy import helpers

from src.services.opensearch.client import OpenSearchClient
from src.services.opensearch.index_config import PAPER_INDEX_BODY
from src.services.opensearch.query_builder import build_chunk_bm25_query, build_chunk_vector_query

logger = logging.getLogger(__name__)

CHUNK_INDEX_VERSION = 1


class EmbeddingModelMismatch(RuntimeError):
    pass


def chunk_index_body(dimensions: int, model_id: str) -> dict:
    return {
        "settings": {
            "index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0},
            "analysis": PAPER_INDEX_BODY["settings"]["analysis"],  # same english analyzer as the paper index
        },
        "mappings": {
            "dynamic": "strict",
            "_meta": {"embedding_model": model_id},
            "properties": {
                "chunk_id": {"type": "keyword"},
                "arxiv_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "section_titles": {"type": "text", "analyzer": "paper_english"},
                "text": {"type": "text", "analyzer": "paper_english"},
                "title": {"type": "text", "analyzer": "paper_english"},
                "authors": {"type": "keyword"},
                "categories": {"type": "keyword"},
                "published_date": {"type": "date"},
                "pdf_url": {"type": "keyword", "index": False},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimensions,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                        "parameters": {"m": 16, "ef_construction": 128},
                    },
                },
            },
        },
    }


class ChunkIndex:
    def __init__(self, opensearch: OpenSearchClient, provider: str, model_id: str, dimensions: int):
        self.client = opensearch.client
        self.model_id = model_id
        self.dimensions = dimensions
        base = opensearch.settings.opensearch.chunk_index_name
        self.name = f"{base}-{provider}-{dimensions}-v{CHUNK_INDEX_VERSION}"

    def setup(self, recreate: bool = False) -> bool:
        """Create the index if missing; refuse to use one built with a different model."""
        if recreate and self.client.indices.exists(index=self.name):
            self.client.indices.delete(index=self.name)
            logger.warning("Deleted chunk index %s", self.name)
        if self.client.indices.exists(index=self.name):
            self.check_model()
            return False
        self.client.indices.create(index=self.name, body=chunk_index_body(self.dimensions, self.model_id))
        logger.info("Created chunk index %s for %s", self.name, self.model_id)
        return True

    def check_model(self) -> None:
        mapping = self.client.indices.get_mapping(index=self.name)[self.name]["mappings"]
        indexed_with = mapping.get("_meta", {}).get("embedding_model")
        if indexed_with != self.model_id:
            raise EmbeddingModelMismatch(
                f"Index {self.name} holds vectors from '{indexed_with}', but the app is configured for "
                f"'{self.model_id}'. Rebuild it: uv run python scripts/index_papers.py --recreate"
            )

    def replace_paper_chunks(self, documents: list[dict]) -> dict:
        """Upsert these chunks, then delete leftover chunks of the same papers
        (a re-parsed paper can end up with fewer chunks than before)."""
        if not documents:
            return {"indexed": 0, "errors": []}
        actions = ({"_index": self.name, "_id": d["chunk_id"], "_source": d} for d in documents)
        indexed, errors = helpers.bulk(self.client, actions, raise_on_error=False, request_timeout=120)
        error_messages = [str(e)[:300] for e in errors] if isinstance(errors, list) else []

        chunk_counts: dict[str, int] = {}
        for d in documents:
            chunk_counts[d["arxiv_id"]] = max(chunk_counts.get(d["arxiv_id"], 0), d["chunk_index"] + 1)
        stale = [
            {"bool": {"filter": [{"term": {"arxiv_id": aid}}, {"range": {"chunk_index": {"gte": n}}}]}}
            for aid, n in chunk_counts.items()
        ]
        self.client.delete_by_query(
            index=self.name, body={"query": {"bool": {"should": stale, "minimum_should_match": 1}}}, refresh=True
        )
        return {"indexed": indexed, "errors": error_messages}

    def refresh(self) -> None:
        self.client.indices.refresh(index=self.name)

    def count(self) -> int:
        return self.client.count(index=self.name)["count"]

    def bm25_search(self, query: str, size: int, filters: list[dict]) -> list[dict]:
        return self.client.search(index=self.name, body=build_chunk_bm25_query(query, size, filters))["hits"]["hits"]

    def vector_search(self, vector: list[float], size: int, filters: list[dict]) -> list[dict]:
        return self.client.search(index=self.name, body=build_chunk_vector_query(vector, size, filters))["hits"]["hits"]
