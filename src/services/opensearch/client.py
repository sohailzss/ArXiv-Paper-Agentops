"""OpenSearch client wrapper: health, index setup, bulk indexing and BM25 search."""

import logging
from datetime import date

from opensearchpy import OpenSearch, helpers

from src.config import Settings, get_settings
from src.schemas.search import SearchHit, SearchResponse
from src.services.opensearch.index_config import INDEX_VERSION, PAPER_INDEX_BODY
from src.services.opensearch.query_builder import build_search_query

logger = logging.getLogger(__name__)


class OpenSearchClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.client = OpenSearch(
            hosts=[self.settings.opensearch.host],
            use_ssl=self.settings.opensearch.host.startswith("https"),
            verify_certs=False,
            ssl_show_warn=False,
            timeout=30,
        )

    @property
    def alias(self) -> str:
        return self.settings.opensearch.index_name

    @property
    def versioned_index(self) -> str:
        return f"{self.alias}-v{INDEX_VERSION}"

    def health(self) -> dict:
        """Returns the cluster health dict, e.g. {"status": "green", ...}."""
        return self.client.cluster.health()

    # ---------- index management ----------

    def setup_index(self, recreate: bool = False) -> bool:
        """Create the versioned index and point the alias at it. Returns True if it created anything.

        Safe to call on every startup. recreate=True drops the current version first (dev only).
        """
        if recreate and self.client.indices.exists(index=self.versioned_index):
            self.client.indices.delete(index=self.versioned_index)
            logger.warning("Deleted index %s", self.versioned_index)

        if self.client.indices.exists_alias(name=self.alias):
            return False
        if self.client.indices.exists(index=self.alias):
            raise RuntimeError(
                f"'{self.alias}' exists as a plain index, not an alias. Delete it once "
                f"(DELETE /{self.alias}) and rerun setup_index()."
            )

        if not self.client.indices.exists(index=self.versioned_index):
            self.client.indices.create(index=self.versioned_index, body=PAPER_INDEX_BODY)
            logger.info("Created index %s", self.versioned_index)
        self.client.indices.put_alias(index=self.versioned_index, name=self.alias)
        logger.info("Alias %s -> %s", self.alias, self.versioned_index)
        return True

    def index_documents(self, documents: list[dict], refresh: bool = False) -> dict:
        """Bulk upsert. Each document's arxiv_id is its _id, so reindexing never duplicates."""
        if not documents:
            return {"indexed": 0, "errors": []}
        actions = (
            {"_op_type": "index", "_index": self.alias, "_id": doc["arxiv_id"], "_source": doc} for doc in documents
        )
        indexed, errors = helpers.bulk(
            self.client, actions, raise_on_error=False, request_timeout=120, refresh="wait_for" if refresh else False
        )
        error_messages = [str(e)[:300] for e in errors] if isinstance(errors, list) else []
        if error_messages:
            logger.error("Bulk indexing had %d errors, first: %s", len(error_messages), error_messages[0])
        return {"indexed": indexed, "errors": error_messages}

    def count(self) -> int:
        return self.client.count(index=self.alias)["count"]

    # ---------- search ----------

    def search_papers(
        self,
        query: str,
        size: int = 10,
        offset: int = 0,
        categories: list[str] | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> SearchResponse:
        body = build_search_query(
            query, size=size, offset=offset, categories=categories, date_from=date_from, date_to=date_to
        )
        response = self.client.search(index=self.alias, body=body)
        hits = [
            SearchHit(**hit["_source"], score=hit.get("_score"), highlights=hit.get("highlight", {}))
            for hit in response["hits"]["hits"]
        ]
        return SearchResponse(
            query=query, total=response["hits"]["total"]["value"], size=size, offset=offset, hits=hits
        )

    def close(self) -> None:
        self.client.close()
