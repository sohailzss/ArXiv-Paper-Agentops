"""Copies papers from Postgres (source of truth) into OpenSearch:
the paper-level BM25 index (Phase 3) and the chunk index with embeddings (Phase 4).

Every paper is indexed, parsed PDF or not: title + abstract alone are enough for keyword
search, and full text is added when Docling has produced it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.db.database import Database
from src.models.paper import Paper
from src.repositories.paper import PaperRepository
from src.services.opensearch.client import OpenSearchClient

if TYPE_CHECKING:
    from src.config import Settings
    from src.services.embeddings.client import EmbeddingsClient
    from src.services.opensearch.chunk_index import ChunkIndex

logger = logging.getLogger(__name__)


def paper_to_document(paper: Paper) -> dict:
    has_full_text = bool(paper.pdf_processed and paper.raw_text)
    published = paper.published_date
    if published.tzinfo is None:  # SQLite (tests) drops tz info; Postgres keeps it
        published = published.replace(tzinfo=UTC)
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": list(paper.authors or []),
        "categories": list(paper.categories or []),
        "published_date": published.isoformat(),
        "pdf_url": paper.pdf_url,
        "raw_text": paper.raw_text if has_full_text else None,
        "has_full_text": has_full_text,
        "indexed_at": datetime.now(UTC).isoformat(),
    }


def index_papers(
    database: Database,
    opensearch: OpenSearchClient,
    arxiv_ids: list[str] | None = None,
    batch_size: int = 50,
) -> dict:
    """Index the given papers (or all papers when arxiv_ids is None). Idempotent."""
    opensearch.setup_index()
    stats = {"indexed": 0, "errors": []}
    if arxiv_ids is not None and not arxiv_ids:
        return stats

    offset = 0
    while True:
        with database.session() as session:
            batch = PaperRepository(session).list_for_indexing(arxiv_ids, limit=batch_size, offset=offset)
            documents = [paper_to_document(p) for p in batch]
        if not documents:
            break
        result = opensearch.index_documents(documents)
        stats["indexed"] += result["indexed"]
        stats["errors"].extend(result["errors"])
        offset += batch_size

    # Make the new documents searchable right away (otherwise up to 1 s delay).
    opensearch.client.indices.refresh(index=opensearch.alias)
    logger.info("Indexed %d papers (%d errors)", stats["indexed"], len(stats["errors"]))
    return stats


# ---------- Phase 4: chunks + embeddings ----------


def build_chunk_index(opensearch: OpenSearchClient, embedder: EmbeddingsClient) -> ChunkIndex:
    from src.services.opensearch.chunk_index import ChunkIndex

    return ChunkIndex(opensearch, embedder.provider, embedder.model_id, embedder.dimensions)


def index_chunks(
    database: Database,
    chunk_index: ChunkIndex,
    embedder: EmbeddingsClient,
    settings: Settings | None = None,
    arxiv_ids: list[str] | None = None,
    papers_per_batch: int = 10,
) -> dict:
    """Chunk papers, embed the chunks with the configured provider, and store them. Idempotent."""
    from src.config import get_settings
    from src.services.chunking import chunk_paper, embedding_text

    settings = settings or get_settings()
    chunk_index.setup()
    stats = {"papers": 0, "chunks": 0, "errors": []}
    if arxiv_ids is not None and not arxiv_ids:
        return stats

    offset = 0
    while True:
        with database.session() as session:
            batch = PaperRepository(session).list_for_indexing(arxiv_ids, limit=papers_per_batch, offset=offset)
            papers = [(paper_to_document(p), p.sections if p.pdf_processed else None) for p in batch]
        if not papers:
            break
        offset += papers_per_batch

        documents, texts = [], []
        for paper, sections in papers:
            for chunk in chunk_paper(paper["arxiv_id"], paper["title"], paper["abstract"], sections, settings.chunking):
                texts.append(embedding_text(paper["title"], chunk))
                documents.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "arxiv_id": chunk.arxiv_id,
                        "chunk_index": chunk.chunk_index,
                        "section_titles": chunk.section_titles,
                        "text": chunk.text,
                        "title": paper["title"],
                        "authors": paper["authors"],
                        "categories": paper["categories"],
                        "published_date": paper["published_date"],
                        "pdf_url": paper["pdf_url"],
                    }
                )
        for document, vector in zip(documents, embedder.embed_documents(texts), strict=True):
            document["embedding"] = vector

        result = chunk_index.replace_paper_chunks(documents)
        stats["papers"] += len(papers)
        stats["chunks"] += result["indexed"]
        stats["errors"].extend(result["errors"])

    chunk_index.refresh()
    logger.info("Indexed %d chunks for %d papers into %s", stats["chunks"], stats["papers"], chunk_index.name)
    return stats
