"""Task logic for the arXiv ingestion DAG, kept out of the DAG file so it's easy to test.

`src/` is mounted into the Airflow container at /opt/airflow/src and PYTHONPATH
includes /opt/airflow, so `import src...` works the same as in the API.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta

logger = logging.getLogger(__name__)


def resolve_date_window(run_date: datetime | None, from_date: str | None, to_date: str | None) -> tuple[date, date]:
    """Explicit params win. Otherwise use the 3 days ending yesterday.

    arXiv only shows a paper in the API after it's announced (roughly a day after
    submission), so a 3-day window catches late arrivals; upserts skip duplicates.
    """
    if from_date or to_date:
        start = date.fromisoformat(from_date or to_date)
        end = date.fromisoformat(to_date or from_date)
        return start, end
    anchor = (run_date or datetime.now()).date()
    return anchor - timedelta(days=3), anchor - timedelta(days=1)


def setup_environment() -> dict:
    """Fail fast if the database is unreachable, and make sure tables exist."""
    from src.db.database import Database

    db = Database()
    try:
        db.ping()
        db.create_tables()
    finally:
        db.close()
    return {"database": "ok"}


def fetch_and_process(from_date: date, to_date: date, max_results: int, process_pdfs: bool = True) -> dict:
    from src.db.database import Database
    from src.services.arxiv.client import ArxivClient
    from src.services.metadata_fetcher import MetadataFetcher
    from src.services.pdf_parser.parser import PDFParserService

    db = Database()
    try:
        fetcher = MetadataFetcher(ArxivClient(), PDFParserService() if process_pdfs else None, db)
        stats = asyncio.run(
            fetcher.fetch_and_process(
                from_date=from_date, to_date=to_date, max_results=max_results, process_pdfs=process_pdfs
            )
        )
    finally:
        db.close()
    stats["window"] = f"{from_date} to {to_date}"
    return stats


def index_papers(stats: dict) -> dict:
    """Push the papers this run touched into OpenSearch: the BM25 paper index, then
    chunks + embeddings (provider from EMBEDDINGS__PROVIDER) for hybrid search."""
    from src.db.database import Database
    from src.services.embeddings.client import get_embeddings_client
    from src.services.indexing import build_chunk_index, index_chunks
    from src.services.indexing import index_papers as run_indexing
    from src.services.opensearch.client import OpenSearchClient

    arxiv_ids = stats.get("arxiv_ids", [])
    db = Database()
    opensearch = OpenSearchClient()
    embedder = get_embeddings_client()
    try:
        papers = run_indexing(db, opensearch, arxiv_ids=arxiv_ids)
        chunks = index_chunks(db, build_chunk_index(opensearch, embedder), embedder, arxiv_ids=arxiv_ids)
    finally:
        embedder.close()
        opensearch.close()
        db.close()
    errors = papers["errors"] + chunks["errors"]
    if errors:
        raise RuntimeError(f"{len(errors)} documents failed to index, first: {errors[0]}")
    return {
        **stats,
        "indexed": papers["indexed"],
        "chunks_indexed": chunks["chunks"],
        "embedding_model": embedder.model_id,
    }


def build_report(stats: dict) -> dict:
    from src.db.database import Database
    from src.repositories.paper import PaperRepository

    db = Database()
    try:
        with db.session() as session:
            repo = PaperRepository(session)
            totals = {"total_papers": repo.count(), "processed_papers": repo.count(processed_only=True)}
    finally:
        db.close()

    report = {**{k: v for k, v in stats.items() if k != "arxiv_ids"}, **totals}
    logger.info("Ingestion report: %s", report)
    if stats.get("fetched", 0) and stats.get("errors") and len(stats["errors"]) >= stats["fetched"]:
        raise RuntimeError(f"Every paper failed: {stats['errors'][:3]}")
    return report
