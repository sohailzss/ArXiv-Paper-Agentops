"""Orchestrates ingestion: arXiv metadata -> Postgres -> PDF download -> Docling -> Postgres.
(Indexing into OpenSearch is a separate DAG task; see src/services/indexing.py.)

One paper failing never stops the run; failures are counted and stored on the row.
"""

import asyncio
import logging
from datetime import date

from src.config import Settings, get_settings
from src.db.database import Database
from src.repositories.paper import PaperRepository
from src.schemas.arxiv import ArxivPaper
from src.services.arxiv.client import ArxivClient
from src.services.pdf_parser.parser import PDFParserService

logger = logging.getLogger(__name__)


class MetadataFetcher:
    def __init__(
        self,
        arxiv_client: ArxivClient,
        pdf_parser: PDFParserService | None,
        database: Database,
        settings: Settings | None = None,
    ):
        self.arxiv = arxiv_client
        self.parser = pdf_parser
        self.db = database
        self.settings = settings or get_settings()

    async def fetch_and_process(
        self,
        from_date: date | None = None,
        to_date: date | None = None,
        max_results: int | None = None,
        process_pdfs: bool = True,
    ) -> dict:
        stats = {"fetched": 0, "stored": 0, "already_processed": 0, "pdfs_parsed": 0, "errors": [], "arxiv_ids": []}

        papers = await self.arxiv.fetch_papers(from_date=from_date, to_date=to_date, max_results=max_results)
        stats["fetched"] = len(papers)
        # The DAG's index task reindexes exactly these papers.
        stats["arxiv_ids"] = [p.arxiv_id for p in papers]

        to_process: list[ArxivPaper] = []
        with self.db.session() as session:
            repo = PaperRepository(session)
            for paper in papers:
                row = repo.upsert_metadata(paper)
                stats["stored"] += 1
                if row.pdf_processed:
                    stats["already_processed"] += 1
                else:
                    to_process.append(paper)

        if process_pdfs and self.parser and to_process:
            download_slots = asyncio.Semaphore(self.settings.arxiv.max_concurrent_downloads)
            parse_slots = asyncio.Semaphore(self.settings.pdf_parser.max_concurrent_parsing)
            results = await asyncio.gather(
                *(self._process_one(p, download_slots, parse_slots) for p in to_process),
                return_exceptions=True,
            )
            for paper, result in zip(to_process, results, strict=True):
                if isinstance(result, BaseException):
                    stats["errors"].append(f"{paper.arxiv_id}: {result}")
                    with self.db.session() as session:
                        PaperRepository(session).mark_failed(paper.arxiv_id, str(result))
                else:
                    stats["pdfs_parsed"] += 1

        logger.info(
            "Ingestion done: fetched=%d stored=%d parsed=%d errors=%d",
            stats["fetched"],
            stats["stored"],
            stats["pdfs_parsed"],
            len(stats["errors"]),
        )
        return stats

    async def _process_one(self, paper: ArxivPaper, download_slots: asyncio.Semaphore, parse_slots: asyncio.Semaphore):
        async with download_slots:
            pdf_path = await self.arxiv.download_pdf(paper, max_size_mb=self.settings.pdf_parser.max_file_size_mb)
        async with parse_slots:
            parsed = await asyncio.to_thread(self.parser.parse, pdf_path)
        with self.db.session() as session:
            PaperRepository(session).save_parsed_content(paper.arxiv_id, parsed)
        logger.info("Parsed %s: %d sections", paper.arxiv_id, len(parsed.sections))
