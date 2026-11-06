from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.repositories.paper import PaperRepository
from src.schemas.arxiv import ArxivPaper, PaperSection, ParsedPDF
from src.services.metadata_fetcher import MetadataFetcher
from src.services.pdf_parser.parser import PDFParserService


def make_paper(arxiv_id: str) -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        authors=["A. Author"],
        abstract="Abstract",
        categories=["cs.AI"],
        published_date=datetime(2026, 11, 1, tzinfo=UTC),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


class FakeArxiv:
    def __init__(self, papers):
        self.papers = papers

    async def fetch_papers(self, **kwargs):
        return self.papers

    async def download_pdf(self, paper, max_size_mb=20):
        if paper.arxiv_id == "bad":
            raise RuntimeError("download failed")
        return Path(f"/tmp/{paper.arxiv_id}.pdf")


class FakeParser:
    def parse(self, path):
        return ParsedPDF(raw_text="# Intro\nhello", sections=[PaperSection(title="Intro", content="hello")])


def test_split_sections_on_markdown_headings():
    md = "Title text\n\n## 1 Introduction\nWe start.\n\n## 2 Method\nWe do.\n### 2.1 Detail\nMore."
    sections = PDFParserService.split_sections(md)
    assert [s.title for s in sections] == ["Preamble", "1 Introduction", "2 Method", "2.1 Detail"]
    assert sections[3].level == 3


def test_repository_upsert_is_idempotent(database):
    with database.session() as s:
        PaperRepository(s).upsert_metadata(make_paper("1111.0001"))
    with database.session() as s:
        PaperRepository(s).upsert_metadata(make_paper("1111.0001"))
        assert PaperRepository(s).count() == 1


@pytest.mark.asyncio
async def test_fetcher_parses_good_papers_and_records_failures(database):
    fetcher = MetadataFetcher(FakeArxiv([make_paper("2611.0001"), make_paper("bad")]), FakeParser(), database)
    stats = await fetcher.fetch_and_process(from_date=date(2026, 11, 1), to_date=date(2026, 11, 1))

    assert stats["fetched"] == 2
    assert stats["pdfs_parsed"] == 1
    assert len(stats["errors"]) == 1

    with database.session() as s:
        repo = PaperRepository(s)
        assert repo.get_by_arxiv_id("2611.0001").pdf_processed is True
        assert "download failed" in repo.get_by_arxiv_id("bad").processing_error

    # A second run skips the already-processed paper.
    stats2 = await fetcher.fetch_and_process()
    assert stats2["already_processed"] == 1
