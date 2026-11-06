"""All database reads and writes for papers live here, so services never write SQL."""

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.models.paper import Paper
from src.schemas.arxiv import ArxivPaper, ParsedPDF


class PaperRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_arxiv_id(self, arxiv_id: str) -> Paper | None:
        return self.session.scalar(select(Paper).where(Paper.arxiv_id == arxiv_id))

    def upsert_metadata(self, paper: ArxivPaper) -> Paper:
        """Insert a new paper, or refresh the metadata of an existing one."""
        existing = self.get_by_arxiv_id(paper.arxiv_id)
        data = paper.model_dump()
        if existing:
            for key, value in data.items():
                setattr(existing, key, value)
            row = existing
        else:
            row = Paper(**data)
            self.session.add(row)
        self.session.flush()
        return row

    def save_parsed_content(self, arxiv_id: str, parsed: ParsedPDF) -> None:
        row = self.get_by_arxiv_id(arxiv_id)
        if row is None:
            raise ValueError(f"Paper {arxiv_id} not found")
        row.raw_text = parsed.raw_text
        row.sections = [s.model_dump() for s in parsed.sections]
        row.parser_used = parsed.parser_used
        row.pdf_processed = True
        row.pdf_processing_date = datetime.now(UTC)
        row.processing_error = None
        self.session.flush()

    def mark_failed(self, arxiv_id: str, error: str) -> None:
        row = self.get_by_arxiv_id(arxiv_id)
        if row is not None:
            row.processing_error = error[:2000]
            self.session.flush()

    def list_unprocessed(self, limit: int = 50) -> list[Paper]:
        stmt = select(Paper).where(Paper.pdf_processed.is_(False)).order_by(Paper.published_date.desc()).limit(limit)
        return list(self.session.scalars(stmt))

    def list_for_indexing(self, arxiv_ids: list[str] | None = None, limit: int = 100, offset: int = 0) -> list[Paper]:
        """Papers in a stable order so callers can page through with offset."""
        stmt = select(Paper).order_by(Paper.published_date.desc(), Paper.arxiv_id)
        if arxiv_ids is not None:
            stmt = stmt.where(Paper.arxiv_id.in_(arxiv_ids))
        return list(self.session.scalars(stmt.limit(limit).offset(offset)))

    def count(self, processed_only: bool = False) -> int:
        stmt = select(func.count()).select_from(Paper)
        if processed_only:
            stmt = stmt.where(Paper.pdf_processed.is_(True))
        return self.session.scalar(stmt) or 0
