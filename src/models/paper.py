"""The `papers` table: one row per arXiv paper, plus its parsed PDF content."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from src.db.database import Base


def _now() -> datetime:
    return datetime.now(UTC)


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)

    # arXiv metadata
    arxiv_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(JSON, default=list)
    abstract: Mapped[str] = mapped_column(Text)
    categories: Mapped[list[str]] = mapped_column(JSON, default=list)
    published_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    pdf_url: Mapped[str] = mapped_column(String(255))

    # Parsed PDF content (filled by the PDF parser)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    sections: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    parser_used: Mapped[str | None] = mapped_column(String(32), nullable=True)
    pdf_processed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    pdf_processing_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    def __repr__(self) -> str:
        return f"<Paper {self.arxiv_id} processed={self.pdf_processed}>"
