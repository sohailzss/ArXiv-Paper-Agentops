from datetime import datetime

from pydantic import BaseModel, Field


class ArxivPaper(BaseModel):
    """A paper as returned by the arXiv API (metadata only)."""

    arxiv_id: str = Field(description="Base id without version, e.g. 2411.01234")
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_date: datetime
    pdf_url: str


class PaperSection(BaseModel):
    title: str
    content: str
    level: int = 1


class ParsedPDF(BaseModel):
    """Output of the PDF parser."""

    raw_text: str
    sections: list[PaperSection]
    parser_used: str = "docling"
    page_count: int | None = None
