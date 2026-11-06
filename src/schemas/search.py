from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SearchRequest(BaseModel):
    query: str = Field(default="", max_length=500, description="Keywords. Empty = newest papers first.")
    size: int = Field(default=10, ge=1, le=50)
    offset: int = Field(default=0, ge=0, le=1000, description="For paging: skip this many results")
    categories: list[str] | None = Field(default=None, description='e.g. ["cs.AI", "cs.LG"]')
    date_from: date | None = None
    date_to: date | None = None

    @model_validator(mode="after")
    def check_dates(self) -> "SearchRequest":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to")
        return self


class SearchHit(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_date: datetime
    pdf_url: str
    has_full_text: bool = False
    score: float | None = None
    highlights: dict[str, list[str]] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    total: int
    size: int
    offset: int
    hits: list[SearchHit]


# ---------- Phase 4: hybrid search over chunks ----------

SearchMode = Literal["hybrid", "bm25", "vector"]


class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    mode: SearchMode = Field(default="hybrid", description="hybrid = BM25 + vectors fused with RRF")
    size: int = Field(default=10, ge=1, le=50)
    categories: list[str] | None = None
    date_from: date | None = None
    date_to: date | None = None

    @model_validator(mode="after")
    def check_dates(self) -> "HybridSearchRequest":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to")
        return self


class ChunkHit(BaseModel):
    chunk_id: str
    arxiv_id: str
    chunk_index: int
    section_titles: list[str]
    text: str
    title: str
    authors: list[str]
    categories: list[str]
    published_date: datetime
    pdf_url: str
    score: float = Field(description="RRF score (higher is better)")
    bm25_rank: int | None = Field(default=None, description="1-based rank in the keyword list, if present")
    vector_rank: int | None = Field(default=None, description="1-based rank in the vector list, if present")


class HybridSearchResponse(BaseModel):
    query: str
    mode: SearchMode
    embedding_model: str
    total: int
    hits: list[ChunkHit]
