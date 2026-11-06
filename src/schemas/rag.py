from datetime import date

from pydantic import BaseModel, Field, model_validator

from src.schemas.guardrail import GuardrailResult
from src.schemas.search import SearchMode


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000, description="A question about the indexed papers")
    top_k: int | None = Field(
        default=None, ge=1, le=10, description="Passages given to the LLM (default from settings)"
    )
    mode: SearchMode = "hybrid"
    categories: list[str] | None = None
    date_from: date | None = None
    date_to: date | None = None

    @model_validator(mode="after")
    def check_dates(self) -> "AskRequest":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to")
        return self


class Source(BaseModel):
    number: int = Field(description="The [n] used in the answer")
    arxiv_id: str
    title: str
    section_titles: list[str]
    chunk_id: str
    url: str
    pdf_url: str
    text: str


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]
    cited: list[int] = Field(description="Source numbers the answer actually cites")
    llm_model: str | None
    embedding_model: str
    retrieval_ms: int
    generation_ms: int
    cached: bool = Field(default=False, description="Served from the Redis cache (no search or LLM call)")
    trace_id: str | None = Field(default=None, description="Langfuse trace id, when tracing is on")
    guardrail: GuardrailResult | None = Field(default=None, description="AWS Guardrails results, when enabled")
