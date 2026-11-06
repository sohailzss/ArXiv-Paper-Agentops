from typing import Any, Literal

from pydantic import BaseModel, Field

from src.schemas.guardrail import GuardrailResult
from src.schemas.rag import Source


class AgentStep(BaseModel):
    node: Literal["guardrail", "route", "retrieve", "grade", "rewrite", "generate", "direct", "give_up"]
    detail: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    query: str
    answer: str
    route: Literal["search", "direct", "blocked"]
    search_queries: list[str] = Field(description="Every query the agent searched with, in order")
    steps: list[AgentStep] = Field(description="What the agent did and why, in order")
    sources: list[Source] = Field(description="Only the passages judged relevant, numbered as cited")
    cited: list[int]
    llm_model: str
    embedding_model: str
    llm_calls: int
    total_ms: int
    trace_id: str | None = None
    guardrail: GuardrailResult | None = None
