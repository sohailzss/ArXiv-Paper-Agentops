from typing import Literal

from pydantic import BaseModel, Field

GuardrailAction = Literal["none", "masked", "blocked", "flagged", "error"]


class GuardrailCheck(BaseModel):
    action: GuardrailAction = Field(
        description="none = passed; masked = PII replaced; blocked = refused; "
        "flagged = answer kept but not well supported by the passages; error = check failed (fail-open)"
    )
    reasons: list[str] = Field(default_factory=list, description="e.g. PROMPT_ATTACK, EMAIL, GROUNDING")
    grounding: float | None = Field(default=None, description="0..1, how well the answer is supported by the passages")
    relevance: float | None = Field(default=None, description="0..1, how well the answer addresses the question")


class GuardrailResult(BaseModel):
    input: GuardrailCheck | None = None
    output: GuardrailCheck | None = None
