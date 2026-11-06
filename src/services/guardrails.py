"""AWS Bedrock Guardrails around RAG and the agent.

Uses the standalone ApplyGuardrail API (not Converse's built-in guardrail), so the same checks run
whether answers come from OpenAI or Bedrock.

- Input check (before search/LLM): content filters, prompt attacks, denied topics -> refuse;
  PII -> masked question is used from then on (so it never reaches the LLM, cache or traces).
- Output check (after generation): content filters + PII on the answer, plus the contextual
  grounding check against the retrieved passages and the question. Grounding failures are
  "flagged" by default rather than blocked, because the check is strict about paraphrasing.

AWS returns one combined verdict, so this module reads the per-policy assessments itself to tell
a block from a mask from a grounding failure. The app always shows its own messages (from settings),
not the guardrail's configured ones.

Failure policy: if the guardrail call fails (network, credentials, throttling), fail_open=True lets
the request through and records action="error"; fail_open=False refuses it.
"""

import logging
from dataclasses import dataclass

from opentelemetry import trace

from src.config import GuardrailSettings
from src.schemas.guardrail import GuardrailCheck

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)

# ApplyGuardrail limits for contextual grounding (characters).
MAX_GROUNDING_SOURCE = 100_000
MAX_QUERY = 1_000
MAX_CONTENT = 5_000


@dataclass
class Verdict:
    check: GuardrailCheck
    text: str  # the text to use from now on (original, masked, or the refusal message)

    @property
    def blocked(self) -> bool:
        return self.check.action == "blocked"


def _policy_findings(assessments: list[dict]) -> tuple[list[str], list[str], dict[str, tuple[float, str]]]:
    """Split AWS assessments into (blocking reasons, masked PII types, grounding scores)."""
    blocked: list[str] = []
    masked: list[str] = []
    grounding: dict[str, tuple[float, str]] = {}
    for a in assessments:
        for f in a.get("contentPolicy", {}).get("filters", []):
            if f.get("action") == "BLOCKED":
                blocked.append(f["type"])
        for t in a.get("topicPolicy", {}).get("topics", []):
            if t.get("action") == "BLOCKED":
                blocked.append(f"TOPIC:{t['name']}")
        wp = a.get("wordPolicy", {})
        for w in wp.get("customWords", []) + wp.get("managedWordLists", []):
            if w.get("action") == "BLOCKED":
                blocked.append("WORDS")
        sp = a.get("sensitiveInformationPolicy", {})
        for e in sp.get("piiEntities", []) + sp.get("regexes", []):
            kind = e.get("type") or e.get("name", "REGEX")
            if e.get("action") == "BLOCKED":
                blocked.append(kind)
            elif e.get("action") == "ANONYMIZED":
                masked.append(kind)
        for f in a.get("contextualGroundingPolicy", {}).get("filters", []):
            grounding[f["type"]] = (float(f.get("score", 0.0)), f.get("action", "NONE"))
    return sorted(set(blocked)), sorted(set(masked)), grounding


class Guardrails:
    def __init__(self, settings: GuardrailSettings, client):
        self.settings = settings
        self.client = client

    @classmethod
    def from_settings(cls, settings: GuardrailSettings) -> "Guardrails | None":
        """None when disabled, unconfigured, or no AWS credentials are available."""
        if not settings.enabled:
            return None
        if not settings.id:
            logger.warning("GUARDRAILS__ENABLED=true but GUARDRAILS__ID is empty; guardrails off")
            return None
        import boto3
        from botocore.config import Config

        session = boto3.Session()
        if session.get_credentials() is None:
            logger.warning("No AWS credentials found (IAM role or ~/.aws); guardrails off")
            return None
        client = session.client(
            "bedrock-runtime",
            region_name=settings.region,
            config=Config(
                connect_timeout=3,
                read_timeout=settings.timeout_seconds,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )
        return cls(settings, client)

    # ---------- AWS call ----------

    def _apply(self, source: str, content: list[dict]) -> dict:
        return self.client.apply_guardrail(
            guardrailIdentifier=self.settings.id,
            guardrailVersion=self.settings.version,
            source=source,
            content=content,
        )

    def _failed(self, exc: Exception, original: str, refusal: str) -> Verdict:
        logger.error("Guardrail check failed (%s): %s", "allowing" if self.settings.fail_open else "refusing", exc)
        if self.settings.fail_open:
            return Verdict(GuardrailCheck(action="error", reasons=[type(exc).__name__]), original)
        return Verdict(GuardrailCheck(action="blocked", reasons=["GUARDRAIL_UNAVAILABLE"]), refusal)

    # ---------- checks ----------

    def check_input(self, question: str) -> Verdict:
        refusal = self.settings.blocked_input_message
        with _tracer.start_as_current_span("guardrail.input"):
            try:
                response = self._apply("INPUT", [{"text": {"text": question[:MAX_CONTENT]}}])
            except Exception as exc:
                return self._failed(exc, question, refusal)

        if response.get("action") != "GUARDRAIL_INTERVENED":
            return Verdict(GuardrailCheck(action="none"), question)
        blocked, masked, _ = _policy_findings(response.get("assessments", []))
        if blocked:
            return Verdict(GuardrailCheck(action="blocked", reasons=blocked), refusal)
        if masked:
            outputs = response.get("outputs") or []
            masked_text = outputs[0]["text"] if outputs else question
            return Verdict(GuardrailCheck(action="masked", reasons=masked), masked_text)
        return Verdict(GuardrailCheck(action="none"), question)

    def check_output(self, answer: str, question: str, passages: list[str]) -> Verdict:
        refusal = self.settings.blocked_output_message
        use_grounding = self.settings.grounding_action != "off" and bool(passages)
        content = [{"text": {"text": answer[:MAX_CONTENT], "qualifiers": ["guard_content"]}}]
        if use_grounding:
            source = "\n\n".join(passages)[:MAX_GROUNDING_SOURCE]
            content = [
                {"text": {"text": source, "qualifiers": ["grounding_source"]}},
                {"text": {"text": question[:MAX_QUERY], "qualifiers": ["query"]}},
                *content,
            ]

        with _tracer.start_as_current_span("guardrail.output", attributes={"guardrail.grounding": use_grounding}):
            try:
                response = self._apply("OUTPUT", content)
            except Exception as exc:
                return self._failed(exc, answer, refusal)

        blocked, masked, grounding = _policy_findings(response.get("assessments", []))
        scores = {
            "grounding": grounding.get("GROUNDING", (None, None))[0],
            "relevance": grounding.get("RELEVANCE", (None, None))[0],
        }
        if blocked:
            return Verdict(GuardrailCheck(action="blocked", reasons=blocked, **scores), refusal)

        weak = [kind for kind, (_, action) in grounding.items() if action == "BLOCKED"]
        if weak and self.settings.grounding_action == "block":
            return Verdict(GuardrailCheck(action="blocked", reasons=weak, **scores), refusal)

        text = answer
        if masked:
            # When grounding also intervened, AWS's output is its refusal text, not the masked answer:
            # ask again without grounding to get the masked version.
            if weak or not response.get("outputs"):
                text = self._mask_only(answer)
            else:
                text = response["outputs"][0]["text"]

        if weak:  # grounding_action == "flag"
            return Verdict(GuardrailCheck(action="flagged", reasons=weak + masked, **scores), text)
        if masked:
            return Verdict(GuardrailCheck(action="masked", reasons=masked, **scores), text)
        return Verdict(GuardrailCheck(action="none", **scores), text)

    def _mask_only(self, answer: str) -> str:
        try:
            response = self._apply("OUTPUT", [{"text": {"text": answer[:MAX_CONTENT]}}])
        except Exception as exc:
            logger.warning("PII masking retry failed, returning refusal instead of unmasked text: %s", exc)
            return self.settings.blocked_output_message
        outputs = response.get("outputs") or []
        return outputs[0]["text"] if outputs else answer


# ---------- helpers shared by RAG and the agent (record each check in the Langfuse trace) ----------


def guard_input(guardrails: "Guardrails | None", trace, question: str) -> Verdict | None:
    if guardrails is None or not guardrails.settings.check_input:
        return None
    span = trace.span("guardrail.input", as_type="guardrail", input={"length": len(question)})
    verdict = guardrails.check_input(question)
    span.end(
        output=verdict.check.model_dump(),
        level="WARNING" if verdict.check.action in ("blocked", "error") else None,
    )
    return verdict


def guard_output(
    guardrails: "Guardrails | None", trace, answer: str, question: str, passages: list[str]
) -> Verdict | None:
    if guardrails is None or not guardrails.settings.check_output or not answer:
        return None
    span = trace.span("guardrail.output", as_type="guardrail", input={"passages": len(passages)})
    verdict = guardrails.check_output(answer, question, passages)
    span.end(
        output=verdict.check.model_dump(),
        level="WARNING" if verdict.check.action in ("blocked", "flagged", "error") else None,
    )
    return verdict
