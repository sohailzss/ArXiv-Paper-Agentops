"""Retrieval-augmented answers: hybrid search -> numbered passages -> LLM -> cited answer.

The LLM only sees the retrieved passages and must cite them as [n]; if nothing relevant is
retrieved we answer "not found" without calling the LLM at all (no cost, no made-up answer).

Phase 6: answers are cached in Redis (see services/cache.py) and every question is traced in
Langfuse (see services/tracing.py). Both are optional and never break a request.
Guardrails (services/guardrails.py): input check before anything else, output check before caching.
"""

import logging
import re
import time
from collections.abc import Iterator

from src.config import Settings, get_settings
from src.schemas.guardrail import GuardrailCheck, GuardrailResult
from src.schemas.rag import AskRequest, AskResponse, Source
from src.schemas.search import ChunkHit
from src.services.cache import RAGCache, cache_key
from src.services.guardrails import Guardrails, guard_input, guard_output
from src.services.hybrid_search import HybridSearchService
from src.services.llm.client import LLMClient
from src.services.online_eval import OnlineEvaluator
from src.services.tracing import Observation, Trace, Tracer, now, usage_details

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research assistant answering questions about recent arXiv papers.

Rules:
- Use ONLY the numbered passages provided. Do not use outside knowledge.
- Cite every factual claim with the passage number in square brackets, e.g. [1] or [2][3].
- If the message is a topic or a list of keywords rather than a question, summarise what the passages say
  about it, with citations.
- If the passages do not contain the answer, say so plainly instead of guessing.
- Mention which paper a finding comes from when it helps the reader.
- Be concise: give the direct answer first, then supporting detail. Use markdown."""

NO_RESULTS_ANSWER = (
    "I couldn't find any passages in the indexed papers that match this question. "
    "Try different keywords, or widen the category/date filters."
)

_CITATION = re.compile(r"\[(\d+)\]")

# Timings, the cached flag and the trace id describe one request, not the answer, so they aren't cached.
NOT_CACHED_FIELDS = {"retrieval_ms", "generation_ms", "cached", "trace_id"}


def to_sources(hits: list[ChunkHit]) -> list[Source]:
    return [
        Source(
            number=i,
            arxiv_id=h.arxiv_id,
            title=h.title,
            section_titles=h.section_titles,
            chunk_id=h.chunk_id,
            url=f"https://arxiv.org/abs/{h.arxiv_id}",
            pdf_url=h.pdf_url,
            text=h.text,
        )
        for i, h in enumerate(hits, start=1)
    ]


def format_passage(s: Source) -> str:
    """How a passage is shown to the LLM. The guardrail's grounding check gets the exact same text:
    if the answer mentions something the LLM saw (e.g. the arXiv id) but the check didn't, a correct
    answer scores as ungrounded (measured: 0.08 with bare text vs 0.98 with this format)."""
    section = "; ".join(s.section_titles)
    return f'[{s.number}] "{s.title}" (arXiv:{s.arxiv_id}), section: {section}\n{s.text}'


def build_user_prompt(query: str, sources: list[Source]) -> str:
    passages = "\n\n".join(format_passage(s) for s in sources)
    return f"Passages:\n\n{passages}\n\nQuestion: {query}"


def extract_citations(answer: str, source_count: int) -> list[int]:
    """Valid [n] citations in order of first appearance (ignores numbers outside 1..source_count)."""
    seen: list[int] = []
    for match in _CITATION.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= source_count and n not in seen:
            seen.append(n)
    return seen


def done_event(response: AskResponse) -> dict:
    return {
        "type": "done",
        "cited": response.cited,
        "llm_model": response.llm_model,
        "retrieval_ms": response.retrieval_ms,
        "generation_ms": response.generation_ms,
        "cached": response.cached,
        "trace_id": response.trace_id,
        "guardrail": response.guardrail.model_dump() if response.guardrail else None,
    }


class RAGService:
    def __init__(
        self,
        search: HybridSearchService,
        llm: LLMClient,
        settings: Settings | None = None,
        cache: RAGCache | None = None,
        tracer: Tracer | None = None,
        guardrails: Guardrails | None = None,
        online_eval: OnlineEvaluator | None = None,
    ):
        self.search = search
        self.llm = llm
        self.settings = settings or get_settings()
        self.cache = cache
        self.tracer = tracer or Tracer()
        self.guardrails = guardrails
        self.online_eval = online_eval

    def record_online(self, kind: str, request: AskRequest, answer: str, sources: list[Source], trace: Trace) -> None:
        """Queue a finished, grounded answer for DeepEval online scoring (sampled; scores go to Langfuse).
        Skips answers with nothing to judge: no passages, or no citations (refusals, blocked answers)."""
        if self.online_eval is None or not sources or not extract_citations(answer, len(sources)):
            return
        self.online_eval.submit(
            trace_id=trace.trace_id,
            question=request.query,
            answer=answer,
            passages=[format_passage(s) for s in sources],  # exactly what the LLM saw
        )

    # ---------- guardrails ----------

    def guard_request(self, trace: Trace, request: AskRequest) -> tuple[AskRequest, GuardrailCheck | None, str | None]:
        """Input check. Returns (request to use from now on, the check, refusal text if blocked)."""
        verdict = guard_input(self.guardrails, trace, request.query)
        if verdict is None:
            return request, None, None
        if verdict.blocked:
            return request, verdict.check, verdict.text
        if verdict.check.action == "masked":
            request = request.model_copy(update={"query": verdict.text})  # PII never reaches search/LLM/cache
        return request, verdict.check, None

    def guard_answer(
        self, trace: Trace, request: AskRequest, answer: str, sources: list[Source]
    ) -> tuple[str, GuardrailCheck | None]:
        """Output check (content, PII, grounding against the passages). Returns (answer to show, check)."""
        if not sources:  # the canned "nothing found" text needs no check
            return answer, None
        # Grounding only makes sense for answers that claim something from the passages. An answer that
        # cites nothing ("the passages don't cover this") would always score as ungrounded, so for those
        # only the content/PII checks run.
        cites_passages = bool(extract_citations(answer, len(sources)))
        passages = [format_passage(s) for s in sources] if cites_passages else []
        verdict = guard_output(self.guardrails, trace, answer, request.query, passages)
        if verdict is None:
            return answer, None
        return verdict.text, verdict.check

    def _blocked_response(self, request: AskRequest, refusal: str, check: GuardrailCheck, trace: Trace) -> AskResponse:
        response = self._response(request, refusal, [], None, 0, 0, trace)
        response.guardrail = GuardrailResult(input=check)
        return response

    # ---------- building blocks ----------

    def _top_k(self, request: AskRequest) -> int:
        return request.top_k or self.settings.rag.top_k

    def _cache_key(self, request: AskRequest) -> str | None:
        if self.cache is None:
            return None
        try:
            corpus_version = self.search.chunks.count()  # changes whenever new chunks are indexed
        except Exception as exc:
            logger.warning("Cache skipped (could not read corpus size): %s", exc)
            return None
        return cache_key(
            request.query,
            self._top_k(request),
            request.mode,
            request.categories,
            request.date_from.isoformat() if request.date_from else None,
            request.date_to.isoformat() if request.date_to else None,
            self.llm.model_id,
            self.search.embedder.model_id,
            corpus_version,
        )

    def _cache_lookup(self, trace: Trace, request: AskRequest) -> tuple[str | None, dict | None]:
        key = self._cache_key(request)
        cached = self.cache.get(key) if key else None
        trace.span("cache", input={"enabled": key is not None}).end(output={"hit": cached is not None})
        return key, cached

    def _cache_store(self, key: str | None, response: AskResponse) -> None:
        if key:
            value = response.model_dump(mode="json", exclude=NOT_CACHED_FIELDS)
            # The output check belongs to the answer; the input check is redone on every request.
            if value.get("guardrail"):
                value["guardrail"] = {"output": value["guardrail"].get("output")}
            self.cache.set(key, value)

    def retrieve(self, request: AskRequest, trace: Trace | None = None) -> tuple[list[Source], int]:
        trace = trace or Trace()
        span = trace.span(
            "retrieval",
            as_type="retriever",
            input={"query": request.query, "mode": request.mode, "top_k": self._top_k(request)},
            metadata={"categories": request.categories, "embedding_model": self.search.embedder.model_id},
        )
        started = time.perf_counter()
        try:
            result = self.search.search(
                request.query,
                mode=request.mode,
                size=self._top_k(request),
                categories=request.categories,
                date_from=request.date_from,
                date_to=request.date_to,
            )
        except Exception as exc:
            span.end(level="ERROR", status_message=str(exc)[:300])
            raise
        sources = to_sources(result.hits)
        span.end(output=[{"n": s.number, "chunk_id": s.chunk_id, "title": s.title} for s in sources])
        return sources, int((time.perf_counter() - started) * 1000)

    def _messages(self, query: str, sources: list[Source]) -> list[dict]:
        return [{"role": "user", "content": build_user_prompt(query, sources)}]

    def _generation(self, trace: Trace, messages: list[dict]) -> Observation:
        return trace.generation(
            "llm",
            model=self.llm.model,
            input=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            model_parameters={"max_tokens": self.settings.rag.max_tokens, "temperature": self.settings.rag.temperature},
            metadata={"provider": self.llm.provider},
        )

    def _start_trace(self, request: AskRequest, streaming: bool) -> Trace:
        return self.tracer.start_trace(
            "rag.ask",
            tags=["rag", "stream" if streaming else "json", request.mode],
            input={"query": request.query},
            metadata=request.model_dump(mode="json", exclude={"query"}),
        )

    def _cached_response(self, cached: dict, trace: Trace, input_check: GuardrailCheck | None = None) -> AskResponse:
        response = AskResponse(**cached, cached=True, retrieval_ms=0, generation_ms=0, trace_id=trace.trace_id)
        if input_check is not None or response.guardrail is not None:
            output_check = response.guardrail.output if response.guardrail else None
            response.guardrail = GuardrailResult(input=input_check, output=output_check)
        return response

    # ---------- JSON answer ----------

    def answer(self, request: AskRequest) -> AskResponse:
        trace = self._start_trace(request, streaming=False)
        try:
            request, input_check, refusal = self.guard_request(trace, request)
            if refusal is not None:
                response = self._blocked_response(request, refusal, input_check, trace)
                trace.end(output=refusal, metadata={"guardrail": "input blocked"})
                return response

            key, cached = self._cache_lookup(trace, request)
            if cached:
                response = self._cached_response(cached, trace, input_check)
                trace.end(output=response.answer, metadata={"cached": True})
                return response

            sources, retrieval_ms = self.retrieve(request, trace)
            if not sources:
                response = self._response(request, NO_RESULTS_ANSWER, [], None, retrieval_ms, 0, trace)
            else:
                messages = self._messages(request.query, sources)
                generation = self._generation(trace, messages)
                started = time.perf_counter()
                try:
                    result = self.llm.generate(SYSTEM_PROMPT, messages)
                except Exception as exc:
                    generation.end(level="ERROR", status_message=str(exc)[:300])
                    raise
                generation.end(output=result.text, usage_details=usage_details(result.usage))
                generation_ms = int((time.perf_counter() - started) * 1000)
                answer, output_check = self.guard_answer(trace, request, result.text, sources)
                response = self._response(
                    request, answer, sources, self.llm.model_id, retrieval_ms, generation_ms, trace
                )
                if input_check or output_check:
                    response.guardrail = GuardrailResult(input=input_check, output=output_check)

            if response.guardrail is None and input_check:
                response.guardrail = GuardrailResult(input=input_check)
            self._cache_store(key, response)
            self.record_online("rag.ask", request, response.answer, response.sources, trace)
            trace.end(output=response.answer, metadata={"cached": False, "cited": response.cited})
            return response
        except Exception as exc:
            trace.end(level="ERROR", status_message=str(exc)[:300])
            raise

    # ---------- streamed answer ----------

    def stream(self, request: AskRequest) -> Iterator[dict]:
        """Events: {"type": "sources"} first (so the UI can show them right away),
        then {"type": "token"} pieces, then {"type": "done"} with citations, timings and cache flag."""
        trace = self._start_trace(request, streaming=True)
        try:
            request, input_check, refusal = self.guard_request(trace, request)
            if refusal is not None:
                response = self._blocked_response(request, refusal, input_check, trace)
                yield {"type": "guardrail", "stage": "input", "check": input_check.model_dump()}
                yield {
                    "type": "sources",
                    "sources": [],
                    "embedding_model": self.search.embedder.model_id,
                    "retrieval_ms": 0,
                }
                yield {"type": "token", "text": refusal}
                yield done_event(response)
                trace.end(output=refusal, metadata={"guardrail": "input blocked"})
                return

            key, cached = self._cache_lookup(trace, request)
            if cached:
                # Shaped exactly like a live stream, so clients need no special case.
                yield {
                    "type": "sources",
                    "sources": cached["sources"],
                    "embedding_model": cached["embedding_model"],
                    "retrieval_ms": 0,
                }
                yield {"type": "token", "text": cached["answer"]}
                yield done_event(self._cached_response(cached, trace, input_check))
                trace.end(output=cached["answer"], metadata={"cached": True})
                return

            sources, retrieval_ms = self.retrieve(request, trace)
            yield {
                "type": "sources",
                "sources": [s.model_dump() for s in sources],
                "embedding_model": self.search.embedder.model_id,
                "retrieval_ms": retrieval_ms,
            }
            if not sources:
                answer, llm_model, generation_ms = NO_RESULTS_ANSWER, None, 0
                yield {"type": "token", "text": answer}
            else:
                messages = self._messages(request.query, sources)
                generation = self._generation(trace, messages)
                started, pieces, first_token_at = time.perf_counter(), [], None
                try:
                    for text in self.llm.stream(SYSTEM_PROMPT, messages):
                        if first_token_at is None:
                            first_token_at = now()  # Langfuse shows this as time to first token
                        pieces.append(text)
                        yield {"type": "token", "text": text}
                except Exception as exc:
                    generation.end(level="ERROR", status_message=str(exc)[:300], output="".join(pieces))
                    raise
                answer, llm_model = "".join(pieces), self.llm.model_id
                generation.end(output=answer, completion_start_time=first_token_at)
                generation_ms = int((time.perf_counter() - started) * 1000)

            # The answer has already been streamed, so the output check can't un-send it: it reports the
            # verdict, and if the answer must change (blocked or PII masked) clients replace what they showed.
            checked, output_check = self.guard_answer(trace, request, answer, sources)
            if output_check is not None:
                yield {
                    "type": "guardrail",
                    "stage": "output",
                    "check": output_check.model_dump(),
                    "replace_answer": checked if checked != answer else None,
                }
            response = self._response(request, checked, sources, llm_model, retrieval_ms, generation_ms, trace)
            if input_check or output_check:
                response.guardrail = GuardrailResult(input=input_check, output=output_check)
            self._cache_store(key, response)
            yield done_event(response)
            self.record_online("rag.ask", request, response.answer, response.sources, trace)
            trace.end(output=answer, metadata={"cached": False, "cited": response.cited})
        except Exception as exc:
            trace.end(level="ERROR", status_message=str(exc)[:300])
            raise

    def _response(self, request, answer, sources, llm_model, retrieval_ms, generation_ms, trace) -> AskResponse:
        return AskResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            cited=extract_citations(answer, len(sources)),
            llm_model=llm_model,
            embedding_model=self.search.embedder.model_id,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            trace_id=trace.trace_id,
        )
