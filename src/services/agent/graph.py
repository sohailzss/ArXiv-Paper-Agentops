"""Agentic RAG with LangGraph.

    START -> route --search--> retrieve -> grade --relevant--> generate -> END
               |                  ^           |--none, tries left--> rewrite --> (retrieve)
               |                  |           '--none, no tries----> give_up -> END
               '--direct--> direct -> END

- route:    is this a question for the paper index at all? Greetings, thanks and off-topic requests
            get a short direct reply, with no search and no made-up "research".
- grade:    one LLM call judges which retrieved passages actually help answer the question; only those
            reach the answer prompt. This is what stops weakly related hits from being cited.
- rewrite:  if nothing relevant came back, the LLM rewrites the question into a better search query
            (it sees the queries already tried) and the agent searches again, up to max_retrievals.

Nodes call the project's own LLMClient (OpenAI or Bedrock) and hybrid search, so the agent follows
LLM_PROVIDER / EMBEDDINGS__PROVIDER like the rest of the app. The final answer is streamed token by
token through LangGraph's custom stream; every node also appears as a span in the Langfuse trace.
"""

import json
import logging
import operator
import re
import time
from collections.abc import Iterator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from src.config import Settings, get_settings
from src.schemas.agent import AgentResponse, AgentStep
from src.schemas.guardrail import GuardrailResult
from src.schemas.rag import AskRequest, Source
from src.services.llm.client import LLMClient
from src.services.rag import SYSTEM_PROMPT, RAGService, build_user_prompt, extract_citations
from src.services.tracing import Trace, Tracer, now, usage_details

logger = logging.getLogger(__name__)

ROUTE_PROMPT = """You route messages for a research assistant that answers questions using an index of \
recent arXiv AI papers.

Reply with JSON only: {"route": "search" | "direct", "reason": "<a few words>"}

- "search": any question that could be answered from research papers (methods, results, benchmarks, \
models, comparisons, "which papers...", definitions of research concepts).
- "direct": greetings, thanks, questions about the assistant itself, or requests clearly unrelated to \
research papers (recipes, weather, personal advice)."""

DIRECT_PROMPT = """You are a research assistant for recent arXiv AI papers. The user's message does not \
need a paper search. Reply briefly and politely. If it is off-topic, say you can only answer questions \
about the indexed research papers and suggest what they could ask instead. Never invent research findings."""

GRADE_PROMPT = """You check retrieved passages for a research question.

Reply with JSON only: {"relevant": [<passage numbers>]}

A passage is relevant if it contains information that helps answer the question, even partially. \
Passages that only share keywords but are about something else are NOT relevant. \
If none are relevant, reply {"relevant": []}."""

REWRITE_PROMPT = """A search over arXiv AI papers found nothing relevant for the user's question.
Write ONE better search query: use the technical terms a paper would use, add likely synonyms, drop \
filler words. Do not repeat a query that was already tried.

Reply with JSON only: {"query": "<new search query>"}"""

GIVE_UP_ANSWER = (
    "I searched the indexed papers but couldn't find passages that answer this question "
    "(tried: {queries}). The papers ingested so far may not cover it; try rephrasing, or widen "
    "the category/date filters."
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class AgentState(TypedDict, total=False):
    question: str
    options: dict  # mode, top_k, categories, date_from, date_to from the request
    route: Literal["search", "direct"]
    search_query: str
    search_queries: Annotated[list[str], operator.add]
    attempts: int
    retrieved: list[Source]
    relevant: list[Source]
    answer: str
    steps: Annotated[list[dict], operator.add]
    llm_calls: Annotated[int, operator.add]


def parse_json(text: str) -> dict | None:
    """LLMs sometimes wrap JSON in prose or code fences; take the first {...} block."""
    match = _JSON_OBJECT.search(text or "")
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def renumber(sources: list[Source]) -> list[Source]:
    """Relevant passages become [1]..[n] so the answer's citations match what the user sees."""
    return [s.model_copy(update={"number": i}) for i, s in enumerate(sources, start=1)]


class AgentService:
    def __init__(
        self,
        rag: RAGService,
        llm: LLMClient,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
    ):
        self.rag = rag  # reused for hybrid retrieval and its prompt helpers
        self.llm = llm
        self.settings = settings or get_settings()
        self.tracer = tracer or Tracer()
        self.graph = self._build()

    # ---------- LLM helpers (each call is a Langfuse generation) ----------

    @staticmethod
    def _trace(config: RunnableConfig) -> Trace:
        return config.get("configurable", {}).get("trace") or Trace()

    def _llm_json(self, config: RunnableConfig, name: str, system: str, user: str) -> dict | None:
        messages = [{"role": "user", "content": user}]
        generation = self._trace(config).generation(
            name, model=self.llm.model, input=[{"role": "system", "content": system}, *messages]
        )
        try:
            result = self.llm.generate(system, messages)
        except Exception as exc:
            generation.end(level="ERROR", status_message=str(exc)[:300])
            raise
        generation.end(output=result.text, usage_details=usage_details(result.usage))
        parsed = parse_json(result.text)
        if parsed is None:
            logger.warning("%s: LLM did not return JSON: %r", name, result.text[:200])
        return parsed

    def _llm_stream(self, config: RunnableConfig, name: str, system: str, user: str) -> str:
        """Streams tokens to the client via LangGraph's custom stream, returns the full text."""
        writer = get_stream_writer()
        messages = [{"role": "user", "content": user}]
        generation = self._trace(config).generation(
            name, model=self.llm.model, input=[{"role": "system", "content": system}, *messages]
        )
        pieces, first_token_at = [], None
        try:
            for text in self.llm.stream(system, messages):
                first_token_at = first_token_at or now()
                pieces.append(text)
                writer({"type": "token", "text": text})
        except Exception as exc:
            generation.end(level="ERROR", status_message=str(exc)[:300], output="".join(pieces))
            raise
        answer = "".join(pieces)
        generation.end(output=answer, completion_start_time=first_token_at)
        return answer

    # ---------- nodes ----------

    def route(self, state: AgentState, config: RunnableConfig) -> dict:
        decision = self._llm_json(config, "route", ROUTE_PROMPT, f"Message: {state['question']}") or {}
        # Fail open: if the router answers badly, searching is the safe default.
        route = "direct" if decision.get("route") == "direct" else "search"
        reason = str(decision.get("reason", ""))[:200]
        return {
            "route": route,
            "search_query": state["question"],
            "attempts": 0,
            "steps": [{"node": "route", "detail": {"route": route, "reason": reason}}],
            "llm_calls": 1,
        }

    def retrieve(self, state: AgentState, config: RunnableConfig) -> dict:
        options = state.get("options", {})
        request = AskRequest(query=state["search_query"], **options)
        sources, retrieval_ms = self.rag.retrieve(request, self._trace(config))
        attempt = state.get("attempts", 0) + 1
        return {
            "retrieved": sources,
            "attempts": attempt,
            "search_queries": [state["search_query"]],
            "steps": [
                {
                    "node": "retrieve",
                    "detail": {
                        "query": state["search_query"],
                        "attempt": attempt,
                        "found": len(sources),
                        "ms": retrieval_ms,
                    },
                }
            ],
        }

    def grade(self, state: AgentState, config: RunnableConfig) -> dict:
        retrieved = state.get("retrieved", [])
        if not retrieved:
            return {"relevant": [], "steps": [{"node": "grade", "detail": {"relevant": [], "of": 0}}]}

        limit = self.settings.agent.grade_passage_words
        passages = "\n\n".join(
            f'[{s.number}] "{s.title}" ({"; ".join(s.section_titles)}): {" ".join(s.text.split()[:limit])}'
            for s in retrieved
        )
        user = f"Question: {state['question']}\n\nPassages:\n\n{passages}"
        verdict = self._llm_json(config, "grade", GRADE_PROMPT, user)
        if verdict is None or not isinstance(verdict.get("relevant"), list):
            keep = [s.number for s in retrieved]  # can't tell: keep everything rather than lose answers
        else:
            valid = {s.number for s in retrieved}
            keep = [n for n in verdict["relevant"] if isinstance(n, int) and n in valid]
        relevant = [s for s in retrieved if s.number in keep]
        return {
            "relevant": relevant,
            "steps": [{"node": "grade", "detail": {"relevant": keep, "of": len(retrieved)}}],
            "llm_calls": 1,
        }

    def rewrite(self, state: AgentState, config: RunnableConfig) -> dict:
        tried = "\n".join(f"- {q}" for q in state.get("search_queries", []))
        user = f"Question: {state['question']}\n\nQueries already tried:\n{tried}"
        result = self._llm_json(config, "rewrite", REWRITE_PROMPT, user) or {}
        query = str(result.get("query") or "").strip()
        if not query or query.lower() in {q.lower() for q in state.get("search_queries", [])}:
            query = f"{state['question']} methods results"  # still a different query than before
        return {
            "search_query": query[:300],
            "steps": [{"node": "rewrite", "detail": {"query": query[:300]}}],
            "llm_calls": 1,
        }

    def generate(self, state: AgentState, config: RunnableConfig) -> dict:
        sources = renumber(state["relevant"])
        get_stream_writer()({"type": "sources", "sources": [s.model_dump() for s in sources]})
        answer = self._llm_stream(config, "answer", SYSTEM_PROMPT, build_user_prompt(state["question"], sources))
        return {
            "relevant": sources,
            "answer": answer,
            "steps": [{"node": "generate", "detail": {"passages": len(sources)}}],
            "llm_calls": 1,
        }

    def direct(self, state: AgentState, config: RunnableConfig) -> dict:
        get_stream_writer()({"type": "sources", "sources": []})
        answer = self._llm_stream(config, "direct", DIRECT_PROMPT, state["question"])
        return {"relevant": [], "answer": answer, "steps": [{"node": "direct", "detail": {}}], "llm_calls": 1}

    def give_up(self, state: AgentState, config: RunnableConfig) -> dict:
        writer = get_stream_writer()
        writer({"type": "sources", "sources": []})
        queries = "; ".join(f'"{q}"' for q in state.get("search_queries", []))
        answer = GIVE_UP_ANSWER.format(queries=queries)
        writer({"type": "token", "text": answer})
        return {
            "relevant": [],
            "answer": answer,
            "steps": [{"node": "give_up", "detail": {"attempts": state.get("attempts", 0)}}],
        }

    # ---------- edges ----------

    @staticmethod
    def after_route(state: AgentState) -> str:
        return "retrieve" if state["route"] == "search" else "direct"

    def after_grade(self, state: AgentState) -> str:
        if state.get("relevant"):
            return "generate"
        if state.get("attempts", 0) < self.settings.agent.max_retrievals:
            return "rewrite"
        return "give_up"

    def _build(self):
        graph = StateGraph(AgentState)
        for name in ("route", "retrieve", "grade", "rewrite", "generate", "direct", "give_up"):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, "route")
        graph.add_conditional_edges("route", self.after_route, ["retrieve", "direct"])
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges("grade", self.after_grade, ["generate", "rewrite", "give_up"])
        graph.add_edge("rewrite", "retrieve")
        for name in ("generate", "direct", "give_up"):
            graph.add_edge(name, END)
        return graph.compile()

    # ---------- public API ----------

    def stream(self, request: AskRequest) -> Iterator[dict]:
        """Events: `step` after each node, `sources` before the answer, `token`s, then `done`."""
        yield from self._events(request, result={})

    def run(self, request: AskRequest) -> AgentResponse:
        """Non-streaming: drives the same graph run and returns the assembled response."""
        result: dict[str, AgentResponse] = {}  # per call, so concurrent requests never share state
        for _ in self._events(request, result):
            pass
        return result["response"]

    def _events(self, request: AskRequest, result: dict) -> Iterator[dict]:
        started = time.perf_counter()
        trace = self.tracer.start_trace(
            "agent.ask",
            tags=["agent", request.mode],
            input={"query": request.query},
            metadata=request.model_dump(mode="json", exclude={"query"}),
        )
        # Guardrail input check first: blocked questions never reach the router, search or LLM;
        # masked questions (PII replaced) are what the agent works with from here on.
        try:
            request, input_check, refusal = self.rag.guard_request(trace, request)
        except Exception as exc:
            trace.end(level="ERROR", status_message=str(exc)[:300])
            raise
        guardrail = GuardrailResult(input=input_check) if input_check else None
        if input_check is not None:
            step = {"node": "guardrail", "detail": {"stage": "input", **input_check.model_dump(exclude_none=True)}}
            yield {"type": "step", **step}
        if refusal is not None:
            yield {"type": "sources", "sources": []}
            yield {"type": "token", "text": refusal}
            final = {"route": "blocked", "answer": refusal, "steps": [step], "relevant": []}
            response = self._response(request, final, trace, started, guardrail)
            result["response"] = response
            trace.end(output=refusal, metadata={"guardrail": "input blocked"})
            yield {"type": "done", **response.model_dump(mode="json", exclude={"answer", "sources", "query"})}
            return

        options = request.model_dump(exclude={"query"}, exclude_none=True)
        state: dict[str, Any] = {"question": request.query, "options": options}
        final: dict[str, Any] = dict(state)
        if input_check is not None:
            final["steps"] = [step]
        try:
            for mode, chunk in self.graph.stream(
                state, config={"configurable": {"trace": trace}}, stream_mode=["updates", "custom"]
            ):
                if mode == "custom":
                    yield chunk
                    continue
                # chunk is {node_name: state_update}; merge updates the same way the graph's reducers do.
                for update in chunk.values():
                    for key, value in (update or {}).items():
                        if key in ("steps", "search_queries"):
                            final[key] = final.get(key, []) + value
                        elif key == "llm_calls":
                            final[key] = final.get(key, 0) + value
                        else:
                            final[key] = value
                    for step in (update or {}).get("steps", []):
                        yield {"type": "step", **step}
            # Output check on the final, grounded answer (direct replies and give-ups have no passages).
            if final.get("route") == "search" and final.get("relevant"):
                answer = final.get("answer", "")
                checked, output_check = self.rag.guard_answer(trace, request, answer, final["relevant"])
                if output_check is not None:
                    guardrail = GuardrailResult(input=input_check, output=output_check)
                    step = {
                        "node": "guardrail",
                        "detail": {"stage": "output", **output_check.model_dump(exclude_none=True)},
                    }
                    final["steps"] = final.get("steps", []) + [step]
                    yield {"type": "step", **step}
                    yield {
                        "type": "guardrail",
                        "stage": "output",
                        "check": output_check.model_dump(),
                        "replace_answer": checked if checked != answer else None,
                    }
                    final["answer"] = checked
        except Exception as exc:
            trace.end(level="ERROR", status_message=str(exc)[:300])
            raise

        response = self._response(request, final, trace, started, guardrail)
        result["response"] = response
        if response.route == "search":
            self.rag.record_online("agent.ask", request, response.answer, response.sources, trace)
        trace.end(
            output=response.answer,
            metadata={"route": response.route, "search_queries": response.search_queries, "cited": response.cited},
        )
        yield {"type": "done", **response.model_dump(mode="json", exclude={"answer", "sources", "query"})}

    def _response(
        self, request: AskRequest, final: dict, trace: Trace, started: float, guardrail: GuardrailResult | None = None
    ) -> AgentResponse:
        sources = final.get("relevant", [])
        answer = final.get("answer", "")
        return AgentResponse(
            query=request.query,
            answer=answer,
            route=final.get("route", "search"),
            search_queries=final.get("search_queries", []),
            steps=[AgentStep(**s) for s in final.get("steps", [])],
            sources=sources,
            cited=extract_citations(answer, len(sources)),
            llm_model=self.llm.model_id,
            embedding_model=self.rag.search.embedder.model_id,
            llm_calls=final.get("llm_calls", 0),
            total_ms=int((time.perf_counter() - started) * 1000),
            trace_id=trace.trace_id,
            guardrail=guardrail,
        )
