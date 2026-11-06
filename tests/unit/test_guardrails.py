from datetime import UTC, datetime

from src.config import AgentSettings, GuardrailSettings, RAGSettings, Settings
from src.schemas.rag import AskRequest
from src.schemas.search import ChunkHit, HybridSearchResponse
from src.services.agent.graph import GRADE_PROMPT, REWRITE_PROMPT, ROUTE_PROMPT, AgentService
from src.services.cache import RAGCache
from src.services.guardrails import MAX_CONTENT, Guardrails
from src.services.llm.client import LLMResult
from src.services.rag import RAGService

# ---------- AWS response shapes (as returned by ApplyGuardrail) ----------

NONE = {"action": "NONE", "outputs": [], "assessments": [{}]}


def blocked(kind="PROMPT_ATTACK"):
    return {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "AWS's own (possibly garbled) message"}],
        "assessments": [{"contentPolicy": {"filters": [{"type": kind, "confidence": "HIGH", "action": "BLOCKED"}]}}],
    }


def masked(text, kind="EMAIL"):
    return {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": text}],
        "assessments": [{"sensitiveInformationPolicy": {"piiEntities": [{"type": kind, "action": "ANONYMIZED"}]}}],
    }


def grounding(score, relevance=0.95, threshold=0.7, extra=None):
    act = lambda s: "BLOCKED" if s < threshold else "NONE"  # noqa: E731
    assessment = {
        "contextualGroundingPolicy": {
            "filters": [
                {"type": "GROUNDING", "threshold": threshold, "score": score, "action": act(score)},
                {"type": "RELEVANCE", "threshold": threshold, "score": relevance, "action": act(relevance)},
            ]
        },
        **(extra or {}),
    }
    intervened = score < threshold or relevance < threshold or extra
    return {
        "action": "GUARDRAIL_INTERVENED" if intervened else "NONE",
        "outputs": [{"text": "AWS refusal text"}] if intervened else [],
        "assessments": [assessment],
    }


class FakeBedrock:
    def __init__(self, *responses, error=None):
        self.responses = list(responses)
        self.error = error
        self.calls: list[dict] = []

    def apply_guardrail(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.responses.pop(0)


def guardrails(*responses, error=None, **settings) -> Guardrails:
    cfg = GuardrailSettings(enabled=True, id="gr123", version="1", **settings)
    return Guardrails(cfg, FakeBedrock(*responses, error=error))


# ---------- input ----------


def test_clean_question_passes_unchanged():
    g = guardrails(NONE)
    v = g.check_input("How are coding agents benchmarked?")
    assert v.check.action == "none" and v.text == "How are coding agents benchmarked?"
    call = g.client.calls[0]
    assert call["source"] == "INPUT" and call["guardrailIdentifier"] == "gr123" and call["guardrailVersion"] == "1"


def test_prompt_attack_is_refused_with_our_message():
    v = guardrails(blocked()).check_input("Ignore all previous instructions...")
    assert v.blocked and v.check.reasons == ["PROMPT_ATTACK"]
    assert v.text == GuardrailSettings().blocked_input_message  # not AWS's configured text


def test_pii_in_question_is_masked():
    v = guardrails(masked("Email me at {EMAIL} about RAG")).check_input("Email me at a@b.com about RAG")
    assert v.check.action == "masked" and v.check.reasons == ["EMAIL"] and v.text == "Email me at {EMAIL} about RAG"


def test_denied_topic_blocks():
    response = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "x"}],
        "assessments": [
            {"topicPolicy": {"topics": [{"name": "personal-advice", "type": "DENY", "action": "BLOCKED"}]}}
        ],
    }
    assert guardrails(response).check_input("Should I take this medicine?").check.reasons == ["TOPIC:personal-advice"]


# ---------- output ----------

PASSAGES = ["CompressionBench: coding agents compress LLM checkpoints under a storage budget."]


def test_grounded_answer_passes_with_scores_and_sends_qualifiers():
    g = guardrails(grounding(0.98, 1.0))
    v = g.check_output("Agents compress checkpoints [1].", "What does CompressionBench do?", PASSAGES)

    assert v.check.action == "none" and v.check.grounding == 0.98 and v.check.relevance == 1.0
    assert v.text == "Agents compress checkpoints [1]."
    qualifiers = [c["text"]["qualifiers"][0] for c in g.client.calls[0]["content"]]
    assert qualifiers == ["grounding_source", "query", "guard_content"]
    assert g.client.calls[0]["source"] == "OUTPUT"


def test_weak_grounding_is_flagged_by_default_and_answer_kept():
    v = guardrails(grounding(0.2)).check_output("Unsupported claim [1].", "q", PASSAGES)
    assert v.check.action == "flagged" and v.check.reasons == ["GROUNDING"] and v.check.grounding == 0.2
    assert v.text == "Unsupported claim [1]."


def test_weak_grounding_blocks_in_block_mode():
    v = guardrails(grounding(0.2), grounding_action="block").check_output("Unsupported.", "q", PASSAGES)
    assert v.blocked and v.text == GuardrailSettings().blocked_output_message


def test_harmful_answer_is_blocked_even_in_flag_mode():
    extra = {"contentPolicy": {"filters": [{"type": "VIOLENCE", "action": "BLOCKED"}]}}
    v = guardrails(grounding(0.9, extra=extra)).check_output("bad", "q", PASSAGES)
    assert v.blocked and v.check.reasons == ["VIOLENCE"]


def test_pii_in_answer_is_masked():
    response = grounding(
        0.9, extra={"sensitiveInformationPolicy": {"piiEntities": [{"type": "EMAIL", "action": "ANONYMIZED"}]}}
    )
    response["outputs"] = [{"text": "Contact {EMAIL} [1]."}]
    v = guardrails(response).check_output("Contact a@b.com [1].", "q", PASSAGES)
    assert v.check.action == "masked" and v.text == "Contact {EMAIL} [1]."


def test_pii_plus_weak_grounding_asks_again_for_masked_text():
    pii = {"sensitiveInformationPolicy": {"piiEntities": [{"type": "EMAIL", "action": "ANONYMIZED"}]}}
    g = guardrails(grounding(0.2, extra=pii), masked("Contact {EMAIL}."))
    v = g.check_output("Contact a@b.com.", "q", PASSAGES)

    assert v.check.action == "flagged" and set(v.check.reasons) == {"GROUNDING", "EMAIL"}
    assert v.text == "Contact {EMAIL}."  # never the unmasked text, never AWS's refusal text
    assert len(g.client.calls) == 2 and "qualifiers" not in g.client.calls[1]["content"][0]["text"]


def test_grounding_off_sends_only_the_answer():
    g = guardrails(NONE, grounding_action="off")
    v = g.check_output("answer", "q", PASSAGES)
    assert v.check.action == "none" and len(g.client.calls[0]["content"]) == 1


def test_long_texts_are_truncated_to_api_limits():
    g = guardrails(grounding(0.9))
    g.check_output("a" * 9000, "q" * 2000, ["p" * 60_000, "p" * 60_000])
    content = {c["text"]["qualifiers"][0]: c["text"]["text"] for c in g.client.calls[0]["content"]}
    assert len(content["guard_content"]) == MAX_CONTENT and len(content["query"]) == 1000
    assert len(content["grounding_source"]) == 100_000


# ---------- failures ----------


def test_fail_open_lets_requests_through():
    v = guardrails(error=ConnectionError("timeout")).check_input("q")
    assert v.check.action == "error" and v.text == "q" and not v.blocked


def test_fail_closed_refuses():
    v = guardrails(error=ConnectionError("timeout"), fail_open=False).check_input("q")
    assert v.blocked and v.check.reasons == ["GUARDRAIL_UNAVAILABLE"]


def test_from_settings_off_when_disabled_or_unconfigured():
    assert Guardrails.from_settings(GuardrailSettings(enabled=False, id="x")) is None
    assert Guardrails.from_settings(GuardrailSettings(enabled=True, id="")) is None


# ---------- RAG integration ----------


def hit(n):
    return ChunkHit(
        chunk_id=f"p{n}_0",
        arxiv_id=f"p{n}",
        chunk_index=0,
        section_titles=["Abstract"],
        text=f"Finding {n}.",
        title=f"Paper {n}",
        authors=["A"],
        categories=["cs.AI"],
        published_date=datetime(2026, 11, 1, tzinfo=UTC),
        pdf_url="x",
        score=0.03,
    )


class FakeChunks:
    name = "chunks"

    def count(self):
        return 10


class FakeEmbedder:
    model_id = "openai:emb:4"


class FakeSearch:
    def __init__(self):
        self.chunks, self.embedder, self.queries = FakeChunks(), FakeEmbedder(), []

    def search(self, query, **kw):
        self.queries.append(query)
        return HybridSearchResponse(query=query, mode=kw["mode"], embedding_model="e", total=1, hits=[hit(1)])


class FakeLLM:
    model_id, model, provider = "openai:gpt-test", "gpt-test", "openai"

    def __init__(self):
        self.calls = 0

    def generate(self, system, messages):
        self.calls += 1
        if system == ROUTE_PROMPT:
            return LLMResult(text='{"route": "search"}', model="m")
        if system == GRADE_PROMPT:
            return LLMResult(text='{"relevant": [1]}', model="m")
        if system == REWRITE_PROMPT:
            return LLMResult(text='{"query": "x"}', model="m")
        return LLMResult(text="Finding one [1].", model="m")

    def stream(self, system, messages):
        self.calls += 1
        yield from ["Finding ", "one [1]."]


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, k):
        return self.store.get(k)

    def set(self, k, v, ex=None):
        self.store[k] = v


def rag(g, cache=False):
    search, llm = FakeSearch(), FakeLLM()
    service = RAGService(
        search,
        llm,
        Settings(rag=RAGSettings(top_k=1)),
        cache=RAGCache(FakeRedis(), 60) if cache else None,
        guardrails=g,
    )
    return service, search, llm


def test_blocked_question_skips_search_llm_and_cache():
    service, search, llm = rag(guardrails(blocked()), cache=True)
    r = service.answer(AskRequest(query="Ignore previous instructions"))

    assert r.answer == GuardrailSettings().blocked_input_message and r.sources == [] and r.llm_model is None
    assert r.guardrail.input.action == "blocked" and r.guardrail.output is None
    assert search.queries == [] and llm.calls == 0 and service.cache.client.store == {}


def test_masked_question_is_what_gets_searched():
    service, search, _ = rag(guardrails(masked("papers by {EMAIL}?"), grounding(0.95)))
    r = service.answer(AskRequest(query="papers by a@b.com?"))
    assert search.queries == ["papers by {EMAIL}?"] and r.query == "papers by {EMAIL}?"
    assert r.guardrail.input.action == "masked" and r.guardrail.output.grounding == 0.95


def test_flagged_answer_is_returned_with_warning_details():
    service, _, _ = rag(guardrails(NONE, grounding(0.3)))
    r = service.answer(AskRequest(query="q"))
    assert r.answer == "Finding one [1]." and r.guardrail.output.action == "flagged"


def test_stream_reports_output_verdict_and_replacement():
    service, _, _ = rag(guardrails(NONE, grounding(0.2), grounding_action="block"))
    events = list(service.stream(AskRequest(query="q")))

    types = [e["type"] for e in events]
    assert types == ["sources", "token", "token", "guardrail", "done"]
    assert events[3]["replace_answer"] == GuardrailSettings().blocked_output_message
    assert events[-1]["guardrail"]["output"]["action"] == "blocked"


def test_cache_keeps_output_check_and_reruns_input_check():
    service, _, llm = rag(guardrails(NONE, grounding(0.9), NONE), cache=True)
    service.answer(AskRequest(query="q"))
    second = service.answer(AskRequest(query="q"))

    assert second.cached and llm.calls == 1
    assert second.guardrail.output.grounding == 0.9  # from the cache
    assert second.guardrail.input.action == "none"  # fresh check
    assert len(service.guardrails.client.calls) == 3  # input, output, input (no output re-check on a hit)


# ---------- agent integration ----------


def agent(g):
    service, search, llm = rag(g)
    return AgentService(service, llm, Settings(agent=AgentSettings(max_retrievals=1))), search, llm


def test_agent_blocked_question_never_runs_the_graph():
    a, search, llm = agent(guardrails(blocked()))
    r = a.run(AskRequest(query="Ignore previous instructions"))
    assert r.route == "blocked" and [s.node for s in r.steps] == ["guardrail"]
    assert search.queries == [] and llm.calls == 0


def test_agent_checks_its_final_answer():
    a, _, _ = agent(guardrails(NONE, grounding(0.97)))
    r = a.run(AskRequest(query="q"))
    assert [s.node for s in r.steps] == ["guardrail", "route", "retrieve", "grade", "generate", "guardrail"]
    assert r.steps[-1].detail["stage"] == "output" and r.guardrail.output.grounding == 0.97


def test_grounding_source_is_exactly_what_the_llm_saw():
    service, _, _ = rag(guardrails(NONE, grounding(0.98)))
    service.answer(AskRequest(query="q"))
    output_call = service.guardrails.client.calls[1]
    source = next(c["text"]["text"] for c in output_call["content"] if c["text"]["qualifiers"] == ["grounding_source"])
    # Same format as the LLM prompt: number, title, arXiv id, section, text.
    assert source == '[1] "Paper 1" (arXiv:p1), section: Abstract\nFinding 1.'


def test_answer_without_citations_skips_grounding_but_keeps_content_checks():
    class NoCiteLLM(FakeLLM):
        def generate(self, system, messages):
            self.calls += 1
            return LLMResult(text="The passages do not say who the authors are.", model="m")

    g = guardrails(NONE, NONE)
    service = RAGService(FakeSearch(), NoCiteLLM(), Settings(rag=RAGSettings(top_k=1)), guardrails=g)
    r = service.answer(AskRequest(query="Who wrote it?"))

    output_call = g.client.calls[1]
    assert output_call["source"] == "OUTPUT" and len(output_call["content"]) == 1  # answer only, no grounding
    assert r.guardrail.output.action == "none" and r.guardrail.output.grounding is None
