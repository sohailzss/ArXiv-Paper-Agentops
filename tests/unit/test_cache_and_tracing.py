from datetime import UTC, datetime

from src.config import RAGSettings, Settings
from src.schemas.rag import AskRequest
from src.schemas.search import ChunkHit, HybridSearchResponse
from src.services.cache import KEY_PREFIX, RAGCache, cache_key, normalise_query
from src.services.llm.client import LLMResult
from src.services.rag import RAGService
from src.services.tracing import Observation, Trace, Tracer, build_tracer, usage_details

# ---------- fakes ----------


class FakeRedis:
    def __init__(self, fail: bool = False):
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.fail = fail

    def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return self.store.get(key)

    def set(self, key, value, ex=None):
        if self.fail:
            raise ConnectionError("redis down")
        self.store[key], self.ttls[key] = value, ex

    def scan_iter(self, match, count):
        prefix = match.rstrip("*")
        return [k for k in list(self.store) if k.startswith(prefix)]

    def delete(self, key):
        return 1 if self.store.pop(key, None) is not None else 0


class FakeChunks:
    def __init__(self):
        self.size = 60

    def count(self):
        return self.size


class FakeEmbedder:
    model_id = "openai:emb:4"


def hit(n: int) -> ChunkHit:
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
        pdf_url="https://arxiv.org/pdf/x",
        score=0.03,
    )


class FakeSearch:
    def __init__(self):
        self.chunks = FakeChunks()
        self.embedder = FakeEmbedder()
        self.calls = 0

    def search(self, query, **kwargs):
        self.calls += 1
        hits = [hit(1), hit(2)]
        return HybridSearchResponse(query=query, mode=kwargs["mode"], embedding_model="e", total=2, hits=hits)


class FakeLLM:
    model_id, model, provider = "openai:gpt-test", "gpt-test", "openai"

    def __init__(self):
        self.calls = 0

    def generate(self, system, messages):
        self.calls += 1
        return LLMResult(text="Answer [1].", model="gpt-test", usage={"prompt_tokens": 50, "completion_tokens": 5})

    def stream(self, system, messages):
        self.calls += 1
        yield from ["Answer ", "[2]."]


class RecordingObservation(Observation):
    def __init__(self, log, name, fields):
        self.log, self.name = log, name
        log.append(("start", name, fields))

    def end(self, **fields):
        self.log.append(("end", self.name, fields))


class RecordingTrace(Trace, RecordingObservation):
    trace_id = "trace-123"

    def __init__(self, log, name, fields):
        RecordingObservation.__init__(self, log, name, fields)

    def span(self, name, as_type="span", **fields):
        return RecordingObservation(self.log, name, {"as_type": as_type, **fields})

    def generation(self, name, **fields):
        return RecordingObservation(self.log, name, {"as_type": "generation", **fields})


class RecordingTracer(Tracer):
    enabled = True

    def __init__(self):
        self.log: list = []

    def start_trace(self, name, **fields):
        return RecordingTrace(self.log, name, fields)


def service(redis=None, tracer=None):
    search, llm = FakeSearch(), FakeLLM()
    cache = RAGCache(redis if redis is not None else FakeRedis(), ttl_seconds=600)
    rag = RAGService(search, llm, Settings(rag=RAGSettings(top_k=2)), cache=cache, tracer=tracer)
    return rag, search, llm


def starts(tracer):
    return [(name, fields.get("as_type")) for kind, name, fields in tracer.log if kind == "start"]


def end_fields(tracer, name):
    return next(fields for kind, n, fields in tracer.log if kind == "end" and n == name)


# ---------- cache keys ----------


def test_normalise_query_ignores_case_spacing_and_trailing_punctuation():
    assert normalise_query("  What is   RAG? ") == normalise_query("what is rag") == "what is rag"


def test_cache_key_changes_with_anything_that_changes_the_answer():
    base = {
        "query": "q",
        "top_k": 5,
        "mode": "hybrid",
        "categories": ["cs.LG", "cs.AI"],
        "date_from": None,
        "date_to": None,
        "llm_model": "openai:a",
        "embedding_model": "openai:e",
        "corpus_version": 60,
    }
    key = cache_key(**base)
    assert key.startswith(KEY_PREFIX)
    assert cache_key(**(base | {"categories": ["cs.AI", "cs.LG"]})) == key  # order doesn't matter
    for change in ({"top_k": 3}, {"mode": "bm25"}, {"llm_model": "bedrock:x"}, {"corpus_version": 61}):
        assert cache_key(**(base | change)) != key


# ---------- RAG + cache ----------


def test_second_identical_question_is_served_from_cache():
    redis = FakeRedis()
    rag, search, llm = service(redis)

    first = rag.answer(AskRequest(query="What is found?"))
    second = rag.answer(AskRequest(query="  what is FOUND "))

    assert first.cached is False and second.cached is True
    assert second.answer == first.answer and second.cited == first.cited == [1]
    assert search.calls == 1 and llm.calls == 1  # no second search or LLM call
    assert second.retrieval_ms == 0 and second.generation_ms == 0
    assert list(redis.ttls.values()) == [600]
    assert '"retrieval_ms"' not in next(iter(redis.store.values()))  # per-request fields aren't cached


def test_new_papers_invalidate_the_cache():
    rag, search, llm = service()
    rag.answer(AskRequest(query="q"))
    search.chunks.size = 62  # the DAG indexed 2 more chunks

    assert rag.answer(AskRequest(query="q")).cached is False
    assert llm.calls == 2


def test_stream_populates_and_replays_cache():
    rag, _, llm = service()

    live = list(rag.stream(AskRequest(query="q")))
    replay = list(rag.stream(AskRequest(query="q")))

    assert [e["type"] for e in live] == ["sources", "token", "token", "done"]
    assert [e["type"] for e in replay] == ["sources", "token", "done"]
    assert replay[1]["text"] == "Answer [2]."
    assert live[-1]["cached"] is False and replay[-1]["cached"] is True
    assert replay[-1]["cited"] == [2] and llm.calls == 1


def test_json_and_stream_share_cache_entries():
    rag, _, llm = service()
    rag.answer(AskRequest(query="q"))
    events = list(rag.stream(AskRequest(query="q")))
    assert events[-1]["cached"] is True and llm.calls == 1


def test_redis_down_never_breaks_answers():
    rag, _, llm = service(FakeRedis(fail=True))
    assert rag.answer(AskRequest(query="q")).answer == "Answer [1]."
    assert rag.answer(AskRequest(query="q")).cached is False
    assert llm.calls == 2


def test_clear_removes_only_answers():
    redis = FakeRedis()
    redis.store["other:key"] = "keep"
    rag, _, _ = service(redis)
    rag.answer(AskRequest(query="q"))
    assert rag.cache.clear() == 1
    assert list(redis.store) == ["other:key"]


# ---------- tracing ----------


def test_trace_structure_for_a_fresh_answer():
    tracer = RecordingTracer()
    rag, _, _ = service(tracer=tracer)

    response = rag.answer(AskRequest(query="What is found?"))

    assert starts(tracer) == [("rag.ask", None), ("cache", "span"), ("retrieval", "retriever"), ("llm", "generation")]
    llm_end = end_fields(tracer, "llm")
    assert llm_end["output"] == "Answer [1]."
    assert llm_end["usage_details"] == {"input": 50, "output": 5}
    llm_start = next(f for kind, name, f in tracer.log if kind == "start" and name == "llm")
    assert llm_start["model"] == "gpt-test" and llm_start["input"][0]["role"] == "system"
    assert response.trace_id == "trace-123"


def test_cache_hit_trace_has_no_retrieval_or_llm():
    tracer = RecordingTracer()
    rag, _, _ = service(tracer=tracer)
    rag.answer(AskRequest(query="q"))
    tracer.log.clear()

    rag.answer(AskRequest(query="q"))

    assert [name for name, _ in starts(tracer)] == ["rag.ask", "cache"]
    assert end_fields(tracer, "rag.ask")["metadata"]["cached"] is True


def test_stream_records_time_to_first_token():
    tracer = RecordingTracer()
    rag, _, _ = service(tracer=tracer)
    list(rag.stream(AskRequest(query="q")))
    llm_end = end_fields(tracer, "llm")
    assert llm_end["completion_start_time"] is not None and llm_end["output"] == "Answer [2]."


def test_tracer_is_noop_without_keys_or_when_disabled():
    no_keys = Settings(langfuse_enabled=True, langfuse_public_key="", langfuse_secret_key="")
    disabled = Settings(langfuse_enabled=False, langfuse_public_key="pk", langfuse_secret_key="sk")
    assert build_tracer(no_keys).enabled is False
    assert build_tracer(disabled).enabled is False


def test_usage_mapping_for_both_providers():
    openai_usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    assert usage_details(openai_usage) == {"input": 10, "output": 2, "total": 12}
    assert usage_details({"inputTokens": 7, "outputTokens": 3}) == {"input": 7, "output": 3}
    assert usage_details({}) is None
