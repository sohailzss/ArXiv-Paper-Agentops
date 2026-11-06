from datetime import UTC, datetime

from src.config import RAGSettings, Settings
from src.schemas.rag import AskRequest
from src.schemas.search import ChunkHit, HybridSearchResponse
from src.services.llm.client import LLMResult
from src.services.rag import (
    NO_RESULTS_ANSWER,
    SYSTEM_PROMPT,
    RAGService,
    build_user_prompt,
    extract_citations,
    to_sources,
)


def hit(n: int) -> ChunkHit:
    return ChunkHit(
        chunk_id=f"2611.0000{n}_0",
        arxiv_id=f"2611.0000{n}",
        chunk_index=0,
        section_titles=["Results"],
        text=f"Finding number {n}.",
        title=f"Paper {n}",
        authors=["A. Author"],
        categories=["cs.AI"],
        published_date=datetime(2026, 11, 1, tzinfo=UTC),
        pdf_url=f"https://arxiv.org/pdf/2611.0000{n}",
        score=0.03,
    )


class FakeEmbedder:
    model_id = "openai:emb:4"


class FakeSearch:
    def __init__(self, hits):
        self.hits = hits
        self.embedder = FakeEmbedder()
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return HybridSearchResponse(
            query=query, mode=kwargs["mode"], embedding_model="openai:emb:4", total=len(self.hits), hits=self.hits
        )


class FakeLLM:
    model_id = "openai:gpt-test"
    model = "gpt-test"
    provider = "openai"

    def __init__(self, answer="Paper 2 shows this [2], see also [1] and [9]."):
        self.answer = answer
        self.calls = []

    def generate(self, system, messages):
        self.calls.append((system, messages))
        return LLMResult(text=self.answer, model="gpt-test")

    def stream(self, system, messages):
        self.calls.append((system, messages))
        yield from ["Paper 2 ", "shows [2]."]


SETTINGS = Settings(rag=RAGSettings(top_k=3))


def test_prompt_numbers_passages_with_paper_and_section():
    prompt = build_user_prompt("What was found?", to_sources([hit(1), hit(2)]))

    assert '[1] "Paper 1" (arXiv:2611.00001), section: Results\nFinding number 1.' in prompt
    assert "[2]" in prompt
    assert prompt.endswith("Question: What was found?")


def test_extract_citations_keeps_valid_numbers_in_order():
    assert extract_citations("A [3] B [1][3] C [7]", source_count=3) == [3, 1]
    assert extract_citations("no citations", 3) == []


def test_answer_uses_top_k_filters_and_returns_cited_sources():
    search, llm = FakeSearch([hit(1), hit(2)]), FakeLLM()
    request = AskRequest(query="What was found?", categories=["cs.AI"])

    response = RAGService(search, llm, SETTINGS).answer(request)

    assert search.calls[0]["size"] == 3 and search.calls[0]["categories"] == ["cs.AI"]
    system, messages = llm.calls[0]
    assert system == SYSTEM_PROMPT and "Finding number 2." in messages[0]["content"]
    assert response.cited == [2, 1]  # [9] is not a real source
    assert [s.number for s in response.sources] == [1, 2]
    assert response.sources[0].url == "https://arxiv.org/abs/2611.00001"
    assert response.llm_model == "openai:gpt-test" and response.embedding_model == "openai:emb:4"


def test_request_top_k_overrides_settings():
    search = FakeSearch([hit(1)])
    RAGService(search, FakeLLM(), SETTINGS).answer(AskRequest(query="q", top_k=7))
    assert search.calls[0]["size"] == 7


def test_no_hits_means_no_llm_call():
    llm = FakeLLM()
    response = RAGService(FakeSearch([]), llm, SETTINGS).answer(AskRequest(query="unknown topic"))

    assert response.answer == NO_RESULTS_ANSWER
    assert response.sources == [] and response.llm_model is None
    assert llm.calls == []


def test_stream_event_order():
    events = list(RAGService(FakeSearch([hit(1), hit(2)]), FakeLLM(), SETTINGS).stream(AskRequest(query="q")))

    assert [e["type"] for e in events] == ["sources", "token", "token", "done"]
    assert len(events[0]["sources"]) == 2
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Paper 2 shows [2]."
    assert events[-1]["cited"] == [2] and events[-1]["llm_model"] == "openai:gpt-test"


def test_stream_without_hits():
    llm = FakeLLM()
    events = list(RAGService(FakeSearch([]), llm, SETTINGS).stream(AskRequest(query="q")))
    assert [e["type"] for e in events] == ["sources", "token", "done"]
    assert llm.calls == []
