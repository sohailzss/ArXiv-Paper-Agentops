import json

from src.config import AgentSettings, Settings
from src.schemas.rag import AskRequest, Source
from src.services.agent.graph import (
    DIRECT_PROMPT,
    GRADE_PROMPT,
    REWRITE_PROMPT,
    ROUTE_PROMPT,
    AgentService,
    parse_json,
)
from src.services.llm.client import LLMResult
from src.services.rag import SYSTEM_PROMPT


def source(n: int, topic: str) -> Source:
    return Source(
        number=n,
        arxiv_id=f"2611.0{n}",
        title=f"Paper about {topic}",
        section_titles=["Abstract"],
        chunk_id=f"2611.0{n}_0",
        url=f"https://arxiv.org/abs/2611.0{n}",
        pdf_url=f"https://arxiv.org/pdf/2611.0{n}",
        text=f"This passage discusses {topic}.",
    )


class FakeEmbedder:
    model_id = "openai:emb:4"


class FakeSearch:
    embedder = FakeEmbedder()


class FakeRAG:
    """Returns passages per search query; records what was searched."""

    def __init__(self, results: dict[str, list[Source]]):
        self.results = results
        self.search = FakeSearch()
        self.queries: list[str] = []

    def retrieve(self, request, trace=None):
        self.queries.append(request.query)
        return self.results.get(request.query, []), 5

    # Guardrails off in these tests (covered in test_guardrails.py).
    def guard_request(self, trace, request):
        return request, None, None

    def guard_answer(self, trace, request, answer, sources):
        return answer, None

    def record_online(self, kind, request, answer, sources, trace):  # online evals off in these tests
        pass


class ScriptedLLM:
    """Answers each agent prompt from a script; `grades` is consumed one per grade call."""

    model_id, model, provider = "openai:gpt-test", "gpt-test", "openai"

    def __init__(self, route='{"route": "search", "reason": "research question"}', grades=None, rewrite=None):
        self.route_reply = route
        self.grades = list(grades or [])
        self.rewrite_reply = rewrite or '{"query": "better query"}'
        self.calls: list[str] = []

    def generate(self, system, messages):
        name = {ROUTE_PROMPT: "route", GRADE_PROMPT: "grade", REWRITE_PROMPT: "rewrite"}[system]
        self.calls.append(name)
        text = {"route": self.route_reply, "rewrite": self.rewrite_reply}.get(name) or self.grades.pop(0)
        return LLMResult(text=text, model="gpt-test", usage={"prompt_tokens": 10, "completion_tokens": 2})

    def stream(self, system, messages):
        self.calls.append({SYSTEM_PROMPT: "answer", DIRECT_PROMPT: "direct"}[system])
        if system == DIRECT_PROMPT:
            yield from ["I only answer ", "questions about papers."]
        else:
            yield from ["Walking is learned in simulation ", "[1]."]


def agent(rag, llm, max_retrievals=2) -> AgentService:
    return AgentService(rag, llm, Settings(agent=AgentSettings(max_retrievals=max_retrievals)))


def nodes(response) -> list[str]:
    return [step.node for step in response.steps]


def test_happy_path_keeps_only_relevant_passages_and_renumbers_them():
    rag = FakeRAG(
        {"How do robots learn to walk?": [source(1, "cooking"), source(2, "robot walking"), source(3, "chess")]}
    )
    llm = ScriptedLLM(grades=['{"relevant": [2]}'])

    response = agent(rag, llm).run(AskRequest(query="How do robots learn to walk?"))

    assert nodes(response) == ["route", "retrieve", "grade", "generate"]
    assert [s.title for s in response.sources] == ["Paper about robot walking"]
    assert response.sources[0].number == 1  # was [2] in retrieval, [1] in the answer
    assert response.cited == [1]
    assert response.answer == "Walking is learned in simulation [1]."
    assert response.route == "search" and response.search_queries == ["How do robots learn to walk?"]
    assert llm.calls == ["route", "grade", "answer"] and response.llm_calls == 3


def test_rewrites_query_and_retries_when_nothing_is_relevant():
    rag = FakeRAG(
        {"walking bots?": [source(1, "cooking")], "humanoid locomotion learning": [source(1, "humanoid locomotion")]}
    )
    llm = ScriptedLLM(
        grades=['{"relevant": []}', '{"relevant": [1]}'], rewrite='{"query": "humanoid locomotion learning"}'
    )

    response = agent(rag, llm).run(AskRequest(query="walking bots?"))

    assert nodes(response) == ["route", "retrieve", "grade", "rewrite", "retrieve", "grade", "generate"]
    assert response.search_queries == ["walking bots?", "humanoid locomotion learning"]
    assert rag.queries == ["walking bots?", "humanoid locomotion learning"]
    assert response.steps[3].detail == {"query": "humanoid locomotion learning"}
    assert response.sources[0].title == "Paper about humanoid locomotion"


def test_gives_up_honestly_after_max_retrievals():
    rag = FakeRAG({"q": [source(1, "cooking")]})
    llm = ScriptedLLM(grades=['{"relevant": []}', '{"relevant": []}'], rewrite='{"query": "q2"}')

    response = agent(rag, llm, max_retrievals=2).run(AskRequest(query="q"))

    assert nodes(response) == ["route", "retrieve", "grade", "rewrite", "retrieve", "grade", "give_up"]
    assert "couldn't find" in response.answer and '"q"; "q2"' in response.answer
    assert response.sources == [] and response.cited == []
    assert "answer" not in llm.calls  # no generation call when there's nothing to ground it


def test_direct_route_skips_search():
    rag = FakeRAG({})
    llm = ScriptedLLM(route='{"route": "direct", "reason": "greeting"}')

    response = agent(rag, llm).run(AskRequest(query="hi there!"))

    assert nodes(response) == ["route", "direct"]
    assert response.route == "direct" and rag.queries == []
    assert response.answer == "I only answer questions about papers."
    assert response.steps[0].detail == {"route": "direct", "reason": "greeting"}


def test_bad_router_output_fails_open_to_search():
    rag = FakeRAG({"q": [source(1, "x")]})
    llm = ScriptedLLM(route="Sure! I think you should search.", grades=['{"relevant": [1]}'])
    assert agent(rag, llm).run(AskRequest(query="q")).route == "search"


def test_bad_grader_output_keeps_all_passages():
    rag = FakeRAG({"q": [source(1, "a"), source(2, "b")]})
    llm = ScriptedLLM(grades=["I can't decide"])
    response = agent(rag, llm).run(AskRequest(query="q"))
    assert len(response.sources) == 2


def test_grader_numbers_outside_the_list_are_ignored():
    rag = FakeRAG({"q": [source(1, "a")]})
    llm = ScriptedLLM(grades=['{"relevant": [1, 7, "x"]}'])
    response = agent(rag, llm).run(AskRequest(query="q"))
    assert response.steps[2].detail == {"relevant": [1], "of": 1}


def test_rewrite_never_repeats_a_tried_query():
    rag = FakeRAG({})
    llm = ScriptedLLM(grades=[], rewrite='{"query": "Q"}')  # same as the question, ignoring case
    response = agent(rag, llm).run(AskRequest(query="q"))
    assert response.search_queries[0] == "q" and response.search_queries[1] != "q"


def test_search_options_are_passed_to_retrieval():
    class RecordingRAG(FakeRAG):
        def retrieve(self, request, trace=None):
            self.request = request
            return [source(1, "x")], 1

    rag = RecordingRAG({})
    agent(rag, ScriptedLLM(grades=['{"relevant": [1]}'])).run(
        AskRequest(query="q", mode="bm25", top_k=3, categories=["cs.AI"])
    )
    assert rag.request.mode == "bm25" and rag.request.top_k == 3 and rag.request.categories == ["cs.AI"]


def test_stream_event_order():
    rag = FakeRAG({"q": [source(1, "x")]})
    events = list(agent(rag, ScriptedLLM(grades=['{"relevant": [1]}'])).stream(AskRequest(query="q")))

    types = [e["type"] for e in events]
    assert types[:3] == ["step", "step", "step"]  # route, retrieve, grade
    assert types[3:] == ["sources", "token", "token", "step", "done"]
    assert [e["node"] for e in events if e["type"] == "step"] == ["route", "retrieve", "grade", "generate"]
    done = events[-1]
    assert done["cited"] == [1] and done["llm_calls"] == 3 and done["route"] == "search"
    json.dumps(events)  # every event must be JSON-serialisable for SSE


def test_parse_json_handles_fences_and_prose():
    assert parse_json('```json\n{"route": "direct"}\n```') == {"route": "direct"}
    assert parse_json('Here you go: {"relevant": [1, 2]} hope that helps') == {"relevant": [1, 2]}
    assert parse_json("no json here") is None
    assert parse_json("[1, 2]") is None
