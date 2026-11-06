import json

import httpx
import pytest

from src.config import RAGSettings, Settings
from src.services.llm import client as llm_module
from src.services.llm.client import BedrockLLM, LLMError, OpenAILLM, get_llm_client

MESSAGES = [{"role": "user", "content": "What is RAG?"}]


def make_settings(**overrides) -> Settings:
    base = {
        "openai_api_key": "sk-test",
        "openai_model": "gpt-test",
        "bedrock_model_id": "anthropic.test-model",
        "rag": RAGSettings(max_tokens=100, temperature=0.2, max_retries=2),
    }
    return Settings(**(base | overrides))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)


def mock_http(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def sse_body(pieces: list[str]) -> bytes:
    events = [{"choices": [{"delta": {"content": p}}]} for p in pieces]
    events.append({"choices": [], "usage": {"total_tokens": 5}})  # final usage-only chunk
    return ("".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n").encode()


def test_openai_generate_payload_and_result():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "gpt-test-2026",
                "choices": [{"message": {"content": "Answer [1]"}}],
                "usage": {"total_tokens": 9},
            },
        )

    llm = OpenAILLM(make_settings(), http_client=mock_http(handler))
    result = llm.generate("be precise", MESSAGES)

    assert result.text == "Answer [1]" and result.usage == {"total_tokens": 9}
    body = seen[0]
    assert body["messages"][0] == {"role": "system", "content": "be precise"}
    assert body["messages"][1:] == MESSAGES
    assert body["max_completion_tokens"] == 100 and body["temperature"] == 0.2
    assert "max_tokens" not in body and "stream" not in body
    assert llm.model_id == "openai:gpt-test"


def test_openai_omits_temperature_when_none():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm = OpenAILLM(make_settings(rag=RAGSettings(temperature=None)), http_client=mock_http(handler))
    llm.generate("s", MESSAGES)
    assert "temperature" not in seen[0]


def test_openai_stream_yields_pieces():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=sse_body(["Hel", "lo", " [1]"]))

    llm = OpenAILLM(make_settings(), http_client=mock_http(handler))
    assert list(llm.stream("s", MESSAGES)) == ["Hel", "lo", " [1]"]
    assert seen[0]["stream"] is True


def test_openai_stream_retries_rate_limit_before_first_token():
    statuses = [429, 200]

    def handler(request):
        status = statuses.pop(0)
        return httpx.Response(status, content=sse_body(["ok"]) if status == 200 else b"slow down")

    llm = OpenAILLM(make_settings(), http_client=mock_http(handler))
    assert list(llm.stream("s", MESSAGES)) == ["ok"]


def test_openai_retries_then_raises():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, text="busy")

    llm = OpenAILLM(make_settings(), http_client=mock_http(handler))
    with pytest.raises(LLMError, match="503"):
        llm.generate("s", MESSAGES)
    assert len(calls) == 3  # 1 + max_retries


def test_openai_auth_error_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, text="bad key")

    with pytest.raises(LLMError, match="401"):
        list(OpenAILLM(make_settings(), http_client=mock_http(handler)).stream("s", MESSAGES))
    assert len(calls) == 1


class FakeBedrock:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.requests: list[dict] = []

    def converse(self, **request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("AccessDeniedException")
        return {
            "output": {"message": {"content": [{"text": "Bedrock "}, {"text": "answer"}]}},
            "usage": {"inputTokens": 3},
        }

    def converse_stream(self, **request):
        self.requests.append(request)
        events = [
            {"messageStart": {}},
            *({"contentBlockDelta": {"delta": {"text": t}}} for t in ["a", "b"]),
            {"messageStop": {}},
        ]
        return {"stream": iter(events)}


def test_bedrock_converse_request_shape():
    fake = FakeBedrock()
    llm = BedrockLLM(make_settings(), boto_client=fake)

    result = llm.generate("sys", MESSAGES)

    assert result.text == "Bedrock answer"
    request = fake.requests[0]
    assert request["modelId"] == "anthropic.test-model"
    assert request["system"] == [{"text": "sys"}]
    assert request["messages"] == [{"role": "user", "content": [{"text": "What is RAG?"}]}]
    assert request["inferenceConfig"] == {"maxTokens": 100, "temperature": 0.2}


def test_bedrock_stream_and_errors():
    assert list(BedrockLLM(make_settings(), boto_client=FakeBedrock()).stream("s", MESSAGES)) == ["a", "b"]
    with pytest.raises(LLMError, match="AccessDenied"):
        BedrockLLM(make_settings(), boto_client=FakeBedrock(fail=True)).generate("s", MESSAGES)


def test_factory_and_config_errors():
    assert isinstance(get_llm_client(make_settings(llm_provider="openai")), OpenAILLM)
    with pytest.raises(LLMError, match="Unknown"):
        get_llm_client(make_settings(llm_provider="gemini"))
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        get_llm_client(make_settings(openai_api_key=""))
    with pytest.raises(LLMError, match="BEDROCK_MODEL_ID"):
        get_llm_client(make_settings(llm_provider="bedrock", bedrock_model_id=""))
