"""LLM providers behind one interface. Pick one with LLM_PROVIDER=openai|bedrock.

- OpenAI: Chat Completions over plain HTTP (httpx), streaming via server-sent events.
- Bedrock: the Converse API through boto3, which works the same way for every Bedrock model
  (Claude, Llama, Mistral, Nova...). Credentials come from the normal AWS chain
  (env vars, ~/.aws/credentials, or an instance role on EC2).

Messages are [{"role": "user"|"assistant", "content": str}]; the system prompt is separate
because both APIs treat it separately.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(Exception):
    pass


@dataclass
class LLMResult:
    text: str
    model: str
    usage: dict = field(default_factory=dict)


class LLMClient(ABC):
    provider: str
    model: str

    def __init__(self, settings: Settings):
        self.rag = settings.rag

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model}"

    @abstractmethod
    def generate(self, system: str, messages: list[dict]) -> LLMResult: ...

    @abstractmethod
    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        """Yields text pieces as the model produces them."""

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


class OpenAILLM(LLMClient):
    provider = "openai"
    url = "https://api.openai.com/v1/chat/completions"

    def __init__(self, settings: Settings, http_client: httpx.Client | None = None):
        super().__init__(settings)
        if not settings.openai_api_key:
            raise LLMError("LLM_PROVIDER=openai but OPENAI_API_KEY is empty")
        self.model = settings.openai_model
        self.http = http_client or httpx.Client(timeout=self.rag.timeout_seconds)
        self.http.headers["Authorization"] = f"Bearer {settings.openai_api_key}"

    def _payload(self, system: str, messages: list[dict], stream: bool) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            # max_completion_tokens works for every current chat model; max_tokens is rejected by newer ones.
            "max_completion_tokens": self.rag.max_tokens,
        }
        if self.rag.temperature is not None:
            payload["temperature"] = self.rag.temperature
        if stream:
            payload |= {"stream": True, "stream_options": {"include_usage": True}}
        return payload

    def _retry_delay(self, attempt: int, reason: str) -> None:
        delay = 2**attempt
        logger.warning("OpenAI attempt %d failed (%s), retrying in %ds", attempt + 1, reason, delay)
        time.sleep(delay)

    def generate(self, system: str, messages: list[dict]) -> LLMResult:
        payload = self._payload(system, messages, stream=False)
        for attempt in range(self.rag.max_retries + 1):
            try:
                response = self.http.post(self.url, json=payload)
            except httpx.TransportError as exc:
                if attempt == self.rag.max_retries:
                    raise LLMError(f"OpenAI unreachable: {exc}") from exc
                self._retry_delay(attempt, str(exc))
                continue
            if response.status_code == 200:
                body = response.json()
                text = body["choices"][0]["message"].get("content") or ""
                return LLMResult(text=text, model=body.get("model", self.model), usage=body.get("usage", {}))
            if response.status_code not in RETRYABLE_STATUS or attempt == self.rag.max_retries:
                raise LLMError(f"OpenAI error {response.status_code}: {response.text[:300]}")
            self._retry_delay(attempt, f"HTTP {response.status_code}")
        raise LLMError("unreachable")  # pragma: no cover

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        payload = self._payload(system, messages, stream=True)
        # Retries only cover opening the stream; once text has been sent we can't take it back.
        for attempt in range(self.rag.max_retries + 1):
            try:
                with self.http.stream("POST", self.url, json=payload) as response:
                    if response.status_code != 200:
                        body = response.read().decode(errors="replace")
                        if response.status_code in RETRYABLE_STATUS and attempt < self.rag.max_retries:
                            self._retry_delay(attempt, f"HTTP {response.status_code}")
                            continue
                        raise LLMError(f"OpenAI error {response.status_code}: {body[:300]}")
                    yield from self._parse_sse(response.iter_lines())
                    return
            except httpx.TransportError as exc:
                if attempt == self.rag.max_retries:
                    raise LLMError(f"OpenAI stream failed: {exc}") from exc
                self._retry_delay(attempt, str(exc))

    @staticmethod
    def _parse_sse(lines: Iterator[str]) -> Iterator[str]:
        for line in lines:
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            chunk = json.loads(data)
            for choice in chunk.get("choices") or []:  # the final usage chunk has no choices
                if text := (choice.get("delta") or {}).get("content"):
                    yield text

    def close(self) -> None:
        self.http.close()


class BedrockLLM(LLMClient):
    provider = "bedrock"

    def __init__(self, settings: Settings, boto_client=None):
        super().__init__(settings)
        if not settings.bedrock_model_id:
            raise LLMError("LLM_PROVIDER=bedrock but BEDROCK_MODEL_ID is empty")
        self.model = settings.bedrock_model_id
        if boto_client is None:
            import boto3
            from botocore.config import Config

            boto_client = boto3.client(
                "bedrock-runtime",
                region_name=settings.aws_region,
                config=Config(
                    read_timeout=self.rag.timeout_seconds,
                    retries={"max_attempts": self.rag.max_retries + 1, "mode": "adaptive"},
                ),
            )
        self.client = boto_client

    def _request(self, system: str, messages: list[dict]) -> dict:
        inference: dict = {"maxTokens": self.rag.max_tokens}
        if self.rag.temperature is not None:
            inference["temperature"] = self.rag.temperature
        return {
            "modelId": self.model,
            "system": [{"text": system}],
            "messages": [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages],
            "inferenceConfig": inference,
        }

    def generate(self, system: str, messages: list[dict]) -> LLMResult:
        try:
            response = self.client.converse(**self._request(system, messages))
        except Exception as exc:  # botocore ClientError, credentials errors, timeouts
            raise LLMError(f"Bedrock error: {exc}") from exc
        blocks = response["output"]["message"]["content"]
        text = "".join(block.get("text", "") for block in blocks)
        return LLMResult(text=text, model=self.model, usage=response.get("usage", {}))

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        try:
            response = self.client.converse_stream(**self._request(system, messages))
            for event in response["stream"]:
                if text := event.get("contentBlockDelta", {}).get("delta", {}).get("text"):
                    yield text
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Bedrock stream error: {exc}") from exc


PROVIDERS: dict[str, type[LLMClient]] = {"openai": OpenAILLM, "bedrock": BedrockLLM}


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(PROVIDERS)}")
    return PROVIDERS[provider](settings)
