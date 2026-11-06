"""Application settings, loaded from environment variables and the .env file.

Nested settings use a double underscore in env vars:
    OPENSEARCH__HOST=http://opensearch:9200  ->  settings.opensearch.host
"""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenSearchSettings(BaseModel):
    host: str = "http://localhost:9200"
    index_name: str = "arxiv-papers"
    chunk_index_name: str = "arxiv-papers-chunks"


class ArxivSettings(BaseModel):
    base_url: str = "https://export.arxiv.org/api/query"
    search_category: str = "cs.AI"
    max_results: int = 15
    # arXiv asks API users to wait 3 seconds between requests.
    rate_limit_delay: float = 3.0
    timeout_seconds: int = 30
    max_retries: int = 3
    pdf_cache_dir: Path = Path("data/arxiv_pdfs")
    max_concurrent_downloads: int = 5


class PDFParserSettings(BaseModel):
    max_pages: int = 30
    max_file_size_mb: int = 20
    do_ocr: bool = False
    do_table_structure: bool = True
    # Docling is CPU/RAM heavy, so parse one PDF at a time by default.
    max_concurrent_parsing: int = 1


class EmbeddingsSettings(BaseModel):
    # "openai" or "jina". Each provider gets its own chunk index, so switching never mixes vectors.
    provider: str = "openai"
    openai_model: str = "text-embedding-3-small"
    jina_model: str = "jina-embeddings-v3"
    # Both models support 1024 dims (OpenAI shortens its 1536-dim vectors natively).
    dimensions: int = 1024
    batch_size: int = 64
    timeout_seconds: int = 60
    max_retries: int = 3


class ChunkingSettings(BaseModel):
    chunk_words: int = 400  # ~530 tokens: one focused idea per chunk, well under model limits
    overlap_words: int = 50  # so a sentence cut at a boundary still appears whole in one chunk
    min_words: int = 30  # smaller trailing pieces get merged into the previous chunk


class RAGSettings(BaseModel):
    top_k: int = 5  # chunks given to the LLM (~400 words each, so ~3k tokens of context)
    max_tokens: int = 800
    # Low temperature: answers should stick to the passages. Set to None for models that
    # reject the parameter (some reasoning models only allow the default).
    temperature: float | None = 0.2
    timeout_seconds: int = 120
    max_retries: int = 2


class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379/0"  # compose.yml points the API at redis://redis:6379/0
    enabled: bool = True
    # New papers already invalidate the cache (the key includes the chunk count), so the TTL only
    # bounds how long an answer to an unchanged corpus is reused.
    ttl_seconds: int = 6 * 60 * 60


class AgentSettings(BaseModel):
    max_retrievals: int = 2  # first search + this many minus one query rewrites
    # Passages are trimmed to this many words for the relevance check. Must cover a whole chunk
    # (chunk_words=400 + title): at 120, facts at the end of abstracts were cut off and the grader
    # rejected the right paper (found by the golden evaluation: 3 false refusals).
    grade_passage_words: int = 450


class LogfireSettings(BaseModel):
    """Pydantic Logfire: app-level tracing (HTTP requests, SQL, Redis, outbound HTTP, OpenSearch).
    Complements Langfuse, which traces the LLM side (prompts, tokens, cost)."""

    enabled: bool = True
    token: str = ""
    service_name: str = "arxiv-rag-api"
    environment: str = "development"
    # "if-token-present" (default), "true" or "false"
    send_to_logfire: str = "if-token-present"


class GuardrailSettings(BaseModel):
    """AWS Bedrock Guardrails, applied with the standalone ApplyGuardrail API, so it works whichever
    LLM_PROVIDER is used. Credentials come from the normal AWS chain (IAM role on EC2, ~/.aws locally)."""

    enabled: bool = False
    id: str = ""
    version: str = "DRAFT"  # use a numbered version ("1") in production
    region: str = "us-east-2"
    check_input: bool = True
    check_output: bool = True
    # What to do when the contextual grounding check says the answer isn't supported by the passages:
    #   "flag"  keep the answer, mark it with a warning (default; the check is strict about wording)
    #   "block" replace the answer
    #   "off"   don't run the grounding check
    grounding_action: str = "flag"
    # If AWS is unreachable: true = let the request through (logged), false = refuse it.
    fail_open: bool = True
    timeout_seconds: int = 10
    blocked_input_message: str = (
        "Sorry, I can't help with that request. I answer questions about the indexed arXiv papers."
    )
    blocked_output_message: str = "Sorry, I can't share that answer."


class OnlineEvalSettings(BaseModel):
    """DeepEval online evals on live answers; scores go to Langfuse (see services/online_eval.py)."""

    enabled: bool = True
    sample_rate: float = Field(default=0.2, ge=0.0, le=1.0)  # share of live answers scored
    # gpt-4o: gpt-4o-mini misjudged answers; gpt-4.1 hung inside DeepEval (see scripts/run_eval.py).
    judge_model: str = "gpt-4o"
    metrics: list[str] = Field(default_factory=lambda: ["faithfulness", "answer_relevancy", "contextual_relevancy"])
    max_pending: int = 20  # scoring jobs allowed to wait; extra samples are dropped


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    app_name: str = "Agentic RAG arXiv Curator"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False

    # Neon gives you a URL like postgresql://user:pass@ep-xxx.region.aws.neon.tech/neondb?sslmode=require
    postgres_database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    postgres_echo_sql: bool = False

    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    logfire: LogfireSettings = Field(default_factory=LogfireSettings)
    guardrails: GuardrailSettings = Field(default_factory=GuardrailSettings)
    online_eval: OnlineEvalSettings = Field(default_factory=OnlineEvalSettings)

    jina_api_key: str = ""
    llm_provider: str = "openai"  # "openai" or "bedrock"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    bedrock_model_id: str = ""
    aws_region: str = "us-east-1"

    # Langfuse tracing (Phase 6). Tracing turns on when both keys are set.
    langfuse_enabled: bool = True
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()
