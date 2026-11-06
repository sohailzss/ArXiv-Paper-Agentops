"""FastAPI entry point. Run locally with:  uv run fastapi dev src/main.py"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import get_settings
from src.db.database import Database
from src.routers import agent, ask, hybrid_search, ping, search
from src.services.agent.graph import AgentService
from src.services.cache import RAGCache
from src.services.embeddings.client import get_embeddings_client
from src.services.guardrails import Guardrails
from src.services.hybrid_search import HybridSearchService
from src.services.indexing import build_chunk_index
from src.services.llm.client import get_llm_client
from src.services.observability import setup_logfire
from src.services.online_eval import OnlineEvaluator
from src.services.opensearch.client import OpenSearchClient
from src.services.rag import RAGService
from src.services.tracing import build_tracer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting %s (%s)", settings.app_name, settings.environment)

    app.state.database = Database(settings)
    try:
        app.state.database.create_tables()
        logger.info("Database tables ready")
    except Exception as exc:
        # Keep the API up so /health can show what's wrong.
        logger.error("Could not reach the database at startup: %s", exc)

    app.state.opensearch = OpenSearchClient(settings)
    try:
        if app.state.opensearch.setup_index():
            logger.info("Created search index; fill it with: uv run python scripts/index_papers.py")
    except Exception as exc:
        logger.error("Could not set up the OpenSearch index at startup: %s", exc)

    # Hybrid search needs an embeddings provider; if its key is missing the rest of the API still works.
    app.state.embedder, app.state.hybrid_search, app.state.hybrid_search_error = None, None, None
    try:
        app.state.embedder = get_embeddings_client(settings)
        chunk_index = build_chunk_index(app.state.opensearch, app.state.embedder)
        chunk_index.setup()
        app.state.hybrid_search = HybridSearchService(chunk_index, app.state.embedder)
        logger.info("Hybrid search ready (%s, index %s)", app.state.embedder.model_id, chunk_index.name)
    except Exception as exc:
        app.state.hybrid_search_error = str(exc)[:300]
        logger.error("Hybrid search disabled: %s", exc)

    # Phase 6: Redis answer cache + Langfuse tracing. Both optional; failures only log.
    app.state.cache = None
    try:
        app.state.cache = RAGCache.from_settings(settings.redis)
        if app.state.cache is not None:
            app.state.cache.ping()
            logger.info("Answer cache ready (Redis, TTL %ss)", settings.redis.ttl_seconds)
    except Exception as exc:
        # Keep the client: it reconnects by itself once Redis is back.
        logger.error("Redis not reachable at startup (cache will be skipped until it is): %s", exc)
    app.state.tracer = build_tracer(settings)
    logger.info("Langfuse tracing %s", "on" if app.state.tracer.enabled else "off")

    # AWS Bedrock Guardrails (input + output checks). Off unless configured and AWS credentials exist.
    app.state.guardrails = None
    try:
        app.state.guardrails = Guardrails.from_settings(settings.guardrails)
    except Exception as exc:
        logger.error("Guardrails disabled: %s", exc)
    if app.state.guardrails is not None:
        g = settings.guardrails
        logger.info(
            "Guardrails on (%s v%s, %s; grounding=%s, fail_open=%s)",
            g.id,
            g.version,
            g.region,
            g.grounding_action,
            g.fail_open,
        )

    # DeepEval online evals: a sample of live answers scored in the background, scores on the Langfuse trace.
    app.state.online_eval = OnlineEvaluator.from_settings(settings, app.state.tracer)

    # RAG = hybrid search + an LLM (LLM_PROVIDER=openai|bedrock).
    app.state.llm, app.state.rag, app.state.rag_error, app.state.agent = None, None, None, None
    if app.state.hybrid_search is None:
        app.state.rag_error = f"hybrid search unavailable ({app.state.hybrid_search_error})"
    else:
        try:
            app.state.llm = get_llm_client(settings)
            app.state.rag = RAGService(
                app.state.hybrid_search,
                app.state.llm,
                settings,
                cache=app.state.cache,
                tracer=app.state.tracer,
                guardrails=app.state.guardrails,
                online_eval=app.state.online_eval,
            )
            logger.info("RAG ready (LLM %s)", app.state.llm.model_id)
            app.state.agent = AgentService(app.state.rag, app.state.llm, settings, tracer=app.state.tracer)
            logger.info("Agent ready (max %d retrievals)", settings.agent.max_retrievals)
        except Exception as exc:
            app.state.rag_error = str(exc)[:300]
            logger.error("RAG disabled: %s", exc)

    yield

    if app.state.online_eval is not None:
        app.state.online_eval.shutdown()
    app.state.tracer.shutdown()  # flushes pending traces
    if app.state.cache is not None:
        app.state.cache.close()
    if app.state.llm is not None:
        app.state.llm.close()
    if app.state.embedder is not None:
        app.state.embedder.close()
    app.state.opensearch.close()
    app.state.database.close()
    logger.info("Shutdown complete")


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Agentic RAG over arXiv papers",
    lifespan=lifespan,
)

app.include_router(ping.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(hybrid_search.router, prefix="/api/v1")
app.include_router(ask.router, prefix="/api/v1")
app.include_router(agent.router, prefix="/api/v1")

# Logfire wraps the app, so it's set up here (after routes) rather than in the lifespan.
app.state.logfire = setup_logfire(settings, app)
