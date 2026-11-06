"""Pydantic Logfire setup: app-level traces next to Langfuse's LLM traces.

Logfire owns the *global* OpenTelemetry tracer provider and sees:
  - every FastAPI request (route, status, latency)
  - every SQL query to Neon (psycopg2 driver)
  - every outbound HTTP call (OpenAI, Jina) via httpx: timings only, no bodies or auth headers
  - Redis commands, and OpenSearch searches + query embedding (spans in hybrid_search.py)

Langfuse runs on its *own* tracer provider (see tracing.py), so the two never mix: Logfire doesn't
receive prompts, and Langfuse traces never get attached to Logfire's request spans. The request span
in Logfire carries a `langfuse.trace_id` attribute, to jump from a slow request to its LLM trace.

Without a token (or with LOGFIRE__ENABLED=false) nothing is sent anywhere.
"""

import logging

from src.config import Settings

logger = logging.getLogger(__name__)

# Health checks hit these every few seconds; tracing them would only add noise and cost.
EXCLUDED_URLS = "/api/v1/ping,/api/v1/health"


def _send_mode(value: str) -> bool | str:
    value = value.strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    return "if-token-present"


def setup_logfire(settings: Settings, app=None) -> bool:
    """Configure Logfire and instrument the app. Returns True if traces will be sent."""
    cfg = settings.logfire
    if not cfg.enabled or not cfg.token:
        logger.info("Logfire off (%s)", "disabled" if not cfg.enabled else "no LOGFIRE__TOKEN")
        return False
    try:
        import logfire

        logfire.configure(
            token=cfg.token,
            service_name=cfg.service_name,
            service_version=settings.app_version,
            environment=cfg.environment,
            send_to_logfire=_send_mode(cfg.send_to_logfire),
            console=False,  # keep the container logs as they are
        )
        if app is not None:
            logfire.instrument_fastapi(app, excluded_urls=EXCLUDED_URLS)
        logfire.instrument_httpx()  # OpenAI + Jina calls (timings, status codes)
        logfire.instrument_redis()
        # SQL spans for Neon. Instrumented at the psycopg2 driver level because the SQLAlchemy
        # instrumentor only supports SQLAlchemy < 2.1; this works for any SQLAlchemy version.
        logfire.instrument_psycopg("psycopg2")
        logger.info("Logfire on (service %s, env %s)", cfg.service_name, cfg.environment)
        return True
    except Exception as exc:  # observability must never stop the API from starting
        logger.error("Logfire setup failed, continuing without it: %s", exc)
        return False
