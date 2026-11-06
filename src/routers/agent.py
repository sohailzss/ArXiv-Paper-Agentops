import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from opensearchpy.exceptions import TransportError

from src.dependencies import AgentDep
from src.schemas.agent import AgentResponse
from src.schemas.rag import AskRequest
from src.services.embeddings.client import EmbeddingError
from src.services.llm.client import LLMError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])

UPSTREAM_ERRORS = (LLMError, EmbeddingError, TransportError)


@router.post("/ask", response_model=AgentResponse)
def agent_ask(request: AskRequest, agent: AgentDep) -> AgentResponse:
    """Agentic RAG: route the question, search, grade passages, rewrite the query and retry if needed,
    then answer with citations. `steps` shows every decision the agent made."""
    try:
        return agent.run(request)
    except UPSTREAM_ERRORS as exc:
        logger.error("Agent failed: %s", exc)
        raise HTTPException(503, f"{type(exc).__name__}: {str(exc)[:300]}") from exc


@router.post("/stream")
def agent_stream(request: AskRequest, agent: AgentDep) -> StreamingResponse:
    """Server-sent events: `step` (after each agent node), `sources`, `token`s, then `done` (or `error`)."""

    def events() -> Iterator[str]:
        try:
            for event in agent.stream(request):
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        except UPSTREAM_ERRORS as exc:
            logger.error("Agent stream failed: %s", exc)
            detail = f"{type(exc).__name__}: {str(exc)[:300]}"
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'detail': detail})}\n\n"

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
