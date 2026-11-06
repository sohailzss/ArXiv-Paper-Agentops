import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from opensearchpy.exceptions import TransportError

from src.dependencies import RAGDep
from src.schemas.rag import AskRequest, AskResponse
from src.services.embeddings.client import EmbeddingError
from src.services.llm.client import LLMError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ask", tags=["rag"])

UPSTREAM_ERRORS = (LLMError, EmbeddingError, TransportError)


@router.post("", response_model=AskResponse)
def ask(request: AskRequest, rag: RAGDep) -> AskResponse:
    """Answer a question from the indexed papers, with [n] citations to the returned sources."""
    try:
        return rag.answer(request)
    except UPSTREAM_ERRORS as exc:
        logger.error("RAG failed: %s", exc)
        raise HTTPException(503, f"{type(exc).__name__}: {str(exc)[:300]}") from exc


def _sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


@router.post("/stream")
def ask_stream(request: AskRequest, rag: RAGDep) -> StreamingResponse:
    """Same as /ask, streamed as server-sent events: `sources`, then `token`s, then `done`
    (or `error`). Each event's data is JSON, so newlines in the answer are safe."""

    def events() -> Iterator[str]:
        try:
            for event in rag.stream(request):
                yield _sse(event)
        except UPSTREAM_ERRORS as exc:
            # Headers (200) are already sent, so errors travel as an event.
            logger.error("RAG stream failed: %s", exc)
            yield _sse({"type": "error", "detail": f"{type(exc).__name__}: {str(exc)[:300]}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
