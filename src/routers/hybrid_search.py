import logging

from fastapi import APIRouter, HTTPException
from opensearchpy.exceptions import NotFoundError, TransportError

from src.dependencies import HybridSearchDep
from src.schemas.search import HybridSearchRequest, HybridSearchResponse
from src.services.embeddings.client import EmbeddingError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/hybrid-search", tags=["search"])


@router.post("", response_model=HybridSearchResponse)
def hybrid_search(request: HybridSearchRequest, service: HybridSearchDep) -> HybridSearchResponse:
    """Chunk-level search: BM25 + vector similarity fused with RRF (or either one alone via `mode`).

    Returns passages, not whole papers: this is what the RAG step (Phase 5) feeds to the LLM.
    """
    try:
        return service.search(
            request.query,
            mode=request.mode,
            size=request.size,
            categories=request.categories,
            date_from=request.date_from,
            date_to=request.date_to,
        )
    except EmbeddingError as exc:
        logger.error("Query embedding failed: %s", exc)
        raise HTTPException(503, f"Embedding provider error: {exc}") from exc
    except NotFoundError as exc:
        raise HTTPException(503, "Chunk index not found. Run: uv run python scripts/index_papers.py") from exc
    except TransportError as exc:
        logger.error("OpenSearch hybrid search failed: %s", exc)
        raise HTTPException(503, "Search service unavailable") from exc
