import logging
from datetime import date
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError
from opensearchpy.exceptions import NotFoundError, TransportError

from src.dependencies import OpenSearchDep
from src.schemas.search import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


def _run_search(opensearch: OpenSearchDep, request: SearchRequest) -> SearchResponse:
    try:
        return opensearch.search_papers(
            request.query,
            size=request.size,
            offset=request.offset,
            categories=request.categories,
            date_from=request.date_from,
            date_to=request.date_to,
        )
    except NotFoundError as exc:
        raise HTTPException(503, "Search index not found. Run: uv run python scripts/index_papers.py") from exc
    except (OpenSearchConnectionError, TransportError) as exc:
        logger.error("OpenSearch search failed: %s", exc)
        raise HTTPException(503, "Search service unavailable") from exc


@router.post("", response_model=SearchResponse)
def search(request: SearchRequest, opensearch: OpenSearchDep) -> SearchResponse:
    """BM25 keyword search over titles, abstracts, authors and full text."""
    return _run_search(opensearch, request)


@router.get("", response_model=SearchResponse)
def search_get(
    opensearch: OpenSearchDep,
    q: Annotated[str, Query(max_length=500)] = "",
    size: Annotated[int, Query(ge=1, le=50)] = 10,
    offset: Annotated[int, Query(ge=0, le=1000)] = 0,
    category: Annotated[list[str] | None, Query()] = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> SearchResponse:
    """Same search as POST, for quick tries in the browser: /api/v1/search?q=llm+agents"""
    try:
        request = SearchRequest(
            query=q, size=size, offset=offset, categories=category, date_from=date_from, date_to=date_to
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _run_search(opensearch, request)
