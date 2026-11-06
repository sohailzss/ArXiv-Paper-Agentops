"""FastAPI dependency providers. Tests swap these out with app.dependency_overrides."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from src.config import Settings, get_settings
from src.db.database import Database
from src.services.agent.graph import AgentService
from src.services.hybrid_search import HybridSearchService
from src.services.opensearch.client import OpenSearchClient
from src.services.rag import RAGService


def get_database(request: Request) -> Database:
    return request.app.state.database


def get_opensearch(request: Request) -> OpenSearchClient:
    return request.app.state.opensearch


def get_hybrid_search(request: Request) -> HybridSearchService:
    service = getattr(request.app.state, "hybrid_search", None)
    if service is None:
        reason = getattr(request.app.state, "hybrid_search_error", "not initialised")
        raise HTTPException(503, f"Hybrid search unavailable: {reason}")
    return service


def get_rag(request: Request) -> RAGService:
    service = getattr(request.app.state, "rag", None)
    if service is None:
        reason = getattr(request.app.state, "rag_error", "not initialised")
        raise HTTPException(503, f"RAG unavailable: {reason}")
    return service


def get_agent(request: Request) -> AgentService:
    service = getattr(request.app.state, "agent", None)
    if service is None:
        raise HTTPException(
            503, f"Agent unavailable: {getattr(request.app.state, 'rag_error', None) or 'not initialised'}"
        )
    return service


SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[Database, Depends(get_database)]
OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch)]
HybridSearchDep = Annotated[HybridSearchService, Depends(get_hybrid_search)]
RAGDep = Annotated[RAGService, Depends(get_rag)]
AgentDep = Annotated[AgentService, Depends(get_agent)]
