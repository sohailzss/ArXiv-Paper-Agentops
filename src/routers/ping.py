from fastapi import APIRouter, Request

from src.dependencies import DatabaseDep, OpenSearchDep, SettingsDep
from src.schemas.health import HealthResponse, ServiceStatus

router = APIRouter(tags=["health"])


@router.get("/ping")
def ping() -> dict[str, str]:
    return {"message": "pong"}


@router.get("/health", response_model=HealthResponse)
def health(request: Request, settings: SettingsDep, database: DatabaseDep, opensearch: OpenSearchDep) -> HealthResponse:
    services: dict[str, ServiceStatus] = {}

    try:
        database.ping()
        services["database"] = ServiceStatus(status="healthy")
    except Exception as exc:  # report, don't crash the health check
        services["database"] = ServiceStatus(status="unhealthy", message=str(exc)[:200])

    try:
        cluster = opensearch.health()
        services["opensearch"] = ServiceStatus(status="healthy", message=f"cluster status: {cluster.get('status')}")
    except Exception as exc:
        services["opensearch"] = ServiceStatus(status="unhealthy", message=str(exc)[:200])

    # Redis is optional (the API works without the cache), so it's reported but doesn't degrade status.
    cache = getattr(request.app.state, "cache", None)
    if cache is not None:
        try:
            cache.ping()
            stats = cache.stats()
            services["redis"] = ServiceStatus(
                status="healthy", message=f"hits={stats['hits']} misses={stats['misses']}"
            )
        except Exception as exc:
            services["redis"] = ServiceStatus(status="unhealthy", message=str(exc)[:200])

    core = {k: v for k, v in services.items() if k != "redis"}
    overall = "ok" if all(s.status == "healthy" for s in core.values()) else "degraded"
    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.environment,
        services=services,
    )
