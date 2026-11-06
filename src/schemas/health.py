from typing import Literal

from pydantic import BaseModel


class ServiceStatus(BaseModel):
    status: Literal["healthy", "unhealthy"]
    message: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    services: dict[str, ServiceStatus]
