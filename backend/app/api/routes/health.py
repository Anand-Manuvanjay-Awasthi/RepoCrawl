"""Health check endpoint."""

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.responses import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Lightweight liveness check used by the frontend and deployment tooling."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.environment,
    )
