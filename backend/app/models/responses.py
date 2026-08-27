"""
Shared response models.

Phase 1 defines the generic building blocks (health check, structured error
envelope) used across the API. Endpoint-specific response models (index
results, chat answers, etc.) are added in later phases.
"""

from typing import Any

from pydantic import BaseModel


class BaseResponse(BaseModel):
    """Base class for all API response bodies."""


class HealthResponse(BaseResponse):
    """Response returned by GET /api/health."""

    status: str
    app_name: str
    environment: str


class ErrorDetail(BaseModel):
    """A single structured error detail."""

    code: str
    message: str
    context: dict[str, Any] | None = None


class ErrorResponse(BaseResponse):
    """
    Structured error envelope returned for all handled API errors.

    Keeping this consistent means the frontend can rely on `error.code` and
    `error.message` being present for every failure response.
    """

    error: ErrorDetail
