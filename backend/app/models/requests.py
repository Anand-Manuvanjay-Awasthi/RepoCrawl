"""
Shared request models.

Phase 1 only defines the base building blocks. Endpoint-specific request
models (index requests, chat requests, etc.) are added in later phases as
those endpoints are implemented.
"""

from pydantic import BaseModel, ConfigDict


class BaseRequest(BaseModel):
    """Base class for all API request bodies."""

    model_config = ConfigDict(extra="forbid")
