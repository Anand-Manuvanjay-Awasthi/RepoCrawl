"""
Custom application exceptions.

Using dedicated exception types (instead of raising raw HTTPException
everywhere) keeps error semantics consistent and lets a single exception
handler in main.py translate them into the structured ErrorResponse shape.
"""


class RepoCrawlError(Exception):
    """Base class for all application-level errors."""

    code: str = "internal_error"
    status_code: int = 500

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.message = message
        self.context = context


class InvalidRepositoryURLError(RepoCrawlError):
    code = "invalid_repository_url"
    status_code = 400


class RepositoryCloneError(RepoCrawlError):
    code = "repository_clone_failed"
    status_code = 502


class EmptyRepositoryError(RepoCrawlError):
    code = "empty_repository"
    status_code = 422


class IndexingError(RepoCrawlError):
    code = "indexing_failed"
    status_code = 500


class NoRepositoryIndexedError(RepoCrawlError):
    code = "no_repository_indexed"
    status_code = 409


class GenerationError(RepoCrawlError):
    code = "generation_failed"
    status_code = 502
