"""
Custom application exceptions.

Using dedicated exception types (instead of raising raw HTTPException
everywhere) keeps error semantics consistent and lets a single exception
handler in main.py translate them into the structured ErrorResponse shape.
"""


class RepoLensError(Exception):
    """Base class for all application-level errors."""

    code: str = "internal_error"
    status_code: int = 500

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.message = message
        self.context = context


class InvalidRepositoryURLError(RepoLensError):
    code = "invalid_repository_url"
    status_code = 400


class RepositoryCloneError(RepoLensError):
    code = "repository_clone_failed"
    status_code = 502


class EmptyRepositoryError(RepoLensError):
    code = "empty_repository"
    status_code = 422


class IndexingError(RepoLensError):
    code = "indexing_failed"
    status_code = 500


class NoRepositoryIndexedError(RepoLensError):
    code = "no_repository_indexed"
    status_code = 409


class GenerationError(RepoLensError):
    code = "generation_failed"
    status_code = 502
