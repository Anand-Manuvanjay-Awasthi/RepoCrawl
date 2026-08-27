"""
Index orchestration service.

Coordinates the full ingestion-to-indexing pipeline for a single repository:

    validate URL -> fetch metadata -> clone -> load files -> chunk
    -> embed -> store in Chroma -> statistics

This is the one place that knows the *order* of those steps. Each step's
actual logic lives in its own module (github_service, repository_loader,
text_splitter, embedding_service, chroma_store) so this service stays a
thin, readable coordinator rather than a giant do-everything class.

Technology detection, repository maps, and suggested questions (Phase 4)
are intentionally not part of this module — they consume its output.
"""

import time
from dataclasses import dataclass

from app.core.logging import get_logger
from app.loaders.repository_loader import LoadStats, load_repository
from app.rag.text_splitter import chunk_documents
from app.embeddings.embedding_service import embed_documents
from app.services.github_service import (
    ParsedRepository,
    RepositoryMetadata,
    clone_repository,
    fetch_repository_metadata,
    parse_repo_url,
)
from app.vectorstores.chroma_store import get_chroma_store

logger = get_logger(__name__)


@dataclass
class IndexingStats:
    repository_name: str
    total_files_discovered: int
    files_indexed: int
    files_skipped: int
    chunks_created: int
    indexing_duration_seconds: float


@dataclass
class IndexResult:
    parsed_repository: ParsedRepository
    repository_metadata: RepositoryMetadata
    stats: IndexingStats
    load_stats: LoadStats


def _files_skipped(load_stats: LoadStats) -> int:
    return (
        load_stats.files_skipped_unsupported_type
        + load_stats.files_skipped_binary
        + load_stats.files_skipped_too_large
        + load_stats.files_skipped_unreadable
    )


def index_repository(repo_url: str) -> IndexResult:
    """
    Run the full ingestion + indexing pipeline for `repo_url`.

    Raises InvalidRepositoryURLError, RepositoryCloneError,
    EmptyRepositoryError, or IndexingError depending on where the pipeline
    fails — all are RepoCrawlError subclasses with structured messages, so
    callers (API routes) can surface them directly.
    """
    started_at = time.monotonic()

    parsed = parse_repo_url(repo_url)
    logger.info("Starting indexing pipeline for %s", parsed.full_name)

    metadata = fetch_repository_metadata(parsed)
    clone_path = clone_repository(parsed, default_branch=metadata.default_branch)

    load_result = load_repository(clone_path, repo_id=parsed.repo_id)
    logger.info(
        "Loaded %d files for %s, chunking now", len(load_result.documents), parsed.full_name
    )

    chunks = chunk_documents(load_result.documents)
    logger.info("Created %d chunks for %s", len(chunks), parsed.full_name)

    embeddings = embed_documents([chunk.content for chunk in chunks])

    store = get_chroma_store()
    store.replace_repository(parsed.repo_id, chunks, embeddings)

    duration = time.monotonic() - started_at

    stats = IndexingStats(
        repository_name=parsed.full_name,
        total_files_discovered=load_result.stats.total_files_discovered,
        files_indexed=load_result.stats.files_loaded,
        files_skipped=_files_skipped(load_result.stats),
        chunks_created=len(chunks),
        indexing_duration_seconds=round(duration, 2),
    )

    logger.info(
        "Finished indexing %s in %.2fs (%d files, %d chunks)",
        parsed.full_name,
        duration,
        stats.files_indexed,
        stats.chunks_created,
    )

    return IndexResult(
        parsed_repository=parsed,
        repository_metadata=metadata,
        stats=stats,
        load_stats=load_result.stats,
    )
