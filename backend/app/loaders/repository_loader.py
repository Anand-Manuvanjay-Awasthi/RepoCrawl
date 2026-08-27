"""
Repository loader.

Walks a cloned repository on disk and produces a filtered set of
`LoadedDocument`s (source/config/docs files only), each carrying metadata
useful for later chunking, embedding, and citation.

This module deliberately does not know about LangChain, embeddings, or
Chroma — it just turns "a directory on disk" into "a clean list of
(content, metadata) documents". Phase 3's chunking layer builds on top of
this.
"""

from dataclasses import dataclass, field
from pathlib import Path

from app.core.exceptions import EmptyRepositoryError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Directories that never contain useful, original source for this project.
EXCLUDED_DIR_NAMES = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    "target",
    "coverage",
    ".next",
    "vendor",
    "chroma_db",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    "egg-info",
    ".tox",
    ".gradle",
}

# File extensions considered supported source/config/documentation.
SUPPORTED_EXTENSIONS = {
    ".py",
    ".java",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".php",
    ".rb",
    ".sql",
    ".html",
    ".css",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".properties",
}

# Exact filenames (no extension, or where the name itself is the signal)
# that should always be included when supported.
SUPPORTED_FILENAMES = {
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Makefile",
}

# Anything larger than this is almost certainly not something a chat answer
# should quote from directly, and risks blowing up chunking/embedding cost.
MAX_FILE_SIZE_BYTES = 750_000

# Read this many bytes to sniff whether a file is binary.
_BINARY_SNIFF_SIZE = 8192


@dataclass
class LoadedDocument:
    """A single loaded, filtered repository file, ready for chunking."""

    content: str
    repo_id: str
    relative_path: str
    file_name: str
    extension: str
    size_bytes: int


@dataclass
class LoadStats:
    """Counts describing what happened during a load, for transparency."""

    total_files_discovered: int = 0
    files_loaded: int = 0
    files_skipped_directory: int = 0
    files_skipped_unsupported_type: int = 0
    files_skipped_binary: int = 0
    files_skipped_too_large: int = 0
    files_skipped_unreadable: int = 0
    skipped_by_extension: dict[str, int] = field(default_factory=dict)


@dataclass
class LoadResult:
    documents: list[LoadedDocument]
    stats: LoadStats


def _is_supported_file(path: Path) -> bool:
    if path.name in SUPPORTED_FILENAMES:
        return True
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _looks_binary(path: Path) -> bool:
    """Heuristic binary sniff: presence of a NUL byte in the first chunk."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(_BINARY_SNIFF_SIZE)
    except OSError:
        return True
    return b"\x00" in chunk


def _read_text(path: Path) -> str | None:
    """Read a file as UTF-8 text, tolerating minor encoding issues."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def load_repository(repo_path: Path, repo_id: str) -> LoadResult:
    """
    Walk `repo_path`, filter out irrelevant directories/files, and return
    the supported files as `LoadedDocument`s plus load statistics.

    Raises EmptyRepositoryError if no supported, non-empty files are found
    (either the repository truly has no files, or everything present is
    filtered out as unsupported/binary/too large).
    """
    if not repo_path.exists() or not repo_path.is_dir():
        raise EmptyRepositoryError(
            "Repository directory does not exist or is not a directory.",
            context={"path": str(repo_path)},
        )

    stats = LoadStats()
    documents: list[LoadedDocument] = []

    for current_dir, dir_names, file_names in _walk(repo_path):
        # Prune excluded directories in place so os.walk doesn't descend
        # into them at all.
        pruned = [d for d in dir_names if d in EXCLUDED_DIR_NAMES]
        for d in pruned:
            dir_names.remove(d)
        stats.files_skipped_directory += _count_files_under(current_dir, pruned)

        for file_name in file_names:
            stats.total_files_discovered += 1
            file_path = current_dir / file_name

            if not _is_supported_file(file_path):
                stats.files_skipped_unsupported_type += 1
                ext = file_path.suffix.lower() or "(no extension)"
                stats.skipped_by_extension[ext] = stats.skipped_by_extension.get(ext, 0) + 1
                continue

            try:
                size_bytes = file_path.stat().st_size
            except OSError:
                stats.files_skipped_unreadable += 1
                continue

            if size_bytes == 0:
                stats.files_skipped_unreadable += 1
                continue

            if size_bytes > MAX_FILE_SIZE_BYTES:
                stats.files_skipped_too_large += 1
                continue

            if _looks_binary(file_path):
                stats.files_skipped_binary += 1
                continue

            content = _read_text(file_path)
            if content is None or not content.strip():
                stats.files_skipped_unreadable += 1
                continue

            relative_path = file_path.relative_to(repo_path).as_posix()
            documents.append(
                LoadedDocument(
                    content=content,
                    repo_id=repo_id,
                    relative_path=relative_path,
                    file_name=file_path.name,
                    extension=file_path.suffix.lower() or file_path.name,
                    size_bytes=size_bytes,
                )
            )
            stats.files_loaded += 1

    logger.info(
        "Loaded repository %s: %d files loaded, %d discovered, %d skipped (unsupported), "
        "%d skipped (binary), %d skipped (too large)",
        repo_id,
        stats.files_loaded,
        stats.total_files_discovered,
        stats.files_skipped_unsupported_type,
        stats.files_skipped_binary,
        stats.files_skipped_too_large,
    )

    if not documents:
        raise EmptyRepositoryError(
            "No supported source, configuration, or documentation files were "
            "found in this repository after filtering.",
            context={
                "total_files_discovered": stats.total_files_discovered,
                "files_skipped_unsupported_type": stats.files_skipped_unsupported_type,
            },
        )

    return LoadResult(documents=documents, stats=stats)


def _walk(root: Path):
    """Thin wrapper around os.walk that yields Path objects."""
    import os

    for current_dir, dir_names, file_names in os.walk(root):
        yield Path(current_dir), dir_names, file_names


def _count_files_under(directory: Path, subdir_names: list[str]) -> int:
    """Best-effort count of files inside pruned subdirectories, for stats only."""
    total = 0
    for name in subdir_names:
        sub = directory / name
        try:
            total += sum(1 for p in sub.rglob("*") if p.is_file())
        except OSError:
            continue
    return total
