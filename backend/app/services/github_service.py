"""
GitHub integration service.

Responsible for:
* validating and parsing public GitHub repository URLs
* looking up repository metadata via the GitHub REST API
* cloning a repository to local disk (shallow clone)
* cleaning up / replacing the currently cloned repository

This module intentionally knows nothing about embeddings, chunking, or
chat — it only deals with getting a repository's files safely onto disk
plus a small amount of GitHub-provided metadata.
"""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import requests
from git import GitCommandError, Repo

from app.core.config import get_settings
from app.core.exceptions import InvalidRepositoryURLError, RepositoryCloneError
from app.core.logging import get_logger

logger = get_logger(__name__)

GITHUB_API_BASE = "https://api.github.com"
CLONE_TIMEOUT_SECONDS = 120

# Matches https://github.com/{owner}/{repo}, with or without a trailing
# ".git", trailing slash, or "www." prefix. Owner/repo segments follow
# GitHub's allowed charset (alphanumerics, hyphens, underscores, dots).
_GITHUB_URL_PATTERN = re.compile(
    r"^https?://(www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)/"
    r"(?P<repo>[A-Za-z0-9._-]+?)"
    r"(\.git)?/?$"
)


@dataclass(frozen=True)
class ParsedRepository:
    owner: str
    repo: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def repo_id(self) -> str:
        """Filesystem/collection-safe deterministic identifier."""
        return f"{self.owner}__{self.repo}".lower()

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}.git"


@dataclass
class RepositoryMetadata:
    full_name: str
    description: str | None
    default_branch: str
    stars: int
    forks: int
    primary_language: str | None
    topics: list[str]
    size_kb: int
    is_private: bool
    html_url: str


def parse_repo_url(repo_url: str) -> ParsedRepository:
    """
    Validate that `repo_url` is a well-formed public GitHub repository URL
    and extract the owner/repo segments.

    Raises InvalidRepositoryURLError if the URL is malformed or is not a
    github.com repository URL.
    """
    if not repo_url or not repo_url.strip():
        raise InvalidRepositoryURLError("Repository URL is required.")

    candidate = repo_url.strip()
    match = _GITHUB_URL_PATTERN.match(candidate)
    if not match:
        raise InvalidRepositoryURLError(
            "That doesn't look like a valid GitHub repository URL. "
            "Expected something like https://github.com/owner/repository",
            context={"repo_url": repo_url},
        )

    owner = match.group("owner")
    repo = match.group("repo")

    if repo.lower() == ".git" or not repo:
        raise InvalidRepositoryURLError(
            "Could not determine the repository name from that URL.",
            context={"repo_url": repo_url},
        )

    return ParsedRepository(owner=owner, repo=repo)


def fetch_repository_metadata(parsed: ParsedRepository) -> RepositoryMetadata:
    """
    Look up repository metadata from the GitHub REST API.

    Raises InvalidRepositoryURLError if the repository does not exist or is
    inaccessible (private/removed), and RepositoryCloneError for network or
    unexpected API failures.
    """
    url = f"{GITHUB_API_BASE}/repos/{parsed.full_name}"
    logger.info("Fetching GitHub metadata for %s", parsed.full_name)

    try:
        response = requests.get(
            url,
            headers={"Accept": "application/vnd.github+json"},
            timeout=15,
        )
    except requests.RequestException as exc:
        raise RepositoryCloneError(
            f"Could not reach GitHub to look up '{parsed.full_name}'.",
            context={"reason": str(exc)},
        ) from exc

    if response.status_code == 404:
        raise InvalidRepositoryURLError(
            f"GitHub repository '{parsed.full_name}' was not found. "
            "It may be private, misspelled, or removed.",
            context={"repo": parsed.full_name},
        )

    if response.status_code == 403:
        raise RepositoryCloneError(
            "GitHub API rate limit was hit while looking up repository metadata. "
            "Please try again shortly.",
            context={"repo": parsed.full_name},
        )

    if response.status_code != 200:
        raise RepositoryCloneError(
            f"Unexpected response from GitHub API ({response.status_code}) "
            f"while looking up '{parsed.full_name}'.",
            context={"status_code": response.status_code},
        )

    data = response.json()

    if data.get("private"):
        raise InvalidRepositoryURLError(
            f"'{parsed.full_name}' is a private repository. RepoCrawl only "
            "supports public repositories.",
            context={"repo": parsed.full_name},
        )

    return RepositoryMetadata(
        full_name=data.get("full_name", parsed.full_name),
        description=data.get("description"),
        default_branch=data.get("default_branch", "main"),
        stars=data.get("stargazers_count", 0),
        forks=data.get("forks_count", 0),
        primary_language=data.get("language"),
        topics=data.get("topics", []) or [],
        size_kb=data.get("size", 0),
        is_private=bool(data.get("private", False)),
        html_url=data.get("html_url", f"https://github.com/{parsed.full_name}"),
    )


def _repo_clone_path(parsed: ParsedRepository) -> Path:
    settings = get_settings()
    return Path(settings.repo_clone_dir) / parsed.repo_id


def cleanup_repository(clone_path: Path) -> None:
    """Remove a previously cloned repository directory, if present."""
    if clone_path.exists():
        logger.info("Removing existing cloned repository at %s", clone_path)
        shutil.rmtree(clone_path, ignore_errors=True)


def cleanup_all_repositories() -> None:
    """
    Remove everything under the configured clone directory.

    Used to enforce the MVP constraint that only one repository is actively
    indexed at a time: before cloning a new repository, any previously
    cloned repository is discarded.
    """
    settings = get_settings()
    clone_root = Path(settings.repo_clone_dir)
    if clone_root.exists():
        for entry in clone_root.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)


def clone_repository(parsed: ParsedRepository, default_branch: str = "main") -> Path:
    """
    Shallow-clone the given repository to a deterministic local path,
    replacing any previously cloned repository first.

    Raises RepositoryCloneError on failure (bad branch, network error,
    corrupt repo, etc).
    """
    settings = get_settings()
    settings.ensure_runtime_dirs()

    # MVP supports a single actively-indexed repository at a time.
    cleanup_all_repositories()

    clone_path = _repo_clone_path(parsed)
    logger.info("Cloning %s into %s", parsed.clone_url, clone_path)

    try:
        Repo.clone_from(
            parsed.clone_url,
            str(clone_path),
            depth=1,
            branch=default_branch,
            single_branch=True,
        )
    except GitCommandError as exc:
        cleanup_repository(clone_path)
        raise RepositoryCloneError(
            f"Failed to clone '{parsed.full_name}'. It may not exist, may be "
            "empty, or the default branch could not be determined.",
            context={"repo": parsed.full_name, "reason": str(exc)[:500]},
        ) from exc
    except Exception as exc:  # noqa: BLE001 - convert any unexpected failure
        cleanup_repository(clone_path)
        raise RepositoryCloneError(
            f"Unexpected error while cloning '{parsed.full_name}'.",
            context={"reason": str(exc)[:500]},
        ) from exc

    # Remove the .git directory: we only need the working tree, and this
    # keeps the loader from ever having to walk/skip a huge .git folder.
    git_dir = clone_path / ".git"
    shutil.rmtree(git_dir, ignore_errors=True)

    logger.info("Successfully cloned %s (%d files on disk)", parsed.full_name, _count_files(clone_path))
    return clone_path


def _count_files(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file())
