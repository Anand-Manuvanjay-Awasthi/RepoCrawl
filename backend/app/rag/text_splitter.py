"""
Chunking layer.

Turns a `LoadedDocument` (one whole file's content + metadata) into a list
of `Chunk`s suitable for embedding: reasonably sized, overlapping where it
helps preserve context, and never tiny slivers.

Implemented as a small, dependency-free recursive splitter (rather than
pulling in a heavy text-splitting library) so chunking has no dependency
on ML frameworks. The strategy is picked per file category, and the split
implementation is isolated behind `chunk_document()` / `RecursiveTextSplitter`
so it can be swapped out or improved later without touching callers.
"""

import hashlib
from dataclasses import dataclass
from enum import Enum

from app.loaders.repository_loader import LoadedDocument

# Below this many characters, a trailing chunk gets merged into its
# predecessor rather than kept as its own tiny fragment.
MIN_CHUNK_CHARS = 80


class FileCategory(str, Enum):
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONFIG = "config"
    DEFAULT = "default"


_CODE_EXTENSIONS = {
    ".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".cpp",
    ".c", ".h", ".hpp", ".cs", ".php", ".rb", ".sql", ".html", ".css",
}
_DOC_EXTENSIONS = {".md"}
_CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".xml", ".properties"}

# (chunk_size, chunk_overlap, separators) per category. Separators are
# tried in order, from "most meaningful boundary" to "last resort".
_STRATEGY_PARAMS: dict[FileCategory, tuple[int, int, list[str]]] = {
    FileCategory.CODE: (
        1200,
        200,
        ["\nclass ", "\ndef ", "\nfunction ", "\n\n", "\n", " ", ""],
    ),
    FileCategory.DOCUMENTATION: (
        1000,
        150,
        ["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    ),
    FileCategory.CONFIG: (
        800,
        80,
        ["\n\n", "\n", ", ", " ", ""],
    ),
    FileCategory.DEFAULT: (
        1000,
        150,
        ["\n\n", "\n", " ", ""],
    ),
}


def categorize_file(extension: str, file_name: str) -> FileCategory:
    ext = extension.lower()
    if ext in _CODE_EXTENSIONS:
        return FileCategory.CODE
    if ext in _DOC_EXTENSIONS:
        return FileCategory.DOCUMENTATION
    if ext in _CONFIG_EXTENSIONS:
        return FileCategory.CONFIG
    if file_name in {"Dockerfile", "Makefile"}:
        return FileCategory.CONFIG
    return FileCategory.DEFAULT


@dataclass
class Chunk:
    """A single embeddable unit of text, with metadata for retrieval/citation."""

    id: str
    content: str
    repo_id: str
    relative_path: str
    file_name: str
    extension: str
    chunk_index: int
    total_chunks: int

    def to_metadata(self) -> dict:
        """Metadata payload stored alongside the vector in Chroma."""
        return {
            "repo_id": self.repo_id,
            "relative_path": self.relative_path,
            "file_name": self.file_name,
            "extension": self.extension,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }


class RecursiveTextSplitter:
    """
    Minimal recursive character splitter.

    Tries each separator in order; if a piece is still larger than
    `chunk_size`, it recurses using the next (finer-grained) separator.
    Adjacent chunks overlap by `chunk_overlap` characters so context isn't
    lost at boundaries.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str]):
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)
        self.separators = separators

    def split(self, text: str) -> list[str]:
        pieces = self._split_recursive(text, self.separators)
        return self._merge_with_overlap(pieces)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # Last resort: hard-cut on chunk_size.
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
                if text[i : i + self.chunk_size].strip()
            ]

        sep, *rest_separators = separators
        if sep == "":
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
                if text[i : i + self.chunk_size].strip()
            ]

        parts = text.split(sep)
        results: list[str] = []
        buffer = ""

        for i, part in enumerate(parts):
            candidate = part if i == 0 else sep + part
            if len(buffer) + len(candidate) <= self.chunk_size:
                buffer += candidate
            else:
                if buffer.strip():
                    results.append(buffer)
                if len(candidate) > self.chunk_size:
                    results.extend(self._split_recursive(candidate, rest_separators))
                    buffer = ""
                else:
                    buffer = candidate

        if buffer.strip():
            results.append(buffer)

        return results

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Add trailing-context overlap between consecutive pieces."""
        if len(pieces) <= 1 or self.chunk_overlap == 0:
            return [p.strip() for p in pieces if p.strip()]

        merged: list[str] = []
        for i, piece in enumerate(pieces):
            if i == 0:
                merged.append(piece)
                continue
            previous = pieces[i - 1]
            overlap_text = previous[-self.chunk_overlap :]
            merged.append(overlap_text + piece)

        return [p.strip() for p in merged if p.strip()]


def _merge_tiny_pieces(pieces: list[str]) -> list[str]:
    """
    Merge any piece smaller than MIN_CHUNK_CHARS into a neighbor, wherever
    it occurs in the sequence (not just at the end) — a lone short line in
    the middle of a file is just as low-value a chunk as a short trailing
    fragment.
    """
    if len(pieces) <= 1:
        return pieces

    merged: list[str] = []
    for piece in pieces:
        if merged and len(piece) < MIN_CHUNK_CHARS:
            merged[-1] = merged[-1] + "\n" + piece
        else:
            merged.append(piece)

    # A merge can occasionally leave the very first piece undersized (if it
    # was the only one before anything else arrived) — fold it forward.
    if len(merged) > 1 and len(merged[0]) < MIN_CHUNK_CHARS:
        merged[1] = merged[0] + "\n" + merged[1]
        merged.pop(0)

    return merged


def _make_chunk_id(repo_id: str, relative_path: str, chunk_index: int) -> str:
    raw = f"{repo_id}:{relative_path}:{chunk_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def chunk_document(document: LoadedDocument) -> list[Chunk]:
    """
    Split a single loaded document into a list of `Chunk`s using a strategy
    chosen by file category (code vs. documentation vs. config vs. default).
    """
    category = categorize_file(document.extension, document.file_name)
    chunk_size, chunk_overlap, separators = _STRATEGY_PARAMS[category]

    splitter = RecursiveTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
    )
    raw_pieces = splitter.split(document.content)

    if not raw_pieces:
        return []

    raw_pieces = _merge_tiny_pieces(raw_pieces)
    total = len(raw_pieces)
    chunks: list[Chunk] = []
    for index, piece in enumerate(raw_pieces):
        chunks.append(
            Chunk(
                id=_make_chunk_id(document.repo_id, document.relative_path, index),
                content=piece,
                repo_id=document.repo_id,
                relative_path=document.relative_path,
                file_name=document.file_name,
                extension=document.extension,
                chunk_index=index,
                total_chunks=total,
            )
        )
    return chunks


def chunk_documents(documents: list[LoadedDocument]) -> list[Chunk]:
    """Chunk every document in a repository load result."""
    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(chunk_document(document))
    return all_chunks
