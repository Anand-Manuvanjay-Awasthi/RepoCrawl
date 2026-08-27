"""
Chroma vector store wrapper.

Owns all direct interaction with ChromaDB so the rest of the app never
touches the chromadb client directly. Embeddings are always computed
upstream (embedding_service) and passed in explicitly — this store never
auto-embeds — which keeps the embedding provider swappable and this class
simple and testable.

Collection naming is deterministic per repository (`repo_<repo_id>`), so
re-indexing the same repository reuses/replaces the same collection, and
the design does not preclude multiple repositories' collections coexisting
later even though the MVP only keeps one active at a time.
"""

import re

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import get_settings
from app.core.exceptions import IndexingError
from app.core.logging import get_logger
from app.rag.text_splitter import Chunk

logger = get_logger(__name__)

_COLLECTION_PREFIX = "repo_"
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")

# Chroma writes are batched to stay well under request/payload limits.
_WRITE_BATCH_SIZE = 128


def _collection_name(repo_id: str) -> str:
    """Deterministic, Chroma-safe collection name for a given repository."""
    sanitized = _INVALID_NAME_CHARS.sub("_", repo_id.lower())
    name = f"{_COLLECTION_PREFIX}{sanitized}"
    # Chroma collection names must be between 3 and 63 characters.
    return name[:63]


class ChromaStore:
    """Thin, purpose-built wrapper around a persistent Chroma client."""

    def __init__(self):
        settings = get_settings()
        settings.ensure_runtime_dirs()
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    def replace_repository(self, repo_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> Collection:
        """
        Delete any existing collection for this repository and (re)create it
        with the given chunks + embeddings. This is the standard entry point
        for indexing/re-indexing a repository.
        """
        if len(chunks) != len(embeddings):
            raise IndexingError(
                "Number of chunks and embeddings did not match while writing to Chroma.",
                context={"chunks": len(chunks), "embeddings": len(embeddings)},
            )

        name = _collection_name(repo_id)
        self._client.delete_collection(name) if self._collection_exists(name) else None
        collection = self._client.create_collection(
            name=name,
            metadata={"repo_id": repo_id},
        )

        logger.info("Writing %d chunks to Chroma collection '%s'", len(chunks), name)

        for start in range(0, len(chunks), _WRITE_BATCH_SIZE):
            batch_chunks = chunks[start : start + _WRITE_BATCH_SIZE]
            batch_embeddings = embeddings[start : start + _WRITE_BATCH_SIZE]
            collection.add(
                ids=[c.id for c in batch_chunks],
                documents=[c.content for c in batch_chunks],
                metadatas=[c.to_metadata() for c in batch_chunks],
                embeddings=batch_embeddings,
            )

        return collection

    def delete_all_repositories(self) -> None:
        """Remove every RepoCrawl-managed collection (single-active-repo MVP cleanup)."""
        for name in self._client.list_collections():
            collection_name = name if isinstance(name, str) else name.name
            if collection_name.startswith(_COLLECTION_PREFIX):
                self._client.delete_collection(collection_name)

    def _collection_exists(self, name: str) -> bool:
        try:
            self._client.get_collection(name)
            return True
        except Exception:  # noqa: BLE001 - chromadb raises a generic lookup error
            return False

    def get_collection(self, repo_id: str) -> Collection | None:
        name = _collection_name(repo_id)
        if not self._collection_exists(name):
            return None
        return self._client.get_collection(name)

    def similarity_search(
        self, repo_id: str, query_embedding: list[float], top_k: int = 5
    ) -> list[dict]:
        """
        Return up to `top_k` most similar chunks for a query embedding.

        Each result dict has: content, metadata, distance, similarity
        (similarity is a 0-1-ish score derived from distance; higher is
        more relevant).
        """
        collection = self.get_collection(repo_id)
        if collection is None:
            return []

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches: list[dict] = []
        for content, metadata, distance in zip(documents, metadatas, distances):
            # Chroma's default space is squared L2 for arbitrary embeddings;
            # convert to a bounded, intuitive similarity score.
            similarity = 1.0 / (1.0 + max(distance, 0.0))
            matches.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "distance": distance,
                    "similarity": similarity,
                }
            )
        return matches

    def count(self, repo_id: str) -> int:
        collection = self.get_collection(repo_id)
        return collection.count() if collection else 0


_store: ChromaStore | None = None


def get_chroma_store() -> ChromaStore:
    """Process-wide singleton accessor (the underlying client is itself persistent/stateful)."""
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store
