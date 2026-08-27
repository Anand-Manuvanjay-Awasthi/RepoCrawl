"""
Embedding service.

Generates vector embeddings via the Hugging Face Inference API's
feature-extraction task, rather than loading a model locally. This keeps
the backend lightweight (no local ML runtime dependency for embeddings)
and reuses the same Hugging Face credentials as the LLM service (Phase 6).

If a local, self-hosted embedding model is preferred later, this module's
public functions (`embed_documents` / `embed_query`) are the only surface
callers depend on, so the implementation can be swapped without touching
the rest of the app.
"""

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from app.core.config import get_settings
from app.core.exceptions import IndexingError
from app.core.logging import get_logger

logger = get_logger(__name__)

# The Inference API batches reasonably well, but very large single requests
# are slower to retry and harder to debug on failure, so we chunk requests.
_EMBED_BATCH_SIZE = 32


def _get_client() -> InferenceClient:
    settings = get_settings()
    if not settings.hf_api_token:
        raise IndexingError(
            "No Hugging Face API token configured. Set HUGGINGFACEHUB_API_TOKEN "
            "(or HF_TOKEN) in the backend .env file to enable embeddings.",
        )
    return InferenceClient(token=settings.hf_api_token)


def _embed_batch(client: InferenceClient, texts: list[str], model: str) -> list[list[float]]:
    try:
        result = client.feature_extraction(text=texts, model=model)
    except HfHubHTTPError as exc:
        raise IndexingError(
            f"Embedding request to Hugging Face model '{model}' failed.",
            context={"reason": str(exc)[:500]},
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise IndexingError(
            "Unexpected error while generating embeddings.",
            context={"reason": str(exc)[:500]},
        ) from exc

    vectors = _normalize_feature_extraction_output(result)
    if len(vectors) != len(texts):
        raise IndexingError(
            "Embedding response shape did not match the number of inputs sent.",
            context={"expected": len(texts), "received": len(vectors)},
        )
    return vectors


def _normalize_feature_extraction_output(result) -> list[list[float]]:
    """
    The feature-extraction endpoint can return per-token embeddings
    (needing mean-pooling) or already-pooled sentence embeddings depending
    on the model. Normalize both shapes down to one vector per input text.
    """
    array = result
    # huggingface_hub returns a numpy array or nested list; normalize to list.
    try:
        array = array.tolist()  # numpy -> nested python lists
    except AttributeError:
        pass

    vectors: list[list[float]] = []
    for item in array:
        # item is either: a flat vector [float, ...], or a token matrix
        # [[float, ...], [float, ...], ...] that needs mean-pooling.
        if item and isinstance(item[0], (list, tuple)):
            dim = len(item[0])
            sums = [0.0] * dim
            for token_vec in item:
                for i, v in enumerate(token_vec):
                    sums[i] += v
            count = len(item)
            vectors.append([s / count for s in sums])
        else:
            vectors.append(list(item))
    return vectors


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of document/chunk texts. Order-preserving."""
    if not texts:
        return []

    settings = get_settings()
    client = _get_client()
    logger.info("Embedding %d chunks with model %s", len(texts), settings.embedding_model)

    all_vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        all_vectors.extend(_embed_batch(client, batch, settings.embedding_model))

    return all_vectors


def embed_query(text: str) -> list[float]:
    """Embed a single query string (e.g. a user's chat question)."""
    settings = get_settings()
    client = _get_client()
    vectors = _embed_batch(client, [text], settings.embedding_model)
    return vectors[0]
