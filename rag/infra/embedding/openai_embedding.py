"""OpenAI embedding provider — wraps the OpenAI Embeddings API."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError as _openai_import_error:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]
    _openai_import_error_msg = str(_openai_import_error)
else:
    _openai_import_error_msg = ""

from rag.core.interfaces.embedding import BaseEmbeddingProvider
from rag.core.utils.batching import EmbedBatchAccumulator, iter_batches
from rag.infra.embedding.base_embedding import EmbeddingResult

logger = logging.getLogger(__name__)

# Each embedding batch is one HTTP round-trip, independent of the others, so
# concurrent requests parallelize well (same reasoning as Vision page
# transcription — see PdfVisionParser.parse_pages_parallel). 4 is a
# conservative default that stays well under typical per-org rate limits.
_DEFAULT_MAX_WORKERS = 4


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider backed by the OpenAI Embeddings API.

    Sends text batches to ``client.embeddings.create`` and returns dense
    float vectors alongside token usage statistics.

    Usage::

        provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", dim=1536)
        vectors = provider.embed(["Hello world", "Goodbye world"])

    Args:
        model: OpenAI embedding model identifier.
        dim: Expected vector dimensionality for the chosen model.
        api_key: OpenAI API key. If None, the ``OPENAI_API_KEY`` environment
            variable is used (standard OpenAI client behaviour).
        batch_size: Maximum number of texts per API call. Defaults to 64.
        max_workers: Maximum concurrent batch-embedding requests. Batches are
            network-bound (one HTTP round-trip each), so multiple batches are
            sent concurrently via a thread pool. Defaults to 4.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        api_key: str | None = None,
        batch_size: int = 64,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        **kwargs,
    ) -> None:
        if OpenAI is None:
            raise ImportError(
                "openai package is required for OpenAIEmbeddingProvider. "
                f"Install it with: pip install openai ({_openai_import_error_msg})"
            )

        self._model = model
        self._dim = dim
        self._batch_size = batch_size
        self._max_workers = max_workers
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors produced by this provider."""
        return self._dim

    @property
    def model(self) -> str:
        """The OpenAI embedding model id used by this provider (for cost estimation)."""
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the OpenAI Embeddings API.

        Args:
            texts: Non-empty list of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            ValueError: If ``texts`` is empty.
            openai.OpenAIError: On API-level errors.
        """
        return self.embed_with_usage(texts).vectors

    def _embed_batch(self, batch: list[str]) -> tuple[list[list[float]], int]:
        """Embed one batch via a single API call.

        Args:
            batch: Texts for this batch (at most ``batch_size``).

        Returns:
            ``(vectors, prompt_tokens)`` for this batch, vectors in input order.
        """
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
        except Exception as exc:
            logger.error("OpenAI embeddings API call failed: %s", exc)
            raise

        # Response items carry an index field — sort to guarantee order.
        sorted_data = sorted(response.data, key=lambda item: item.index)
        batch_vectors = [item.embedding for item in sorted_data]

        logger.debug(
            "Embedded %d text(s), tokens used: %d",
            len(batch),
            response.usage.prompt_tokens,
        )
        return batch_vectors, response.usage.prompt_tokens

    def embed_with_usage(self, texts: list[str]) -> EmbeddingResult:
        """Embed texts and return vectors plus aggregated token usage.

        Splits ``texts`` into batches of ``batch_size`` and sends them
        concurrently via a thread pool — each batch is an independent HTTP
        round-trip, so this is network-bound rather than CPU-bound (same
        reasoning as ``PdfVisionParser.parse_pages_parallel``).

        Args:
            texts: Non-empty list of text strings to embed.

        Returns:
            EmbeddingResult with vectors, model name, and total prompt tokens.
            Vectors are in the same order as the input ``texts``.

        Raises:
            ValueError: If ``texts`` is empty.
            openai.OpenAIError: On API-level errors.
        """
        if not texts:
            raise ValueError("texts must not be empty")

        batches = list(iter_batches(texts, self._batch_size))
        results: list[tuple[list[list[float]], int]] = [None] * len(batches)  # type: ignore[list-item]

        workers = max(1, min(self._max_workers, len(batches)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(self._embed_batch, batch): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()

        acc = EmbedBatchAccumulator()
        for batch_vectors, prompt_tokens in results:
            acc.add(batch_vectors, prompt_tokens)

        return EmbeddingResult(
            vectors=acc.vectors,
            model=self._model,
            prompt_tokens=acc.total_tokens,
        )
