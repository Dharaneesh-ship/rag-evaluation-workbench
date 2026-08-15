"""Index compatibility validation — the R2 guard against "假 hybrid" failures.

Checks that a request's resolved embedding configuration is actually
compatible with what's on disk before running (or writing to) a vector
index, per ECOSYSTEM_INTEGRATION_PLAN.md 定案 #13's INDEX_NOT_FOUND /
INDEX_CONFIG_MISMATCH error codes. Without this, "the config says vectors
are enabled" and "vectors actually exist and mean what the config thinks
they mean" can silently diverge — the corpus was never embedded, or was
embedded with a different provider/model/dimension — and the caller gets a
degraded BM25-only result with no signal that anything was skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from rag.app.mcp_server.retrieval_config import ResolvedEmbedding, index_compat_mode
from rag.core.errors import IndexConfigMismatchError, IndexNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class IndexValidationResult:
    """Outcome of validating a query-time embedding config against an index.

    Attributes:
        use_vector_search: True when vector search should actually run.
        degraded_reason: Set only when ``RAG_INDEX_COMPAT_MODE=degraded``
            silently downgraded an incompatible request to BM25-only —
            callers must record this (response field + trace), never drop it,
            per 定案 #13's "必须在结构化返回值和日志记录降级原因".
    """

    use_vector_search: bool
    degraded_reason: Optional[str] = None


def validate_index_compatibility(index_manager, resolved: ResolvedEmbedding) -> IndexValidationResult:
    """Validate a query-time embedding config against an index_dir's manifest.

    Args:
        index_manager: An ``IndexManager`` whose indexes/manifest reflect the
            corpus about to be queried.
        resolved: The caller's resolved embedding config (see
            ``retrieval_config.resolve_embedding_config``). If
            ``resolved.provider`` is None (BM25-only requested), this is
            always a no-op success.

    Returns:
        ``IndexValidationResult(use_vector_search=True)`` when compatible.
        In degraded mode, an incompatibility instead returns
        ``use_vector_search=False`` with ``degraded_reason`` set.

    Raises:
        IndexNotFoundError: No vectors were ever indexed for this index_dir
            but vector search was requested (strict mode only).
        IndexConfigMismatchError: Vectors exist but were built with a
            different provider/model/dimension (strict mode only).
    """
    if resolved.provider is None:
        return IndexValidationResult(use_vector_search=False)

    manifest = index_manager.manifest
    faiss_dim = index_manager.faiss.dimension

    reason: Optional[str] = None
    not_found = False
    if manifest.embedding_provider is None or faiss_dim == 0:
        not_found = True
        reason = (
            f"No vector index found for index_dir (requested provider="
            f"'{resolved.provider}') — the corpus was never embedded, or "
            "index_dir is empty/BM25-only."
        )
    elif manifest.embedding_provider != resolved.provider:
        reason = (
            f"Index was built with embedding_provider='{manifest.embedding_provider}' "
            f"but query requested embedding_provider='{resolved.provider}'."
        )
    elif manifest.embedding_model != resolved.model:
        reason = (
            f"Index was built with embedding_model='{manifest.embedding_model}' "
            f"but query requested embedding_model='{resolved.model}'."
        )
    elif faiss_dim != resolved.dimension:
        reason = (
            f"Index vector_dimension={faiss_dim} but query requested "
            f"vector_dimension={resolved.dimension}."
        )

    if reason is None:
        return IndexValidationResult(use_vector_search=True)

    if index_compat_mode() == "degraded":
        logger.warning("Index compatibility check degraded to BM25-only: %s", reason)
        return IndexValidationResult(use_vector_search=False, degraded_reason=reason)

    if not_found:
        raise IndexNotFoundError(reason)
    raise IndexConfigMismatchError(reason)


def validate_ingest_embedding_config(index_manager, resolved: ResolvedEmbedding) -> None:
    """Reject an ingest whose embedding config conflicts with this index_dir's manifest.

    FAISS requires uniform vector dimensionality across the whole index, and
    mixing embedding spaces (even at matching dimension) makes distances
    meaningless — so unlike the query-time check, this is always strict
    regardless of ``RAG_INDEX_COMPAT_MODE``: silently mixing providers in
    one FAISS index isn't a legitimate degrade path, it's index corruption.

    Args:
        index_manager: The ``IndexManager`` this ingest will write into.
        resolved: This ingest call's resolved embedding config. A no-op
            when ``resolved.provider`` is None (BM25-only ingest never
            conflicts with existing vectors).

    Raises:
        IndexConfigMismatchError: This index_dir already has vectors built
            with a different provider/model/dimension.
    """
    if resolved.provider is None:
        return
    manifest = index_manager.manifest
    if manifest.embedding_provider is None:
        return  # first embedding config for this index_dir — nothing to conflict with
    if (
        manifest.embedding_provider != resolved.provider
        or manifest.embedding_model != resolved.model
        or manifest.vector_dimension != resolved.dimension
    ):
        raise IndexConfigMismatchError(
            f"index_dir was already built with embedding_provider="
            f"'{manifest.embedding_provider}' embedding_model='{manifest.embedding_model}' "
            f"vector_dimension={manifest.vector_dimension}, but this ingest requested "
            f"embedding_provider='{resolved.provider}' embedding_model='{resolved.model}' "
            f"vector_dimension={resolved.dimension}. Use a separate index_dir per "
            "embedding configuration."
        )
