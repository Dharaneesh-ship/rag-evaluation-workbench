"""Shared retrieval configuration resolution.

Single source of truth for turning a request's ``embedding_provider`` /
``reranker_provider`` (the ``server_default`` / ``none`` / explicit-provider
enum defined by ECOSYSTEM_INTEGRATION_PLAN.md 定案 #7) into concrete
provider instances. Every call site that can trigger embedding or reranking
— ``rag.ingest``, ``rag.sync_source``, ``retrieve``, ``retrieve_with_metadata``,
``rag.query``, and the ``rag.cli.eval``/``rag.cli.query`` entry points — must
resolve through this module instead of each re-implementing the same
``if provider == "openai": ...`` branch, per R2's "统一...配置解析,不允许各自
实现一份" requirement. A single implementation also means a single place to
enforce the index-compatibility and dependency-availability checks in
``rag.app.mcp_server.index_validation``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from rag.core.errors import EmbeddingProviderUnavailableError, RerankerUnavailableError

logger = logging.getLogger(__name__)

EMBEDDING_PROVIDERS = ("server_default", "none", "openai")
RERANKER_PROVIDERS = ("server_default", "none", "crossencoder", "voyage")

_DEFAULT_RERANKER_MODEL = {
    "crossencoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "voyage": "rerank-2",
}


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var without the ``bool("false") is True`` trap.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset.

    Returns:
        True only for a recognised truthy string (case-insensitive
        ``"1"``/``"true"``/``"yes"``/``"on"``); everything else — including
        ``"false"``, ``"0"``, ``""`` — is False.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def index_compat_mode() -> str:
    """Return the configured index-compatibility mode: ``"strict"`` or ``"degraded"``.

    Defaults to ``"strict"`` — incompatible/missing indexes fail explicitly
    rather than silently falling back to BM25-only, per 定案 #13's "不兼容
    默认明确失败". Set ``RAG_INDEX_COMPAT_MODE=degraded`` to allow a logged
    fallback instead (demo use only, mirrors MAEDA's own
    ``MCP_STRICT_MODE`` strict/degraded split).
    """
    mode = os.environ.get("RAG_INDEX_COMPAT_MODE", "strict").strip().lower()
    return "degraded" if mode == "degraded" else "strict"


# ---------------------------------------------------------------------------
# Embedding provider resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedEmbedding:
    """Result of resolving an ``embedding_provider`` enum value.

    Attributes:
        provider: Concrete provider key (``"openai"``) or None for BM25-only.
        model: Embedding model identifier (meaningless when provider is None).
        dimension: Expected vector dimensionality (meaningless when provider is None).
    """

    provider: Optional[str]
    model: str
    dimension: int


def _server_default_embedding_provider() -> Optional[str]:
    raw = os.environ.get("RAG_EMBEDDING_PROVIDER_DEFAULT", "none").strip().lower()
    return None if raw in ("", "none") else raw


def resolve_embedding_config(
    embedding_provider: Optional[str],
    embedding_model: str,
    vector_dimension: int,
) -> ResolvedEmbedding:
    """Resolve an ``embedding_provider`` enum value to a concrete provider or None.

    Args:
        embedding_provider: One of ``EMBEDDING_PROVIDERS``, or None (legacy
            clients that never set the field) — normalized to
            ``"server_default"`` per 定案 #7's "旧客户端未传字段归一化为
            server_default".
        embedding_model: Model identifier, passed through unchanged —
            provider and model are independent knobs (定案 #7).
        vector_dimension: Expected vector dimensionality, passed through unchanged.

    Returns:
        ``ResolvedEmbedding`` with ``provider=None`` meaning "BM25 only".
    """
    value = (embedding_provider or "server_default").strip().lower()
    if value == "none":
        provider: Optional[str] = None
    elif value == "server_default":
        provider = _server_default_embedding_provider()
    else:
        provider = value
    return ResolvedEmbedding(provider=provider, model=embedding_model, dimension=vector_dimension)


def build_embedding_provider(resolved: ResolvedEmbedding):
    """Construct the embedding provider instance for a resolved config.

    Args:
        resolved: Output of :func:`resolve_embedding_config`.

    Returns:
        A ``BaseEmbeddingProvider`` instance, or None when
        ``resolved.provider`` is None (BM25-only).

    Raises:
        EmbeddingProviderUnavailableError: The provider's dependency (API
            key, package) is unavailable — never a bare exception, so
            callers can raise it straight through to the MCP client with a
            stable error code.
    """
    if resolved.provider is None:
        return None
    if resolved.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise EmbeddingProviderUnavailableError(
                "OPENAI_API_KEY is not set; cannot construct the openai embedding provider."
            )
        try:
            from rag.infra.embedding.openai_embedding import OpenAIEmbeddingProvider

            return OpenAIEmbeddingProvider(model=resolved.model, dim=resolved.dimension)
        except Exception as exc:  # ImportError, OpenAIError, etc.
            raise EmbeddingProviderUnavailableError(str(exc)) from exc
    raise EmbeddingProviderUnavailableError(f"Unknown embedding provider '{resolved.provider}'.")


# ---------------------------------------------------------------------------
# Reranker resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedReranker:
    """Result of resolving a ``reranker_provider`` enum value.

    Attributes:
        provider: Concrete provider key (``"crossencoder"``/``"voyage"``) or
            None for no reranking.
        model: Reranker model identifier (meaningless when provider is None).
    """

    provider: Optional[str]
    model: str


def _server_default_reranker_provider() -> Optional[str]:
    raw = os.environ.get("RAG_RERANKER_PROVIDER_DEFAULT", "none").strip().lower()
    return None if raw in ("", "none") else raw


def resolve_reranker_config(
    reranker_provider: Optional[str],
    reranker_model: str = "",
) -> ResolvedReranker:
    """Resolve a ``reranker_provider`` enum value to a concrete provider or None.

    Args:
        reranker_provider: One of ``RERANKER_PROVIDERS``, or None (legacy
            clients) — normalized to ``"server_default"``.
        reranker_model: Explicit model identifier, or ``""`` to use the
            resolved provider's own default model.

    Returns:
        ``ResolvedReranker`` with ``provider=None`` meaning "no reranking".
    """
    value = (reranker_provider or "server_default").strip().lower()
    if value == "none":
        provider: Optional[str] = None
    elif value == "server_default":
        provider = _server_default_reranker_provider()
    else:
        provider = value
    model = reranker_model or (_DEFAULT_RERANKER_MODEL.get(provider, "") if provider else "")
    return ResolvedReranker(provider=provider, model=model)


def build_reranker(resolved: ResolvedReranker):
    """Construct the reranker instance for a resolved config.

    Args:
        resolved: Output of :func:`resolve_reranker_config`.

    Returns:
        A ``BaseReranker`` instance, or None when ``resolved.provider`` is
        None (no reranking).

    Raises:
        RerankerUnavailableError: The provider's dependency (API key,
            package) is unavailable.
    """
    if resolved.provider is None:
        return None
    if resolved.provider == "crossencoder":
        try:
            from rag.infra.rerank.crossencoder_reranker import CrossEncoderReranker

            return CrossEncoderReranker(model=resolved.model)
        except Exception as exc:  # ImportError, model download failure, etc.
            raise RerankerUnavailableError(str(exc)) from exc
    if resolved.provider == "voyage":
        if not os.environ.get("VOYAGE_API_KEY"):
            raise RerankerUnavailableError(
                "VOYAGE_API_KEY is not set; cannot construct the voyage reranker."
            )
        try:
            from rag.infra.rerank.voyage_rerank import VoyageReranker

            return VoyageReranker(model=resolved.model)
        except Exception as exc:
            raise RerankerUnavailableError(str(exc)) from exc
    raise RerankerUnavailableError(f"Unknown reranker provider '{resolved.provider}'.")
