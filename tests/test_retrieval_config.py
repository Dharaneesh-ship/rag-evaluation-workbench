"""Tests for R1 — shared retrieval configuration resolution (定案 #7)."""

from __future__ import annotations

import pytest

from rag.app.mcp_server.retrieval_config import (
    ResolvedEmbedding,
    ResolvedReranker,
    _env_bool,
    build_embedding_provider,
    build_reranker,
    index_compat_mode,
    resolve_embedding_config,
    resolve_reranker_config,
)
from rag.core.errors import EmbeddingProviderUnavailableError, RerankerUnavailableError


# ---------------------------------------------------------------------------
# _env_bool — must not fall into the bool("false") is True trap
# ---------------------------------------------------------------------------


class TestEnvBool:
    @pytest.mark.parametrize("raw", ["1", "true", "True", "TRUE", "yes", "on"])
    def test_truthy_strings(self, monkeypatch, raw):
        monkeypatch.setenv("X", raw)
        assert _env_bool("X") is True

    @pytest.mark.parametrize("raw", ["0", "false", "False", "FALSE", "no", "off", ""])
    def test_falsy_strings_including_the_word_false(self, monkeypatch, raw):
        monkeypatch.setenv("X", raw)
        assert _env_bool("X") is False

    def test_unset_uses_default(self, monkeypatch):
        monkeypatch.delenv("X", raising=False)
        assert _env_bool("X", default=True) is True
        assert _env_bool("X", default=False) is False


# ---------------------------------------------------------------------------
# index_compat_mode
# ---------------------------------------------------------------------------


class TestIndexCompatMode:
    def test_defaults_to_strict(self, monkeypatch):
        monkeypatch.delenv("RAG_INDEX_COMPAT_MODE", raising=False)
        assert index_compat_mode() == "strict"

    def test_degraded_when_set(self, monkeypatch):
        monkeypatch.setenv("RAG_INDEX_COMPAT_MODE", "degraded")
        assert index_compat_mode() == "degraded"

    def test_unrecognized_value_falls_back_to_strict(self, monkeypatch):
        monkeypatch.setenv("RAG_INDEX_COMPAT_MODE", "yolo")
        assert index_compat_mode() == "strict"


# ---------------------------------------------------------------------------
# Embedding provider resolution matrix:
# (unspecified/none/explicit) x (server default set/unset)
# ---------------------------------------------------------------------------


class TestResolveEmbeddingConfig:
    def test_none_value_normalized_to_server_default_then_off(self, monkeypatch):
        monkeypatch.delenv("RAG_EMBEDDING_PROVIDER_DEFAULT", raising=False)
        resolved = resolve_embedding_config(None, "text-embedding-3-small", 1536)
        assert resolved.provider is None

    def test_explicit_none_disables_regardless_of_server_default(self, monkeypatch):
        monkeypatch.setenv("RAG_EMBEDDING_PROVIDER_DEFAULT", "openai")
        resolved = resolve_embedding_config("none", "text-embedding-3-small", 1536)
        assert resolved.provider is None

    def test_server_default_with_default_unset_is_bm25_only(self, monkeypatch):
        monkeypatch.delenv("RAG_EMBEDDING_PROVIDER_DEFAULT", raising=False)
        resolved = resolve_embedding_config("server_default", "m", 10)
        assert resolved.provider is None

    def test_server_default_with_default_set_resolves_to_it(self, monkeypatch):
        monkeypatch.setenv("RAG_EMBEDDING_PROVIDER_DEFAULT", "openai")
        resolved = resolve_embedding_config("server_default", "m", 10)
        assert resolved.provider == "openai"

    def test_explicit_provider_overrides_server_default(self, monkeypatch):
        monkeypatch.setenv("RAG_EMBEDDING_PROVIDER_DEFAULT", "none")
        resolved = resolve_embedding_config("openai", "m", 10)
        assert resolved.provider == "openai"

    def test_model_and_dimension_pass_through_unchanged(self, monkeypatch):
        monkeypatch.setenv("RAG_EMBEDDING_PROVIDER_DEFAULT", "openai")
        resolved = resolve_embedding_config("server_default", "text-embedding-3-large", 3072)
        assert resolved.model == "text-embedding-3-large"
        assert resolved.dimension == 3072


class TestBuildEmbeddingProvider:
    def test_none_provider_returns_none(self):
        assert build_embedding_provider(ResolvedEmbedding(None, "m", 10)) is None

    def test_openai_without_api_key_raises_unavailable(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EmbeddingProviderUnavailableError) as exc_info:
            build_embedding_provider(ResolvedEmbedding("openai", "text-embedding-3-small", 1536))
        assert "EMBEDDING_PROVIDER_UNAVAILABLE" in str(exc_info.value)

    def test_openai_with_api_key_constructs_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test-key")
        provider = build_embedding_provider(
            ResolvedEmbedding("openai", "text-embedding-3-small", 1536)
        )
        assert provider is not None
        assert provider.dim == 1536

    def test_unknown_provider_raises_unavailable(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test-key")
        with pytest.raises(EmbeddingProviderUnavailableError):
            build_embedding_provider(ResolvedEmbedding("not-a-real-provider", "m", 10))


# ---------------------------------------------------------------------------
# Reranker resolution matrix
# ---------------------------------------------------------------------------


class TestResolveRerankerConfig:
    def test_none_value_normalized_to_server_default_then_off(self, monkeypatch):
        monkeypatch.delenv("RAG_RERANKER_PROVIDER_DEFAULT", raising=False)
        resolved = resolve_reranker_config(None)
        assert resolved.provider is None

    def test_explicit_none_disables_regardless_of_server_default(self, monkeypatch):
        monkeypatch.setenv("RAG_RERANKER_PROVIDER_DEFAULT", "crossencoder")
        resolved = resolve_reranker_config("none")
        assert resolved.provider is None

    def test_server_default_with_default_unset_is_off(self, monkeypatch):
        monkeypatch.delenv("RAG_RERANKER_PROVIDER_DEFAULT", raising=False)
        resolved = resolve_reranker_config("server_default")
        assert resolved.provider is None

    def test_server_default_with_default_set_resolves_to_it(self, monkeypatch):
        monkeypatch.setenv("RAG_RERANKER_PROVIDER_DEFAULT", "voyage")
        resolved = resolve_reranker_config("server_default")
        assert resolved.provider == "voyage"

    def test_explicit_provider_overrides_server_default(self, monkeypatch):
        monkeypatch.setenv("RAG_RERANKER_PROVIDER_DEFAULT", "none")
        resolved = resolve_reranker_config("crossencoder")
        assert resolved.provider == "crossencoder"

    def test_empty_model_falls_back_to_provider_default(self, monkeypatch):
        resolved = resolve_reranker_config("voyage", "")
        assert resolved.model == "rerank-2"
        resolved2 = resolve_reranker_config("crossencoder", "")
        assert resolved2.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_explicit_model_overrides_provider_default(self):
        resolved = resolve_reranker_config("voyage", "rerank-lite-1")
        assert resolved.model == "rerank-lite-1"


class TestBuildReranker:
    def test_none_provider_returns_none(self):
        assert build_reranker(ResolvedReranker(None, "")) is None

    def test_voyage_without_api_key_raises_unavailable(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        with pytest.raises(RerankerUnavailableError) as exc_info:
            build_reranker(ResolvedReranker("voyage", "rerank-2"))
        assert "RERANKER_UNAVAILABLE" in str(exc_info.value)

    def test_unknown_provider_raises_unavailable(self):
        with pytest.raises(RerankerUnavailableError):
            build_reranker(ResolvedReranker("not-a-real-reranker", "m"))
