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
    log_server_default_retrieval_tier,
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


# ---------------------------------------------------------------------------
# 附录 CI.3 (CH.2) — silent server_default degradation must not be silent
# ---------------------------------------------------------------------------
#
# CH.2 found that RAG_EMBEDDING_PROVIDER_DEFAULT/RAG_RERANKER_PROVIDER_DEFAULT
# unset resolves every "server_default" request to BM25-only/no-rerank with
# zero signal anywhere. CI.3's ruling: not fail-fast (an unset default is a
# legitimate, already-tested deployment shape — see
# test_server_default_with_default_unset_is_bm25_only above — not
# necessarily a misconfiguration), but it must stop being *silent*: a log at
# resolution time (these tests) plus the already-existing response echo
# field (RetrieveOutput.embedding_provider/retrieval_mode, covered in
# tests/test_mcp_server.py and tests/test_retrieve_tool.py).


class TestServerDefaultResolutionIsLogged:
    def test_embedding_default_unset_logs_a_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("RAG_EMBEDDING_PROVIDER_DEFAULT", raising=False)
        with caplog.at_level("WARNING", logger="rag.app.mcp_server.retrieval_config"):
            resolve_embedding_config("server_default", "m", 10)
        assert any(
            "RAG_EMBEDDING_PROVIDER_DEFAULT" in r.message and "BM25-only" in r.message
            for r in caplog.records
        )

    def test_embedding_default_set_logs_info_not_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("RAG_EMBEDDING_PROVIDER_DEFAULT", "openai")
        with caplog.at_level("INFO", logger="rag.app.mcp_server.retrieval_config"):
            resolve_embedding_config("server_default", "m", 10)
        assert not any(r.levelname == "WARNING" for r in caplog.records)
        assert any("openai" in r.message for r in caplog.records)

    def test_explicit_none_does_not_touch_server_default_logging(self, monkeypatch, caplog):
        """Explicit embedding_provider='none' is a deliberate caller
        choice, not the silent-default gap CI.3 is about — it must not go
        through (and therefore not warn from) the server_default path."""
        monkeypatch.delenv("RAG_EMBEDDING_PROVIDER_DEFAULT", raising=False)
        with caplog.at_level("WARNING", logger="rag.app.mcp_server.retrieval_config"):
            resolve_embedding_config("none", "m", 10)
        assert caplog.records == []

    def test_reranker_default_unset_logs_a_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("RAG_RERANKER_PROVIDER_DEFAULT", raising=False)
        with caplog.at_level("WARNING", logger="rag.app.mcp_server.retrieval_config"):
            resolve_reranker_config("server_default")
        assert any(
            "RAG_RERANKER_PROVIDER_DEFAULT" in r.message and "no reranking" in r.message
            for r in caplog.records
        )

    def test_reranker_default_set_logs_info_not_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("RAG_RERANKER_PROVIDER_DEFAULT", "voyage")
        with caplog.at_level("INFO", logger="rag.app.mcp_server.retrieval_config"):
            resolve_reranker_config("server_default")
        assert not any(r.levelname == "WARNING" for r in caplog.records)
        assert any("voyage" in r.message for r in caplog.records)


class TestLogServerDefaultRetrievalTier:
    def test_both_unset_logs_one_combined_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("RAG_EMBEDDING_PROVIDER_DEFAULT", raising=False)
        monkeypatch.delenv("RAG_RERANKER_PROVIDER_DEFAULT", raising=False)
        with caplog.at_level("WARNING", logger="rag.app.mcp_server.retrieval_config"):
            log_server_default_retrieval_tier()
        assert any("BM25-only" in r.message for r in caplog.records)
        assert any("Startup retrieval tier check" in r.message for r in caplog.records)

    def test_both_configured_logs_info_summary_not_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("RAG_EMBEDDING_PROVIDER_DEFAULT", "openai")
        monkeypatch.setenv("RAG_RERANKER_PROVIDER_DEFAULT", "voyage")
        with caplog.at_level("INFO", logger="rag.app.mcp_server.retrieval_config"):
            log_server_default_retrieval_tier()
        startup_records = [r for r in caplog.records if "Startup retrieval tier check" in r.message]
        assert len(startup_records) == 1
        assert startup_records[0].levelname == "INFO"
        assert "openai" in startup_records[0].message
        assert "voyage" in startup_records[0].message
