"""Tests for R2 — index compatibility validation (定案 #13's INDEX_NOT_FOUND /
INDEX_CONFIG_MISMATCH), covering the (index state) x (requested config) x
(strict/degraded mode) matrix."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rag.app.mcp_server.index_validation import (
    validate_index_compatibility,
    validate_ingest_embedding_config,
)
from rag.app.mcp_server.retrieval_config import ResolvedEmbedding
from rag.core.errors import IndexConfigMismatchError, IndexNotFoundError
from rag.infra.indexes.manifest import IndexManifest


def _fake_manager(*, manifest: IndexManifest, faiss_dim: int):
    """A minimal stand-in exposing only what validate_* touches."""
    return SimpleNamespace(manifest=manifest, faiss=SimpleNamespace(dimension=faiss_dim))


# ---------------------------------------------------------------------------
# Query-time validation — validate_index_compatibility
# ---------------------------------------------------------------------------


class TestValidateIndexCompatibilityBM25Only:
    def test_no_provider_requested_is_always_a_noop_success(self):
        mgr = _fake_manager(manifest=IndexManifest.empty(), faiss_dim=0)
        result = validate_index_compatibility(mgr, ResolvedEmbedding(None, "m", 10))
        assert result.use_vector_search is False
        assert result.degraded_reason is None


class TestValidateIndexCompatibilityStrictMode:
    """Default mode: incompatible/missing indexes raise, never silently degrade."""

    def test_compatible_index_succeeds(self, monkeypatch):
        monkeypatch.delenv("RAG_INDEX_COMPAT_MODE", raising=False)
        manifest = IndexManifest(embedding_provider="openai", embedding_model="m", vector_dimension=10)
        mgr = _fake_manager(manifest=manifest, faiss_dim=10)
        result = validate_index_compatibility(mgr, ResolvedEmbedding("openai", "m", 10))
        assert result.use_vector_search is True
        assert result.degraded_reason is None

    def test_no_index_ever_built_raises_index_not_found(self, monkeypatch):
        monkeypatch.delenv("RAG_INDEX_COMPAT_MODE", raising=False)
        mgr = _fake_manager(manifest=IndexManifest.empty(), faiss_dim=0)
        with pytest.raises(IndexNotFoundError) as exc_info:
            validate_index_compatibility(mgr, ResolvedEmbedding("openai", "m", 10))
        assert "INDEX_NOT_FOUND" in str(exc_info.value)

    def test_provider_mismatch_raises_config_mismatch(self, monkeypatch):
        monkeypatch.delenv("RAG_INDEX_COMPAT_MODE", raising=False)
        manifest = IndexManifest(embedding_provider="openai", embedding_model="m", vector_dimension=10)
        mgr = _fake_manager(manifest=manifest, faiss_dim=10)
        with pytest.raises(IndexConfigMismatchError) as exc_info:
            validate_index_compatibility(mgr, ResolvedEmbedding("cohere", "m", 10))
        assert "INDEX_CONFIG_MISMATCH" in str(exc_info.value)

    def test_model_mismatch_raises_config_mismatch(self, monkeypatch):
        monkeypatch.delenv("RAG_INDEX_COMPAT_MODE", raising=False)
        manifest = IndexManifest(
            embedding_provider="openai", embedding_model="text-embedding-3-small", vector_dimension=1536
        )
        mgr = _fake_manager(manifest=manifest, faiss_dim=1536)
        with pytest.raises(IndexConfigMismatchError):
            validate_index_compatibility(
                mgr, ResolvedEmbedding("openai", "text-embedding-3-large", 1536)
            )

    def test_dimension_mismatch_raises_config_mismatch(self, monkeypatch):
        monkeypatch.delenv("RAG_INDEX_COMPAT_MODE", raising=False)
        manifest = IndexManifest(embedding_provider="openai", embedding_model="m", vector_dimension=1536)
        mgr = _fake_manager(manifest=manifest, faiss_dim=1536)
        with pytest.raises(IndexConfigMismatchError):
            validate_index_compatibility(mgr, ResolvedEmbedding("openai", "m", 3072))


class TestValidateIndexCompatibilityDegradedMode:
    """RAG_INDEX_COMPAT_MODE=degraded: incompatibility logs and falls back instead of raising."""

    def test_no_index_degrades_to_bm25_with_reason(self, monkeypatch):
        monkeypatch.setenv("RAG_INDEX_COMPAT_MODE", "degraded")
        mgr = _fake_manager(manifest=IndexManifest.empty(), faiss_dim=0)
        result = validate_index_compatibility(mgr, ResolvedEmbedding("openai", "m", 10))
        assert result.use_vector_search is False
        assert result.degraded_reason is not None
        assert "openai" in result.degraded_reason

    def test_mismatch_degrades_to_bm25_with_reason(self, monkeypatch):
        monkeypatch.setenv("RAG_INDEX_COMPAT_MODE", "degraded")
        manifest = IndexManifest(embedding_provider="openai", embedding_model="m", vector_dimension=10)
        mgr = _fake_manager(manifest=manifest, faiss_dim=10)
        result = validate_index_compatibility(mgr, ResolvedEmbedding("openai", "m", 999))
        assert result.use_vector_search is False
        assert result.degraded_reason is not None

    def test_compatible_index_still_succeeds_in_degraded_mode(self, monkeypatch):
        monkeypatch.setenv("RAG_INDEX_COMPAT_MODE", "degraded")
        manifest = IndexManifest(embedding_provider="openai", embedding_model="m", vector_dimension=10)
        mgr = _fake_manager(manifest=manifest, faiss_dim=10)
        result = validate_index_compatibility(mgr, ResolvedEmbedding("openai", "m", 10))
        assert result.use_vector_search is True
        assert result.degraded_reason is None


# ---------------------------------------------------------------------------
# Ingest-time validation — validate_ingest_embedding_config (always strict:
# mixing embedding spaces in one FAISS index is corruption, not a degrade path)
# ---------------------------------------------------------------------------


class TestValidateIngestEmbeddingConfig:
    def test_no_provider_requested_is_always_a_noop(self):
        mgr = _fake_manager(manifest=IndexManifest.empty(), faiss_dim=0)
        validate_ingest_embedding_config(mgr, ResolvedEmbedding(None, "m", 10))  # no raise

    def test_first_ever_embedding_config_is_accepted(self):
        mgr = _fake_manager(manifest=IndexManifest.empty(), faiss_dim=0)
        validate_ingest_embedding_config(mgr, ResolvedEmbedding("openai", "m", 10))  # no raise

    def test_matching_config_is_accepted(self):
        manifest = IndexManifest(embedding_provider="openai", embedding_model="m", vector_dimension=10)
        mgr = _fake_manager(manifest=manifest, faiss_dim=10)
        validate_ingest_embedding_config(mgr, ResolvedEmbedding("openai", "m", 10))  # no raise

    def test_conflicting_provider_raises_even_in_degraded_mode(self, monkeypatch):
        monkeypatch.setenv("RAG_INDEX_COMPAT_MODE", "degraded")
        manifest = IndexManifest(embedding_provider="openai", embedding_model="m", vector_dimension=10)
        mgr = _fake_manager(manifest=manifest, faiss_dim=10)
        with pytest.raises(IndexConfigMismatchError):
            validate_ingest_embedding_config(mgr, ResolvedEmbedding("cohere", "m", 10))

    def test_conflicting_dimension_raises(self):
        manifest = IndexManifest(embedding_provider="openai", embedding_model="m", vector_dimension=10)
        mgr = _fake_manager(manifest=manifest, faiss_dim=10)
        with pytest.raises(IndexConfigMismatchError):
            validate_ingest_embedding_config(mgr, ResolvedEmbedding("openai", "m", 999))
