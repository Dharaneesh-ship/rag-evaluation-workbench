"""Tests for R3 — retrieve/retrieve_with_metadata error semantics and R2 wiring.

Covers: legal zero hits stay success, internal failures raise real MCP tool
errors with stable codes, collection validation, both retrieve tools share
one resolution path, and the actual resolved config lands in the trace.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rag.app.mcp_server.schemas import IngestToolInput, RetrieveInput
from rag.app.mcp_server.server import rag_ingest, retrieve, retrieve_with_metadata
from rag.core.errors import (
    IndexNotFoundError,
    InvalidCollectionError,
    RetrievalInternalError,
)


@pytest.fixture()
def corpus(tmp_path: Path):
    """A tiny real (BM25-only) corpus tagged with collection='knowledge_base'."""
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "The quick brown fox jumps over the lazy dog. "
        "RAG frameworks combine retrieval with generation."
    )
    db_path = str(tmp_path / "rag.db")
    index_dir = str(tmp_path / "indexes")
    out = rag_ingest(
        IngestToolInput(
            source_path=str(doc),
            db_path=db_path,
            index_dir=index_dir,
            collection="knowledge_base",
        )
    )
    assert out.error is None
    assert out.chunk_count > 0
    return {"db_path": db_path, "index_dir": index_dir}


# ---------------------------------------------------------------------------
# Legal zero hits vs. internal failure — the core R3 distinction
# ---------------------------------------------------------------------------


class TestLegalZeroHitsAreSuccess:
    def test_retrieve_returns_empty_chunks_not_an_error(self, corpus):
        result = retrieve(
            RetrieveInput(
                query="completely unrelated gibberish query aaaaaaaaaaaaaaaaaaaaaa",
                collection="knowledge_base",
                **corpus,
            )
        )
        assert result.error is None
        assert isinstance(result.chunks, list)

    def test_retrieve_with_metadata_populates_resolved_config_on_success(self, corpus):
        result = retrieve_with_metadata(
            RetrieveInput(query="fox", collection="knowledge_base", **corpus)
        )
        assert result.error is None
        assert result.retrieval_mode == "bm25_only"
        assert result.embedding_provider is None
        assert result.degraded_reason is None


class TestInternalFailuresRaiseRealToolErrors:
    def test_pipeline_error_result_raises_retrieval_internal_error(self, corpus):
        """QueryPipeline.query() swallows its own exceptions into .error —
        previously that silently became RetrieveOutput(chunks=[]), which is
        indistinguishable from a legitimate empty result. It must now raise.
        """
        from rag.pipelines.query_pipeline import QueryResult

        broken_result = QueryResult(query="fox", error="boom: simulated internal failure")
        with patch(
            "rag.pipelines.query_pipeline.QueryPipeline.query", return_value=broken_result
        ):
            with pytest.raises(RetrievalInternalError) as exc_info:
                retrieve(RetrieveInput(query="fox", collection="knowledge_base", **corpus))
        assert "RETRIEVAL_INTERNAL_ERROR" in str(exc_info.value)
        assert "boom" in str(exc_info.value)

    def test_unexpected_exception_wrapped_as_retrieval_internal_error(self, corpus):
        with patch(
            "rag.app.mcp_server.server._run_retrieval", side_effect=RuntimeError("kaboom")
        ):
            with pytest.raises(RetrievalInternalError):
                retrieve(RetrieveInput(query="fox", collection="knowledge_base", **corpus))

    def test_vector_search_requested_without_index_raises_index_not_found(self, corpus):
        with pytest.raises(IndexNotFoundError):
            retrieve(
                RetrieveInput(
                    query="fox",
                    collection="knowledge_base",
                    embedding_provider="openai",
                    **corpus,
                )
            )

    def test_index_not_found_error_propagates_from_retrieve_with_metadata_too(self, corpus):
        with pytest.raises(IndexNotFoundError):
            retrieve_with_metadata(
                RetrieveInput(
                    query="fox",
                    collection="knowledge_base",
                    embedding_provider="openai",
                    **corpus,
                )
            )


# ---------------------------------------------------------------------------
# Collection validation (INVALID_COLLECTION)
# ---------------------------------------------------------------------------


class TestCollectionValidation:
    def test_unknown_collection_raises_invalid_collection(self, corpus):
        with pytest.raises(InvalidCollectionError) as exc_info:
            retrieve(RetrieveInput(query="fox", collection="does_not_exist", **corpus))
        assert "INVALID_COLLECTION" in str(exc_info.value)

    def test_known_collection_succeeds(self, corpus):
        result = retrieve(RetrieveInput(query="fox", collection="knowledge_base", **corpus))
        assert result.error is None

    def test_unscoped_query_never_triggers_collection_validation(self, corpus):
        # collection=None must never be checked against known collections —
        # it means "search everything," not "an unnamed collection."
        result = retrieve(RetrieveInput(query="fox", collection=None, **corpus))
        assert result.error is None

    def test_fresh_empty_corpus_does_not_false_positive_on_any_collection_name(self, tmp_path):
        # No documents tagged with any collection yet — must not raise, since
        # "nothing tagged" is different from "this specific collection is wrong."
        db_path = str(tmp_path / "rag.db")
        index_dir = str(tmp_path / "indexes")
        result = retrieve(
            RetrieveInput(
                query="anything", collection="brand_new", db_path=db_path, index_dir=index_dir
            )
        )
        assert result.error is None


# ---------------------------------------------------------------------------
# Both retrieve tools share one resolution path (R2's "两个 retrieve 工具
# 确实共用同一套配置解析" requirement)
# ---------------------------------------------------------------------------


class TestSharedResolutionAcrossBothRetrieveTools:
    def test_retrieve_calls_shared_run_retrieval_helper(self, corpus):
        from rag.pipelines.query_pipeline import QueryResult

        with patch(
            "rag.app.mcp_server.server._run_retrieval",
            wraps=__import__(
                "rag.app.mcp_server.server", fromlist=["_run_retrieval"]
            )._run_retrieval,
        ) as spy:
            retrieve(RetrieveInput(query="fox", collection="knowledge_base", **corpus))
        assert spy.called

    def test_retrieve_with_metadata_calls_shared_run_retrieval_helper(self, corpus):
        with patch(
            "rag.app.mcp_server.server._run_retrieval",
            wraps=__import__(
                "rag.app.mcp_server.server", fromlist=["_run_retrieval"]
            )._run_retrieval,
        ) as spy:
            retrieve_with_metadata(RetrieveInput(query="fox", collection="knowledge_base", **corpus))
        assert spy.called


# ---------------------------------------------------------------------------
# Trace records the actual resolved config (TB1a/TB2's observability requirement)
# ---------------------------------------------------------------------------


class TestRerankerWiring:
    """retrieve/retrieve_with_metadata previously hardcoded reranker=None —
    R1's reranker_provider field must actually reach the pipeline."""

    def test_reranker_provider_is_constructed_and_reported_on_bm25_only(self, corpus):
        from rag.core.contracts.candidate import Candidate

        class _StubReranker:
            def rerank(self, query, candidates, top_k):
                for c in candidates:
                    c.final_score = 1.0
                return candidates[:top_k]

        with patch(
            "rag.app.mcp_server.retrieval_config.build_reranker",
            return_value=_StubReranker(),
        ):
            result = retrieve(
                RetrieveInput(
                    query="fox",
                    collection="knowledge_base",
                    reranker_provider="crossencoder",
                    **corpus,
                )
            )
        assert result.error is None
        assert result.reranker_provider == "crossencoder"
        # No vector index was involved — rerank alone doesn't promote the
        # named retrieval_mode past "bm25_only" (定案 #17's three named modes
        # are about the retrieval fan-in; reranker_provider carries the rest).
        assert result.retrieval_mode == "bm25_only"


class TestDegradedIndexCompatMode:
    def test_degraded_mode_returns_success_with_reason_instead_of_raising(self, corpus, monkeypatch):
        monkeypatch.setenv("RAG_INDEX_COMPAT_MODE", "degraded")
        result = retrieve(
            RetrieveInput(
                query="fox",
                collection="knowledge_base",
                embedding_provider="openai",
                **corpus,
            )
        )
        assert result.error is None
        assert result.retrieval_mode == "bm25_only"
        assert result.embedding_provider is None
        assert result.degraded_reason is not None
        assert "openai" in result.degraded_reason


class TestTraceRecordsResolvedConfig:
    def test_trace_has_a_retrieval_config_run_with_resolved_fields(self, corpus):
        from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore

        retrieve(RetrieveInput(query="fox", collection="knowledge_base", **corpus))

        trace_store = SQLiteTraceStore(corpus["db_path"])
        runs = [r for r in trace_store.list_runs() if r.get("run_type") == "retrieval_config"]
        assert runs, "expected at least one retrieval_config trace run"
        metadata = runs[0]["metadata"]  # list_runs() orders most-recent-first
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        assert metadata["retrieval_mode"] == "bm25_only"
        assert metadata["embedding_provider"] is None
        assert metadata["reranker_provider"] is None
        assert metadata["degraded_reason"] is None
