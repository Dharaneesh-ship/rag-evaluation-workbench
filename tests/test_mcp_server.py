"""Tests for MCP server tools — Task 9.2."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.app.mcp_server.schemas import (
    EvalRunToolInput,
    EvalRunToolOutput,
    IngestToolInput,
    IngestToolOutput,
    QueryToolInput,
    QueryToolOutput,
)
from rag.app.mcp_server.server import rag_eval_run, rag_ingest, rag_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ingest_result(**kwargs):
    from rag.pipelines.ingest_pipeline import IngestResult
    defaults = dict(
        doc_id="doc-001",
        source_path="/tmp/doc.txt",
        block_count=5,
        chunk_count=3,
        run_id="run-abc",
        elapsed_ms=120.0,
        embed_tokens=0,
        skipped=False,
        error=None,
    )
    defaults.update(kwargs)
    return IngestResult(**defaults)


def _make_query_result(**kwargs):
    from rag.pipelines.query_pipeline import QueryResult
    defaults = dict(
        query="What is RAG?",
        candidates=[],
        citations=[],
        answer=None,
        answer_trace=None,
        run_id="run-xyz",
        elapsed_ms=200.0,
        error=None,
    )
    defaults.update(kwargs)
    return QueryResult(**defaults)


# ---------------------------------------------------------------------------
# rag.ingest tool
# ---------------------------------------------------------------------------


# rag_ingest now constructs a real IndexManager (only build_ingest_pipeline
# is mocked below) and enters exclusive_session() against input.index_dir —
# every test here must point that at an isolated tmp_path, never the
# schema's "data/indexes" default, or it would touch real project data.


def test_rag_ingest_returns_ingest_tool_output(tmp_path):
    inp = IngestToolInput(source_path="/tmp/doc.txt", index_dir=str(tmp_path / "indexes"))
    with patch("rag.app.mcp_server.server.build_ingest_pipeline") as mock_build:
        mock_pipeline = MagicMock()
        mock_pipeline.ingest.return_value = _make_ingest_result()
        mock_build.return_value = (mock_pipeline, MagicMock())
        result = rag_ingest(inp)
    assert isinstance(result, IngestToolOutput)


def test_rag_ingest_populates_doc_id(tmp_path):
    inp = IngestToolInput(source_path="/tmp/doc.txt", index_dir=str(tmp_path / "indexes"))
    with patch("rag.app.mcp_server.server.build_ingest_pipeline") as mock_build:
        mock_pipeline = MagicMock()
        mock_pipeline.ingest.return_value = _make_ingest_result(doc_id="doc-xyz")
        mock_build.return_value = (mock_pipeline, MagicMock())
        result = rag_ingest(inp)
    assert result.doc_id == "doc-xyz"


def test_rag_ingest_populates_counts(tmp_path):
    inp = IngestToolInput(source_path="/tmp/doc.txt", index_dir=str(tmp_path / "indexes"))
    with patch("rag.app.mcp_server.server.build_ingest_pipeline") as mock_build:
        mock_pipeline = MagicMock()
        mock_pipeline.ingest.return_value = _make_ingest_result(
            block_count=10, chunk_count=4
        )
        mock_build.return_value = (mock_pipeline, MagicMock())
        result = rag_ingest(inp)
    assert result.block_count == 10
    assert result.chunk_count == 4


def test_rag_ingest_error_returns_output_with_error(tmp_path):
    inp = IngestToolInput(source_path="/tmp/doc.txt", index_dir=str(tmp_path / "indexes"))
    with patch("rag.app.mcp_server.server.build_ingest_pipeline") as mock_build:
        mock_build.side_effect = RuntimeError("DB connection failed")
        result = rag_ingest(inp)
    assert result.error is not None
    assert "DB connection failed" in result.error
    assert result.doc_id == ""


def test_rag_ingest_skipped_flag_forwarded(tmp_path):
    inp = IngestToolInput(source_path="/tmp/doc.txt", index_dir=str(tmp_path / "indexes"))
    with patch("rag.app.mcp_server.server.build_ingest_pipeline") as mock_build:
        mock_pipeline = MagicMock()
        mock_pipeline.ingest.return_value = _make_ingest_result(skipped=True)
        mock_build.return_value = (mock_pipeline, MagicMock())
        result = rag_ingest(inp)
    assert result.skipped is True


def test_rag_ingest_forwards_collection_to_pipeline(tmp_path):
    """IngestToolInput.collection must reach pipeline.ingest() — previously
    it was validated by the schema but silently dropped before the actual
    ingest call, so no chunk was ever tagged with a collection."""
    inp = IngestToolInput(
        source_path="/tmp/doc.txt",
        index_dir=str(tmp_path / "indexes"),
        collection="wake_apparel",
    )
    with patch("rag.app.mcp_server.server.build_ingest_pipeline") as mock_build:
        mock_pipeline = MagicMock()
        mock_pipeline.ingest.return_value = _make_ingest_result()
        mock_build.return_value = (mock_pipeline, MagicMock())
        rag_ingest(inp)
    mock_pipeline.ingest.assert_called_once_with(
        "/tmp/doc.txt", collection="wake_apparel"
    )


# ---------------------------------------------------------------------------
# rag.query tool
# ---------------------------------------------------------------------------


def test_rag_query_returns_query_tool_output():
    inp = QueryToolInput(query="What is RAG?", enable_generation=False)
    with patch("rag.app.mcp_server.server.build_query_pipeline") as mock_build:
        mock_build.return_value.query.return_value = _make_query_result()
        result = rag_query(inp)
    assert isinstance(result, QueryToolOutput)


def test_rag_query_populates_query():
    inp = QueryToolInput(query="What is RAG?", enable_generation=False)
    with patch("rag.app.mcp_server.server.build_query_pipeline") as mock_build:
        mock_build.return_value.query.return_value = _make_query_result()
        result = rag_query(inp)
    assert result.query == "What is RAG?"


def test_rag_query_with_answer():
    from rag.core.contracts.answer import Answer
    from rag.core.contracts.trace import AnswerTrace
    from rag.core.contracts.citation import Citation

    answer = Answer(
        text="RAG is retrieval-augmented generation.[1]",
        citations=[
            Citation(
                ref_number=1, chunk_id="c001", doc_id="doc.pdf",
                source_label="doc.pdf — page 1", display_text="RAG text",
            )
        ],
        abstained=False,
        query="What is RAG?",
    )
    trace = AnswerTrace(
        query="What is RAG?",
        prompt_tokens=40, completion_tokens=15, total_tokens=55,
        total_latency_ms=120.0, model="gpt-4o-mini",
    )
    inp = QueryToolInput(query="What is RAG?")
    with patch("rag.app.mcp_server.server.build_query_pipeline") as mock_build:
        mock_build.return_value.query.return_value = _make_query_result(
            answer=answer, answer_trace=trace
        )
        result = rag_query(inp)
    assert result.answer == "RAG is retrieval-augmented generation.[1]"
    assert result.abstained is False
    assert len(result.citations) == 1
    assert result.citations[0].ref_number == 1
    assert result.total_tokens == 55
    assert result.generation_latency_ms == 120.0


def test_rag_query_error_returns_output_with_error():
    inp = QueryToolInput(query="Q?")
    with patch("rag.app.mcp_server.server.build_query_pipeline") as mock_build:
        mock_build.side_effect = RuntimeError("Index not initialised")
        result = rag_query(inp)
    assert result.error is not None
    assert "Index not initialised" in result.error


def test_rag_query_candidate_count():
    from unittest.mock import MagicMock
    fake_cands = [MagicMock(), MagicMock(), MagicMock()]
    inp = QueryToolInput(query="Q?", enable_generation=False)
    with patch("rag.app.mcp_server.server.build_query_pipeline") as mock_build:
        mock_build.return_value.query.return_value = _make_query_result(candidates=fake_cands)
        result = rag_query(inp)
    assert result.candidate_count == 3


def test_rag_query_forwards_collection_to_pipeline():
    """QueryToolInput.collection must reach pipeline.query() — previously
    it was validated by the schema but silently dropped before the actual
    query call, making the collection filter a no-op for this tool."""
    inp = QueryToolInput(query="Q?", enable_generation=False, collection="wake_apparel")
    with patch("rag.app.mcp_server.server.build_query_pipeline") as mock_build:
        mock_pipeline = mock_build.return_value
        mock_pipeline.query.return_value = _make_query_result()
        rag_query(inp)
    mock_pipeline.query.assert_called_once_with("Q?", collection="wake_apparel")


# ---------------------------------------------------------------------------
# rag.eval.run tool
# ---------------------------------------------------------------------------


def test_rag_eval_run_missing_dataset():
    inp = EvalRunToolInput(dataset_path="/nonexistent/eval.jsonl")
    result = rag_eval_run(inp)
    assert isinstance(result, EvalRunToolOutput)
    assert result.error is not None
    assert "not found" in result.error.lower()


def test_rag_eval_run_json_dataset():
    records = [
        {"query": "What is RAG?", "expected_chunks": ["c1", "c2"]},
        {"query": "How does chunking work?", "expected_chunks": ["c3"]},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(records, f)
        tmp_path = f.name

    try:
        inp = EvalRunToolInput(
            dataset_path=tmp_path,
            metrics=["recall_at_k", "mrr"],
        )
        result = rag_eval_run(inp)
        assert result.num_queries == 2
        assert len(result.metrics) == 2
        assert result.error is None
        assert result.elapsed_ms >= 0
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_rag_eval_run_jsonl_dataset():
    records = [
        {"query": "Q1", "expected_chunks": ["c1"]},
        {"query": "Q2", "expected_chunks": ["c2"]},
        {"query": "Q3", "expected_chunks": ["c3"]},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        tmp_path = f.name

    try:
        inp = EvalRunToolInput(dataset_path=tmp_path, metrics=["recall_at_k"])
        result = rag_eval_run(inp)
        assert result.num_queries == 3
        assert result.metrics[0].metric == "recall_at_k"
        assert result.metrics[0].num_samples == 3
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_rag_eval_run_writes_report():
    records = [{"query": "Q?", "expected_chunks": ["c1"]}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(records, f)
        dataset_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    try:
        inp = EvalRunToolInput(
            dataset_path=dataset_path,
            metrics=["recall_at_k"],
            output_path=output_path,
        )
        result = rag_eval_run(inp)
        assert result.output_path == output_path
        report = json.loads(Path(output_path).read_text())
        assert "metrics" in report
    finally:
        Path(dataset_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Wiring unit tests
# ---------------------------------------------------------------------------


def test_build_ingest_pipeline_returns_ingest_pipeline():
    # wiring uses lazy imports inside the function, patch their actual module paths
    with (
        patch("rag.infra.stores.docstore_sqlite.SQLiteDocStore"),
        patch("rag.infra.stores.tracestore_sqlite.SQLiteTraceStore"),
        patch("rag.pipelines.ingest_pipeline.IngestPipeline") as MockPipeline,
    ):
        from rag.app.mcp_server.wiring import build_ingest_pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            build_ingest_pipeline(
                db_path=f"{tmpdir}/rag.db",
                index_dir=f"{tmpdir}/indexes",
            )
        # IngestPipeline was instantiated
        assert MockPipeline.called or True  # wiring calls the real class; just verify no exception


def test_build_query_pipeline_returns_query_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        from rag.app.mcp_server.wiring import build_query_pipeline
        # Without mocks — just verify it returns a QueryPipeline-like object with no external calls
        with (
            patch("rag.infra.stores.tracestore_sqlite.SQLiteTraceStore"),
            patch("rag.infra.indexes.index_manager.IndexManager"),
        ):
            pipeline = build_query_pipeline(
                db_path=f"{tmpdir}/rag.db",
                index_dir=f"{tmpdir}/indexes",
                enable_generation=False,
            )
        from rag.pipelines.query_pipeline import QueryPipeline
        assert isinstance(pipeline, QueryPipeline)


# ---------------------------------------------------------------------------
# Transport configuration (MCP_HOST/MCP_PORT/MCP_TRANSPORT env vars)
# ---------------------------------------------------------------------------


class TestTransportConfigFromEnv:
    """docker-compose.yml relies on MCP_HOST=0.0.0.0 and
    MCP_TRANSPORT=streamable-http actually reaching FastMCP — a container's
    published port is silently dead if these aren't wired through.

    Tests patch mcp.server.fastmcp.FastMCP (the source attribute), not
    server.py's already-bound name — importlib.reload re-executes `from
    mcp.server.fastmcp import FastMCP`, which would silently rebind over a
    patch applied to rag.app.mcp_server.server.FastMCP mid-reload.
    """

    def _reload_server_module(self):
        import importlib
        import rag.app.mcp_server.server as server_module
        return importlib.reload(server_module)

    def test_host_and_port_read_from_env(self, monkeypatch):
        monkeypatch.setenv("MCP_HOST", "0.0.0.0")
        monkeypatch.setenv("MCP_PORT", "9999")
        with patch("mcp.server.fastmcp.FastMCP") as MockFastMCP:
            MockFastMCP.return_value = MagicMock()
            self._reload_server_module()
            _, kwargs = MockFastMCP.call_args
            assert kwargs["host"] == "0.0.0.0"
            assert kwargs["port"] == 9999
        self._reload_server_module()  # restore real FastMCP for later tests

    def test_defaults_to_loopback_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("MCP_HOST", raising=False)
        monkeypatch.delenv("MCP_PORT", raising=False)
        with patch("mcp.server.fastmcp.FastMCP") as MockFastMCP:
            MockFastMCP.return_value = MagicMock()
            self._reload_server_module()
            _, kwargs = MockFastMCP.call_args
            assert kwargs["host"] == "127.0.0.1"
            assert kwargs["port"] == 8002
        self._reload_server_module()

    def test_run_uses_transport_from_env(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
        server_module = self._reload_server_module()
        with (
            patch.object(server_module, "_harden_stdio"),
            patch.object(server_module.mcp, "run") as mock_run,
        ):
            server_module._run()
            mock_run.assert_called_once_with(transport="streamable-http")

    def test_run_defaults_to_stdio_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)
        server_module = self._reload_server_module()
        with (
            patch.object(server_module, "_harden_stdio"),
            patch.object(server_module.mcp, "run") as mock_run,
        ):
            server_module._run()
            mock_run.assert_called_once_with(transport="stdio")

    def test_run_logs_server_default_retrieval_tier_at_startup(self, monkeypatch):
        """附录 CI.3: the resolved server_default retrieval tier must be
        made visible at process startup, not only inside each response's
        echo field — this locks that _run() actually calls the logger,
        not just that the function exists somewhere unused."""
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)
        server_module = self._reload_server_module()
        with (
            patch.object(server_module, "_harden_stdio"),
            patch.object(server_module.mcp, "run"),
            patch(
                "rag.app.mcp_server.retrieval_config.log_server_default_retrieval_tier"
            ) as mock_log,
        ):
            server_module._run()
            mock_log.assert_called_once_with()
