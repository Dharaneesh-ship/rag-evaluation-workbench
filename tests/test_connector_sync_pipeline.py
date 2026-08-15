"""Tests for ConnectorSyncPipeline and rag.sync_source MCP tool — Task 15.6.

Uses the FakeConnector from the contract tests and a lightweight
IngestPipeline double so no real files, indexes, or LLMs are needed.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.core.contracts.source_artifact import SourceArtifact
from rag.core.interfaces.source_connector import BaseSourceConnector
from rag.pipelines.connector_sync_pipeline import (
    ConnectorSyncPipeline,
    SyncResult,
    _mime_to_suffix,
)


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class FakeConnector(BaseSourceConnector):
    """Minimal connector stub for sync pipeline tests."""

    connector_name = "fake"

    def __init__(self, artifacts=None, cursor="cursor-v2"):
        self._artifacts = artifacts or []
        self._cursor = cursor

    def list_items(self, since_cursor=""):
        return list(self._artifacts)

    def next_cursor(self):
        return self._cursor

    def healthcheck(self):
        return {"status": "ok", "connector": self.connector_name, "detail": ""}


def _artifact(
    source_id="art-1",
    text="Hello world",
    mime="text/plain",
    cursor_after="cursor-v2",
):
    return SourceArtifact(
        source_type="fake",
        source_id=source_id,
        content_text=text,
        mime_type=mime,
        cursor_after=cursor_after,
    )


def _binary_artifact(source_id="bin-1", data=b"PDF bytes"):
    return SourceArtifact(
        source_type="fake",
        source_id=source_id,
        content_bytes=data,
        mime_type="application/pdf",
        cursor_after="cursor-v2",
    )


class FakeIngestResult:
    def __init__(self, doc_id="doc-1", chunk_count=3, error=None, skipped=False):
        self.doc_id = doc_id
        self.chunk_count = chunk_count
        self.error = error
        self.skipped = skipped
        self.run_id = str(uuid.uuid4())


class FakeIngestPipeline:
    def __init__(self, results=None, raise_on=None):
        self._results = iter(results or [])
        self._raise_on = raise_on or set()
        self.ingested_paths = []

    def ingest(self, path):
        self.ingested_paths.append(str(path))
        if path in self._raise_on:
            raise RuntimeError("ingest error")
        try:
            return next(self._results)
        except StopIteration:
            return FakeIngestResult()


class FakeDocStore:
    def __init__(self):
        self._cursors = {}

    def load_connector_cursor(self, name):
        return self._cursors.get(name, "")

    def save_connector_cursor(self, name, cursor):
        self._cursors[name] = cursor


class FakeTraceStore:
    def __init__(self):
        self.runs = []

    def save_run(self, run_type, metadata):
        self.runs.append({"run_type": run_type, "metadata": metadata})


def _make_pipeline(artifacts=None, cursor="cur-2", ingest_results=None):
    connector = FakeConnector(artifacts=artifacts, cursor=cursor)
    ingest = FakeIngestPipeline(results=ingest_results or [FakeIngestResult()] * (len(artifacts or [])))
    doc_store = FakeDocStore()
    trace_store = FakeTraceStore()
    pipeline = ConnectorSyncPipeline(
        connector=connector,
        ingest_pipeline=ingest,
        doc_store=doc_store,
        trace_store=trace_store,
    )
    return pipeline, ingest, doc_store, trace_store


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------


class TestMimeToSuffix:
    def test_plain_text(self):
        assert _mime_to_suffix("text/plain") == ".txt"

    def test_html(self):
        assert _mime_to_suffix("text/html") == ".html"

    def test_pdf(self):
        assert _mime_to_suffix("application/pdf") == ".pdf"

    def test_unknown_defaults_to_txt(self):
        assert _mime_to_suffix("application/octet-stream") == ".txt"


# ---------------------------------------------------------------------------
# ConnectorSyncPipeline
# ---------------------------------------------------------------------------


class TestConnectorSyncPipeline:

    # ------ basic run -------------------------------------------------------

    def test_run_returns_sync_result(self):
        pipeline, *_ = _make_pipeline(artifacts=[_artifact()])
        result = pipeline.run()
        assert isinstance(result, SyncResult)

    def test_fetched_count_matches_artifacts(self):
        arts = [_artifact("a1"), _artifact("a2"), _artifact("a3")]
        pipeline, *_ = _make_pipeline(artifacts=arts)
        result = pipeline.run()
        assert result.fetched == 3

    def test_ingested_count(self):
        arts = [_artifact("a1"), _artifact("a2")]
        pipeline, *_ = _make_pipeline(artifacts=arts)
        result = pipeline.run()
        assert result.ingested == 2
        assert result.skipped == 0
        assert result.failed == 0

    def test_empty_connector_returns_zero_counts(self):
        pipeline, *_ = _make_pipeline(artifacts=[])
        result = pipeline.run()
        assert result.fetched == 0
        assert result.ingested == 0

    # ------ cursor handling -------------------------------------------------

    def test_cursor_loaded_from_doc_store(self):
        pipeline, ingest, doc_store, trace_store = _make_pipeline(artifacts=[])
        doc_store._cursors["fake"] = "prev-cursor"
        pipeline.run()
        # Connector was called with the stored cursor (verified via trace)
        start_trace = trace_store.runs[0]
        assert start_trace["metadata"]["cursor_before"] == "prev-cursor"

    def test_cursor_persisted_after_run(self):
        pipeline, ingest, doc_store, trace_store = _make_pipeline(
            artifacts=[_artifact()], cursor="new-cursor"
        )
        pipeline.run()
        assert doc_store.load_connector_cursor("fake") == "new-cursor"

    def test_since_cursor_override(self):
        pipeline, ingest, doc_store, trace_store = _make_pipeline(artifacts=[])
        doc_store._cursors["fake"] = "old"
        pipeline.run(since_cursor="override-cursor")
        start_trace = trace_store.runs[0]
        assert start_trace["metadata"]["cursor_before"] == "override-cursor"

    def test_cursor_unchanged_when_connector_returns_empty(self):
        pipeline, ingest, doc_store, trace_store = _make_pipeline(
            artifacts=[], cursor=""
        )
        doc_store._cursors["fake"] = "stable"
        pipeline.run()
        # Empty next_cursor → keep old cursor
        assert doc_store.load_connector_cursor("fake") == "stable"

    # ------ ingest errors ---------------------------------------------------

    def test_failed_ingest_counted(self):
        arts = [_artifact("a1"), _artifact("a2")]
        results = [FakeIngestResult(error="parse failed"), FakeIngestResult()]
        pipeline, *_ = _make_pipeline(artifacts=arts, ingest_results=results)
        result = pipeline.run()
        assert result.failed == 1
        assert result.ingested == 1

    def test_ingest_exception_counted_as_failed(self):
        arts = [_artifact("a1")]

        class RaisingIngest:
            def ingest(self, path):
                raise RuntimeError("disk full")

        connector = FakeConnector(artifacts=arts)
        pipeline = ConnectorSyncPipeline(
            connector=connector,
            ingest_pipeline=RaisingIngest(),
            doc_store=FakeDocStore(),
            trace_store=FakeTraceStore(),
        )
        result = pipeline.run()
        assert result.failed == 1
        assert result.ingested == 0

    # ------ skipped artifacts -----------------------------------------------

    def test_artifact_with_no_content_is_skipped(self):
        empty = SourceArtifact(source_type="fake", source_id="empty")
        pipeline, *_ = _make_pipeline(artifacts=[empty])
        result = pipeline.run()
        assert result.skipped == 1
        assert result.ingested == 0

    # ------ temp file handling ----------------------------------------------

    def test_text_artifact_written_as_txt(self, tmp_path):
        art = _artifact(text="Some text content")
        pipeline, ingest, *_ = _make_pipeline(artifacts=[art])
        pipeline._tmp_dir = str(tmp_path)
        pipeline.run()
        # Ingest was called with a .txt path
        assert ingest.ingested_paths[0].endswith(".txt")

    def test_binary_artifact_written_as_pdf(self, tmp_path):
        art = _binary_artifact(data=b"%PDF-fake")
        pipeline, ingest, *_ = _make_pipeline(artifacts=[art])
        pipeline._tmp_dir = str(tmp_path)
        pipeline.run()
        assert ingest.ingested_paths[0].endswith(".pdf")

    def test_temp_file_deleted_after_ingest(self, tmp_path):
        art = _artifact(text="temporary")
        pipeline, ingest, *_ = _make_pipeline(artifacts=[art])
        pipeline._tmp_dir = str(tmp_path)
        pipeline.run()
        # No leftover files in tmp_path
        remaining = list(tmp_path.iterdir())
        assert remaining == []

    # ------ trace events ----------------------------------------------------

    def test_two_trace_events_written(self):
        pipeline, ingest, doc_store, trace_store = _make_pipeline(artifacts=[_artifact()])
        pipeline.run()
        run_types = [r["run_type"] for r in trace_store.runs]
        assert "connector_sync_start" in run_types
        assert "connector_sync" in run_types

    def test_sync_trace_contains_counts(self):
        arts = [_artifact("a1"), _artifact("a2")]
        pipeline, *rest = _make_pipeline(artifacts=arts)
        result = pipeline.run()
        trace_store = rest[2]
        sync_trace = next(r for r in trace_store.runs if r["run_type"] == "connector_sync")
        md = sync_trace["metadata"]
        assert md["fetched"] == 2
        assert md["ingested"] == 2

    def test_run_id_in_result(self):
        pipeline, *_ = _make_pipeline()
        result = pipeline.run()
        assert result.run_id != ""

    def test_connector_name_in_result(self):
        pipeline, *_ = _make_pipeline()
        result = pipeline.run()
        assert result.connector_name == "fake"

    def test_elapsed_ms_positive(self):
        pipeline, *_ = _make_pipeline(artifacts=[_artifact()])
        result = pipeline.run()
        assert result.elapsed_ms >= 0.0

    # ------ top-level connector error ---------------------------------------

    def test_connector_error_captured_in_result(self):
        class BrokenConnector(BaseSourceConnector):
            connector_name = "broken"

            def list_items(self, since_cursor=""):
                raise RuntimeError("connection refused")

            def next_cursor(self):
                return ""

            def healthcheck(self):
                return {"status": "error", "connector": "broken", "detail": ""}

        pipeline = ConnectorSyncPipeline(
            connector=BrokenConnector(),
            ingest_pipeline=FakeIngestPipeline(),
            doc_store=FakeDocStore(),
            trace_store=FakeTraceStore(),
        )
        result = pipeline.run()
        assert result.error is not None
        assert "connection refused" in result.error


# ---------------------------------------------------------------------------
# SyncSourceToolInput / SyncSourceToolOutput schema tests
# ---------------------------------------------------------------------------


class TestSyncSourceSchemas:
    def test_valid_connector_names(self):
        from rag.app.mcp_server.schemas import SyncSourceToolInput

        for name in ("email", "slack", "notion", "google_docs"):
            inp = SyncSourceToolInput(connector=name)
            assert inp.connector == name

    def test_invalid_connector_raises(self):
        from pydantic import ValidationError

        from rag.app.mcp_server.schemas import SyncSourceToolInput

        with pytest.raises(ValidationError):
            SyncSourceToolInput(connector="twitter")

    def test_defaults(self):
        from rag.app.mcp_server.schemas import SyncSourceToolInput

        inp = SyncSourceToolInput(connector="slack")
        assert inp.db_path == "data/rag.db"
        assert inp.index_dir == "data/indexes"
        assert inp.since_cursor is None
        # 定案 #7: unset embedding_provider now normalizes to "server_default"
        # (deferring to the service's configured default) rather than None.
        assert inp.embedding_provider == "server_default"

    def test_output_model(self):
        from rag.app.mcp_server.schemas import SyncSourceToolOutput

        out = SyncSourceToolOutput(
            connector="notion",
            fetched=5,
            ingested=4,
            skipped=1,
            failed=0,
            cursor_before="2024-01-01T00:00:00.000Z",
            cursor_after="2024-06-01T00:00:00.000Z",
            elapsed_ms=123.4,
            run_id="run-xyz",
        )
        assert out.connector == "notion"
        assert out.fetched == 5
        assert out.error is None
