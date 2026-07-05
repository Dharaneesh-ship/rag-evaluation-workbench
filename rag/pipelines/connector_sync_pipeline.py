"""Connector sync pipeline — Task 15.6.

Orchestrates a full sync cycle for a single ``BaseSourceConnector``:

1. Load the last cursor from the DocStore ``connector_state`` table.
2. Call ``connector.list_items(since_cursor)`` to fetch new/updated artifacts.
3. Write each ``SourceArtifact`` to a temporary file and run the existing
   ``IngestPipeline`` on it.
4. After all artifacts are ingested, persist the updated cursor via
   ``DocStore.save_connector_cursor()``.
5. Write a ``connector_sync`` trace event with summary stats
   (fetched / ingested / skipped / failed).

The pipeline is intentionally stateless between ``run()`` calls — cursor
state lives entirely in the DocStore.

Usage::

    from rag.infra.connectors.slack_connector import SlackConnector
    from rag.pipelines.connector_sync_pipeline import ConnectorSyncPipeline

    connector = SlackConnector(token="xoxb-...", channel_ids=["C123"])
    ingest_pipeline, index_mgr = build_ingest_pipeline(...)
    pipeline = ConnectorSyncPipeline(
        connector=connector,
        ingest_pipeline=ingest_pipeline,
        doc_store=SQLiteDocStore("data/rag.db"),
        trace_store=SQLiteTraceStore("data/rag.db"),
    )
    result = pipeline.run()
    index_mgr.save()  # persist the same IndexManager instance that was mutated
    print(result.fetched, result.ingested, result.skipped, result.failed)
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rag.core.contracts.source_artifact import SourceArtifact
from rag.core.interfaces.source_connector import BaseSourceConnector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------


@dataclass
class SyncResult:
    """Summary of a single connector sync run.

    Attributes:
        connector_name: Stable connector identifier.
        fetched: Total artifacts returned by the connector.
        ingested: Artifacts successfully ingested into the pipeline.
        skipped: Artifacts skipped (empty content, already up-to-date, etc.).
        failed: Artifacts that raised an exception during ingest.
        cursor_before: Cursor value at the start of the run.
        cursor_after: Cursor value persisted at the end of the run.
        elapsed_ms: Total wall-clock time for the sync in milliseconds.
        run_id: TraceStore run identifier.
        error: Top-level error message if the entire run failed, else None.
        per_artifact: Per-artifact ingest results (for debugging).
    """

    connector_name: str = ""
    fetched: int = 0
    ingested: int = 0
    skipped: int = 0
    failed: int = 0
    cursor_before: str = ""
    cursor_after: str = ""
    elapsed_ms: float = 0.0
    run_id: str = ""
    error: Optional[str] = None
    per_artifact: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "connector_name": self.connector_name,
            "fetched": self.fetched,
            "ingested": self.ingested,
            "skipped": self.skipped,
            "failed": self.failed,
            "cursor_before": self.cursor_before,
            "cursor_after": self.cursor_after,
            "elapsed_ms": self.elapsed_ms,
            "run_id": self.run_id,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ConnectorSyncPipeline:
    """Unified connector sync pipeline.

    Args:
        connector: A ``BaseSourceConnector`` implementation to sync from.
        ingest_pipeline: A configured ``IngestPipeline`` instance.
        doc_store: ``SQLiteDocStore`` for cursor persistence.
        trace_store: ``BaseTraceStore`` for writing sync trace events.
        tmp_dir: Directory for temporary artifact files.  Defaults to the
            system temp directory.
    """

    def __init__(
        self,
        connector: BaseSourceConnector,
        ingest_pipeline: Any,
        doc_store: Any,
        trace_store: Any,
        tmp_dir: Optional[str] = None,
    ) -> None:
        self._connector = connector
        self._ingest = ingest_pipeline
        self._doc_store = doc_store
        self._trace_store = trace_store
        self._tmp_dir = tmp_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, since_cursor: Optional[str] = None) -> SyncResult:
        """Execute a full sync cycle.

        Args:
            since_cursor: Override cursor for this run.  If ``None`` (default),
                the last persisted cursor is loaded from the DocStore.

        Returns:
            ``SyncResult`` with counts and the new cursor.
        """
        run_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        connector_name = self._connector.connector_name or type(self._connector).__name__

        # Load cursor
        if since_cursor is None:
            cursor_before = self._doc_store.load_connector_cursor(connector_name)
        else:
            cursor_before = since_cursor

        result = SyncResult(
            connector_name=connector_name,
            cursor_before=cursor_before,
            run_id=run_id,
        )

        try:
            self._trace_store.save_run(
                run_type="connector_sync_start",
                metadata={
                    "connector_name": connector_name,
                    "cursor_before": cursor_before,
                    "run_id": run_id,
                },
            )

            # Fetch artifacts from connector
            artifacts = self._connector.list_items(since_cursor=cursor_before)
            result.fetched = len(artifacts)
            logger.info(
                "Connector '%s' returned %d artifact(s)", connector_name, result.fetched
            )

            # Ingest each artifact
            for artifact in artifacts:
                art_result = self._ingest_artifact(artifact, run_id)
                result.per_artifact.append(art_result)
                if art_result.get("status") == "ingested":
                    result.ingested += 1
                elif art_result.get("status") == "skipped":
                    result.skipped += 1
                else:
                    result.failed += 1

            # Persist cursor
            new_cursor = self._connector.next_cursor()
            if new_cursor:
                self._doc_store.save_connector_cursor(connector_name, new_cursor)
            result.cursor_after = new_cursor or cursor_before

        except Exception as exc:
            logger.exception("ConnectorSyncPipeline.run failed: %s", exc)
            result.error = str(exc)

        result.elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Write summary trace
        try:
            self._trace_store.save_run(
                run_type="connector_sync",
                metadata=result.as_dict(),
            )
        except Exception as exc:
            logger.warning("Failed to write sync trace: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ingest_artifact(self, artifact: SourceArtifact, run_id: str) -> dict:
        """Write artifact to a temp file and ingest it.

        Returns a dict with ``source_id``, ``status``, and optional ``error``.
        """
        if not artifact.has_content():
            logger.debug("Skipping artifact %s — no content", artifact.source_id)
            return {
                "source_id": artifact.source_id,
                "status": "skipped",
                "reason": "no_content",
            }

        # Choose file extension from MIME type
        suffix = _mime_to_suffix(artifact.mime_type)

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb" if artifact.content_bytes else "w",
                suffix=suffix,
                dir=self._tmp_dir,
                delete=False,
                encoding=None if artifact.content_bytes else "utf-8",
            ) as tmp:
                if artifact.content_bytes:
                    tmp.write(artifact.content_bytes)
                else:
                    tmp.write(artifact.as_text())
                tmp_path = tmp.name

            ingest_result = self._ingest.ingest(tmp_path)

            # Clean up temp file
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

            if ingest_result.error:
                return {
                    "source_id": artifact.source_id,
                    "status": "failed",
                    "error": ingest_result.error,
                }

            return {
                "source_id": artifact.source_id,
                "status": "ingested",
                "doc_id": ingest_result.doc_id,
                "chunk_count": ingest_result.chunk_count,
                "skipped": ingest_result.skipped,
            }

        except Exception as exc:
            logger.warning("Failed to ingest artifact %s: %s", artifact.source_id, exc)
            return {
                "source_id": artifact.source_id,
                "status": "failed",
                "error": str(exc),
            }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mime_to_suffix(mime_type: str) -> str:
    """Return a file extension for a given MIME type."""
    _MAP = {
        "text/plain": ".txt",
        "text/html": ".html",
        "text/markdown": ".md",
        "application/pdf": ".pdf",
        "application/json": ".json",
    }
    return _MAP.get(mime_type, ".txt")
