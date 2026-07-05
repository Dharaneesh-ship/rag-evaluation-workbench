"""Batch ingestion runner — ingests a directory tree and reports per-file outcomes.

The core guarantee: no file disappears silently. Every file discovered under
the input directory ends up in exactly one bucket of the ``BatchIngestReport``
(ok / skipped / unsupported / quality_failed / empty_content / failed), and
the aggregate summary is persisted to the TraceStore.

Usage::

    pipeline = IngestPipeline(doc_store, trace_store)
    runner = BatchIngestRunner(pipeline, trace_store)
    report = runner.run("/path/to/docs")
    print(report.format_summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rag.core.interfaces.trace_store import BaseTraceStore
from rag.pipelines.ingest_pipeline import IngestPipeline, IngestResult

logger = logging.getLogger(__name__)

# Statuses that mean the file's content is NOT retrievable afterwards.
PROBLEM_STATUSES = ("failed", "unsupported", "empty_content")


@dataclass
class BatchIngestReport:
    """Aggregated outcome of a batch ingest run.

    Attributes:
        root: Directory that was ingested.
        results: One IngestResult per discovered file, in ingest order.
    """

    root: str
    results: list[IngestResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of files processed."""
        return len(self.results)

    def by_status(self, status: str) -> list[IngestResult]:
        """Return all results with the given status."""
        return [r for r in self.results if r.status == status]

    @property
    def status_counts(self) -> dict[str, int]:
        """Map of status → number of files with that status."""
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    @property
    def problem_results(self) -> list[IngestResult]:
        """Results whose content is not retrievable (failed/unsupported/empty)."""
        return [r for r in self.results if r.status in PROBLEM_STATUSES]

    @property
    def has_problems(self) -> bool:
        """True when at least one file is not retrievable after the run."""
        return bool(self.problem_results)

    def to_metadata(self) -> dict:
        """Serialise the report summary for TraceStore persistence."""
        return {
            "root": self.root,
            "total": self.total,
            "status_counts": self.status_counts,
            "problem_files": [
                {"source_path": r.source_path, "status": r.status, "error": r.error}
                for r in self.problem_results
            ],
        }

    def format_summary(self) -> str:
        """Render a human-readable multi-line summary of the run."""
        counts = self.status_counts
        ok = counts.get("ok", 0)
        skipped = counts.get("skipped", 0)
        lines = [
            f"Batch ingest of '{self.root}': {self.total} file(s)",
            f"  ok: {ok}   skipped (unchanged): {skipped}",
        ]

        def _section(status: str, label: str) -> None:
            items = self.by_status(status)
            if not items:
                return
            lines.append(f"  {label}: {len(items)}")
            for r in items:
                detail = f" — {r.error}" if r.error else ""
                lines.append(f"    ✗ {r.source_path}{detail}")

        _section("quality_failed", "quality gate failed (ingested, degraded)")
        _section("empty_content", "empty content (NOT retrievable)")
        _section("unsupported", "unsupported format (NOT ingested)")
        _section("failed", "failed (NOT ingested)")

        if not self.has_problems:
            lines.append("  all files retrievable ✓")
        else:
            lines.append(
                f"  WARNING: {len(self.problem_results)} file(s) are NOT "
                "retrievable — see list above."
            )
        return "\n".join(lines)


def discover_files(root: str | Path, recursive: bool = True) -> list[Path]:
    """List candidate files under a directory, skipping hidden entries.

    Hidden files/directories (dot-prefixed) are excluded so artefacts like
    ``.DS_Store`` or ``.git`` don't show up as unsupported-format noise.

    Args:
        root: Directory to scan.
        recursive: Descend into subdirectories. Defaults to True.

    Returns:
        Sorted list of file paths.
    """
    root = Path(root)
    pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in sorted(root.glob(pattern))
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    ]
    return files


class BatchIngestRunner:
    """Ingest every file under a directory and aggregate the outcomes.

    Args:
        pipeline: Configured IngestPipeline used for each file.
        trace_store: TraceStore that receives the batch summary record.
    """

    def __init__(
        self,
        pipeline: IngestPipeline,
        trace_store: Optional[BaseTraceStore] = None,
    ) -> None:
        self._pipeline = pipeline
        self._trace_store = trace_store

    def run(
        self,
        root: str | Path,
        recursive: bool = True,
        progress: bool = False,
        workers: int = 1,
    ) -> BatchIngestReport:
        """Ingest all files under ``root`` and return the aggregated report.

        Args:
            root: Directory containing the files to ingest.
            recursive: Descend into subdirectories. Defaults to True.
            progress: Print one line per file while running.
            workers: Concurrent prepare-phase threads (parse/clean/chunk/
                embed). ``1`` ingests sequentially. DocStore writes and
                index updates always run one file at a time regardless of
                this setting — see ``IngestPipeline.ingest_many()``.

        Returns:
            BatchIngestReport with one entry per discovered file.
        """
        root = Path(root).resolve()
        files = discover_files(root, recursive=recursive)
        report = BatchIngestReport(root=str(root))

        if progress:
            print(f"Ingesting {len(files)} file(s) (workers={workers})...", flush=True)

        # Batch mode defers index rebuild/disk-save from "every document" to
        # "once at the end" — see IngestPipeline.begin_batch() docstring for
        # why the naive per-document version is O(N^2) for N documents.
        self._pipeline.begin_batch()
        try:
            report.results = self._pipeline.ingest_many(files, max_workers=workers)
        finally:
            self._pipeline.end_batch()

        if progress:
            for i, (path, result) in enumerate(zip(files, report.results), start=1):
                print(f"[{i}/{len(files)}] {path.name}: {result.status}", flush=True)

        if self._trace_store is not None:
            self._trace_store.save_run(
                run_type="batch_ingest_complete",
                metadata=report.to_metadata(),
            )

        logger.info(
            "Batch ingest of '%s' finished: %s", root, report.status_counts
        )
        return report
