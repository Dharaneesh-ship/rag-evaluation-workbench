"""Ingest coverage audit — reconciles files on disk against the DocStore.

Independent safety net for silent ingest gaps: instead of trusting the
ingest run's own reporting, this walks the source directory and checks,
for every file, that the DocStore actually contains a current version
with retrievable content.

Buckets:
    - ``ok``: document present with chunks and a plausible amount of text.
    - ``missing``: no document for this file's current content — it was
      never ingested, or it changed on disk after the last ingest.
    - ``empty``: document exists but has zero chunks (invisible to retrieval).
    - ``suspicious``: chunks exist but the extracted text is tiny relative
      to the file size (e.g. a 8 MB PDF that yielded 200 characters —
      likely a scanned document parsed without OCR).

Usage::

    doc_store = SQLiteDocStore("data/default.db")
    report = verify_directory(doc_store, "./docs")
    print(report.format_summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from rag.core.interfaces.doc_store import BaseDocStore
from rag.core.utils.hashing import fingerprint_bytes, make_doc_id
from rag.pipelines.batch_ingest import discover_files

logger = logging.getLogger(__name__)

# Files larger than this can reasonably be expected to contain real text.
_SUSPICION_MIN_FILE_BYTES = 50 * 1024
# Below this many extracted characters per KB of file, flag as suspicious.
_DEFAULT_MIN_CHARS_PER_KB = 5.0


@dataclass
class FileAudit:
    """Audit outcome for a single file.

    Attributes:
        source_path: Absolute path of the file on disk.
        status: One of ``ok`` / ``missing`` / ``empty`` / ``suspicious``.
        chunk_count: Number of chunks stored for the current file content.
        extracted_chars: Total characters across the stored chunks.
        file_size_bytes: Size of the file on disk.
        detail: Human-readable explanation for non-ok statuses.
    """

    source_path: str
    status: str
    chunk_count: int = 0
    extracted_chars: int = 0
    file_size_bytes: int = 0
    detail: str = ""


@dataclass
class VerifyReport:
    """Aggregated audit of a directory against the DocStore."""

    root: str
    audits: list[FileAudit] = field(default_factory=list)

    def by_status(self, status: str) -> list[FileAudit]:
        """Return all audits with the given status."""
        return [a for a in self.audits if a.status == status]

    @property
    def total(self) -> int:
        """Total number of files audited."""
        return len(self.audits)

    @property
    def problem_audits(self) -> list[FileAudit]:
        """All audits that are not ``ok``."""
        return [a for a in self.audits if a.status != "ok"]

    @property
    def has_problems(self) -> bool:
        """True when at least one file is missing, empty, or suspicious."""
        return bool(self.problem_audits)

    def format_summary(self) -> str:
        """Render a human-readable multi-line audit summary."""
        ok = len(self.by_status("ok"))
        lines = [
            f"Verify '{self.root}': {self.total} file(s) checked",
            f"  ok: {ok}",
        ]

        sections = [
            ("missing", "missing from DocStore (never ingested, or changed since)"),
            ("empty", "in DocStore but 0 chunks (invisible to retrieval)"),
            ("suspicious", "suspiciously little text extracted"),
        ]
        for status, label in sections:
            items = self.by_status(status)
            if not items:
                continue
            lines.append(f"  {label}: {len(items)}")
            for a in items:
                detail = f" — {a.detail}" if a.detail else ""
                lines.append(f"    ✗ {a.source_path}{detail}")

        if not self.has_problems:
            lines.append("  coverage complete ✓")
        else:
            lines.append(
                f"  WARNING: {len(self.problem_audits)} file(s) need attention."
            )
        return "\n".join(lines)


def _audit_file(
    doc_store: BaseDocStore,
    path: Path,
    min_chars_per_kb: float,
) -> FileAudit:
    """Audit one file against the DocStore.

    Args:
        doc_store: DocStore to reconcile against.
        path: Resolved path of the file on disk.
        min_chars_per_kb: Suspicion threshold for extracted-text density.

    Returns:
        Populated FileAudit.
    """
    source_path = str(path)
    raw = path.read_bytes()
    file_size = len(raw)
    doc_id = make_doc_id(source_path, fingerprint_bytes(raw))

    if not doc_store.document_exists(doc_id):
        return FileAudit(
            source_path=source_path,
            status="missing",
            file_size_bytes=file_size,
            detail="no document for current file content",
        )

    chunks = doc_store.get_chunks(doc_id)
    extracted_chars = sum(len(c.stable_text or "") for c in chunks)

    if not chunks:
        return FileAudit(
            source_path=source_path,
            status="empty",
            file_size_bytes=file_size,
            detail="document stored but produced no chunks",
        )

    if file_size >= _SUSPICION_MIN_FILE_BYTES:
        chars_per_kb = extracted_chars / (file_size / 1024)
        if chars_per_kb < min_chars_per_kb:
            return FileAudit(
                source_path=source_path,
                status="suspicious",
                chunk_count=len(chunks),
                extracted_chars=extracted_chars,
                file_size_bytes=file_size,
                detail=(
                    f"only {extracted_chars} chars from "
                    f"{file_size / 1024:.0f} KB "
                    f"({chars_per_kb:.1f} chars/KB < {min_chars_per_kb}) — "
                    "possibly a scanned document parsed without OCR"
                ),
            )

    return FileAudit(
        source_path=source_path,
        status="ok",
        chunk_count=len(chunks),
        extracted_chars=extracted_chars,
        file_size_bytes=file_size,
    )


def verify_directory(
    doc_store: BaseDocStore,
    root: str | Path,
    recursive: bool = True,
    min_chars_per_kb: float = _DEFAULT_MIN_CHARS_PER_KB,
) -> VerifyReport:
    """Reconcile every file under ``root`` against the DocStore.

    Args:
        doc_store: DocStore to audit.
        root: Directory whose files should all be ingested.
        recursive: Descend into subdirectories. Defaults to True.
        min_chars_per_kb: Text-density threshold below which a large file
            is flagged as suspicious. Defaults to 5 chars/KB.

    Returns:
        VerifyReport with one FileAudit per discovered file.
    """
    root = Path(root).resolve()
    report = VerifyReport(root=str(root))
    for path in discover_files(root, recursive=recursive):
        try:
            report.audits.append(_audit_file(doc_store, path, min_chars_per_kb))
        except Exception as exc:
            logger.warning("Audit failed for '%s': %s", path, exc)
            report.audits.append(
                FileAudit(
                    source_path=str(path),
                    status="missing",
                    detail=f"audit error: {exc}",
                )
            )
    return report
