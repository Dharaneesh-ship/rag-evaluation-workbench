"""Tests for ingest hardening — three-tier PDF fallback and silent-failure fixes.

Covers:
- PdfVisionParser (mocked OpenAI client)
- ParserOrchestrator escalation: pymupdf → pdf_ocr → pdf_vision
- Explicit IngestResult.status (ok/skipped/unsupported/empty_content/failed)
- BatchIngestRunner report aggregation
- verify_audit directory reconciliation
- CLI directory ingest and verify exit codes
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag.core.contracts.document import Document
from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.core.contracts.parse_report import ParseReport
from rag.infra.loading.local_file_loader import RawArtifact
from rag.pipelines.parsing.plans import ParsePlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(parser_used: str, char_count: int, metadata: dict | None = None) -> Document:
    """Build a minimal Document with a given parser id and char count."""
    return Document(
        doc_id="d1",
        source_path="/tmp/f.pdf",
        mime_type="application/pdf",
        blocks=[IRBlock(text="x" * char_count, block_type=BlockType.PARAGRAPH)],
        parse_report=ParseReport(
            char_count=char_count,
            block_count=1,
            non_printable_ratio=0.0,
            repetition_score=0.0,
            parser_used=parser_used,
        ),
        metadata=metadata or {},
    )


def _three_tier_config(tmp_path, ocr_enabled=True, vision_enabled=True,
                       min_confidence=0.6):
    cfg = tmp_path / "parser_candidates.yaml"
    cfg.write_text(f"""
ocr:
  enabled: {str(ocr_enabled).lower()}
  min_chars_threshold: 100
vision:
  enabled: {str(vision_enabled).lower()}
  min_ocr_confidence: {min_confidence}
routes:
  pdf:
    candidates: [pymupdf, pdf_ocr, pdf_vision]
""")
    return cfg


def _pdf_plan() -> ParsePlan:
    return ParsePlan(
        detected_type="pdf",
        mime_type="application/pdf",
        candidates=["pymupdf", "pdf_ocr", "pdf_vision"],
        unsupported=False,
    )


def _artifact() -> RawArtifact:
    return RawArtifact(source_path="/tmp/f.pdf", raw_bytes=b"%PDF", metadata={})


# ---------------------------------------------------------------------------
# PdfVisionParser
# ---------------------------------------------------------------------------


class TestMarkdownToBlocks:
    def test_ordinal_detection(self):
        from rag.infra.parsing.pdf_vision_parser import _starts_with_ordinal

        assert _starts_with_ordinal("1. first item")
        assert _starts_with_ordinal("12) twelfth item")
        assert not _starts_with_ordinal("Plain sentence.")
        assert not _starts_with_ordinal("2024 was a good year")

    def test_classifies_block_types(self):
        from rag.infra.parsing.pdf_vision_parser import _markdown_to_blocks

        markdown = (
            "# Heading\n\n"
            "A normal paragraph of text.\n\n"
            "| col1 | col2 |\n|------|------|\n| a | b |\n\n"
            "- list item one\n\n"
            "Figure: a bar chart showing revenue by quarter."
        )
        blocks = _markdown_to_blocks(markdown, page_num=3)
        types = [b.block_type for b in blocks]
        assert types == [
            BlockType.HEADING,
            BlockType.PARAGRAPH,
            BlockType.TABLE,
            BlockType.LIST_ITEM,
            BlockType.IMAGE_CAPTION,
        ]
        assert all(b.page == 3 for b in blocks)

    def test_blank_page_yields_no_blocks(self):
        from rag.infra.parsing.pdf_vision_parser import _markdown_to_blocks

        assert _markdown_to_blocks("[BLANK_PAGE]", page_num=1) == []
        assert _markdown_to_blocks("   \n\n  ", page_num=1) == []


class TestPdfVisionParser:
    def _make_parser(self, page_markdowns: list[str]):
        from rag.infra.parsing.pdf_vision_parser import PdfVisionParser

        mock_renderer = MagicMock()
        mock_renderer.page_count.return_value = len(page_markdowns)
        mock_image = MagicMock()
        mock_image.save = MagicMock()
        mock_renderer.render.return_value = mock_image

        parser = PdfVisionParser(renderer=mock_renderer, api_key="test-key")

        responses = []
        for md in page_markdowns:
            resp = MagicMock()
            resp.choices[0].message.content = md
            responses.append(resp)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = responses
        parser._client = mock_client
        return parser, mock_renderer, mock_client

    def test_supports_pdf_mime(self):
        parser, _, _ = self._make_parser(["text"])
        assert parser.supports("application/pdf")
        assert not parser.supports("text/html")

    def test_parse_returns_document_with_vision_report(self, tmp_path):
        parser, _, _ = self._make_parser(["# Title\n\nSome page content."])
        pdf = tmp_path / "complex.pdf"
        pdf.write_bytes(b"%PDF")
        doc = parser.parse(str(pdf))
        assert doc.parse_report.parser_used == "pdf_vision"
        assert doc.parse_report.char_count > 0
        assert doc.metadata["page_count"] == 1

    def test_parse_stamps_page_numbers(self, tmp_path):
        parser, _, _ = self._make_parser(["Page one text.", "Page two text."])
        pdf = tmp_path / "complex.pdf"
        pdf.write_bytes(b"%PDF")
        doc = parser.parse(str(pdf))
        assert {b.page for b in doc.blocks} == {1, 2}

    def test_blank_pages_produce_no_blocks(self, tmp_path):
        parser, _, _ = self._make_parser(["[BLANK_PAGE]"])
        pdf = tmp_path / "blank.pdf"
        pdf.write_bytes(b"%PDF")
        doc = parser.parse(str(pdf))
        assert doc.blocks == []
        assert doc.parse_report.char_count == 0

    def test_max_pages_caps_api_calls(self, tmp_path):
        from rag.infra.parsing.pdf_vision_parser import PdfVisionParser

        mock_renderer = MagicMock()
        mock_renderer.page_count.return_value = 10
        mock_renderer.render.return_value = MagicMock()
        parser = PdfVisionParser(
            renderer=mock_renderer, api_key="test-key", max_pages=2
        )
        resp = MagicMock()
        resp.choices[0].message.content = "text"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        parser._client = mock_client

        pdf = tmp_path / "long.pdf"
        pdf.write_bytes(b"%PDF")
        parser.parse(str(pdf))
        assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# Orchestrator three-tier escalation
# ---------------------------------------------------------------------------


class TestThreeTierEscalation:
    def _orchestrator(self, tmp_path, registry, **cfg_kwargs):
        from rag.pipelines.parsing.orchestrator import ParserOrchestrator

        cfg = _three_tier_config(tmp_path, **cfg_kwargs)
        return ParserOrchestrator(registry, router_config_path=cfg)

    def test_low_ocr_confidence_escalates_to_vision(self, tmp_path):
        mock_pymupdf = MagicMock()
        mock_pymupdf.parse.return_value = _make_doc("pymupdf", 10)
        mock_ocr = MagicMock()
        mock_ocr.parse.return_value = _make_doc(
            "pdf_ocr", 500, {"mean_ocr_confidence": 0.3}
        )
        mock_vision = MagicMock()
        mock_vision.parse.return_value = _make_doc("pdf_vision", 800)

        orch = self._orchestrator(
            tmp_path,
            {"pymupdf": mock_pymupdf, "pdf_ocr": mock_ocr, "pdf_vision": mock_vision},
        )
        doc = orch.parse(_artifact(), _pdf_plan())
        assert doc.parse_report.parser_used == "pdf_vision"
        assert doc.parse_report.fallback_triggered is True

    def test_confident_ocr_result_is_accepted(self, tmp_path):
        mock_pymupdf = MagicMock()
        mock_pymupdf.parse.return_value = _make_doc("pymupdf", 10)
        mock_ocr = MagicMock()
        mock_ocr.parse.return_value = _make_doc(
            "pdf_ocr", 500, {"mean_ocr_confidence": 0.95}
        )
        mock_vision = MagicMock()

        orch = self._orchestrator(
            tmp_path,
            {"pymupdf": mock_pymupdf, "pdf_ocr": mock_ocr, "pdf_vision": mock_vision},
        )
        doc = orch.parse(_artifact(), _pdf_plan())
        assert doc.parse_report.parser_used == "pdf_ocr"
        mock_vision.parse.assert_not_called()

    def test_vision_disabled_keeps_low_confidence_ocr(self, tmp_path):
        mock_pymupdf = MagicMock()
        mock_pymupdf.parse.return_value = _make_doc("pymupdf", 10)
        mock_ocr = MagicMock()
        mock_ocr.parse.return_value = _make_doc(
            "pdf_ocr", 500, {"mean_ocr_confidence": 0.3}
        )
        mock_vision = MagicMock()

        orch = self._orchestrator(
            tmp_path,
            {"pymupdf": mock_pymupdf, "pdf_ocr": mock_ocr, "pdf_vision": mock_vision},
            vision_enabled=False,
        )
        doc = orch.parse(_artifact(), _pdf_plan())
        assert doc.parse_report.parser_used == "pdf_ocr"
        mock_vision.parse.assert_not_called()

    def test_good_text_pdf_never_escalates(self, tmp_path):
        mock_pymupdf = MagicMock()
        mock_pymupdf.parse.return_value = _make_doc("pymupdf", 5000)
        mock_ocr = MagicMock()
        mock_vision = MagicMock()

        orch = self._orchestrator(
            tmp_path,
            {"pymupdf": mock_pymupdf, "pdf_ocr": mock_ocr, "pdf_vision": mock_vision},
        )
        doc = orch.parse(_artifact(), _pdf_plan())
        assert doc.parse_report.parser_used == "pymupdf"
        mock_ocr.parse.assert_not_called()
        mock_vision.parse.assert_not_called()

    def test_no_later_tier_keeps_best_partial_result(self, tmp_path):
        """A low-char result is kept (not discarded as unsupported) when
        no later tier is registered."""
        mock_pymupdf = MagicMock()
        mock_pymupdf.parse.return_value = _make_doc("pymupdf", 10)

        orch = self._orchestrator(tmp_path, {"pymupdf": mock_pymupdf})
        doc = orch.parse(_artifact(), _pdf_plan())
        assert doc.parse_report.parser_used == "pymupdf"
        assert doc.metadata.get("unsupported_format") is None

    def test_escalation_failure_falls_back_to_best_result(self, tmp_path):
        """If vision escalation raises, the OCR result is returned."""
        mock_pymupdf = MagicMock()
        mock_pymupdf.parse.return_value = _make_doc("pymupdf", 10)
        mock_ocr = MagicMock()
        mock_ocr.parse.return_value = _make_doc(
            "pdf_ocr", 500, {"mean_ocr_confidence": 0.3}
        )
        mock_vision = MagicMock()
        mock_vision.parse.side_effect = RuntimeError("API down")

        orch = self._orchestrator(
            tmp_path,
            {"pymupdf": mock_pymupdf, "pdf_ocr": mock_ocr, "pdf_vision": mock_vision},
        )
        doc = orch.parse(_artifact(), _pdf_plan())
        assert doc.parse_report.parser_used == "pdf_ocr"
        assert doc.parse_report.fallback_triggered is True


# ---------------------------------------------------------------------------
# IngestResult.status via the real pipeline
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline(tmp_path):
    from rag.infra.stores.docstore_sqlite import SQLiteDocStore
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    from rag.pipelines.ingest_pipeline import IngestPipeline

    db = tmp_path / "test.db"
    doc_store = SQLiteDocStore(db)
    trace_store = SQLiteTraceStore(db)
    return IngestPipeline(doc_store, trace_store), doc_store, trace_store


class TestIngestStatus:
    def test_ok_status(self, pipeline, tmp_path):
        p, _, _ = pipeline
        f = tmp_path / "good.md"
        f.write_text("# Title\n\nReal content with several meaningful words here.")
        result = p.ingest(f)
        assert result.status == "ok"
        assert result.chunk_count > 0

    def test_skipped_status_on_reingest(self, pipeline, tmp_path):
        p, _, _ = pipeline
        f = tmp_path / "good.md"
        f.write_text("# Title\n\nReal content with several meaningful words here.")
        p.ingest(f)
        result = p.ingest(f)
        assert result.status == "skipped"
        assert result.skipped is True

    def test_unsupported_status_and_not_stored(self, pipeline, tmp_path):
        p, doc_store, _ = pipeline
        f = tmp_path / "weird.xyz"
        f.write_bytes(b"\x00\x01\x02binaryjunk" * 10)
        result = p.ingest(f)
        assert result.status == "unsupported"
        assert result.error is not None
        # The placeholder document must NOT be stored or indexed
        assert not doc_store.document_exists(result.doc_id)
        assert result.chunk_count == 0

    def test_empty_content_status(self, pipeline, tmp_path):
        p, _, _ = pipeline
        f = tmp_path / "empty.md"
        f.write_text("")
        result = p.ingest(f)
        assert result.status == "empty_content"
        assert result.chunk_count == 0

    def test_failed_status_on_missing_file(self, pipeline, tmp_path):
        p, _, _ = pipeline
        result = p.ingest(tmp_path / "does_not_exist.md")
        assert result.status == "failed"
        assert result.error is not None

    def test_status_recorded_in_trace(self, pipeline, tmp_path):
        p, _, trace_store = pipeline
        f = tmp_path / "weird.xyz"
        f.write_bytes(b"\x00\x01\x02binaryjunk" * 10)
        p.ingest(f)
        runs = trace_store.list_runs(run_type="ingest_complete")
        assert any(
            r.get("metadata", {}).get("status") == "unsupported" for r in runs
        )


# ---------------------------------------------------------------------------
# Batch ingest report
# ---------------------------------------------------------------------------


class TestBatchIngest:
    def _make_corpus(self, tmp_path):
        docs = tmp_path / "corpus"
        docs.mkdir()
        body = "This document contains enough real prose to pass every quality gate.\n"
        (docs / "a.md").write_text("# A\n\n" + body * 3)
        (docs / "b.md").write_text("# B\n\n" + body * 3)
        (docs / "broken.xyz").write_bytes(b"\x00\x01binaryjunk" * 10)
        (docs / ".hidden.md").write_text("should be skipped")
        sub = docs / "sub"
        sub.mkdir()
        (sub / "c.md").write_text("# C\n\n" + body * 3)
        return docs

    def test_discover_files_skips_hidden(self, tmp_path):
        from rag.pipelines.batch_ingest import discover_files

        docs = self._make_corpus(tmp_path)
        names = [p.name for p in discover_files(docs)]
        assert ".hidden.md" not in names
        assert "c.md" in names  # recursive
        assert len(names) == 4

    def test_discover_files_non_recursive(self, tmp_path):
        from rag.pipelines.batch_ingest import discover_files

        docs = self._make_corpus(tmp_path)
        names = [p.name for p in discover_files(docs, recursive=False)]
        assert "c.md" not in names

    def test_batch_report_buckets_and_problems(self, pipeline, tmp_path):
        from rag.pipelines.batch_ingest import BatchIngestRunner

        p, _, trace_store = pipeline
        docs = self._make_corpus(tmp_path)
        report = BatchIngestRunner(p, trace_store).run(docs)

        assert report.total == 4
        assert report.status_counts.get("ok") == 3
        assert report.status_counts.get("unsupported") == 1
        assert report.has_problems is True
        assert any("broken.xyz" in r.source_path for r in report.problem_results)

    def test_batch_report_persisted_to_trace(self, pipeline, tmp_path):
        from rag.pipelines.batch_ingest import BatchIngestRunner

        p, _, trace_store = pipeline
        docs = self._make_corpus(tmp_path)
        BatchIngestRunner(p, trace_store).run(docs)
        runs = trace_store.list_runs(run_type="batch_ingest_complete")
        assert len(runs) == 1
        meta = runs[0]["metadata"]
        assert meta["total"] == 4
        assert len(meta["problem_files"]) == 1

    def test_summary_lists_problem_files(self, pipeline, tmp_path):
        from rag.pipelines.batch_ingest import BatchIngestRunner

        p, _, _ = pipeline
        docs = self._make_corpus(tmp_path)
        report = BatchIngestRunner(p).run(docs)
        summary = report.format_summary()
        assert "broken.xyz" in summary
        assert "NOT" in summary

    def test_clean_batch_has_no_problems(self, pipeline, tmp_path):
        from rag.pipelines.batch_ingest import BatchIngestRunner

        p, _, _ = pipeline
        docs = tmp_path / "clean"
        docs.mkdir()
        (docs / "a.md").write_text("# A\n\nContent of document A with words.")
        report = BatchIngestRunner(p).run(docs)
        assert report.has_problems is False
        assert "✓" in report.format_summary()


# ---------------------------------------------------------------------------
# Verify audit
# ---------------------------------------------------------------------------


class TestVerifyAudit:
    def test_ok_and_missing(self, pipeline, tmp_path):
        from rag.pipelines.verify_audit import verify_directory

        p, doc_store, _ = pipeline
        docs = tmp_path / "corpus"
        docs.mkdir()
        ingested = docs / "in.md"
        ingested.write_text("# In\n\nThis file was ingested properly.")
        p.ingest(ingested)
        (docs / "never.md").write_text("# Never\n\nThis file was never ingested.")

        report = verify_directory(doc_store, docs)
        by_name = {a.source_path.split("/")[-1]: a.status for a in report.audits}
        assert by_name["in.md"] == "ok"
        assert by_name["never.md"] == "missing"
        assert report.has_problems is True

    def test_changed_file_reported_missing(self, pipeline, tmp_path):
        from rag.pipelines.verify_audit import verify_directory

        p, doc_store, _ = pipeline
        docs = tmp_path / "corpus"
        docs.mkdir()
        f = docs / "doc.md"
        f.write_text("# V1\n\nOriginal content of this document.")
        p.ingest(f)
        f.write_text("# V2\n\nChanged content after the last ingest run.")

        report = verify_directory(doc_store, docs)
        assert report.audits[0].status == "missing"

    def test_empty_content_not_stored_and_reported_missing(self, pipeline, tmp_path):
        """empty_content docs are removed from the store (so retries work)
        and verify reports the file as missing."""
        from rag.pipelines.verify_audit import verify_directory

        p, doc_store, _ = pipeline
        docs = tmp_path / "corpus"
        docs.mkdir()
        f = docs / "empty.md"
        f.write_text("")
        result = p.ingest(f)
        assert result.status == "empty_content"
        assert not doc_store.document_exists(result.doc_id)
        # Re-ingest retries instead of skipping
        assert p.ingest(f).status == "empty_content"

        report = verify_directory(doc_store, docs)
        assert report.audits[0].status == "missing"
        assert report.has_problems is True

    def test_empty_document_in_store_detected(self, pipeline, tmp_path):
        """A stored document with zero chunks is flagged as 'empty'."""
        from rag.core.utils.hashing import fingerprint_bytes, make_doc_id
        from rag.pipelines.verify_audit import verify_directory

        _, doc_store, _ = pipeline
        docs = tmp_path / "corpus"
        docs.mkdir()
        f = docs / "hollow.md"
        f.write_text("# Hollow\n\nContent that was stored without chunks.")
        doc_id = make_doc_id(str(f.resolve()), fingerprint_bytes(f.read_bytes()))
        doc_store.save_document(
            Document(doc_id=doc_id, source_path=str(f.resolve()), mime_type="text/markdown")
        )

        report = verify_directory(doc_store, docs)
        assert report.audits[0].status == "empty"
        assert report.has_problems is True

    def test_suspicious_low_text_density(self, pipeline, tmp_path):
        from rag.core.utils.hashing import fingerprint_bytes, make_doc_id
        from rag.core.contracts.chunk import Chunk
        from rag.pipelines.verify_audit import verify_directory

        _, doc_store, _ = pipeline
        docs = tmp_path / "corpus"
        docs.mkdir()
        # A 100 KB file whose stored text is only a handful of characters —
        # the signature of a scanned PDF parsed without OCR.
        f = docs / "scan.md"
        f.write_text("# X\n\n" + "y" * 100_000)
        raw = f.read_bytes()
        doc_id = make_doc_id(str(f.resolve()), fingerprint_bytes(raw))
        doc_store.save_document(
            Document(doc_id=doc_id, source_path=str(f.resolve()), mime_type="text/markdown")
        )
        doc_store.save_chunks([
            Chunk(
                chunk_id="c1",
                doc_id=doc_id,
                stable_text="tiny",
                display_text="tiny",
                chunk_signature="sig1",
            )
        ])

        report = verify_directory(doc_store, docs)
        assert report.audits[0].status == "suspicious"
        assert "OCR" in report.audits[0].detail

    def test_clean_directory_passes(self, pipeline, tmp_path):
        from rag.pipelines.verify_audit import verify_directory

        p, doc_store, _ = pipeline
        docs = tmp_path / "corpus"
        docs.mkdir()
        f = docs / "doc.md"
        f.write_text("# Doc\n\nProperly ingested content with words.")
        p.ingest(f)

        report = verify_directory(doc_store, docs)
        assert report.has_problems is False
        assert "✓" in report.format_summary()


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLIBatchAndVerify:
    def test_cli_directory_ingest_exit_codes(self, tmp_path, monkeypatch, capsys):
        from rag.cli.ingest import main as ingest_main

        monkeypatch.chdir(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("# A\n\nContent of document A with words.")

        assert ingest_main(["--path", str(docs), "--collection", "t"]) == 0
        out = capsys.readouterr().out
        assert "Batch ingest" in out

        # Add an unsupported file → exit code 1 and it appears in the report
        (docs / "bad.xyz").write_bytes(b"\x00\x01binaryjunk" * 10)
        assert ingest_main(["--path", str(docs), "--collection", "t"]) == 1
        out = capsys.readouterr().out
        assert "bad.xyz" in out

    def test_cli_verify_exit_codes(self, tmp_path, monkeypatch, capsys):
        from rag.cli.ingest import main as ingest_main
        from rag.cli.verify import main as verify_main

        monkeypatch.chdir(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("# A\n\nContent of document A with words.")
        ingest_main(["--path", str(docs), "--collection", "t"])
        capsys.readouterr()

        assert verify_main(["--path", str(docs), "--collection", "t"]) == 0
        out = capsys.readouterr().out
        assert "coverage complete" in out

        # A file added after ingest must be reported missing → exit 1
        (docs / "new.md").write_text("# New\n\nAdded after the ingest run.")
        assert verify_main(["--path", str(docs), "--collection", "t"]) == 1
        out = capsys.readouterr().out
        assert "new.md" in out

    def test_cli_verify_missing_db(self, tmp_path, monkeypatch, capsys):
        from rag.cli.verify import main as verify_main

        monkeypatch.chdir(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        assert verify_main(["--path", str(docs), "--collection", "nope"]) == 1
