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
# Keyword (BM25) index population must not depend on an embedding provider
# ---------------------------------------------------------------------------


class TestKeywordIndexIndependentOfEmbedding:
    """BM25-only ingest (embedding_provider=None) is a supported, documented
    configuration (Studio's "none" option) and must populate the keyword
    index like any other run — it must not silently no-op."""

    def test_bm25_only_ingest_populates_keyword_index(self, tmp_path):
        from rag.infra.indexes.bm25_local import BM25LocalIndex
        from rag.infra.stores.docstore_sqlite import SQLiteDocStore
        from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
        from rag.pipelines.ingest_pipeline import IngestPipeline

        db = tmp_path / "test.db"
        keyword_index = BM25LocalIndex()
        pipeline = IngestPipeline(
            SQLiteDocStore(db),
            SQLiteTraceStore(db),
            keyword_index=keyword_index,
            embedding_provider=None,
        )
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nContent about cats and dogs with enough words.")

        result = pipeline.ingest(f)

        assert result.status == "ok"
        assert len(keyword_index._chunks) == result.chunk_count > 0
        assert keyword_index.search("cats and dogs", top_k=5)

    def test_keyword_index_populated_alongside_vector_index(self, tmp_path):
        # Both indexes must end up populated when an embedding_provider IS
        # supplied — guards against a regression that re-couples the two.
        from rag.infra.indexes.bm25_local import BM25LocalIndex
        from rag.infra.stores.docstore_sqlite import SQLiteDocStore
        from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
        from rag.pipelines.ingest_pipeline import IngestPipeline

        db = tmp_path / "test.db"
        keyword_index = BM25LocalIndex()
        vector_index = MagicMock()
        embedding_provider = MagicMock()
        embedding_provider.embed.return_value = MagicMock(prompt_tokens=10)
        embedding_provider.embed_with_usage.return_value = MagicMock(
            vectors=[[0.1, 0.2]], prompt_tokens=10
        )

        pipeline = IngestPipeline(
            SQLiteDocStore(db),
            SQLiteTraceStore(db),
            keyword_index=keyword_index,
            vector_index=vector_index,
            embedding_provider=embedding_provider,
        )
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nContent about cats and dogs with enough words.")

        result = pipeline.ingest(f)

        assert result.status == "ok"
        assert len(keyword_index._chunks) == result.chunk_count
        vector_index.add.assert_called_once()


# ---------------------------------------------------------------------------
# Batch mode: deferred index rebuild/persistence (O(N^2) -> O(N) fix)
# ---------------------------------------------------------------------------


class TestBatchMode:
    def _make_pipeline(self, tmp_path, index_dir=None):
        from rag.infra.indexes.bm25_local import BM25LocalIndex
        from rag.infra.stores.docstore_sqlite import SQLiteDocStore
        from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
        from rag.pipelines.ingest_pipeline import IngestPipeline

        db = tmp_path / "test.db"
        keyword_index = BM25LocalIndex()
        pipeline = IngestPipeline(
            SQLiteDocStore(db),
            SQLiteTraceStore(db),
            keyword_index=keyword_index,
            embedding_provider=None,
            index_dir=index_dir,
        )
        return pipeline, keyword_index

    def _write_docs(self, tmp_path, n):
        paths = []
        for i in range(n):
            f = tmp_path / f"doc{i}.md"
            f.write_text(
                f"# Title {i}\n\nContent number {i} about cats and dogs, "
                "with enough words to clear the quality gate threshold."
            )
            paths.append(f)
        return paths

    def test_rebuild_deferred_during_batch_and_applied_at_end(self, tmp_path):
        pipeline, keyword_index = self._make_pipeline(tmp_path)
        paths = self._write_docs(tmp_path, 5)

        pipeline.begin_batch()
        for p in paths:
            result = pipeline.ingest(p)
            assert result.status == "ok"
            # Mid-batch: chunks are recorded but the BM25 model isn't
            # rebuilt yet, so search returns nothing until end_batch().
            assert keyword_index._bm25 is None
        pipeline.end_batch()

        assert keyword_index._bm25 is not None
        assert keyword_index.search("cats and dogs", top_k=10)

    def test_rebuild_called_once_per_batch_not_per_document(self, tmp_path):
        pipeline, keyword_index = self._make_pipeline(tmp_path)
        paths = self._write_docs(tmp_path, 5)

        call_count = {"n": 0}
        original_rebuild = keyword_index._rebuild

        def counting_rebuild():
            call_count["n"] += 1
            original_rebuild()

        keyword_index._rebuild = counting_rebuild

        pipeline.begin_batch()
        for p in paths:
            pipeline.ingest(p)
        assert call_count["n"] == 0  # zero rebuilds mid-batch
        pipeline.end_batch()
        assert call_count["n"] == 1  # exactly one rebuild at the end

    def test_index_persisted_once_at_end_batch_not_per_document(self, tmp_path):
        index_dir = tmp_path / "indexes"
        pipeline, keyword_index = self._make_pipeline(tmp_path, index_dir=index_dir)
        paths = self._write_docs(tmp_path, 4)

        save_calls = {"n": 0}
        original_save = keyword_index.save

        def counting_save(path):
            save_calls["n"] += 1
            original_save(path)

        keyword_index.save = counting_save

        pipeline.begin_batch()
        for p in paths:
            pipeline.ingest(p)
        assert save_calls["n"] == 0  # no disk writes mid-batch
        pipeline.end_batch()
        assert save_calls["n"] == 1  # exactly one disk write at the end
        assert (index_dir / "bm25.index").exists()

    def test_single_document_ingest_unaffected_by_batch_mode(self, tmp_path):
        # Outside begin_batch/end_batch, behavior must be identical to before:
        # every ingest() call rebuilds and is immediately searchable.
        pipeline, keyword_index = self._make_pipeline(tmp_path)
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nContent about cats and dogs with enough words.")

        result = pipeline.ingest(f)

        assert result.status == "ok"
        assert keyword_index._bm25 is not None
        assert keyword_index.search("cats and dogs", top_k=5)

    def test_end_batch_without_begin_batch_is_safe_noop(self, tmp_path):
        pipeline, _ = self._make_pipeline(tmp_path)
        pipeline.end_batch()  # must not raise


# ---------------------------------------------------------------------------
# ingest_many: concurrent prepare, serial commit
# ---------------------------------------------------------------------------


class TestIngestMany:
    def _make_pipeline(self, tmp_path):
        from rag.infra.indexes.bm25_local import BM25LocalIndex
        from rag.infra.stores.docstore_sqlite import SQLiteDocStore
        from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
        from rag.pipelines.ingest_pipeline import IngestPipeline

        db = tmp_path / "test.db"
        keyword_index = BM25LocalIndex()
        pipeline = IngestPipeline(
            SQLiteDocStore(db),
            SQLiteTraceStore(db),
            keyword_index=keyword_index,
            embedding_provider=None,
        )
        return pipeline, keyword_index

    def _write_docs(self, tmp_path, n):
        paths = []
        for i in range(n):
            f = tmp_path / f"doc{i}.md"
            f.write_text(
                f"# Title {i}\n\nContent number {i} about cats and dogs, "
                "with enough words to clear the quality gate threshold."
            )
            paths.append(f)
        return paths

    def test_sequential_max_workers_1_matches_ingest_loop(self, tmp_path):
        pipeline, keyword_index = self._make_pipeline(tmp_path)
        paths = self._write_docs(tmp_path, 5)

        results = pipeline.ingest_many(paths, max_workers=1)

        assert [r.status for r in results] == ["ok"] * 5
        assert len(keyword_index._chunks) == sum(r.chunk_count for r in results)
        assert keyword_index.search("cats and dogs", top_k=10)

    def test_concurrent_prepare_all_documents_committed(self, tmp_path):
        pipeline, keyword_index = self._make_pipeline(tmp_path)
        paths = self._write_docs(tmp_path, 12)

        results = pipeline.ingest_many(paths, max_workers=4)

        assert [r.status for r in results] == ["ok"] * 12
        assert len({r.doc_id for r in results}) == 12  # all distinct, none lost
        assert len(keyword_index._chunks) == sum(r.chunk_count for r in results)
        for i in range(12):
            assert keyword_index.search(f"number {i}", top_k=20)

    def test_results_returned_in_original_file_order(self, tmp_path, monkeypatch):
        # Worker threads may finish out of order; results must still line up
        # with the input paths list, not completion order.
        pipeline, _ = self._make_pipeline(tmp_path)
        paths = self._write_docs(tmp_path, 6)

        import random
        import time as time_mod
        from pathlib import Path as _Path

        original_prepare = pipeline._prepare
        delays = {str(_Path(p).resolve()): random.Random(42).random() * 0.02 for p in paths}

        def jittered_prepare(source_path, run_id, canonical_name=None):
            time_mod.sleep(delays.get(source_path, 0))
            return original_prepare(source_path, run_id, canonical_name=canonical_name)

        monkeypatch.setattr(pipeline, "_prepare", jittered_prepare)

        results = pipeline.ingest_many(paths, max_workers=6)
        expected_order = [f"doc{i}.md" for i in range(6)]
        actual_order = [_Path(r.source_path).name for r in results]
        assert actual_order == expected_order

    def test_one_file_failure_does_not_affect_others(self, tmp_path):
        pipeline, keyword_index = self._make_pipeline(tmp_path)
        paths = self._write_docs(tmp_path, 4)
        missing = tmp_path / "does_not_exist.md"
        paths.insert(2, missing)  # inject a bad path in the middle

        results = pipeline.ingest_many(paths, max_workers=3)

        statuses = [r.status for r in results]
        assert statuses == ["ok", "ok", "failed", "ok", "ok"]
        assert results[2].error is not None
        # The three good docs are still fully indexed despite the failure.
        assert len(keyword_index._chunks) == sum(
            r.chunk_count for r in results if r.status == "ok"
        )

    def test_canonical_names_length_mismatch_raises(self, tmp_path):
        pipeline, _ = self._make_pipeline(tmp_path)
        paths = self._write_docs(tmp_path, 2)
        with pytest.raises(ValueError):
            pipeline.ingest_many(paths, canonical_names=["only-one"])

    def test_commits_interleaved_not_batched_after_all_prepares(self, tmp_path, monkeypatch):
        # Regression guard: ingest_many must commit each document as its
        # prepare finishes, not collect every prepared result before
        # committing any of them. The latter holds N documents' full
        # parsed/chunked state in memory at once for a batch of N — peak
        # memory would scale with batch size instead of max_workers.
        #
        # Proof: track how many _prepare() calls are "in flight" (started,
        # not yet committed) at the moment each _commit() runs. Under
        # streaming commits this must stay <= max_workers; under the old
        # "drain all futures, then commit all" behavior it would climb to
        # the full batch size (12) before any commit happens.
        pipeline, _ = self._make_pipeline(tmp_path)
        paths = self._write_docs(tmp_path, 12)
        max_workers = 3

        in_flight = 0
        peak_in_flight = 0
        lock = __import__("threading").Lock()

        original_prepare = pipeline._prepare
        original_commit = pipeline._commit

        def tracked_prepare(*args, **kwargs):
            nonlocal in_flight, peak_in_flight
            with lock:
                in_flight += 1
                peak_in_flight = max(peak_in_flight, in_flight)
            try:
                return original_prepare(*args, **kwargs)
            finally:
                pass  # decremented in tracked_commit, once actually committed

        def tracked_commit(*args, **kwargs):
            nonlocal in_flight
            try:
                return original_commit(*args, **kwargs)
            finally:
                with lock:
                    in_flight -= 1

        monkeypatch.setattr(pipeline, "_prepare", tracked_prepare)
        monkeypatch.setattr(pipeline, "_commit", tracked_commit)

        results = pipeline.ingest_many(paths, max_workers=max_workers)

        assert [r.status for r in results] == ["ok"] * 12
        assert peak_in_flight <= max_workers, (
            f"peak in-flight prepared-but-uncommitted documents was "
            f"{peak_in_flight}, expected <= max_workers ({max_workers}) — "
            f"commits are not being interleaved with prepares"
        )

    def test_combined_with_batch_mode_defers_rebuild(self, tmp_path):
        pipeline, keyword_index = self._make_pipeline(tmp_path)
        paths = self._write_docs(tmp_path, 8)

        pipeline.begin_batch()
        results = pipeline.ingest_many(paths, max_workers=4)
        assert keyword_index._bm25 is None  # not rebuilt until end_batch
        pipeline.end_batch()

        assert [r.status for r in results] == ["ok"] * 8
        assert keyword_index._bm25 is not None
        assert keyword_index.search("cats and dogs", top_k=20)


# ---------------------------------------------------------------------------
# Per-document stage timing and cost tracking
# ---------------------------------------------------------------------------


class TestStageTimingsAndCost:
    def _make_pipeline(self, tmp_path, **kwargs):
        from rag.infra.stores.docstore_sqlite import SQLiteDocStore
        from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
        from rag.pipelines.ingest_pipeline import IngestPipeline

        db = tmp_path / "test.db"
        return IngestPipeline(SQLiteDocStore(db), SQLiteTraceStore(db), **kwargs)

    def _write_doc(self, tmp_path, name="doc.md"):
        f = tmp_path / name
        f.write_text("# Title\n\nContent about cats and dogs with enough words.")
        return f

    def test_stage_timings_present_for_bm25_only_ingest(self, tmp_path):
        from rag.infra.indexes.bm25_local import BM25LocalIndex

        pipeline = self._make_pipeline(tmp_path, keyword_index=BM25LocalIndex())
        result = pipeline.ingest(self._write_doc(tmp_path))

        assert result.status == "ok"
        for stage in ("load", "parse", "clean", "chunk", "bm25_index", "index_save"):
            assert stage in result.stage_timings_ms, f"missing stage: {stage}"
            assert result.stage_timings_ms[stage] >= 0

    def test_embed_stage_absent_without_embedding_provider(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        result = pipeline.ingest(self._write_doc(tmp_path))
        assert "embed" not in result.stage_timings_ms
        assert "faiss_index" not in result.stage_timings_ms

    def test_embed_stage_present_with_embedding_provider(self, tmp_path):
        embed_result = MagicMock(prompt_tokens=42)
        embed_result.vectors = [[0.1, 0.2]]
        provider = MagicMock()
        provider.model = "text-embedding-3-small"
        provider.embed.return_value = embed_result
        provider.embed_with_usage.return_value = embed_result
        vector_index = MagicMock()

        pipeline = self._make_pipeline(
            tmp_path, embedding_provider=provider, vector_index=vector_index
        )
        result = pipeline.ingest(self._write_doc(tmp_path))

        assert result.status == "ok"
        assert "embed" in result.stage_timings_ms
        assert "faiss_index" in result.stage_timings_ms
        assert result.embed_tokens == 42

    def test_index_save_absent_during_batch_mode(self, tmp_path):
        from rag.infra.indexes.bm25_local import BM25LocalIndex

        pipeline = self._make_pipeline(tmp_path, keyword_index=BM25LocalIndex())
        pipeline.begin_batch()
        result = pipeline.ingest(self._write_doc(tmp_path))
        pipeline.end_batch()

        # Deferred to end_batch(), not per-document — absent, not zero.
        assert "index_save" not in result.stage_timings_ms

    def test_cost_usd_zero_without_embedding_or_vision(self, tmp_path):
        result = self._make_pipeline(tmp_path).ingest(self._write_doc(tmp_path))
        assert result.cost_usd == 0.0

    def test_cost_usd_reflects_embedding_cost(self, tmp_path):
        from rag.infra.llm.pricing import estimate_cost_usd

        embed_result = MagicMock(prompt_tokens=100_000)
        embed_result.vectors = [[0.1, 0.2]]
        provider = MagicMock()
        provider.model = "text-embedding-3-small"
        provider.embed.return_value = embed_result
        provider.embed_with_usage.return_value = embed_result

        pipeline = self._make_pipeline(tmp_path, embedding_provider=provider)
        result = pipeline.ingest(self._write_doc(tmp_path))

        expected = estimate_cost_usd("text-embedding-3-small", 100_000)
        assert result.cost_usd == pytest.approx(expected)
        assert result.cost_usd > 0

    def test_cost_usd_includes_vision_cost_from_document_metadata(self, tmp_path, monkeypatch):
        # Vision-tier parsers (pdf_vision/pdf_hybrid) record their own token
        # usage and cost directly on Document.metadata during parsing (they
        # run inside the orchestrator, not _prepare() itself). _prepare()
        # must roll that into IngestResult.cost_usd alongside embedding cost.
        pipeline = self._make_pipeline(tmp_path)
        original_parse = pipeline._orchestrator.parse

        def fake_parse(artifact, plan):
            doc = original_parse(artifact, plan)
            return doc.model_copy(
                update={"metadata": {**doc.metadata, "vision_cost_usd": 0.0123}}
            )

        monkeypatch.setattr(pipeline._orchestrator, "parse", fake_parse)
        result = pipeline.ingest(self._write_doc(tmp_path))

        assert result.cost_usd == pytest.approx(0.0123)


# ---------------------------------------------------------------------------
# Batch ingest report
# ---------------------------------------------------------------------------


class TestFileSizeGuard:
    def test_oversized_file_fails_cleanly_not_oom_risk(self, tmp_path):
        # LocalFileLoader.read_bytes() has no streaming fallback — without a
        # size guard, an oversized file would be read entirely into memory
        # before any check runs. IngestPipeline must reject it up front and
        # report a clean "failed" status, not propagate an unhandled error.
        from rag.infra.stores.docstore_sqlite import SQLiteDocStore
        from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
        from rag.pipelines.ingest_pipeline import IngestPipeline

        db = tmp_path / "test.db"
        pipeline = IngestPipeline(
            SQLiteDocStore(db), SQLiteTraceStore(db), max_file_size_bytes=500
        )
        f = tmp_path / "big.md"
        f.write_text("x" * 1000)

        result = pipeline.ingest(f)

        assert result.status == "failed"
        assert "500" in result.error or "MB" in result.error

    def test_default_pipeline_allows_normal_sized_files(self, tmp_path):
        from rag.infra.stores.docstore_sqlite import SQLiteDocStore
        from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
        from rag.pipelines.ingest_pipeline import IngestPipeline

        db = tmp_path / "test.db"
        pipeline = IngestPipeline(SQLiteDocStore(db), SQLiteTraceStore(db))
        f = tmp_path / "normal.md"
        f.write_text("# Title\n\nContent about cats and dogs with enough words.")

        result = pipeline.ingest(f)
        assert result.status == "ok"


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

    def test_total_cost_and_tokens_aggregate_across_files(self, tmp_path):
        from rag.infra.stores.docstore_sqlite import SQLiteDocStore
        from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
        from rag.infra.llm.pricing import estimate_cost_usd
        from rag.pipelines.batch_ingest import BatchIngestRunner
        from rag.pipelines.ingest_pipeline import IngestPipeline

        embed_result = MagicMock(prompt_tokens=1000)
        embed_result.vectors = [[0.1, 0.2]]
        provider = MagicMock()
        provider.model = "text-embedding-3-small"
        provider.embed.return_value = embed_result
        provider.embed_with_usage.return_value = embed_result

        db = tmp_path / "t.db"
        p = IngestPipeline(
            SQLiteDocStore(db), SQLiteTraceStore(db), embedding_provider=provider
        )
        docs = tmp_path / "corpus"
        docs.mkdir()
        for name in ("a", "b", "c"):
            (docs / f"{name}.md").write_text(
                f"# {name}\n\nContent about {name} with enough words for the gate."
            )

        report = BatchIngestRunner(p).run(docs)

        assert report.total_embed_tokens == 3000  # 3 files x 1000 tokens each
        expected_cost = 3 * estimate_cost_usd("text-embedding-3-small", 1000)
        assert report.total_cost_usd == pytest.approx(expected_cost)
        assert f"${report.total_cost_usd:.4f}" in report.format_summary()

        meta = report.to_metadata()
        assert meta["total_embed_tokens"] == 3000
        assert meta["total_cost_usd"] == pytest.approx(expected_cost)


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
