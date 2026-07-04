"""Tests for complex-layout PDF hardening.

Covers:
- Heading heuristics for slide-style PDFs (length cap, sentence endings,
  page-level heading-flood demotion)
- Oversized-block splitting and prefix-aware budget accounting in the packer
- PdfHybridParser page-level vision escalation
"""

from __future__ import annotations

from unittest.mock import MagicMock

import fitz
import pytest

from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.core.contracts.text_block import TextBlock
from rag.infra.chunking.chunk_packer_anchor_aware import AnchorAwareChunkPacker
from rag.infra.parsing.pdf_pymupdf import (
    PdfPyMuPDFParser,
    _demote_heading_floods,
    _heading_signal,
)

DOC_ID = "doc-layout-test"


# ---------------------------------------------------------------------------
# Heading heuristics
# ---------------------------------------------------------------------------


class TestHeadingSignal:
    THRESHOLD = 14.4  # e.g. median 12pt * 1.2

    def test_short_large_font_title_is_heading(self):
        assert _heading_signal("What Is The Next Move?", 20.0, self.THRESHOLD) == "font"

    def test_long_large_font_line_is_not_heading(self):
        text = (
            "In 2026, we believe the 'proud' bull market will power on into "
            "the next era of growth and beyond all expectations"
        )
        assert _heading_signal(text, 18.0, self.THRESHOLD) is None

    def test_sentence_ending_large_font_line_is_not_heading(self):
        assert _heading_signal("experienced from 2023 to 2025.", 18.0, self.THRESHOLD) is None

    def test_all_caps_label_is_heading_regardless_of_size(self):
        assert _heading_signal("PROFESSIONAL EXPERIENCE", 10.0, self.THRESHOLD) == "caps"

    def test_normal_body_line_is_not_heading(self):
        assert _heading_signal("Regular body text here", 12.0, self.THRESHOLD) is None


class TestHeadingFloodDemotion:
    def _heading(self, text: str, page: int) -> IRBlock:
        return IRBlock(block_type=BlockType.HEADING, text=text, page=page)

    def _para(self, text: str, page: int) -> IRBlock:
        return IRBlock(block_type=BlockType.PARAGRAPH, text=text, page=page)

    def test_flooded_page_demotes_font_headings(self):
        blocks = [self._heading(f"Body line {i} misread", 1) for i in range(6)]
        blocks.append(self._para("small print", 1))
        font_ids = {id(b) for b in blocks[:6]}
        _demote_heading_floods(blocks, font_ids)
        assert all(b.block_type == BlockType.PARAGRAPH for b in blocks)

    def test_caps_headings_survive_demotion(self):
        blocks = [self._heading(f"Body line {i} misread", 1) for i in range(6)]
        caps = self._heading("SCENARIO OVERVIEW", 1)
        blocks.append(caps)
        font_ids = {id(b) for b in blocks[:6]}  # caps block not font-derived
        _demote_heading_floods(blocks, font_ids)
        assert blocks[-1].block_type == BlockType.HEADING

    def test_normal_page_keeps_headings(self):
        blocks = [
            self._heading("Real Title", 1),
            self._para("Paragraph one content.", 1),
            self._para("Paragraph two content.", 1),
        ]
        font_ids = {id(blocks[0])}
        _demote_heading_floods(blocks, font_ids)
        assert blocks[0].block_type == BlockType.HEADING

    def test_demotion_is_per_page(self):
        flooded = [self._heading(f"Line {i}", 1) for i in range(6)]
        healthy = [
            self._heading("Real Title", 2),
            self._para("Content paragraph.", 2),
            self._para("More content here.", 2),
        ]
        blocks = flooded + healthy
        font_ids = {id(b) for b in flooded} | {id(healthy[0])}
        _demote_heading_floods(blocks, font_ids)
        assert all(b.block_type == BlockType.PARAGRAPH for b in blocks[:6])
        assert blocks[6].block_type == BlockType.HEADING


class TestInvisibleCharsAndLowercase:
    THRESHOLD = 14.4

    def test_zero_width_space_does_not_hide_sentence_ending(self):
        # Real case: 'buildout. ​' — ZWSP after the period defeated the
        # sentence-ending check. _line_text strips it at extraction time.
        from rag.infra.parsing.pdf_pymupdf import _line_text

        line = {"spans": [{"text": "by the ongoing buildout. ​"}]}
        assert _line_text(line).strip().endswith(".")

    def test_lowercase_start_rejected_as_heading(self):
        assert (
            _heading_signal(
                "by the ongoing digital infrastructure buildout", 18.0, self.THRESHOLD
            )
            is None
        )

    def test_unicode_fix_removes_zero_width_chars(self):
        from rag.infra.cleaning.unicode_fix import UnicodeFixer

        blocks = [
            IRBlock(
                block_type=BlockType.PARAGRAPH,
                text="buildout.​ done﻿",
            )
        ]
        cleaned = UnicodeFixer().clean(blocks)
        assert cleaned[0].text == "buildout. done"


class TestVisionTruncationWarning:
    def _make_parser(self, finish_reason: str, content: str = "partial transcript"):
        from rag.infra.parsing.pdf_vision_parser import PdfVisionParser

        parser = PdfVisionParser(renderer=MagicMock(), api_key="test-key")
        resp = MagicMock()
        resp.choices[0].message.content = content
        resp.choices[0].finish_reason = finish_reason
        client = MagicMock()
        client.chat.completions.create.return_value = resp
        parser._client = client
        return parser

    def test_length_truncation_flagged_and_warned(self, caplog):
        import logging

        parser = self._make_parser("length")
        with caplog.at_level(logging.WARNING):
            text, truncated = parser._transcribe_page(MagicMock())
        assert text == "partial transcript"
        assert truncated is True
        assert any("truncated" in r.message for r in caplog.records)

    def test_content_filter_truncation_flagged(self, caplog):
        # Real-world case: OpenAI moderation deterministically cuts the
        # transcript on some financial disclaimer content.
        import logging

        parser = self._make_parser("content_filter")
        with caplog.at_level(logging.WARNING):
            text, truncated = parser._transcribe_page(MagicMock())
        assert truncated is True
        assert any("truncated" in r.message for r in caplog.records)

    def test_complete_transcript_not_flagged(self):
        parser = self._make_parser("stop", content="full page")
        text, truncated = parser._transcribe_page(MagicMock())
        assert text == "full page"
        assert truncated is False


class TestParsePagesParallel:
    """PdfVisionParser.parse_pages_parallel — the throughput primitive shared
    by the full-document tier and PdfHybridParser's per-page escalation."""

    def _make_parser(self, page_side_effect, max_workers=8):
        from rag.infra.parsing.pdf_vision_parser import PdfVisionParser

        parser = PdfVisionParser(
            renderer=MagicMock(), api_key="test-key", max_workers=max_workers
        )
        parser.parse_page_blocks = MagicMock(side_effect=page_side_effect)
        return parser

    def test_returns_result_per_page_in_success_case(self):
        def fake(source_path, page_num):
            return [IRBlock(block_type=BlockType.PARAGRAPH, text=f"page {page_num}", page=page_num)], False

        parser = self._make_parser(fake)
        results = parser.parse_pages_parallel("doc.pdf", [1, 2, 3])

        assert set(results.keys()) == {1, 2, 3}
        for page_num in (1, 2, 3):
            blocks, truncated, exc = results[page_num]
            assert exc is None
            assert truncated is False
            assert blocks[0].text == f"page {page_num}"

    def test_empty_page_list_returns_empty_dict(self):
        parser = self._make_parser(lambda source_path, page_num: ([], False))
        assert parser.parse_pages_parallel("doc.pdf", []) == {}

    def test_per_page_exception_isolated_not_raised(self):
        # A single page's API failure must not prevent other pages from
        # being processed, and must not raise out of parse_pages_parallel —
        # callers (hybrid parser) handle per-page fallback themselves.
        def fake(source_path, page_num):
            if page_num == 2:
                raise RuntimeError("API down")
            return [IRBlock(block_type=BlockType.PARAGRAPH, text="ok", page=page_num)], False

        parser = self._make_parser(fake)
        results = parser.parse_pages_parallel("doc.pdf", [1, 2, 3])

        assert results[1][2] is None and results[3][2] is None
        _, _, exc2 = results[2]
        assert isinstance(exc2, RuntimeError)
        assert results[2][0] == []  # empty blocks on failure

    def test_concurrency_bounded_by_max_workers(self):
        # Track the peak number of concurrently-executing calls to prove
        # pages actually run in parallel (not accidentally serialized) while
        # respecting the max_workers cap.
        import threading
        import time

        lock = threading.Lock()
        state = {"active": 0, "peak": 0}

        def fake(source_path, page_num):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.05)
            with lock:
                state["active"] -= 1
            return [], False

        parser = self._make_parser(fake, max_workers=3)
        parser.parse_pages_parallel("doc.pdf", list(range(1, 7)))

        assert state["peak"] > 1  # actually ran concurrently
        assert state["peak"] <= 3  # respected max_workers


class TestDrawingAreaTrigger:
    def test_large_filled_chart_escalates_despite_few_paths(self, tmp_path):
        """A bar chart drawn with a handful of big rects must escalate via
        the area trigger even when the path count is low."""
        from rag.infra.parsing.pdf_hybrid_parser import PdfHybridParser

        pdf_path = tmp_path / "big_bars.pdf"
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(
            fitz.Rect(50, 30, 545, 120),
            "Chart page with a short intro paragraph of digital text. " * 6,
            fontsize=11,
        )
        for i in range(6):  # 6 fat bars — few paths, large area
            page.draw_rect(
                fitz.Rect(60, 150 + i * 90, 500, 220 + i * 90),
                fill=(0.2, 0.3, 0.6),
            )
        doc.save(pdf_path)
        doc.close()

        vision = MagicMock()
        vision.parse_pages_parallel.return_value = {
            1: (
                [IRBlock(block_type=BlockType.IMAGE_CAPTION, text="Figure: bars.", page=1)],
                False,
                None,
            )
        }
        hybrid = PdfHybridParser(
            text_parser=PdfPyMuPDFParser(),
            vision_parser=vision,
            min_vector_drawings=25,
            min_drawing_area_ratio=0.10,
        )
        document = hybrid.parse(str(pdf_path))
        vision.parse_pages_parallel.assert_called_once()
        reasons = document.metadata["vision_page_reasons"]["1"]
        assert "drawing_area" in reasons


class TestMarkdownArtifactStripping:
    def test_heading_markers_stripped(self):
        from rag.infra.parsing.pdf_vision_parser import _markdown_to_blocks

        blocks = _markdown_to_blocks("## SCENARIO 1 BULL CASE", page_num=2)
        assert blocks[0].text == "SCENARIO 1 BULL CASE"
        assert blocks[0].block_type == BlockType.HEADING

    def test_horizontal_rules_dropped(self):
        from rag.infra.parsing.pdf_vision_parser import _markdown_to_blocks

        assert _markdown_to_blocks("# Title\n\n---\n\nBody text.", page_num=1) == [
            IRBlock(block_type=BlockType.HEADING, text="Title", page=1),
            IRBlock(block_type=BlockType.PARAGRAPH, text="Body text.", page=1),
        ]


class TestSlideStylePdfIntegration:
    """End-to-end: a slide-style PDF must not classify body text as headings."""

    def _make_slide_pdf(self, tmp_path):
        pdf_path = tmp_path / "slide.pdf"
        doc = fitz.open()
        page = doc.new_page(width=842, height=595)  # landscape slide
        # Title: large font, short
        page.insert_text((60, 70), "Quarterly Outlook", fontsize=28)
        # Body: large font (slide body), long sentence lines
        body_lines = [
            "In 2026, we believe the market will power on into the next era of growth.",
            "Four engines are expected to fuel this developing growth cycle overall.",
            "These tailwinds may rev up more than expected as we move through the year.",
        ]
        for i, line in enumerate(body_lines):
            page.insert_text((60, 140 + i * 30), line, fontsize=18)
        # Disclaimer: tiny font, lots of characters (drags the median down)
        disclaimer = (
            "This material is for informational purposes only and is not a "
            "recommendation. " * 6
        )
        for i in range(4):
            page.insert_text(
                (60, 480 + i * 12), disclaimer[:110], fontsize=7
            )
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    def test_body_lines_not_classified_as_headings(self, tmp_path):
        pdf_path = self._make_slide_pdf(tmp_path)
        document = PdfPyMuPDFParser().parse(str(pdf_path))
        headings = [
            b.text for b in document.blocks if b.block_type == BlockType.HEADING
        ]
        assert all("we believe the market" not in h for h in headings)
        assert all("tailwinds" not in h for h in headings)


# ---------------------------------------------------------------------------
# Packer: oversized blocks and budget accounting
# ---------------------------------------------------------------------------


def _tb(text, block_type=BlockType.PARAGRAPH, sequence=0, page=None, block_hash=None):
    return TextBlock(
        doc_id=DOC_ID,
        block_type=block_type,
        text=text,
        block_hash=block_hash or f"{sequence:064x}",
        sequence=sequence,
        page=page,
    )


class TestPackerBudgetEnforcement:
    def test_oversized_block_is_split(self):
        packer = AnchorAwareChunkPacker(token_budget=100)
        long_text = " ".join(
            f"Sentence number {i} contains several meaningful words here."
            for i in range(40)
        )
        chunks = packer.pack([_tb(long_text, sequence=0, page=3)])
        assert len(chunks) > 1
        assert all(c.token_count <= 100 for c in chunks)
        # All parts reference the same source block, with unique chunk ids
        assert all(c.block_hashes == [f"{0:064x}"] for c in chunks)
        assert len({c.chunk_signature for c in chunks}) == len(chunks)
        assert all(c.metadata.get("start_page") == 3 for c in chunks)

    def test_oversized_single_sentence_stays_intact(self):
        packer = AnchorAwareChunkPacker(token_budget=20)
        text = "word " * 100  # no sentence boundaries
        chunks = packer.pack([_tb(text.strip())])
        assert len(chunks) == 1

    def test_budget_accounts_for_heading_prefix_injection(self):
        # stable_text injects "heading: " before every content block; the
        # packed chunks must respect the budget AFTER that injection.
        packer = AnchorAwareChunkPacker(token_budget=60)
        heading = _tb(
            "Long Section Heading About Market Conditions",
            block_type=BlockType.HEADING,
            sequence=0,
        )
        contents = [
            _tb(
                f"Content line {i} with a handful of extra words attached.",
                sequence=i + 1,
            )
            for i in range(10)
        ]
        chunks = packer.pack([heading] + contents)
        assert all(c.token_count <= 60 for c in chunks)

    def test_normal_packing_unchanged(self):
        packer = AnchorAwareChunkPacker(token_budget=512)
        blocks = [
            _tb("Title", block_type=BlockType.HEADING, sequence=0),
            _tb("Short paragraph one.", sequence=1),
            _tb("Short paragraph two.", sequence=2),
        ]
        chunks = packer.pack(blocks)
        assert len(chunks) == 1
        assert chunks[0].token_count <= 512


# ---------------------------------------------------------------------------
# PdfHybridParser
# ---------------------------------------------------------------------------


def _make_mixed_pdf(tmp_path, image_path):
    """Two-page PDF: page 1 text-rich, page 2 dominated by a raster image."""
    pdf_path = tmp_path / "mixed.pdf"
    doc = fitz.open()

    page1 = doc.new_page(width=595, height=842)
    body = (
        "This page is ordinary digital text with plenty of characters. " * 8
    )
    rect = fitz.Rect(50, 50, 545, 700)
    page1.insert_textbox(rect, body, fontsize=11)

    page2 = doc.new_page(width=595, height=842)
    page2.insert_image(fitz.Rect(0, 0, 595, 700), filename=str(image_path))
    page2.insert_text((50, 780), "tiny caption", fontsize=9)

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def chart_image(tmp_path):
    from PIL import Image

    img_path = tmp_path / "chart.png"
    Image.new("RGB", (600, 700), "lightblue").save(img_path)
    return img_path


class TestPdfHybridParser:
    def _make_hybrid(self, vision_blocks=None, truncated=False, exc=None, **kwargs):
        from rag.infra.parsing.pdf_hybrid_parser import PdfHybridParser

        vision = MagicMock()
        blocks = vision_blocks or [
            IRBlock(
                block_type=BlockType.IMAGE_CAPTION,
                text="Figure: a line chart showing scenario outcomes.",
                page=2,
            )
        ]

        def fake_parse_pages_parallel(source_path, page_nums):
            return {p: (blocks, truncated, exc) for p in page_nums}

        vision.parse_pages_parallel.side_effect = fake_parse_pages_parallel
        return PdfHybridParser(
            text_parser=PdfPyMuPDFParser(), vision_parser=vision, **kwargs
        ), vision

    def test_complex_page_escalates_to_vision(self, tmp_path, chart_image):
        pdf_path = _make_mixed_pdf(tmp_path, chart_image)
        hybrid, vision = self._make_hybrid()
        document = hybrid.parse(str(pdf_path))

        vision.parse_pages_parallel.assert_called_once()
        assert vision.parse_pages_parallel.call_args[0][1] == [2]  # page 2 only
        assert document.parse_report.parser_used == "pdf_hybrid"
        assert document.metadata["vision_pages"] == [2]
        # Page 1 text kept, page 2 replaced by vision output
        page1_text = " ".join(b.text for b in document.blocks if b.page == 1)
        page2_text = " ".join(b.text for b in document.blocks if b.page == 2)
        assert "ordinary digital text" in page1_text
        assert "line chart" in page2_text
        assert "tiny caption" not in page2_text

    def test_text_only_pdf_skips_vision_entirely(self, tmp_path):
        pdf_path = tmp_path / "plain.pdf"
        doc = fitz.open()
        for _ in range(2):
            page = doc.new_page(width=595, height=842)
            page.insert_textbox(
                fitz.Rect(50, 50, 545, 780),
                "Plain digital text with plenty of characters on it. " * 10,
                fontsize=11,
            )
        doc.save(pdf_path)
        doc.close()

        hybrid, vision = self._make_hybrid()
        document = hybrid.parse(str(pdf_path))
        vision.parse_pages_parallel.assert_not_called()
        assert document.parse_report.parser_used == "pymupdf"

    def test_vision_failure_keeps_text_blocks(self, tmp_path, chart_image):
        pdf_path = _make_mixed_pdf(tmp_path, chart_image)
        hybrid, vision = self._make_hybrid(exc=RuntimeError("API down"))
        document = hybrid.parse(str(pdf_path))

        page2_text = " ".join(b.text for b in document.blocks if b.page == 2)
        assert "tiny caption" in page2_text  # text-parser blocks preserved
        assert document.metadata["vision_pages"] == []

    def test_truncated_vision_keeps_text_blocks(self, tmp_path, chart_image):
        # A truncated transcript (content_filter / length) may have dropped a
        # whole region. The complete text-parser blocks for that page must be
        # kept rather than replaced by the partial vision output.
        pdf_path = _make_mixed_pdf(tmp_path, chart_image)
        hybrid, vision = self._make_hybrid(truncated=True)
        document = hybrid.parse(str(pdf_path))

        page2_text = " ".join(b.text for b in document.blocks if b.page == 2)
        assert "tiny caption" in page2_text          # text-parser kept
        assert "line chart" not in page2_text        # partial vision discarded
        assert document.metadata["vision_pages"] == []
        assert document.metadata["vision_truncated_pages"] == [2]

    def test_max_vision_pages_caps_api_calls(self, tmp_path, chart_image):
        pdf_path = tmp_path / "many_charts.pdf"
        doc = fitz.open()
        for _ in range(4):
            page = doc.new_page(width=595, height=842)
            page.insert_image(fitz.Rect(0, 0, 595, 700), filename=str(chart_image))
        doc.save(pdf_path)
        doc.close()

        hybrid, vision = self._make_hybrid(max_vision_pages=2)
        hybrid.parse(str(pdf_path))
        vision.parse_pages_parallel.assert_called_once()
        assert len(vision.parse_pages_parallel.call_args[0][1]) == 2  # capped

    def test_pages_escalated_concurrently_via_thread_pool(self, tmp_path, chart_image):
        # Regression guard for the parallelization itself: escalating N
        # complex pages must issue exactly one parse_pages_parallel() call
        # carrying all N page numbers, not N sequential calls.
        pdf_path = tmp_path / "many_charts.pdf"
        doc = fitz.open()
        for _ in range(4):
            page = doc.new_page(width=595, height=842)
            page.insert_image(fitz.Rect(0, 0, 595, 700), filename=str(chart_image))
        doc.save(pdf_path)
        doc.close()

        hybrid, vision = self._make_hybrid(max_vision_pages=4)
        hybrid.parse(str(pdf_path))
        vision.parse_pages_parallel.assert_called_once()
        assert sorted(vision.parse_pages_parallel.call_args[0][1]) == [1, 2, 3, 4]


class TestVisionModelSelection:
    def test_default_model_is_gpt4o_mini(self, monkeypatch):
        from rag.pipelines.ingest_pipeline import _build_vision_parser

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        parser = _build_vision_parser()
        assert parser._model == "gpt-4o-mini"

    def test_custom_model_is_threaded_through(self, monkeypatch):
        from rag.pipelines.ingest_pipeline import _build_vision_parser

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        parser = _build_vision_parser(vision_model="gpt-4.1-nano")
        assert parser._model == "gpt-4.1-nano"

    def test_ingest_pipeline_threads_vision_model(self, monkeypatch, tmp_path):
        from rag.infra.stores.docstore_sqlite import SQLiteDocStore
        from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
        from rag.pipelines.ingest_pipeline import IngestPipeline

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        db = tmp_path / "t.db"
        pipeline = IngestPipeline(
            SQLiteDocStore(db), SQLiteTraceStore(db), vision_model="gpt-4.1-mini"
        )
        vision_parser = pipeline._orchestrator._registry.get("pdf_vision")
        assert vision_parser is not None
        assert vision_parser._model == "gpt-4.1-mini"


class TestHybridConfigGating:
    def test_disabled_page_level_returns_none(self, tmp_path):
        from rag.pipelines.ingest_pipeline import _build_hybrid_parser

        cfg = tmp_path / "parser_candidates.yaml"
        cfg.write_text(
            "vision:\n  page_level:\n    enabled: false\nroutes: {}\n"
        )
        assert _build_hybrid_parser(MagicMock(), MagicMock(), cfg) is None

    def test_enabled_page_level_builds_parser(self, tmp_path):
        from rag.infra.parsing.pdf_hybrid_parser import PdfHybridParser
        from rag.pipelines.ingest_pipeline import _build_hybrid_parser

        cfg = tmp_path / "parser_candidates.yaml"
        cfg.write_text(
            "vision:\n  page_level:\n    enabled: true\n"
            "    max_pages_per_doc: 5\nroutes: {}\n"
        )
        parser = _build_hybrid_parser(
            PdfPyMuPDFParser(), MagicMock(), cfg
        )
        assert isinstance(parser, PdfHybridParser)
        assert parser._max_vision_pages == 5
