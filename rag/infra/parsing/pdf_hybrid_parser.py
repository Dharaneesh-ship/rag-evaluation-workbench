"""Hybrid PDF parser — free text extraction plus per-page vision escalation.

Fixes the blind spot of document-level fallback: a PDF with plenty of text
overall passes the char-count gate even when half its information lives in
charts, images, or multi-column layouts that plain text extraction mangles
(wrong reading order) or drops entirely (figures).

Strategy:
1. Parse the whole document with the free PyMuPDF text parser.
2. Analyse every page for complexity signals:
   - image area coverage (raster charts, photos, scanned regions)
   - vector drawing count (native charts/graphs in financial decks)
   - sparse text (image-only pages)
3. Re-parse only the complex pages with the vision LLM (correct reading
   order, table structure, figure descriptions) and splice those blocks
   into the document in page order.

Cost is bounded by ``max_vision_pages`` per document; simple text pages
never touch the API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from rag.core.contracts.document import Document
from rag.core.contracts.ir_block import IRBlock
from rag.core.contracts.parse_report import ParseReport
from rag.core.interfaces.parser import BaseParser
from rag.infra.parsing.pdf_pymupdf import PdfPyMuPDFParser
from rag.infra.parsing.pdf_vision_parser import PdfVisionParser

logger = logging.getLogger(__name__)


@dataclass
class PageComplexity:
    """Complexity signals for one PDF page.

    Attributes:
        page: 1-based page number.
        char_count: Characters extracted by the text parser for this page.
        image_area_ratio: Fraction of the page area covered by raster images.
        drawing_count: Number of vector drawing paths on the page.
        drawing_area_ratio: Fraction of the page area covered by vector
            drawing bounding boxes (bar/line charts have real area; thin
            table rules and link underlines do not).
        reasons: Which triggers fired (empty = simple page).
    """

    page: int
    char_count: int = 0
    image_area_ratio: float = 0.0
    drawing_count: int = 0
    drawing_area_ratio: float = 0.0
    reasons: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = []

    @property
    def is_complex(self) -> bool:
        return bool(self.reasons)


class PdfHybridParser(BaseParser):
    """PDF parser that escalates individual complex pages to a vision LLM.

    Args:
        text_parser: PyMuPDF text parser used for the base pass.
        vision_parser: Vision parser used for complex pages.
        image_area_ratio: Pages with raster images covering at least this
            fraction of the page escalate. Defaults to 0.15.
        min_vector_drawings: Pages with at least this many vector drawing
            paths (native charts) escalate. Defaults to 25.
        min_drawing_area_ratio: Pages whose vector drawing bounding boxes
            cover at least this fraction of the page escalate — catches
            charts drawn with few but large paths. Defaults to 0.10.
        min_chars_per_page: Pages with fewer extracted characters escalate
            (image-only pages). Defaults to 200.
        max_vision_pages: Hard cap on vision-parsed pages per document —
            bounds API cost for pathological inputs. Defaults to 30.

    Attributes:
        PARSER_ID: Identifier written to ``ParseReport.parser_used`` when
            at least one page was vision-parsed.
    """

    PARSER_ID = "pdf_hybrid"

    def __init__(
        self,
        text_parser: PdfPyMuPDFParser,
        vision_parser: PdfVisionParser,
        image_area_ratio: float = 0.15,
        min_vector_drawings: int = 25,
        min_drawing_area_ratio: float = 0.10,
        min_chars_per_page: int = 200,
        max_vision_pages: int = 30,
    ) -> None:
        self._text_parser = text_parser
        self._vision_parser = vision_parser
        self._image_area_ratio = image_area_ratio
        self._min_vector_drawings = min_vector_drawings
        self._min_drawing_area_ratio = min_drawing_area_ratio
        self._min_chars_per_page = min_chars_per_page
        self._max_vision_pages = max_vision_pages

    def supports(self, mime_type: str) -> bool:
        """Return True for PDF MIME types."""
        return mime_type in ("application/pdf", "application/x-pdf")

    # ── Page analysis ────────────────────────────────────────────────────

    def _analyse_pages(
        self, source_path: str, chars_per_page: dict[int, int]
    ) -> list[PageComplexity]:
        """Compute complexity signals for every page.

        Args:
            source_path: Absolute path to the PDF.
            chars_per_page: Characters extracted by the text pass, per page.

        Returns:
            One PageComplexity per page, in page order.
        """
        import fitz

        results: list[PageComplexity] = []
        doc = fitz.open(source_path)
        try:
            for index in range(len(doc)):
                page = doc[index]
                page_num = index + 1
                info = PageComplexity(
                    page=page_num, char_count=chars_per_page.get(page_num, 0)
                )

                page_area = abs(page.rect.width * page.rect.height) or 1.0
                image_area = 0.0
                try:
                    for img in page.get_images(full=True):
                        for rect in page.get_image_rects(img[0]):
                            image_area += abs(rect.width * rect.height)
                except Exception:  # pragma: no cover - fitz quirk safety
                    pass
                info.image_area_ratio = min(image_area / page_area, 1.0)

                try:
                    drawings = page.get_drawings()
                    info.drawing_count = len(drawings)
                    drawing_area = 0.0
                    for d in drawings:
                        rect = d.get("rect")
                        if rect is not None:
                            drawing_area += abs(rect.width * rect.height)
                    info.drawing_area_ratio = min(drawing_area / page_area, 1.0)
                except Exception:  # pragma: no cover - fitz quirk safety
                    info.drawing_count = 0

                if info.image_area_ratio >= self._image_area_ratio:
                    info.reasons.append(
                        f"image_area {info.image_area_ratio:.0%}"
                    )
                if info.drawing_count >= self._min_vector_drawings:
                    info.reasons.append(f"drawings {info.drawing_count}")
                if info.drawing_area_ratio >= self._min_drawing_area_ratio:
                    info.reasons.append(
                        f"drawing_area {info.drawing_area_ratio:.0%}"
                    )
                if info.char_count < self._min_chars_per_page:
                    info.reasons.append(f"chars {info.char_count}")

                results.append(info)
        finally:
            doc.close()
        return results

    # ── Parse ────────────────────────────────────────────────────────────

    def parse(self, source_path: str) -> Document:
        """Parse a PDF, escalating complex pages to the vision LLM.

        Args:
            source_path: Absolute path to the PDF file.

        Returns:
            Document whose blocks mix text-parsed and vision-parsed pages,
            in page order. When no page is complex, the plain text-parser
            result is returned unchanged.
        """
        source_path = str(Path(source_path).resolve())
        text_doc = self._text_parser.parse(source_path)

        chars_per_page: dict[int, int] = {}
        for block in text_doc.blocks:
            if block.page is not None:
                chars_per_page[block.page] = (
                    chars_per_page.get(block.page, 0) + len(block.text)
                )

        pages = self._analyse_pages(source_path, chars_per_page)
        complex_pages = [p for p in pages if p.is_complex]

        if not complex_pages:
            logger.debug(
                "No complex pages in '%s' — keeping plain text parse.",
                source_path,
            )
            return text_doc

        if len(complex_pages) > self._max_vision_pages:
            logger.warning(
                "'%s' has %d complex pages; vision-parsing only the first %d "
                "(max_vision_pages).",
                source_path,
                len(complex_pages),
                self._max_vision_pages,
            )
            complex_pages = complex_pages[: self._max_vision_pages]

        # Vision-parse the complex pages. Two ways escalation can fail without
        # losing content:
        #   1. API/render exception → keep the text-parser blocks.
        #   2. Truncated transcript (output cap or content-filter cut) → the
        #      vision output is partial and may have dropped whole regions
        #      (e.g. a right-column rating table). A truncated result is
        #      untrustworthy, so we KEEP the complete text-parser extraction
        #      for that page rather than replacing it with the partial vision
        #      output. The text parser often already captures the "missing"
        #      content (PDFs embed accessibility text for charts/dot ratings).
        vision_blocks: dict[int, list[IRBlock]] = {}
        truncated_pages: list[int] = []
        for info in complex_pages:
            logger.info(
                "Vision-escalating page %d of '%s' (%s).",
                info.page,
                source_path,
                "; ".join(info.reasons),
            )
            try:
                blocks, truncated = self._vision_parser.parse_page_blocks(
                    source_path, info.page
                )
            except Exception as exc:
                logger.warning(
                    "Vision parse failed for page %d of '%s': %s — "
                    "keeping text-parser blocks for this page.",
                    info.page,
                    source_path,
                    exc,
                )
                continue

            if truncated:
                truncated_pages.append(info.page)
                logger.warning(
                    "Vision transcript for page %d of '%s' was truncated — "
                    "keeping the complete text-parser blocks for this page "
                    "instead of the partial vision output.",
                    info.page,
                    source_path,
                )
                continue

            vision_blocks[info.page] = blocks

        # Splice: replace text blocks of vision-parsed pages, keep the rest.
        text_blocks_by_page: dict[int, list[IRBlock]] = {}
        for block in text_doc.blocks:
            text_blocks_by_page.setdefault(block.page or 0, []).append(block)

        merged: list[IRBlock] = []
        for page_num in range(1, len(pages) + 1):
            if page_num in vision_blocks:
                merged.extend(vision_blocks[page_num])
            else:
                merged.extend(text_blocks_by_page.get(page_num, []))
        merged.extend(text_blocks_by_page.get(0, []))  # pageless blocks, if any

        all_text = " ".join(b.text for b in merged)
        report = ParseReport(
            char_count=len(all_text),
            block_count=len(merged),
            non_printable_ratio=0.0,
            repetition_score=0.0,
            parser_used=self.PARSER_ID,
            fallback_triggered=False,
        )

        return text_doc.model_copy(
            update={
                "blocks": merged,
                "parse_report": report,
                "metadata": {
                    **text_doc.metadata,
                    "vision_pages": sorted(vision_blocks.keys()),
                    "vision_page_reasons": {
                        str(p.page): "; ".join(p.reasons)
                        for p in complex_pages
                        if p.page in vision_blocks
                    },
                    # Pages we tried to escalate but whose transcript was cut
                    # off; text-parser blocks were kept for these.
                    "vision_truncated_pages": sorted(truncated_pages),
                },
            }
        )
