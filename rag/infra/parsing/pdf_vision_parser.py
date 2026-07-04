"""Vision-LLM PDF parser — renders pages to images and transcribes them.

Third tier of the PDF parsing fallback chain (pymupdf → pdf_ocr → pdf_vision).
Each page is rendered to an image via ``BasePageRenderer`` and sent to a
multimodal LLM that transcribes the page content to Markdown, including
table structure and short descriptions of figures/charts. The Markdown is
converted into ``IRBlock`` objects so the rest of the pipeline stays
format-agnostic.

This parser costs API tokens per page — the orchestrator only escalates to
it when the OCR tier produces low-confidence or near-empty output.

Usage::

    from rag.infra.ocr.renderer_pymupdf import PyMuPDFPageRenderer
    from rag.infra.parsing.pdf_vision_parser import PdfVisionParser

    parser = PdfVisionParser(renderer=PyMuPDFPageRenderer(dpi=150))
    document = parser.parse("/path/to/complex.pdf")
"""

from __future__ import annotations

import base64
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional

from rag.core.contracts.document import Document
from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.core.contracts.parse_report import ParseReport
from rag.core.interfaces.page_renderer import BasePageRenderer
from rag.core.interfaces.parser import BaseParser

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"
# Per-page Vision calls are network-bound (one HTTP round-trip per page), so
# concurrent requests parallelize well. 8 is a conservative default that
# stays well under typical per-org rate limits while still giving a large
# (5-8x observed) speedup on multi-page documents.
_DEFAULT_MAX_WORKERS = 8
# Dense pages (tables of contents, data tables) can need several thousand
# output tokens; a low cap silently truncates the transcript mid-page.
_DEFAULT_MAX_TOKENS = 8192

# finish_reasons that mean the transcript is INCOMPLETE — the model stopped
# before covering the whole page. "length" = output token cap hit;
# "content_filter" = OpenAI moderation cut the response (fires deterministically
# on some financial/legal disclaimer content). A truncated transcript is
# untrustworthy: callers should prefer the complete text-parser extraction.
_TRUNCATION_REASONS = frozenset({"length", "content_filter"})

_PAGE_PROMPT = (
    "Transcribe this document page to Markdown. Rules:\n"
    "- Preserve headings, paragraphs, and list structure.\n"
    "- Convert tables to Markdown tables, keeping every cell value.\n"
    "- For figures, charts, or photos, write a one-paragraph description "
    "starting with 'Figure:'.\n"
    "- Do not add commentary, preamble, or code fences.\n"
    "- If the page is blank, respond with exactly: [BLANK_PAGE]"
)


def _starts_with_ordinal(text: str) -> bool:
    """Return True for ordered-list leads like ``1. item`` or ``12) item``."""
    head = text.split(" ", 1)[0]
    return len(head) >= 2 and head[:-1].isdigit() and head[-1] in ".)"


def _markdown_to_blocks(markdown: str, page_num: int) -> list[IRBlock]:
    """Convert one page of Markdown transcript into IRBlocks.

    Splits on blank lines; classifies headings, tables, list items, and
    figure descriptions so downstream chunking can use block types.

    Args:
        markdown: Markdown text returned by the vision model.
        page_num: 1-based page number to stamp on every block.

    Returns:
        List of IRBlocks (empty for blank pages).
    """
    blocks: list[IRBlock] = []
    for segment in markdown.split("\n\n"):
        text = segment.strip()
        if not text or text == "[BLANK_PAGE]":
            continue
        # Drop horizontal rules — layout artifacts, not content
        if text in ("---", "***", "___"):
            continue

        if text.startswith("#"):
            block_type = BlockType.HEADING
            text = text.lstrip("#").strip()  # strip Markdown heading markers
            if not text:
                continue
        elif text.lstrip().startswith("|"):
            block_type = BlockType.TABLE
        elif text.startswith(("- ", "* ")) or _starts_with_ordinal(text):
            block_type = BlockType.LIST_ITEM
        elif text.startswith("Figure:"):
            block_type = BlockType.IMAGE_CAPTION
        else:
            block_type = BlockType.PARAGRAPH

        blocks.append(IRBlock(block_type=block_type, text=text, page=page_num))
    return blocks


class PdfVisionParser(BaseParser):
    """Parse complex or scanned PDFs with a multimodal (vision) LLM.

    Args:
        renderer: A ``BasePageRenderer`` implementation (e.g. PyMuPDF).
        model: OpenAI vision-capable model id. Defaults to ``"gpt-4o-mini"``.
        api_key: OpenAI API key. If None, reads ``OPENAI_API_KEY`` env var.
        max_pages: Maximum pages to process (None = all). A hard cap keeps
            a single pathological document from burning the API budget.
        max_tokens: Completion token cap per page. Defaults to 8192.
        max_workers: Maximum concurrent page-transcription requests. Vision
            calls are network-bound (one HTTP round-trip per page), so
            pages are transcribed in parallel via a thread pool. Defaults
            to 8.

    Attributes:
        PARSER_ID: Identifier written to ``ParseReport.parser_used``.
    """

    PARSER_ID = "pdf_vision"

    def __init__(
        self,
        renderer: BasePageRenderer,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_pages: Optional[int] = 50,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for PdfVisionParser. "
                "Install with: pip install openai"
            ) from exc

        self._renderer = renderer
        self._model = model
        self._max_pages = max_pages
        self._max_tokens = max_tokens
        self._max_workers = max_workers
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def supports(self, mime_type: str) -> bool:
        """Return True for PDF MIME types."""
        return mime_type in ("application/pdf", "application/x-pdf")

    def _encode_image(self, image: Any) -> str:
        """Encode a PIL Image as a base64 PNG data-URL payload."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _transcribe_page(self, image: Any) -> tuple[str, bool]:
        """Send one page image to the vision model and return its Markdown.

        Args:
            image: PIL Image of the rendered page.

        Returns:
            ``(markdown, truncated)`` where ``truncated`` is True when the
            model stopped early (output token cap or content-filter cut),
            meaning the transcript may be missing page content.
        """
        b64 = self._encode_image(image)
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PAGE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        )
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        truncated = finish_reason in _TRUNCATION_REASONS
        if truncated:
            logger.warning(
                "Vision transcript was truncated (finish_reason=%s) — part of "
                "the page is missing. Callers should keep the text-parser "
                "extraction for this page.",
                finish_reason,
            )
        return choice.message.content or "", truncated

    def parse_page_blocks(
        self, source_path: str, page_num: int
    ) -> tuple[list[IRBlock], bool]:
        """Transcribe a single page and return its IRBlocks.

        Used by the hybrid parser to escalate individual complex pages
        without paying for the whole document.

        Args:
            source_path: Absolute path to the PDF file.
            page_num: 1-based page number.

        Returns:
            ``(blocks, truncated)`` — IRBlocks for the page (empty for blank
            pages), and whether the transcript was cut off early. On a
            truncated transcript the blocks are partial and the caller should
            prefer the text-parser extraction for this page.
        """
        image = self._renderer.render(source_path, page_num)
        markdown, truncated = self._transcribe_page(image)
        return _markdown_to_blocks(markdown, page_num), truncated

    def parse_pages_parallel(
        self, source_path: str, page_nums: list[int]
    ) -> dict[int, tuple[list[IRBlock], bool, Optional[BaseException]]]:
        """Transcribe multiple pages concurrently via a thread pool.

        Each page is an independent render+API-call unit, so pages are
        network-bound rather than CPU-bound — a thread pool parallelizes
        the HTTP round-trips (the OpenAI client and PyMuPDF's per-call
        ``fitz.open()`` are both safe to use from multiple threads at once).

        Args:
            source_path: Absolute path to the PDF file.
            page_nums: 1-based page numbers to transcribe.

        Returns:
            Dict mapping page_num to ``(blocks, truncated, exception)``.
            ``exception`` is None on success; when set, ``blocks`` is empty
            and ``truncated`` is False — callers should fall back to
            another source for that page (e.g. text-parser blocks) exactly
            as they would for a sequential per-page exception.
        """
        results: dict[int, tuple[list[IRBlock], bool, Optional[BaseException]]] = {}
        if not page_nums:
            return results

        workers = max(1, min(self._max_workers, len(page_nums)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_page = {
                executor.submit(self.parse_page_blocks, source_path, page_num): page_num
                for page_num in page_nums
            }
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    blocks, truncated = future.result()
                    results[page_num] = (blocks, truncated, None)
                except Exception as exc:  # noqa: BLE001 - surfaced to caller
                    results[page_num] = ([], False, exc)
        return results

    def parse(self, source_path: str) -> Document:
        """Parse a PDF by transcribing each rendered page with the LLM.

        Args:
            source_path: Absolute path to the PDF file.

        Returns:
            ``Document`` with Markdown-derived IRBlocks and a ``ParseReport``
            recording ``parser_used="pdf_vision"``.

        Raises:
            Exception: Propagates renderer or API errors so the orchestrator
                can record the failure and fall through.
        """
        source_path = str(Path(source_path).resolve())
        n_pages = self._renderer.page_count(source_path)
        end_page = n_pages if self._max_pages is None else min(n_pages, self._max_pages)
        if end_page < n_pages:
            logger.warning(
                "pdf_vision capped at %d of %d pages for '%s'",
                end_page, n_pages, source_path,
            )

        page_nums = list(range(1, end_page + 1))
        page_results = self.parse_pages_parallel(source_path, page_nums)

        # Preserve the pre-parallelism contract: pdf_vision is the last-resort
        # tier (no further fallback exists below it), so any page failure
        # still fails the whole parse and propagates — the orchestrator uses
        # this to record the failure and fall through. We still get the
        # parallelism benefit for the (expected) all-success case.
        for page_num in page_nums:
            _, _, exc = page_results[page_num]
            if exc is not None:
                raise RuntimeError(
                    f"Vision parse failed for page {page_num}/{end_page} of "
                    f"'{source_path}': {exc}"
                ) from exc

        all_blocks: list[IRBlock] = []
        page_char_counts: list[int] = []

        for page_num in page_nums:
            page_blocks, _truncated, _exc = page_results[page_num]
            all_blocks.extend(page_blocks)
            page_chars = sum(len(b.text) for b in page_blocks)
            page_char_counts.append(page_chars)
            logger.debug(
                "Vision page %d/%d: %d blocks, %d chars",
                page_num, end_page, len(page_blocks), page_chars,
            )

        report = ParseReport(
            char_count=sum(page_char_counts),
            block_count=len(all_blocks),
            non_printable_ratio=0.0,
            repetition_score=0.0,
            parser_used=self.PARSER_ID,
            fallback_triggered=False,
        )

        # Stub doc_id from the path (fingerprint assigned by the pipeline)
        doc_id = sha256(source_path.encode()).hexdigest()

        return Document(
            doc_id=doc_id,
            source_path=source_path,
            mime_type="application/pdf",
            blocks=all_blocks,
            parse_report=report,
            metadata={
                "vision_model": self._model,
                "page_count": end_page,
                "page_char_counts": page_char_counts,
            },
        )
