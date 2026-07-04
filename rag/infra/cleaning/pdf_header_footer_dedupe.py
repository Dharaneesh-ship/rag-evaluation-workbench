"""PDF header/footer deduplication cleaner — removes repeating page-level blocks."""

import re
from collections import Counter

from rag.core.contracts.ir_block import IRBlock
from rag.core.interfaces.cleaner import BaseCleaner

# Digits are masked before repetition matching so footers that embed the
# page number ("2 of 55 ... | 1/2026") still count as the same text on
# every page.
_DIGIT_RE = re.compile(r"\d+")


def _normalized_key(text: str) -> str:
    """Return the repetition-matching key for a block text.

    Masks digit runs and collapses whitespace, so per-page variations
    (page numbers, dates, document ids) don't defeat the dedupe.
    """
    masked = _DIGIT_RE.sub("#", text.strip())
    return " ".join(masked.split())

# A block is considered a repeating header/footer if it appears on at least
# this fraction of total pages in the document.
_DEFAULT_PAGE_FRACTION_THRESHOLD = 0.5

# Minimum number of pages required before the heuristic applies.
_MIN_PAGES = 2


class PdfHeaderFooterDedupe(BaseCleaner):
    """Cleaner that strips repeating PDF headers and footers.

    Identifies text blocks that appear on a significant fraction of pages
    (controlled by ``page_fraction_threshold``) and removes all occurrences.
    This targets running titles, page numbers, and boilerplate footers that
    parsers include verbatim on every page.

    The heuristic requires blocks to have a ``page`` attribute set (i.e.,
    produced by a page-aware parser such as PdfPyMuPDFParser). Blocks
    without a page number are passed through unchanged.

    Usage::

        cleaner = PdfHeaderFooterDedupe()
        cleaned_blocks = cleaner.clean(blocks)

        # More aggressive — remove blocks appearing on 30%+ of pages
        cleaner = PdfHeaderFooterDedupe(page_fraction_threshold=0.3)

    Args:
        page_fraction_threshold: Fraction of distinct pages a block text must
            appear on to be considered a repeating header/footer.
            Defaults to 0.5 (appears on at least half of all pages).
    """

    def __init__(self, page_fraction_threshold: float = _DEFAULT_PAGE_FRACTION_THRESHOLD) -> None:
        self._threshold = page_fraction_threshold

    def clean(self, blocks: list[IRBlock]) -> list[IRBlock]:
        """Remove blocks identified as repeating PDF headers or footers.

        Args:
            blocks: Input IRBlocks from the previous pipeline stage.

        Returns:
            Filtered list with repeating header/footer blocks removed.
        """
        # Collect blocks that have page numbers
        paged = [b for b in blocks if b.page is not None]
        if not paged:
            return blocks

        page_numbers = {b.page for b in paged}
        total_pages = len(page_numbers)
        if total_pages < _MIN_PAGES:
            return blocks

        # Count on how many distinct pages each (digit-masked) text appears
        text_to_pages: dict[str, set[int]] = {}
        for block in paged:
            key = _normalized_key(block.text)
            if key:
                text_to_pages.setdefault(key, set()).add(block.page)  # type: ignore[arg-type]

        # Identify texts that appear on strictly more than threshold fraction of pages.
        # Using > (not >=) avoids removing content that appears on exactly half the
        # pages in a two-page document, which is almost certainly not a header/footer.
        repeating: set[str] = {
            text
            for text, pages in text_to_pages.items()
            if len(pages) / total_pages > self._threshold
        }

        if not repeating:
            return blocks

        return [b for b in blocks if _normalized_key(b.text) not in repeating]
