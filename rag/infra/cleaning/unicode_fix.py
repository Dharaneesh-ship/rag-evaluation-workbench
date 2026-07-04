"""Unicode normalisation cleaner — fixes common encoding issues in extracted text."""

import unicodedata

from rag.core.contracts.ir_block import IRBlock
from rag.core.interfaces.cleaner import BaseCleaner

# Mapping of common "fancy" Unicode characters to their plain ASCII equivalents.
# Covers smart quotes, dashes, ellipsis, and common ligatures.
_CHAR_MAP: dict[str, str] = {
    "\u2018": "'",   # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK
    "\u201c": '"',   # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',   # RIGHT DOUBLE QUOTATION MARK
    "\u2013": "-",   # EN DASH
    "\u2014": "--",  # EM DASH
    "\u2026": "...", # HORIZONTAL ELLIPSIS
    "\u00a0": " ",   # NO-BREAK SPACE
    "\u00ad": "",    # SOFT HYPHEN (remove)
    "\u200b": "",    # ZERO WIDTH SPACE (remove)
    "\u200c": "",    # ZERO WIDTH NON-JOINER (remove)
    "\u200d": "",    # ZERO WIDTH JOINER (remove)
    "\u2060": "",    # WORD JOINER (remove)
    "\ufeff": "",    # ZERO WIDTH NO-BREAK SPACE / BOM (remove)
    "\ufb01": "fi",  # LATIN SMALL LIGATURE FI
    "\ufb02": "fl",  # LATIN SMALL LIGATURE FL
    "\ufb00": "ff",  # LATIN SMALL LIGATURE FF
    "\ufb03": "ffi", # LATIN SMALL LIGATURE FFI
    "\ufb04": "ffl", # LATIN SMALL LIGATURE FFL
}

_TRANSLATION_TABLE = str.maketrans(_CHAR_MAP)


def _normalize_text(text: str) -> str:
    """Apply Unicode normalisation and character substitution to a string.

    Steps:
    1. Apply NFC normalisation to canonically compose characters.
    2. Replace known fancy characters with plain ASCII equivalents.

    Args:
        text: Raw extracted text.

    Returns:
        Normalised text string.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_TRANSLATION_TABLE)
    return text


class UnicodeFixer(BaseCleaner):
    """Cleaner that normalises Unicode characters in IRBlock text.

    Converts smart quotes, en/em dashes, ligatures, and other common
    encoding artefacts to their plain ASCII counterparts. Operates
    in-place on block text without removing any blocks.

    Usage::

        cleaner = UnicodeFixer()
        cleaned_blocks = cleaner.clean(blocks)
    """

    def clean(self, blocks: list[IRBlock]) -> list[IRBlock]:
        """Normalise Unicode text in each block.

        Args:
            blocks: Input IRBlocks from the previous pipeline stage.

        Returns:
            New list of IRBlocks with normalised text. Block count
            is unchanged; only text content is modified.
        """
        result: list[IRBlock] = []
        for block in blocks:
            normalised = _normalize_text(block.text)
            if normalised == block.text:
                result.append(block)
            else:
                result.append(block.model_copy(update={"text": normalised}))
        return result
