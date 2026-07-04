"""PyMuPDF page renderer — converts PDF pages to PIL Images — Task 13.2.

Usage::

    from rag.infra.ocr.renderer_pymupdf import PyMuPDFPageRenderer
    renderer = PyMuPDFPageRenderer(dpi=150)
    image = renderer.render("/path/to/doc.pdf", page_num=1)
    images = renderer.render_range("/path/to/doc.pdf", start=1, end=3)
"""

from __future__ import annotations

import logging
from pathlib import Path

from rag.core.interfaces.page_renderer import BasePageRenderer

logger = logging.getLogger(__name__)


class PyMuPDFPageRenderer(BasePageRenderer):
    """Render PDF pages to PIL Images using PyMuPDF (fitz).

    Args:
        dpi: Rendering resolution in dots-per-inch.  150 is fast and
            sufficient for most OCR engines; 300 produces higher-quality
            output for small fonts.  Defaults to 150.
        colorspace: PyMuPDF colorspace name.  ``"RGB"`` for colour,
            ``"GRAY"`` for greyscale.  Defaults to ``"RGB"``.

    Raises:
        ImportError: At instantiation if ``pymupdf`` (fitz) is not installed.
    """

    def __init__(self, dpi: int = 150, colorspace: str = "RGB") -> None:
        try:
            import fitz  # noqa: F401  (just verify it's available)
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is not installed. Install it with:\n  pip install pymupdf"
            ) from exc
        self._dpi = dpi
        self._colorspace = colorspace

    def _open(self, pdf_path: str):  # type: ignore[return]
        """Open a PDF file, raising FileNotFoundError if missing."""
        import fitz

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        return fitz.open(str(path))

    def page_count(self, pdf_path: str) -> int:
        """Return the total number of pages in a PDF.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            Total page count.
        """
        doc = self._open(pdf_path)
        try:
            return len(doc)
        finally:
            doc.close()

    def render(self, pdf_path: str, page_num: int) -> object:
        """Render a single PDF page to a PIL Image.

        Args:
            pdf_path: Absolute path to the PDF file.
            page_num: 1-based page number.

        Returns:
            PIL ``Image`` in the configured colorspace.

        Raises:
            IndexError: If page_num is out of range.
        """
        from PIL import Image  # type: ignore[import]

        doc = self._open(pdf_path)
        try:
            n_pages = len(doc)
            if page_num < 1 or page_num > n_pages:
                raise IndexError(
                    f"Page {page_num} out of range for PDF with {n_pages} pages"
                )
            import fitz

            page = doc[page_num - 1]
            zoom = self._dpi / 72.0  # 72 DPI is PyMuPDF's default
            mat = fitz.Matrix(zoom, zoom)
            try:
                # fitz >= 1.18
                pix = page.get_pixmap(matrix=mat, colorspace=self._colorspace)  # type: ignore[attr-defined]
            except AttributeError:
                # fitz < 1.18 fallback
                cs = fitz.csRGB if self._colorspace == "RGB" else fitz.csGRAY
                pix = page.getPixmap(matrix=mat, colorspace=cs)  # type: ignore[attr-defined]

            mode = "RGB" if self._colorspace == "RGB" else "L"
            image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            logger.debug(
                "Rendered page %d of '%s' at %d DPI → %dx%d",
                page_num, pdf_path, self._dpi, pix.width, pix.height,
            )
            return image
        finally:
            doc.close()

    def render_range(self, pdf_path: str, start: int, end: int) -> list[object]:
        """Render a range of pages to PIL Images.

        Args:
            pdf_path: Absolute path to the PDF file.
            start: 1-based first page (inclusive).
            end: 1-based last page (inclusive).

        Returns:
            List of PIL Images, one per page.
        """
        return [self.render(pdf_path, p) for p in range(start, end + 1)]
