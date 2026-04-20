"""PDF → per-page images and text using PyMuPDF (fitz)."""

from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Render resolution — 150 DPI is a good balance of quality vs. token cost.
# Raise to 200 for scanned/handwritten tables; lower to 100 for large PDFs.
DEFAULT_DPI = 150
# Maximum pages sent to the VLM in a single call.  Pages beyond this are
# batched (see VLMExtractor).
MAX_PAGES_PER_BATCH = 20


@dataclass
class PageData:
    """All data extracted from a single PDF page."""

    page_number: int          # 1-based
    image_bytes: bytes        # PNG bytes at DEFAULT_DPI
    image_base64: str         # base64-encoded PNG (ready for Anthropic API)
    text: str                 # raw text extracted by PyMuPDF
    width_pt: float           # page width in PDF points
    height_pt: float          # page height in PDF points


@dataclass
class PDFDocument:
    """All pages from a PDF, plus document-level metadata."""

    path: Path
    pages: list[PageData] = field(default_factory=list)
    document_hash: str = ""   # SHA-256 of the raw file bytes
    page_count: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    def batches(self, batch_size: int = MAX_PAGES_PER_BATCH) -> list[list[PageData]]:
        """Split pages into batches for multi-call extraction."""
        return [
            self.pages[i : i + batch_size]
            for i in range(0, len(self.pages), batch_size)
        ]


class PDFLoader:
    """Convert a PDF file into per-page images and text."""

    def __init__(self, dpi: int = DEFAULT_DPI) -> None:
        self.dpi = dpi
        self._matrix = fitz.Matrix(dpi / 72, dpi / 72)

    def load(self, pdf_path: str | Path) -> PDFDocument:
        """Load a PDF and render every page to PNG.

        Parameters
        ----------
        pdf_path:
            Absolute or relative path to the PDF file.

        Returns
        -------
        PDFDocument
            Contains one PageData per page.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        raw_bytes = path.read_bytes()
        doc_hash = hashlib.sha256(raw_bytes).hexdigest()

        doc = fitz.open(str(path))
        logger.info("Loading PDF: %s (%d pages)", path.name, len(doc))

        pages: list[PageData] = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1

            # Render to PNG
            pixmap = page.get_pixmap(matrix=self._matrix, alpha=False)
            img_bytes = pixmap.tobytes("png")
            img_b64 = base64.standard_b64encode(img_bytes).decode()

            # Extract text (best-effort; may be empty for pure-image scans)
            text = page.get_text("text").strip()

            pages.append(
                PageData(
                    page_number=page_number,
                    image_bytes=img_bytes,
                    image_base64=img_b64,
                    text=text,
                    width_pt=page.rect.width,
                    height_pt=page.rect.height,
                )
            )
            logger.debug("  Page %d: %d chars text, image %d bytes", page_number, len(text), len(img_bytes))

        doc.close()

        # PDF metadata (title, author, etc.) — values may be empty strings
        raw_meta = doc.metadata or {}
        metadata = {k: v for k, v in raw_meta.items() if v}

        return PDFDocument(
            path=path,
            pages=pages,
            document_hash=doc_hash,
            page_count=len(pages),
            metadata=metadata,
        )
