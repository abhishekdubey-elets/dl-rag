"""PDF → markdown-ish text extraction.

Text layers are read with ``pypdf``; tables are detected with ``pdfplumber`` and
rendered as GitHub markdown tables. Pages with little or no extractable text
fall back to OCR (``pdf2image`` rasterisation + ``pytesseract``), which is lazily
imported and fully guarded — if the libraries or their system binaries
(poppler/tesseract) are missing, OCR is skipped with a warning. Blocking work
runs in a worker thread; the extractor never raises.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

import pdfplumber
from pypdf import PdfReader

from dl_rag.config import Settings
from dl_rag.logging_config import get_logger

logger = get_logger(__name__)

# Below this many characters of extractable text, a page is treated as scanned.
_OCR_TEXT_THRESHOLD = 20
_OCR_DPI = 200


class PDFProcessor:
    """Extract clean text (and tables) from PDFs, OCR-ing scanned pages."""

    def __init__(self, settings: Settings, ocr_enabled: bool = True) -> None:
        self._settings = settings
        self.ocr_enabled = ocr_enabled
        self._ocr_warned = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def extract(self, source: str | bytes) -> str:
        """Extract markdown-ish text from a PDF path or raw bytes (never raises)."""
        try:
            return await asyncio.to_thread(self._extract_sync, source)
        except Exception as exc:  # noqa: BLE001 - contract: return "" on failure
            logger.warning("pdf.extract_failed", error=str(exc))
            return ""

    # ------------------------------------------------------------------ #
    # Blocking implementation
    # ------------------------------------------------------------------ #
    def _extract_sync(self, source: str | bytes) -> str:
        try:
            data = self._load_bytes(source)
        except Exception as exc:  # noqa: BLE001 - bad path/type
            logger.warning("pdf.load_failed", error=str(exc))
            return ""
        if not data:
            return ""

        page_texts = self._extract_text_layers(data)
        page_tables = self._extract_tables(data)

        num_pages = max(len(page_texts), len(page_tables))
        if num_pages == 0:
            return ""

        rendered_pages: list[str] = []
        for index in range(num_pages):
            text = page_texts[index] if index < len(page_texts) else ""
            tables = page_tables[index] if index < len(page_tables) else []

            if len(text.strip()) < _OCR_TEXT_THRESHOLD and self.ocr_enabled:
                ocr_text = self._ocr_page(data, index + 1)
                if ocr_text.strip():
                    text = ocr_text

            tables_md = self._tables_to_markdown(tables)
            body_parts = [part for part in (text.strip(), tables_md.strip()) if part]
            if body_parts:
                header = f"--- Page {index + 1} ---"
                rendered_pages.append(header + "\n\n" + "\n\n".join(body_parts))

        return "\n\n".join(rendered_pages)

    # ------------------------------------------------------------------ #
    # Stage helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_bytes(source: str | bytes) -> bytes:
        if isinstance(source, bytes):
            return source
        if isinstance(source, (bytearray, memoryview)):
            return bytes(source)
        if isinstance(source, str):
            with open(source, "rb") as handle:
                return handle.read()
        raise TypeError(f"Unsupported PDF source type: {type(source)!r}")

    def _extract_text_layers(self, data: bytes) -> list[str]:
        texts: list[str] = []
        try:
            reader = PdfReader(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001 - encrypted/corrupt file
            logger.warning("pdf.pypdf_open_failed", error=str(exc))
            return texts
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception as exc:  # noqa: BLE001 - per-page extraction failure
                logger.debug("pdf.page_text_failed", error=str(exc))
                texts.append("")
        return texts

    def _extract_tables(self, data: bytes) -> list[list[Any]]:
        page_tables: list[list[Any]] = []
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    try:
                        tables = page.extract_tables()
                    except Exception as exc:  # noqa: BLE001 - per-page table failure
                        logger.debug("pdf.page_tables_failed", error=str(exc))
                        tables = []
                    page_tables.append(list(tables or []))
        except Exception as exc:  # noqa: BLE001 - pdfplumber open failure
            logger.warning("pdf.pdfplumber_open_failed", error=str(exc))
        return page_tables

    def _ocr_page(self, data: bytes, page_number: int) -> str:
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
        except ImportError:
            if not self._ocr_warned:
                logger.warning("pdf.ocr_unavailable", reason="import_error")
                self._ocr_warned = True
            return ""

        try:
            images = convert_from_bytes(
                data, dpi=_OCR_DPI, first_page=page_number, last_page=page_number
            )
        except Exception as exc:  # noqa: BLE001 - poppler missing / render error
            if not self._ocr_warned:
                logger.warning("pdf.ocr_render_failed", page=page_number, error=str(exc))
                self._ocr_warned = True
            return ""
        if not images:
            return ""

        try:
            return str(pytesseract.image_to_string(images[0]) or "")
        except Exception as exc:  # noqa: BLE001 - tesseract binary missing / error
            if not self._ocr_warned:
                logger.warning("pdf.ocr_failed", page=page_number, error=str(exc))
                self._ocr_warned = True
            return ""

    def _tables_to_markdown(self, tables: list[Any]) -> str:
        """Render extracted tables as GitHub-flavoured markdown tables."""
        if not tables:
            return ""
        blocks: list[str] = []
        for table in tables:
            if not table:
                continue
            rows: list[list[str]] = []
            for row in table:
                if row is None:
                    continue
                cells = [
                    ("" if cell is None else str(cell)).replace("\n", " ").replace("|", "\\|").strip()
                    for cell in row
                ]
                if any(cells):
                    rows.append(cells)
            if not rows:
                continue
            width = max(len(row) for row in rows)
            normalized = [row + [""] * (width - len(row)) for row in rows]
            header = normalized[0]
            lines = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(["---"] * width) + " |",
            ]
            for row in normalized[1:]:
                lines.append("| " + " | ".join(row) + " |")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
