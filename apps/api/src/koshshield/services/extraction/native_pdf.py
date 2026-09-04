import pymupdf

from koshshield.services.extraction.interfaces import (
    ExtractedBlock,
    ExtractedPage,
    ExtractionLimitExceededError,
    ExtractionResult,
)


class PyMuPdfExtractor:
    """Extracts text, block coordinates, and dimensions from PDFs using PyMuPDF."""

    def __init__(
        self,
        max_pages: int = 50,
        max_text_bytes: int = 5 * 1024 * 1024,
        min_native_chars_per_page: int = 20,
    ) -> None:
        self.max_pages = max_pages
        self.max_text_bytes = max_text_bytes
        self.min_native_chars_per_page = min_native_chars_per_page

    def has_sufficient_native_text(self, content: bytes) -> bool:
        """Determines if the PDF contains sufficient native text to extract without OCR."""
        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
            if len(doc) == 0:
                return False
            # Check the first few pages
            pages_to_check = min(len(doc), 3)
            total_chars = 0
            for i in range(pages_to_check):
                text = doc[i].get_text()
                total_chars += len(text.strip())
            return total_chars >= (self.min_native_chars_per_page * pages_to_check)
        except Exception:
            return False

    def extract(self, content: bytes) -> ExtractionResult:
        doc = pymupdf.open(stream=content, filetype="pdf")
        total_pages = len(doc)
        if total_pages > self.max_pages:
            raise ExtractionLimitExceededError(
                f"Document page count ({total_pages}) exceeds configured limit ({self.max_pages})"
            )

        pages: list[ExtractedPage] = []
        total_extracted_bytes = 0

        for page_idx in range(total_pages):
            page = doc[page_idx]
            rect = page.rect
            raw_blocks = page.get_text("blocks")

            page_blocks: list[ExtractedBlock] = []
            page_text_parts: list[str] = []

            for raw in raw_blocks:
                # raw: (x0, y0, x1, y1, text, block_no, block_type)
                if len(raw) >= 7 and raw[6] != 0:
                    # Skip image blocks (block_type 1)
                    continue
                block_text = raw[4] if len(raw) > 4 else ""
                if not block_text.strip():
                    continue
                bbox = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
                block_no = int(raw[5]) if len(raw) > 5 else len(page_blocks)
                page_blocks.append(
                    ExtractedBlock(
                        text=block_text,
                        bbox=bbox,
                        block_number=block_no,
                    )
                )
                page_text_parts.append(block_text)

            page_full_text = "\n".join(page_text_parts)
            total_extracted_bytes += len(page_full_text.encode("utf-8"))
            if total_extracted_bytes > self.max_text_bytes:
                raise ExtractionLimitExceededError(
                    f"Extracted text exceeds size limit ({self.max_text_bytes} bytes)"
                )

            pages.append(
                ExtractedPage(
                    page_number=page_idx + 1,
                    width=float(rect.width),
                    height=float(rect.height),
                    text=page_full_text,
                    blocks=page_blocks,
                    extraction_method="native_pdf",
                )
            )

        return ExtractionResult(
            pages=pages,
            total_pages=total_pages,
            extraction_method="native_pdf",
        )
