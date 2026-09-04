import pymupdf

from koshshield.config import Settings
from koshshield.services.extraction.interfaces import (
    ExtractionError,
    ExtractionResult,
    OcrUnavailableError,
)
from koshshield.services.extraction.native_pdf import PyMuPdfExtractor
from koshshield.services.extraction.paddle_ocr import PaddleOcrAdapter


class UnifiedDocumentExtractor:
    """Selects native extraction or local OCR based on document type and content."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.native_extractor = PyMuPdfExtractor(
            max_pages=settings.max_extraction_pages,
            max_text_bytes=settings.max_extracted_text_bytes,
        )
        self.ocr_adapter = PaddleOcrAdapter(
            det_model_dir=settings.ocr_det_model_dir,
            rec_model_dir=settings.ocr_rec_model_dir,
            cls_model_dir=settings.ocr_cls_model_dir,
            max_image_dimension=settings.max_image_dimension,
        )

    def extract(self, filename: str, content: bytes, media_type: str) -> ExtractionResult:
        if media_type == "application/pdf":
            # 1. Try native PDF extraction first
            if self.native_extractor.has_sufficient_native_text(content):
                return self.native_extractor.extract(content)

            # 2. Scanned PDF fallback to OCR
            ocr_ready, reason = self.ocr_adapter.is_available()
            if not ocr_ready:
                raise OcrUnavailableError(
                    "PDF contains scanned or minimal native text, "
                    f"and local OCR is unavailable: {reason}"
                )

            # Rasterize pages and run OCR
            doc = pymupdf.open(stream=content, filetype="pdf")
            total_pages = len(doc)
            if total_pages > self.settings.max_extraction_pages:
                raise ExtractionError(
                    f"Page count ({total_pages}) exceeds limit "
                    f"({self.settings.max_extraction_pages})"
                )

            extracted_pages = []
            for idx in range(total_pages):
                page = doc[idx]
                pix = page.get_pixmap(dpi=150)
                image_bytes = pix.tobytes("png")
                page_result = self.ocr_adapter.extract_image(image_bytes, page_number=idx + 1)
                extracted_pages.append(page_result)

            return ExtractionResult(
                pages=extracted_pages,
                total_pages=total_pages,
                extraction_method="paddleocr",
            )

        elif media_type in {"image/png", "image/jpeg"}:
            ocr_ready, reason = self.ocr_adapter.is_available()
            if not ocr_ready:
                raise OcrUnavailableError(
                    f"Image extraction requires local OCR, which is currently unavailable: {reason}"
                )

            page_result = self.ocr_adapter.extract_image(content, page_number=1)
            return ExtractionResult(
                pages=[page_result],
                total_pages=1,
                extraction_method="paddleocr",
            )

        raise ExtractionError(f"Unsupported media type for extraction: {media_type}")
