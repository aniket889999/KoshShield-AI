from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ExtractedBlock:
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    block_number: int


@dataclass
class ExtractedPage:
    page_number: int  # 1-indexed
    width: float
    height: float
    text: str
    blocks: list[ExtractedBlock] = field(default_factory=list)
    extraction_method: str = "native_pdf"


@dataclass
class ExtractionResult:
    pages: list[ExtractedPage]
    total_pages: int
    extraction_method: str


class ExtractionError(Exception):
    """Base error for document extraction failures."""


class OcrUnavailableError(ExtractionError):
    """Raised when OCR is required but unavailable in the air-gapped environment."""


class ExtractionLimitExceededError(ExtractionError):
    """Raised when page count or content size limits are breached."""


class NativePdfExtractor(Protocol):
    def can_extract(self, content: bytes) -> bool: ...

    def extract(self, content: bytes) -> ExtractionResult: ...


class OcrEngine(Protocol):
    def is_available(self) -> tuple[bool, str]: ...

    def extract_image(self, image_bytes: bytes, page_number: int = 1) -> ExtractedPage: ...


class DocumentExtractor(Protocol):
    def extract(self, filename: str, content: bytes, media_type: str) -> ExtractionResult: ...
