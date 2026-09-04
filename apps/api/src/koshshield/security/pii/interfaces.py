from dataclasses import dataclass
from typing import Protocol


@dataclass
class DetectedPii:
    finding_type: str
    value: str  # Transient in-memory only; hashed before persistence
    start: int
    end: int
    confidence: float
    detection_source: str
    page_number: int = 1
    bbox: tuple[float, float, float, float] | None = None


class PiiDetector(Protocol):
    """Protocol for local PII detectors (custom recognizers, Presidio, etc.)."""

    def detect(
        self,
        text: str,
        page_number: int = 1,
        blocks: list[tuple[float, float, float, float, str]] | None = None,
    ) -> list[DetectedPii]: ...
