import importlib.util
import io
import logging
from pathlib import Path
from typing import Any

from koshshield.services.extraction.interfaces import (
    ExtractedBlock,
    ExtractedPage,
    ExtractionLimitExceededError,
    OcrUnavailableError,
)

logger = logging.getLogger(__name__)


class PaddleOcrAdapter:
    """Real local adapter for PaddleOCR with lazy loading and air-gapped guarantees."""

    def __init__(
        self,
        det_model_dir: Path | None = None,
        rec_model_dir: Path | None = None,
        cls_model_dir: Path | None = None,
        max_image_dimension: int = 4096,
    ) -> None:
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.cls_model_dir = cls_model_dir
        self.max_image_dimension = max_image_dimension
        self._engine: Any = None
        self._initialized: bool = False

    def is_available(self) -> tuple[bool, str]:
        """Verifies if PaddleOCR package and local model paths are installed and ready."""
        # 1. Check if paddleocr is installed
        if importlib.util.find_spec("paddleocr") is None:
            return False, "paddleocr package is not installed"

        # 2. In an air-gapped system, local model directories must be explicitly configured
        if not self.det_model_dir or not self.rec_model_dir:
            return (
                False,
                "local OCR model paths not configured "
                "(KOSHSHIELD_OCR_DET_MODEL_DIR, KOSHSHIELD_OCR_REC_MODEL_DIR)",
            )

        if not Path(self.det_model_dir).exists():
            return False, f"OCR detection model dir does not exist: {self.det_model_dir}"
        if not Path(self.rec_model_dir).exists():
            return False, f"OCR recognition model dir does not exist: {self.rec_model_dir}"

        return True, "ready"

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        available, reason = self.is_available()
        if not available:
            raise OcrUnavailableError(f"Local OCR engine is unavailable: {reason}")

        try:
            # Lazy import to avoid loading heavy paddle dependencies unless required
            from paddleocr import PaddleOCR

            kwargs: dict[str, Any] = {
                "use_angle_cls": bool(self.cls_model_dir and Path(self.cls_model_dir).exists()),
                "det_model_dir": str(self.det_model_dir),
                "rec_model_dir": str(self.rec_model_dir),
                "show_log": False,
                "lang": "en",
            }
            if self.cls_model_dir and Path(self.cls_model_dir).exists():
                kwargs["cls_model_dir"] = str(self.cls_model_dir)

            self._engine = PaddleOCR(**kwargs)
            self._initialized = True
            return self._engine
        except Exception as exc:
            logger.error("Failed to initialize PaddleOCR engine: %s", exc)
            raise OcrUnavailableError(f"Failed to initialize local PaddleOCR: {exc}") from exc

    def extract_image(self, image_bytes: bytes, page_number: int = 1) -> ExtractedPage:
        """Extracts text and bounding boxes from an image file using local PaddleOCR."""
        available, reason = self.is_available()
        if not available:
            raise OcrUnavailableError(f"OCR is unavailable: {reason}")

        engine = self._get_engine()

        # Validate image dimensions
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
        except ImportError as exc:
            raise OcrUnavailableError("PIL (Pillow) is required for image processing") from exc
        except Exception as exc:
            raise ValueError("Invalid image content") from exc

        if width > self.max_image_dimension or height > self.max_image_dimension:
            raise ExtractionLimitExceededError(
                f"Image dimensions ({width}x{height}) exceed maximum allowed "
                f"({self.max_image_dimension})"
            )

        # Run OCR
        import numpy as np

        image_np = np.array(image)
        results = engine.ocr(image_np, cls=True)

        blocks: list[ExtractedBlock] = []
        text_lines: list[str] = []

        if results and results[0]:
            for idx, line in enumerate(results[0]):
                # line structure: [bbox, (text, confidence)]
                # bbox is [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                box_coords = line[0]
                text_info = line[1]
                line_text = text_info[0].strip() if text_info else ""
                if not line_text:
                    continue

                xs = [pt[0] for pt in box_coords]
                ys = [pt[1] for pt in box_coords]
                bbox = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

                blocks.append(
                    ExtractedBlock(
                        text=line_text,
                        bbox=bbox,
                        block_number=idx,
                    )
                )
                text_lines.append(line_text)

        full_text = "\n".join(text_lines)
        return ExtractedPage(
            page_number=page_number,
            width=float(width),
            height=float(height),
            text=full_text,
            blocks=blocks,
            extraction_method="paddleocr",
        )
