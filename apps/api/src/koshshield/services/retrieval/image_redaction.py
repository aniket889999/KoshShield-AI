import hashlib
import io
import logging
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from koshshield.models import DocumentPageRecord, RedactionFinding
from koshshield.security.pii.indian_pii import IndianPiiDetector
from koshshield.security.vault import EncryptedVault
from koshshield.services.extraction.paddle_ocr import PaddleOcrAdapter

logger = logging.getLogger(__name__)


def merge_bounding_boxes(
    boxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Merges overlapping or touching bounding boxes into non-overlapping rectangles."""
    if not boxes:
        return []

    current_boxes = list(boxes)
    merged_any = True
    while merged_any:
        merged_any = False
        new_boxes = []
        used = [False] * len(current_boxes)

        for i in range(len(current_boxes)):
            if used[i]:
                continue
            x0, y0, x1, y1 = current_boxes[i]
            for j in range(i + 1, len(current_boxes)):
                if used[j]:
                    continue
                ax0, ay0, ax1, ay1 = current_boxes[j]
                # Check overlap or touch
                if max(x0, ax0) <= min(x1, ax1) and max(y0, ay0) <= min(y1, ay1):
                    x0 = min(x0, ax0)
                    y0 = min(y0, ay0)
                    x1 = max(x1, ax1)
                    y1 = max(y1, ay1)
                    used[j] = True
                    merged_any = True
            new_boxes.append((x0, y0, x1, y1))
            used[i] = True

        current_boxes = new_boxes

    return current_boxes


def generate_masked_page_image_derivative(
    *,
    vault: EncryptedVault,
    document_id: str,
    page: DocumentPageRecord,
    accepted_findings: list[RedactionFinding],
    target_version: int,
    ocr_adapter: PaddleOcrAdapter | None = None,
    padding_points: float = 2.0,
) -> tuple[Path | None, str | None, str | None, str]:
    """Generates an opaque, privacy-safe masked page image derivative.

    Returns:
        (encrypted_path, masked_sha256, media_type, visual_privacy_status)

    visual_privacy_status values:
        - "APPROVED": Successfully generated, verified, and encrypted in vault.
        - "BLOCKED_UNLOCATED_PII": Accepted PII had missing, invalid, or unlocated bounding box.
        - "BLOCKED_RESIDUAL_PII": OCR verification detected sensitive PII on the derivative.
        - "NOT_APPLICABLE": Page has no original image.
    """
    if not page.encrypted_page_image_path or not page.page_image_sha256:
        return None, None, None, "NOT_APPLICABLE"

    raw_boxes: list[tuple[float, float, float, float]] = []
    page_width = float(page.width)
    page_height = float(page.height)

    # If there are accepted findings, every one must have a valid bbox
    for finding in accepted_findings:
        if not finding.bbox_json or not isinstance(finding.bbox_json, dict):
            logger.warning(
                "Accepted finding '%s' has no bbox_json on page %d; blocking visual derivative.",
                finding.id,
                page.page_number,
            )
            return None, None, None, "BLOCKED_UNLOCATED_PII"

        bbox = finding.bbox_json.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            logger.warning(
                "Accepted finding '%s' has invalid bbox format on page %d; blocking derivative.",
                finding.id,
                page.page_number,
            )
            return None, None, None, "BLOCKED_UNLOCATED_PII"

        try:
            x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        except (ValueError, TypeError):
            return None, None, None, "BLOCKED_UNLOCATED_PII"

        # Validate finite coordinates
        if any(math.isnan(v) or math.isinf(v) for v in (x0, y0, x1, y1)):
            return None, None, None, "BLOCKED_UNLOCATED_PII"

        # Validate ordered coordinates
        if x0 >= x1 or y0 >= y1:
            return None, None, None, "BLOCKED_UNLOCATED_PII"

        # Validate in-bounds
        if page_width > 0 and page_height > 0:
            if x1 <= 0 or y1 <= 0 or x0 >= page_width or y0 >= page_height:
                return None, None, None, "BLOCKED_UNLOCATED_PII"
            # Add bounded padding and clamp to page boundaries
            x0 = max(0.0, x0 - padding_points)
            y0 = max(0.0, y0 - padding_points)
            x1 = min(page_width, x1 + padding_points)
            y1 = min(page_height, y1 + padding_points)

        raw_boxes.append((x0, y0, x1, y1))

    # Decrypt original image transiently in memory
    try:
        page_artifact_id = f"{document_id}_p{page.page_number}_image"
        raw_image_bytes = vault.decrypt(
            document_id=page_artifact_id,
            evidence_hash=page.page_image_sha256,
            path=Path(page.encrypted_page_image_path),
        )
    except Exception as err:
        logger.error("Failed to decrypt original page image for doc '%s': %s", document_id, err)
        return None, None, None, "NOT_APPLICABLE"

    # Decode with Pillow, apply EXIF orientation, strip metadata, and normalize to RGB
    try:
        with Image.open(io.BytesIO(raw_image_bytes)) as pil_img:
            img = ImageOps.exif_transpose(pil_img)
            img = img.convert("RGB")
    except Exception as err:
        logger.error("Failed to decode page image with Pillow: %s", err)
        return None, None, None, "NOT_APPLICABLE"

    raster_width, raster_height = img.width, img.height
    scale_x = (raster_width / page_width) if page_width > 0 else 1.0
    scale_y = (raster_height / page_height) if page_height > 0 else 1.0

    # Scale boxes to raster coordinates
    raster_boxes: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in raw_boxes:
        rx0 = max(0.0, x0 * scale_x)
        ry0 = max(0.0, y0 * scale_y)
        rx1 = min(float(raster_width), x1 * scale_x)
        ry1 = min(float(raster_height), y1 * scale_y)
        raster_boxes.append((rx0, ry0, rx1, ry1))

    # Merge overlapping rectangles
    merged_boxes = merge_bounding_boxes(raster_boxes)

    # Draw opaque black redaction rectangles
    draw = ImageDraw.Draw(img)
    for bx0, by0, bx1, by1 in merged_boxes:
        ix0 = max(0, int(math.floor(bx0)))
        iy0 = max(0, int(math.floor(by0)))
        ix1 = min(raster_width, int(math.ceil(bx1)))
        iy1 = min(raster_height, int(math.ceil(by1)))
        draw.rectangle([ix0, iy0, ix1, iy1], fill=(0, 0, 0))

    # Save to normalized PNG bytes without metadata
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    masked_png_bytes = buf.getvalue()

    # Optional local OCR verification
    if ocr_adapter is not None:
        try:
            available, _ = ocr_adapter.is_available()
            if available:
                extracted = ocr_adapter.extract_page_from_image(
                    image_bytes=masked_png_bytes,
                    page_number=page.page_number,
                )
                detector = IndianPiiDetector()
                residual_findings = detector.detect(extracted.text)
                if residual_findings:
                    logger.warning(
                        "Residual PII detected on page %d visual derivative; blocking derivative.",
                        page.page_number,
                    )
                    return None, None, None, "BLOCKED_RESIDUAL_PII"
        except Exception as ocr_err:
            logger.warning("Local OCR verification encountered error: %s", ocr_err)

    # Encrypt masked derivative into vault
    masked_artifact_id = f"{document_id}_p{page.page_number}_v{target_version}_masked_image"
    masked_sha256 = hashlib.sha256(masked_png_bytes).hexdigest()
    encrypted_path = vault.encrypt(
        document_id=masked_artifact_id,
        evidence_hash=masked_sha256,
        plaintext=masked_png_bytes,
    )

    return encrypted_path, masked_sha256, "image/png", "APPROVED"
