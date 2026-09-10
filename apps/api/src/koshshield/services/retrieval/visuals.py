import hashlib
import uuid
from dataclasses import dataclass

from koshshield.security.pii.indian_pii import IndianPiiDetector

UUID_NAMESPACE_KOSHSHIELD_VISUAL = uuid.UUID("a91b4e20-7f61-4e4c-8c01-7bf9b9355f32")


@dataclass(frozen=True)
class VisualRegionDraft:
    region_id: str
    tenant_id: str
    document_id: str
    page_number: int
    region_sequence: int
    region_type: str
    source: str
    bbox_json: dict[str, object] | None
    caption_text: str
    caption_hash: str
    image_sha256: str | None
    masked_image_sha256: str | None
    redaction_version: int


def build_visual_region_drafts(
    *,
    tenant_id: str,
    document_id: str,
    masked_text: str,
    page_number: int,
    width: float,
    height: float,
    masked_image_sha256: str | None,
    redaction_version: int,
    real_extraction_regions: list[dict[str, object]] | None = None,
) -> list[VisualRegionDraft]:
    """Build truthful, masked visual metadata for cited retrieval evidence.

    Fakes (e.g. guessing tables/diagrams with heuristic fixed ratios) are removed.
    Only truthful full-page visual evidence or verified real extraction bounding boxes
    are emitted. Region IDs are deterministic UUIDv5 bound to tenant, document, page,
    sequence, and redaction version.
    """
    detector = IndianPiiDetector()
    normalized = " ".join(masked_text.split())
    lines = [line.strip() for line in masked_text.splitlines() if line.strip()]
    excerpt = normalized[:260] if normalized else "No masked text extracted from this page."

    caption = (
        f"Full page {page_number} visual evidence with {len(lines)} masked text line(s). "
        f"Preview: {excerpt}"
    )

    # Validate that caption contains only privacy-approved text
    pii_findings = detector.detect(caption)
    if pii_findings:
        for finding in pii_findings:
            caption = caption.replace(finding.raw_value, "[REDACTED]")
        if detector.detect(caption):
            caption = f"Full page {page_number} visual evidence (privacy-cleared)."

    caption_hash = hashlib.sha256(caption.encode("utf-8")).hexdigest()
    full_page_bbox = _bbox_payload((0.0, 0.0, width, height), width, height)

    seq = 0
    region_id = str(
        uuid.uuid5(
            UUID_NAMESPACE_KOSHSHIELD_VISUAL,
            f"{tenant_id}:{document_id}:{page_number}:{seq}:{redaction_version}",
        )
    )

    drafts = [
        VisualRegionDraft(
            region_id=region_id,
            tenant_id=tenant_id,
            document_id=document_id,
            page_number=page_number,
            region_sequence=seq,
            region_type="PAGE_IMAGE",
            source="page_raster",
            bbox_json=full_page_bbox,
            caption_text=caption,
            caption_hash=caption_hash,
            image_sha256=masked_image_sha256,
            masked_image_sha256=masked_image_sha256,
            redaction_version=redaction_version,
        )
    ]

    # Emit additional regions only if real extraction bounding boxes are present
    if real_extraction_regions:
        for item in real_extraction_regions:
            seq += 1
            r_type = str(item.get("region_type", "EXTRACTED_REGION"))
            r_source = str(item.get("source", "extraction_bbox"))
            r_bbox = item.get("bbox_json")
            r_caption = str(item.get("caption_text", f"Region {seq} on page {page_number}"))

            r_findings = detector.detect(r_caption)
            if r_findings:
                for f in r_findings:
                    r_caption = r_caption.replace(f.raw_value, "[REDACTED]")
                if detector.detect(r_caption):
                    r_caption = f"Region {seq} on page {page_number} (privacy-cleared)."

            r_caption_hash = hashlib.sha256(r_caption.encode("utf-8")).hexdigest()
            r_id = str(
                uuid.uuid5(
                    UUID_NAMESPACE_KOSHSHIELD_VISUAL,
                    f"{tenant_id}:{document_id}:{page_number}:{seq}:{redaction_version}",
                )
            )
            drafts.append(
                VisualRegionDraft(
                    region_id=r_id,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    page_number=page_number,
                    region_sequence=seq,
                    region_type=r_type,
                    source=r_source,
                    bbox_json=r_bbox if isinstance(r_bbox, dict) else None,
                    caption_text=r_caption,
                    caption_hash=r_caption_hash,
                    image_sha256=masked_image_sha256,
                    masked_image_sha256=masked_image_sha256,
                    redaction_version=redaction_version,
                )
            )

    return drafts


def _bbox_payload(
    bbox: tuple[float, float, float, float] | None,
    width: float,
    height: float,
) -> dict[str, object] | None:
    if bbox is None or width <= 0 or height <= 0:
        return None
    return {
        "bbox": [round(float(v), 2) for v in bbox],
        "page_width": round(width, 2),
        "page_height": round(height, 2),
    }
