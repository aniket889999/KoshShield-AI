import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class VisualRegionDraft:
    region_sequence: int
    region_type: str
    source: str
    bbox_json: dict[str, object] | None
    caption_text: str
    caption_hash: str
    image_sha256: str | None


def build_visual_region_drafts(
    *,
    masked_text: str,
    page_number: int,
    width: float,
    height: float,
    image_sha256: str | None,
) -> list[VisualRegionDraft]:
    """Build masked, local-only visual metadata for cited retrieval evidence."""
    normalized = " ".join(masked_text.split())
    lines = [line.strip() for line in masked_text.splitlines() if line.strip()]
    excerpt = normalized[:260] if normalized else "No masked text extracted from this page."

    drafts: list[tuple[str, str, tuple[float, float, float, float] | None]] = [
        (
            "PAGE_IMAGE",
            f"Full page {page_number} visual evidence with {len(lines)} masked text line(s). "
            f"Preview: {excerpt}",
            _scaled_bbox(width, height, 0.0, 0.0, 1.0, 1.0),
        )
    ]

    if _looks_like_table_or_form(lines):
        drafts.append(
            (
                "TABLE_OR_FORM_REGION",
                f"Structured table or form region on page {page_number}. "
                "Use this crop when the answer depends on tabular fields, clauses, "
                f"or approval columns. Masked preview: {excerpt}",
                _scaled_bbox(width, height, 0.04, 0.12, 0.96, 0.88),
            )
        )

    if _looks_like_diagram_or_map(normalized):
        drafts.append(
            (
                "DIAGRAM_OR_MAP_REGION",
                f"Diagram, map, flow, or figure region on page {page_number}. "
                "Use this crop when the answer depends on visual structure or layout labels. "
                f"Masked preview: {excerpt}",
                _scaled_bbox(width, height, 0.08, 0.18, 0.92, 0.82),
            )
        )

    return [
        VisualRegionDraft(
            region_sequence=idx,
            region_type=region_type,
            source="masked_layout_heuristic",
            bbox_json=_bbox_payload(bbox, width, height),
            caption_text=caption,
            caption_hash=hashlib.sha256(caption.encode("utf-8")).hexdigest(),
            image_sha256=image_sha256,
        )
        for idx, (region_type, caption, bbox) in enumerate(drafts)
    ]


def _looks_like_table_or_form(lines: list[str]) -> bool:
    if not lines:
        return False

    structured_lines = 0
    for line in lines:
        has_columns = "|" in line or "\t" in line or "  " in line
        has_field = ":" in line and len(line.split(":", maxsplit=1)[0]) <= 40
        if has_columns or has_field:
            structured_lines += 1

    keywords = ("table", "annexure", "schedule", "form", "amount", "date", "signature")
    keyword_hits = sum(1 for line in lines if any(keyword in line.lower() for keyword in keywords))
    return structured_lines >= 2 or keyword_hits >= 2


def _looks_like_diagram_or_map(text: str) -> bool:
    lowered = text.lower()
    keywords = ("diagram", "figure", "map", "chart", "workflow", "layout", "schematic")
    return any(keyword in lowered for keyword in keywords)


def _scaled_bbox(
    width: float,
    height: float,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> tuple[float, float, float, float] | None:
    if width <= 0 or height <= 0:
        return None
    return (
        round(width * left_ratio, 2),
        round(height * top_ratio, 2),
        round(width * right_ratio, 2),
        round(height * bottom_ratio, 2),
    )


def _bbox_payload(
    bbox: tuple[float, float, float, float] | None,
    width: float,
    height: float,
) -> dict[str, object] | None:
    if bbox is None:
        return None
    return {
        "bbox": list(bbox),
        "page_width": round(width, 2),
        "page_height": round(height, 2),
    }
