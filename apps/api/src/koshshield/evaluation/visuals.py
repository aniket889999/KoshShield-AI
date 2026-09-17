"""Require the intended, approved masked pages for a synthetic vision probe."""

import base64
import hashlib
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.evaluation.fixtures import FixtureValidationError
from koshshield.models import DocumentPageRecord
from koshshield.security.vault import EncryptedVault


def load_probe_images(
    *,
    session: Session,
    vault: EncryptedVault,
    evidence: list[dict],
    document_id: str,
    redaction_version: int,
    required_pages: list[int],
) -> list[str]:
    """Call only after evidence tenant/document/version validation in the smoke runner."""
    available_pages = {
        item["page_number"] for item in evidence if item["document_id"] == document_id
    }
    if not required_pages or not set(required_pages) <= available_pages:
        raise FixtureValidationError("VISUAL_PROBE_EVIDENCE_MISSING")
    images = []
    for page_number in sorted(set(required_pages)):
        page = session.scalar(
            select(DocumentPageRecord).where(
                DocumentPageRecord.document_id == document_id,
                DocumentPageRecord.page_number == page_number,
            )
        )
        if (
            page is None
            or page.document_id != document_id
            or page.page_number != page_number
            or page.visual_privacy_status != "APPROVED"
            or page.visual_redaction_version != redaction_version
            or not page.encrypted_masked_page_image_path
            or not page.masked_page_image_sha256
            or page.masked_page_image_media_type != "image/png"
        ):
            raise FixtureValidationError("VISUAL_PROBE_NOT_APPROVED")
        try:
            content = vault.decrypt(
                f"{document_id}_p{page_number}_masked",
                page.masked_page_image_sha256,
                Path(page.encrypted_masked_page_image_path),
            )
        except Exception:
            raise FixtureValidationError("VISUAL_PROBE_DECRYPT_FAILED") from None
        if (
            len(content) > 16 * 1024 * 1024
            or not content.startswith(b"\x89PNG\r\n\x1a\n")
            or hashlib.sha256(content).hexdigest() != page.masked_page_image_sha256
        ):
            raise FixtureValidationError("VISUAL_PROBE_IMAGE_INVALID")
        images.append("data:image/png;base64," + base64.b64encode(content).decode("ascii"))
    return images
