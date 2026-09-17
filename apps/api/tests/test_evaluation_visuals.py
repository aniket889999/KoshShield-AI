import base64
import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from koshshield.evaluation.fixtures import FixtureValidationError
from koshshield.evaluation.visuals import load_probe_images

PNG = b"\x89PNG\r\n\x1a\nsynthetic header; client separately decodes images"


def page(**changes):
    return SimpleNamespace(
        **(
            dict(
                document_id="doc-1",
                page_number=2,
                visual_privacy_status="APPROVED",
                visual_redaction_version=4,
                encrypted_masked_page_image_path="/owned/masked.enc",
                masked_page_image_sha256=hashlib.sha256(PNG).hexdigest(),
                masked_page_image_media_type="image/png",
            )
            | changes
        )
    )


def load(session, vault, evidence=None):
    return load_probe_images(
        session=session,
        vault=vault,
        evidence=evidence if evidence is not None else [{"document_id": "doc-1", "page_number": 2}],
        document_id="doc-1",
        redaction_version=4,
        required_pages=[2],
    )


def test_only_required_masked_page_is_sent() -> None:
    session, vault = MagicMock(), MagicMock()
    session.scalar.return_value = page()
    vault.decrypt.return_value = PNG
    result = load(
        session,
        vault,
        [
            {"document_id": "doc-1", "page_number": 1},
            {"document_id": "doc-1", "page_number": 2},
        ],
    )
    assert result == ["data:image/png;base64," + base64.b64encode(PNG).decode("ascii")]
    assert vault.decrypt.call_args.args[0] == "doc-1_p2_masked"
    assert session.scalar.call_count == 1


@pytest.mark.parametrize(
    "change",
    [
        {"document_id": "foreign"},
        {"page_number": 1},
        {"visual_privacy_status": "BLOCKED"},
        {"visual_redaction_version": 3},
        {"encrypted_masked_page_image_path": None},
        {"masked_page_image_sha256": None},
        {"masked_page_image_media_type": "image/jpeg"},
    ],
)
def test_unapproved_or_stale_image_never_reaches_decryption(change: dict) -> None:
    session, vault = MagicMock(), MagicMock()
    session.scalar.return_value = page(**change)
    with pytest.raises(FixtureValidationError, match="VISUAL_PROBE_NOT_APPROVED"):
        load(session, vault)
    vault.decrypt.assert_not_called()


def test_retrieval_must_include_required_page_before_db_lookup() -> None:
    session, vault = MagicMock(), MagicMock()
    with pytest.raises(FixtureValidationError, match="VISUAL_PROBE_EVIDENCE_MISSING"):
        load(session, vault, [{"document_id": "doc-1", "page_number": 1}])
    session.scalar.assert_not_called()
    vault.decrypt.assert_not_called()


def test_corrupt_plaintext_hash_is_rejected() -> None:
    session, vault = MagicMock(), MagicMock()
    session.scalar.return_value = page()
    vault.decrypt.return_value = PNG + b"tampered"
    with pytest.raises(FixtureValidationError, match="VISUAL_PROBE_IMAGE_INVALID"):
        load(session, vault)


def test_decryption_failure_has_no_raw_exception() -> None:
    session, vault = MagicMock(), MagicMock()
    session.scalar.return_value = page()
    vault.decrypt.side_effect = RuntimeError("/private/secret with credential")
    with pytest.raises(FixtureValidationError, match="^VISUAL_PROBE_DECRYPT_FAILED$"):
        load(session, vault)
