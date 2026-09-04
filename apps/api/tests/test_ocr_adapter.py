from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from koshshield.services.extraction.interfaces import OcrUnavailableError
from koshshield.services.extraction.paddle_ocr import PaddleOcrAdapter


def test_ocr_unavailable_when_unconfigured() -> None:
    adapter = PaddleOcrAdapter(
        det_model_dir=None,
        rec_model_dir=None,
    )
    available, reason = adapter.is_available()
    assert available is False
    assert "not configured" in reason or "not installed" in reason

    with pytest.raises(OcrUnavailableError) as exc_info:
        adapter.extract_image(b"fake_image_bytes")
    assert "unavailable" in str(exc_info.value).lower()


def test_ocr_unavailable_when_model_dir_missing() -> None:
    adapter = PaddleOcrAdapter(
        det_model_dir=Path("/nonexistent/models/det"),
        rec_model_dir=Path("/nonexistent/models/rec"),
    )
    available, reason = adapter.is_available()
    assert available is False


def test_system_status_reflects_ocr_state(client: TestClient) -> None:
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    payload = response.json()
    assert "ocr" in payload
    assert payload["ocr"]["status"] in {"ready", "unavailable", "not_configured"}
