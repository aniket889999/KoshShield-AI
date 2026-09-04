from dataclasses import dataclass
from pathlib import Path


class UnsupportedDocumentError(ValueError):
    pass


@dataclass(frozen=True)
class ValidatedDocument:
    filename: str
    media_type: str


SIGNATURES: tuple[tuple[bytes, str, set[str]], ...] = (
    (b"%PDF-", "application/pdf", {".pdf"}),
    (b"\x89PNG\r\n\x1a\n", "image/png", {".png"}),
    (b"\xff\xd8\xff", "image/jpeg", {".jpg", ".jpeg"}),
)


def validate_document(filename: str | None, content: bytes) -> ValidatedDocument:
    safe_name = Path(filename or "document").name.replace("\x00", "").strip()
    if not safe_name:
        safe_name = "document"

    suffix = Path(safe_name).suffix.lower()
    for signature, media_type, allowed_suffixes in SIGNATURES:
        if content.startswith(signature):
            if suffix not in allowed_suffixes:
                raise UnsupportedDocumentError("file extension does not match its content")
            return ValidatedDocument(filename=safe_name, media_type=media_type)

    raise UnsupportedDocumentError("only PDF, PNG, and JPEG documents are accepted")
