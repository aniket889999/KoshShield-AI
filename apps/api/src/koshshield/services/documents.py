import hashlib
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from koshshield.models import DocumentRecord
from koshshield.security.file_validation import validate_document
from koshshield.security.vault import EncryptedVault
from koshshield.services.audit import append_audit_event


def accept_document(
    session: Session,
    *,
    filename: str | None,
    content: bytes,
    actor_id: str,
    vault: EncryptedVault,
) -> DocumentRecord:
    validated = validate_document(filename, content)
    evidence_hash = hashlib.sha256(content).hexdigest()
    document_id = str(uuid4())
    vault_path: Path | None = None

    try:
        vault_path = vault.encrypt(document_id, evidence_hash, content)
        document = DocumentRecord(
            id=document_id,
            filename=validated.filename,
            media_type=validated.media_type,
            size_bytes=len(content),
            sha256=evidence_hash,
            vault_path=str(vault_path),
            status="encrypted",
        )
        session.add(document)
        append_audit_event(
            session,
            actor_id=actor_id,
            event_type="document.accepted",
            resource_type="document",
            resource_id=document_id,
            details={
                "media_type": validated.media_type,
                "size_bytes": len(content),
                "sha256": evidence_hash,
                "storage_state": "encrypted",
            },
        )
        session.commit()
        session.refresh(document)
        return document
    except Exception:
        session.rollback()
        if vault_path and vault_path.exists():
            vault_path.unlink()
        raise
