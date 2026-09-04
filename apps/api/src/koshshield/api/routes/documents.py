from typing import Annotated

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from koshshield.config import Settings, get_settings
from koshshield.database import get_db
from koshshield.models import DocumentRecord
from koshshield.schemas import DocumentResponse
from koshshield.security.file_validation import UnsupportedDocumentError
from koshshield.security.vault import EncryptedVault, VaultConfigurationError
from koshshield.services.documents import accept_document

router = APIRouter()
SessionDependency = Annotated[Session, Depends(get_db)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    session: SessionDependency,
    limit: int = 20,
) -> list[DocumentRecord]:
    safe_limit = max(1, min(limit, 100))
    return list(
        session.scalars(
            select(DocumentRecord).order_by(DocumentRecord.created_at.desc()).limit(safe_limit)
        )
    )


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF, PNG, or JPEG document")],
    session: SessionDependency,
    settings: SettingsDependency,
    actor_id: Annotated[str, Header(alias="X-Actor-ID", max_length=120)] = "local-demo-user",
) -> DocumentRecord:
    content = await file.read(settings.max_upload_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="document is empty")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="document exceeds the configured size limit")

    try:
        vault = EncryptedVault(settings.vault_dir, settings.master_key_base64)
        return accept_document(
            session,
            filename=file.filename,
            content=content,
            actor_id=actor_id,
            vault=vault,
        )
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except VaultConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
