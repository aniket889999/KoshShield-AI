from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="KOSHSHIELD_",
        extra="ignore",
    )

    app_name: str = "KoshShield AI"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./data/koshshield.db"
    qdrant_url: str = "http://localhost:6333"
    llama_base_url: str = "http://localhost:8080/v1"
    vault_dir: Path = Path("./data/vault")
    master_key_base64: str | None = None
    auto_create_schema: bool = True
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, ge=1, le=100 * 1024 * 1024)
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    ocr_det_model_dir: Path | None = None
    ocr_rec_model_dir: Path | None = None
    ocr_cls_model_dir: Path | None = None
    pii_salt: str = "koshshield-default-dev-salt"
    max_extraction_pages: int = Field(default=50, ge=1, le=500)
    max_extracted_text_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)
    max_image_dimension: int = Field(default=4096, ge=256)
    high_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    @field_validator("qdrant_url", "llama_base_url")
    @classmethod
    def require_local_service_url(cls, value: str) -> str:
        parsed = urlparse(value)
        host = parsed.hostname or ""
        is_loopback = host in {"localhost", "127.0.0.1", "::1"}
        is_container_name = bool(host) and "." not in host
        if parsed.scheme not in {"http", "https"} or not (is_loopback or is_container_name):
            raise ValueError("service URLs must target localhost or a private container name")
        return value.rstrip("/")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def vault_configured(self) -> bool:
        return bool(self.master_key_base64)


@lru_cache
def get_settings() -> Settings:
    return Settings()
