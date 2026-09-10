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
    demo_mode: bool = Field(default=True)
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
    agent_integrity_secret: str = "koshshield-default-dev-agent-integrity"
    max_extraction_pages: int = Field(default=50, ge=1, le=500)
    max_extracted_text_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)
    max_image_dimension: int = Field(default=4096, ge=256)
    high_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    embedding_model_dir: Path | None = None
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=16, ge=1, le=128)
    qdrant_collection: str = "koshshield_masked_docs"
    retrieval_rrf_k: int = Field(default=60, ge=1, le=1000)
    max_search_top_k: int = Field(default=50, ge=1, le=100)
    tool_runner_image: str = "koshshield-tool-runner:0.1.0"
    tool_runner_timeout_seconds: int = Field(default=10, ge=1, le=60)
    tool_runner_max_output_bytes: int = Field(default=64 * 1024, ge=1024, le=1024 * 1024)
    enable_multimodal_answering: bool = Field(default=False)
    llama_cpp_model_id: str = "qwen3-vl-4b-instruct"
    llama_cpp_release: str = "v0.4.0"
    llama_cpp_build: str = "b10809"
    llama_cpp_commit: str = "5266f24"
    llama_cpp_service_name: str = "llama-server"
    llama_cpp_max_tokens: int = Field(default=512, ge=64, le=2048)
    llama_cpp_timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    llama_cpp_model_path: Path | None = None
    llama_cpp_mmproj_path: Path | None = None

    @field_validator("qdrant_url", "llama_base_url")
    @classmethod
    def require_local_service_url(cls, value: str) -> str:
        parsed = urlparse(value)
        host = parsed.hostname or ""
        allowed_hosts = {"localhost", "127.0.0.1", "::1", "llama-server", "qdrant"}
        if parsed.scheme not in {"http", "https"} or host not in allowed_hosts:
            raise ValueError(
                "service URLs must strictly target localhost (127.0.0.1, ::1) or "
                "explicit local service names (llama-server, qdrant)"
            )
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
