"""Strictly local-only, fail-closed operational health checks for KoshShield AI.

Enforces network boundary isolation:
1. Rejects any outbound network calls outside loopback (localhost/127.0.0.1) or
   internal Docker Compose service names.
2. Never triggers automatic model downloads, Docker pulls, package installation,
   or background process execution.
3. Fails closed with standardized, sanitized error codes when local dependencies
   are missing or offline.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from koshshield.config import Settings

# Allowlisted local and internal Docker Compose service names
ALLOWED_LOCAL_HOSTS: frozenset[str] = frozenset(
    {
        "localhost",
        "127.0.0.1",
        "::1",
        "0.0.0.0",
        "qdrant",
        "llama-server",
        "db",
        "postgres",
        "api",
        "web",
    }
)


def is_safe_local_endpoint(endpoint_url: str) -> bool:
    """Verifies that a target URL points strictly to loopback or local Compose services."""
    try:
        parsed = urlsplit(endpoint_url)
        hostname = (parsed.hostname or "").lower().strip()
        if not hostname:
            return False
        return hostname in ALLOWED_LOCAL_HOSTS
    except Exception:
        return False


class ServiceHealthProbe(BaseModel):
    service_id: str
    display_name: str
    status: str = Field(description="'ready', 'unavailable', or 'not_configured'")
    failure_code: str | None = None
    details: str
    local_only: bool = True
    endpoint_hostname: str | None = None


class LocalOperationalHealthResponse(BaseModel):
    boundary: str = "local-only"
    network_isolated: bool = True
    external_calls_blocked: bool = True
    overall_healthy: bool
    safe_mode_active: bool
    services: dict[str, ServiceHealthProbe]
    checked_at: str
    disclaimer: str = (
        "Strictly local-only operational health check. Probing is confined to loopback "
        "and internal Compose services. Outbound network requests are rejected. "
        "Missing models or services fail closed."
    )


async def probe_local_http(
    service_id: str,
    display_name: str,
    url: str,
    timeout_seconds: float = 0.5,
) -> ServiceHealthProbe:
    """Safely probes a local HTTP service endpoint, failing closed if non-local or unreachable."""
    if not is_safe_local_endpoint(url):
        return ServiceHealthProbe(
            service_id=service_id,
            display_name=display_name,
            status="unavailable",
            failure_code="NON_LOCAL_DESTINATION_REJECTED",
            details="Probe rejected: target destination is outside the local-only boundary.",
            endpoint_hostname=None,
        )

    parsed = urlsplit(url)
    hostname = parsed.hostname

    try:
        async with httpx.AsyncClient(
            timeout=min(timeout_seconds, 2.0),
            trust_env=False,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        return ServiceHealthProbe(
            service_id=service_id,
            display_name=display_name,
            status="ready",
            failure_code=None,
            details=f"Local {display_name} responsive and healthy.",
            endpoint_hostname=hostname,
        )
    except (TimeoutError, httpx.HTTPError, OSError):
        return ServiceHealthProbe(
            service_id=service_id,
            display_name=display_name,
            status="unavailable",
            failure_code="SERVICE_OFFLINE",
            details=f"Local {display_name} unreachable on local endpoint.",
            endpoint_hostname=hostname,
        )


async def evaluate_local_health(
    settings: Settings,
    session: Session | None = None,
) -> LocalOperationalHealthResponse:
    """Evaluates local service operational health with strict boundary enforcement."""
    services: dict[str, ServiceHealthProbe] = {}

    # 1. Database
    if session is not None:
        try:
            session.execute(text("SELECT 1"))
            services["database"] = ServiceHealthProbe(
                service_id="database",
                display_name="Metadata Database",
                status="ready",
                failure_code=None,
                details="Relational database responsive.",
                endpoint_hostname="local-db",
            )
        except Exception:
            services["database"] = ServiceHealthProbe(
                service_id="database",
                display_name="Metadata Database",
                status="unavailable",
                failure_code="DATABASE_UNAVAILABLE",
                details="Database connection failed.",
                endpoint_hostname="local-db",
            )
    else:
        services["database"] = ServiceHealthProbe(
            service_id="database",
            display_name="Metadata Database",
            status="unavailable",
            failure_code="SESSION_UNAVAILABLE",
            details="Database session was not provided.",
            endpoint_hostname=None,
        )

    # 2. Vault
    if settings.vault_configured:
        services["vault"] = ServiceHealthProbe(
            service_id="vault",
            display_name="Encrypted Document Vault",
            status="ready",
            failure_code=None,
            details="AES-256-GCM vault master key is configured.",
            endpoint_hostname="local-filesystem",
        )
    else:
        services["vault"] = ServiceHealthProbe(
            service_id="vault",
            display_name="Encrypted Document Vault",
            status="not_configured",
            failure_code="VAULT_KEY_UNCONFIGURED",
            details="Vault master key is unconfigured or invalid.",
            endpoint_hostname="local-filesystem",
        )

    # 3. Qdrant & Llama.cpp (async probe with strict localhost guard)
    qdrant_probe = probe_local_http(
        service_id="qdrant",
        display_name="Qdrant Vector Database",
        url=settings.qdrant_url,
        timeout_seconds=0.5,
    )
    llama_probe = probe_local_http(
        service_id="llama_cpp",
        display_name="llama.cpp Multimodal Server",
        url=f"{settings.llama_base_url}/models",
        timeout_seconds=0.5,
    )

    qdrant_res, llama_res = await asyncio.gather(qdrant_probe, llama_probe)
    services["qdrant"] = qdrant_res
    services["llama_cpp"] = llama_res

    # Overall health and safe-mode
    db_ready = services["database"].status == "ready"
    vault_ready = services["vault"].status == "ready"
    all_ready = all(s.status == "ready" for s in services.values())
    safe_mode = db_ready and vault_ready and not all_ready

    return LocalOperationalHealthResponse(
        boundary="local-only",
        network_isolated=True,
        external_calls_blocked=True,
        overall_healthy=all_ready,
        safe_mode_active=safe_mode,
        services=services,
        checked_at=datetime.now(UTC).isoformat(),
    )
