"""Authenticated, tenant-safe demo-readiness API endpoint.

Reports local component readiness, configuration status, artifact availability,
and executable state. Failures are reported with safe, standardized failure codes.
Absolute filesystem paths, model paths, secrets, passwords, raw exceptions, and
prompts are strictly stripped and never exposed.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from koshshield.config import Settings, get_settings
from koshshield.database import get_db
from koshshield.security.context import RequestContextDependency
from koshshield.services.readiness import (
    ComponentReadiness,
    DemoReadinessReport,
    evaluate_demo_readiness,
    sanitize_text,
)

router = APIRouter()
SessionDependency = Annotated[Session, Depends(get_db)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]


class TenantSafeReadinessResponse(DemoReadinessReport):
    tenant_id: str


@router.get(
    "/readiness",
    response_model=TenantSafeReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Get tenant-scoped local demo-readiness report",
)
def get_demo_readiness(
    context: RequestContextDependency,
    session: SessionDependency,
    settings: SettingsDependency,
    probe_services: Annotated[
        bool,
        Query(
            description="Run live local loopback service probes (requires admin/auditor)",
        ),
    ] = False,
) -> TenantSafeReadinessResponse:
    """Returns local demo readiness without exposing internal paths or credentials.

    Active service loopback probes require 'admin', 'auditor', or 'operator' roles.
    """
    if probe_services and not (
        context.has_role("admin")
        or context.has_role("auditor")
        or context.has_role("operator")
        or context.has_role("reviewer")
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: active service probing requires admin or auditor role",
        )

    try:
        report = evaluate_demo_readiness(
            settings=settings,
            session=session,
            probe_services=probe_services,
        )

        # Ensure deep sanitization across all component fields
        sanitized_components = [
            ComponentReadiness(
                component_id=c.component_id,
                display_name=c.display_name,
                category=c.category,
                status=c.status,
                failure_code=c.failure_code,
                details=sanitize_text(c.details),
                executable=c.executable,
                local_only=c.local_only,
                configuration_status=c.configuration_status,
                live_status=c.live_status,
            )
            for c in report.components
        ]

        return TenantSafeReadinessResponse(
            tenant_id=context.tenant_id,
            report_type=report.report_type,
            overall_status=report.overall_status,
            can_run_offline_demo=report.can_run_offline_demo,
            can_run_live_inference=report.can_run_live_inference,
            executable_components_count=report.executable_components_count,
            total_components_count=report.total_components_count,
            components=sanitized_components,
            missing_components=report.missing_components,
            summary=sanitize_text(report.summary),
            timestamp=report.timestamp,
            disclaimer=report.disclaimer,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate sanitized readiness report",
        ) from exc
