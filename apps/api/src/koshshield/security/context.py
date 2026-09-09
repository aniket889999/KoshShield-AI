from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from koshshield.config import Settings, get_settings


@dataclass(frozen=True)
class RequestContext:
    actor_id: str
    tenant_id: str
    roles: tuple[str, ...] = field(default_factory=lambda: ("user",))

    def has_role(self, role: str) -> bool:
        return role.lower() in self.roles


def get_request_context(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID", max_length=120)] = None,
    x_actor_id: Annotated[str | None, Header(alias="X-Actor-ID", max_length=120)] = None,
    x_roles: Annotated[str | None, Header(alias="X-Roles", max_length=255)] = None,
) -> RequestContext:
    """Centralizes actor, tenant, and role information.

    In demo mode, identities may be derived from request headers.
    In production mode (or when demo_mode is False), requests fail closed with 401
    because header-derived identities are rejected and verified authentication is required.
    """
    if not settings.demo_mode or settings.environment.lower() == "production":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Verified authentication required: "
                "header-derived identity disabled in production mode"
            ),
        )

    tenant_id = (x_tenant_id or "default").strip()
    actor_id = (x_actor_id or "local-demo-user").strip()
    if not tenant_id:
        tenant_id = "default"
    if not actor_id:
        actor_id = "local-demo-user"

    if x_roles:
        roles = tuple(r.strip().lower() for r in x_roles.split(",") if r.strip())
    else:
        roles = ("user",)

    return RequestContext(
        actor_id=actor_id,
        tenant_id=tenant_id,
        roles=roles,
    )


RequestContextDependency = Annotated[RequestContext, Depends(get_request_context)]


def require_roles(*allowed_roles: str):
    """Enforces that the request context has at least one of the allowed roles (or admin)."""
    normalized_allowed = {r.strip().lower() for r in allowed_roles}

    def role_checker(context: RequestContextDependency) -> RequestContext:
        if context.has_role("admin") or any(context.has_role(r) for r in normalized_allowed):
            return context
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: requires one of the following roles: {sorted(normalized_allowed)}",
        )

    return role_checker


ReviewerContextDependency = Annotated[RequestContext, Depends(require_roles("reviewer"))]
ApproverContextDependency = Annotated[RequestContext, Depends(require_roles("approver"))]
ExecutorContextDependency = Annotated[RequestContext, Depends(require_roles("executor"))]
AuditorContextDependency = Annotated[RequestContext, Depends(require_roles("auditor"))]
AdminContextDependency = Annotated[RequestContext, Depends(require_roles("admin"))]
