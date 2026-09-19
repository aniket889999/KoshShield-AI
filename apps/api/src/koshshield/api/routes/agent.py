from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from koshshield.config import Settings, get_settings
from koshshield.database import get_db
from koshshield.models import AgentRunRecord
from koshshield.schemas import (
    AgentActionRequest,
    AgentApprovalDecisionRequest,
    AgentApprovalResponse,
    AgentDeliverableResponse,
    AgentExecuteRequest,
    AgentRunResponse,
    ApprovalRequestDetailResponse,
)
from koshshield.security.context import (
    ApproverContextDependency,
    ExecutorContextDependency,
    RequestContextDependency,
    RequesterContextDependency,
)
from koshshield.services.agent.actions import get_action_risk_level
from koshshield.services.agent.lifecycle import (
    DEFAULT_APPROVAL_TTL_SECONDS,
    is_approval_expired,
)
from koshshield.services.agent.runner import RestrictedDeterministicToolRunner
from koshshield.services.agent.service import (
    AgentRunConflictError,
    AgentRunNotFoundError,
    AgentRunService,
    SeparationOfDutiesError,
)
from koshshield.services.agent.tool_runner import DockerToolRunner, ToolRunner

router = APIRouter()
SessionDependency = Annotated[Session, Depends(get_db)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]


def get_tool_runner(settings: SettingsDependency) -> ToolRunner:
    if getattr(settings, "tool_runner_backend", "restricted") == "docker":
        return DockerToolRunner(
            image=settings.tool_runner_image,
            timeout_seconds=settings.tool_runner_timeout_seconds,
            max_output_bytes=settings.tool_runner_max_output_bytes,
        )
    return RestrictedDeterministicToolRunner()


def get_agent_service(
    tool_runner: Annotated[ToolRunner, Depends(get_tool_runner)],
    settings: SettingsDependency,
) -> AgentRunService:
    return AgentRunService(
        tool_runner=tool_runner,
        integrity_secret=settings.agent_integrity_secret,
    )


def to_run_response(
    *,
    session: Session,
    service: AgentRunService,
    run: AgentRunRecord,
) -> AgentRunResponse:
    approval = service.get_approval(session, run.id)
    approval_resp = None
    if approval:
        expires_at = approval.created_at + timedelta(seconds=DEFAULT_APPROVAL_TTL_SECONDS)
        is_expired = is_approval_expired(approval.created_at)
        doc_id = (
            str(run.arguments_json.get("document_id"))
            if "document_id" in run.arguments_json
            else None
        )
        doc_ver = (
            int(run.arguments_json["document_version"])
            if run.arguments_json.get("document_version") is not None
            else None
        )
        approval_resp = AgentApprovalResponse(
            decision=approval.decision,
            reviewer_id=approval.reviewer_id,
            version=approval.version,
            decided_at=approval.decided_at,
            expires_at=expires_at,
            is_expired=is_expired,
            action_name=run.tool_name,
            risk_level=get_action_risk_level(run.tool_name).value,
            input_digest=run.arguments_hash,
            document_id=doc_id,
            document_version=doc_ver,
        )

    return AgentRunResponse(
        id=run.id,
        tenant_id=run.tenant_id,
        actor_id=run.actor_id,
        tool_name=run.tool_name,
        classification=run.classification,
        arguments_hash=run.arguments_hash,
        argument_summary=run.argument_summary,
        status=run.status,
        state_history=list(run.state_history),
        policy_decision=run.policy_decision,
        policy_reason=run.policy_reason,
        approval_required=run.approval_required,
        approval=approval_resp,
        result=run.result_json,
        result_hash=run.result_hash,
        failure_code=run.failure_code,
        version=run.version,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


@router.get("/runs", response_model=list[AgentRunResponse])
def list_agent_runs(
    session: SessionDependency,
    service: Annotated[AgentRunService, Depends(get_agent_service)],
    context: RequestContextDependency,
    limit: int = 50,
) -> list[AgentRunResponse]:
    runs = service.list_runs(session=session, tenant_id=context.tenant_id, limit=limit)
    return [to_run_response(session=session, service=service, run=run) for run in runs]


@router.get("/runs/{run_id}", response_model=AgentRunResponse)
def get_agent_run(
    run_id: str,
    session: SessionDependency,
    service: Annotated[AgentRunService, Depends(get_agent_service)],
    context: RequestContextDependency,
) -> AgentRunResponse:
    try:
        run = service.get_run(session=session, run_id=run_id, tenant_id=context.tenant_id)
    except AgentRunNotFoundError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err)) from err
    return to_run_response(session=session, service=service, run=run)


@router.get("/runs/{run_id}/approval", response_model=ApprovalRequestDetailResponse)
def get_agent_run_approval(
    run_id: str,
    session: SessionDependency,
    service: Annotated[AgentRunService, Depends(get_agent_service)],
    context: RequestContextDependency,
) -> ApprovalRequestDetailResponse:
    try:
        detail = service.get_approval_detail(
            session=session,
            run_id=run_id,
            tenant_id=context.tenant_id,
        )
    except AgentRunNotFoundError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err)) from err
    return ApprovalRequestDetailResponse.model_validate(detail)


@router.get("/runs/{run_id}/deliverable", response_model=AgentDeliverableResponse)
def get_agent_run_deliverable(
    run_id: str,
    session: SessionDependency,
    service: Annotated[AgentRunService, Depends(get_agent_service)],
    context: RequestContextDependency,
) -> AgentDeliverableResponse:
    try:
        deliverable = service.get_deliverable(
            session=session,
            run_id=run_id,
            tenant_id=context.tenant_id,
        )
    except AgentRunNotFoundError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err)) from err
    return AgentDeliverableResponse.model_validate(deliverable.model_dump())


@router.post("/runs", response_model=AgentRunResponse, status_code=status.HTTP_201_CREATED)
def propose_agent_action(
    request: AgentActionRequest,
    session: SessionDependency,
    service: Annotated[AgentRunService, Depends(get_agent_service)],
    context: RequesterContextDependency,
) -> AgentRunResponse:
    run = service.propose_action(
        session=session,
        tenant_id=context.tenant_id,
        actor_id=context.actor_id,
        tool_name=request.tool_name,
        classification=request.classification,
        arguments=request.arguments,
    )
    return to_run_response(session=session, service=service, run=run)


@router.post("/runs/{run_id}/approval", response_model=AgentRunResponse)
def decide_agent_approval(
    run_id: str,
    request: AgentApprovalDecisionRequest,
    session: SessionDependency,
    service: Annotated[AgentRunService, Depends(get_agent_service)],
    context: ApproverContextDependency,
) -> AgentRunResponse:
    try:
        run = service.decide_approval(
            session=session,
            run_id=run_id,
            tenant_id=context.tenant_id,
            reviewer_id=context.actor_id,
            decision=request.decision,
            version=request.version,
        )
    except AgentRunNotFoundError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err)) from err
    except SeparationOfDutiesError as err:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(err)) from err
    except AgentRunConflictError as err:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(err)) from err
    return to_run_response(session=session, service=service, run=run)


@router.post("/runs/{run_id}/execute", response_model=AgentRunResponse)
def execute_agent_action(
    run_id: str,
    request: AgentExecuteRequest,
    session: SessionDependency,
    service: Annotated[AgentRunService, Depends(get_agent_service)],
    context: ExecutorContextDependency,
) -> AgentRunResponse:
    try:
        run = service.execute_action(
            session=session,
            run_id=run_id,
            tenant_id=context.tenant_id,
            executor_id=context.actor_id,
            version=request.version,
        )
    except AgentRunNotFoundError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err)) from err
    except AgentRunConflictError as err:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(err)) from err
    return to_run_response(session=session, service=service, run=run)
