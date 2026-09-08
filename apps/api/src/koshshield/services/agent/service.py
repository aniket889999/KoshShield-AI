import hashlib
import hmac
import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from koshshield.models import (
    AgentApprovalRecord,
    AgentRunRecord,
    AgentRunState,
    ApprovalDecision,
    DocumentChunkRecord,
    DocumentPageRecord,
    DocumentRecord,
    DocumentState,
    RedactionFinding,
)
from koshshield.services.agent.policy import AgentPolicyEngine
from koshshield.services.agent.tool_runner import ToolRunner, ToolRunnerError
from koshshield.services.agent.workflow import AgentPolicyWorkflow
from koshshield.services.audit import append_audit_event


class AgentRunNotFoundError(ValueError):
    pass


class AgentRunConflictError(ValueError):
    pass


class SeparationOfDutiesError(ValueError):
    pass


def canonical_json_hash(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def argument_integrity_hash(value: object, secret: str) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()


class AgentRunService:
    def __init__(
        self,
        *,
        tool_runner: ToolRunner,
        integrity_secret: str,
        policy_engine: AgentPolicyEngine | None = None,
    ) -> None:
        self.tool_runner = tool_runner
        self.integrity_secret = integrity_secret
        self.policy_engine = policy_engine or AgentPolicyEngine()
        self.policy_workflow = AgentPolicyWorkflow(self.policy_engine)

    def list_runs(
        self,
        *,
        session: Session,
        tenant_id: str,
        limit: int = 50,
    ) -> list[AgentRunRecord]:
        return list(
            session.scalars(
                select(AgentRunRecord)
                .where(AgentRunRecord.tenant_id == tenant_id)
                .order_by(AgentRunRecord.created_at.desc())
                .limit(max(1, min(limit, 100)))
            )
        )

    def get_run(
        self,
        *,
        session: Session,
        run_id: str,
        tenant_id: str,
        lock: bool = False,
    ) -> AgentRunRecord:
        query = select(AgentRunRecord).where(
            AgentRunRecord.id == run_id,
            AgentRunRecord.tenant_id == tenant_id,
        )
        if lock:
            query = query.with_for_update()
        run = session.scalar(query)
        if not run:
            raise AgentRunNotFoundError("Agent run not found")
        return run

    def get_approval(self, session: Session, run_id: str) -> AgentApprovalRecord | None:
        return session.scalar(
            select(AgentApprovalRecord).where(AgentApprovalRecord.agent_run_id == run_id)
        )

    def propose_action(
        self,
        *,
        session: Session,
        tenant_id: str,
        actor_id: str,
        tool_name: str,
        classification: str,
        arguments: dict[str, object],
    ) -> AgentRunRecord:
        resource_authorized = self._resource_is_authorized(
            session=session,
            tool_name=tool_name,
            arguments=arguments,
        )
        workflow_state = self.policy_workflow.evaluate(
            tool_name=tool_name,
            arguments=arguments,
            classification=classification,
            resource_authorized=resource_authorized,
        )
        assessment = workflow_state["assessment"]
        allowed = assessment.decision == "ALLOWED"
        stored_arguments = assessment.normalized_arguments if allowed else {}
        argument_hash = argument_integrity_hash(
            stored_arguments if allowed else arguments,
            self.integrity_secret,
        )
        run = AgentRunRecord(
            id=str(uuid4()),
            tenant_id=tenant_id,
            actor_id=actor_id,
            tool_name=tool_name,
            classification=classification,
            arguments_json=stored_arguments,
            arguments_hash=argument_hash,
            argument_summary=assessment.argument_summary,
            status=workflow_state["status"],
            state_history=workflow_state["state_history"],
            policy_decision=assessment.decision,
            policy_reason=assessment.reason,
            approval_required=assessment.approval_required,
        )
        session.add(run)
        if allowed:
            session.add(
                AgentApprovalRecord(
                    id=str(uuid4()),
                    agent_run_id=run.id,
                    decision=ApprovalDecision.PENDING,
                )
            )

        append_audit_event(
            session,
            actor_id=actor_id,
            event_type="AGENT_ACTION_PROPOSED" if allowed else "AGENT_ACTION_REJECTED",
            resource_type="agent_run",
            resource_id=None,
            details={
                "run_id": run.id,
                "tenant_id": tenant_id,
                "tool_name": tool_name,
                "classification": classification,
                "arguments_hash": argument_hash,
                "policy_decision": assessment.decision,
                "policy_reason_code": assessment.reason_code,
            },
        )
        session.commit()
        session.refresh(run)
        return run

    def decide_approval(
        self,
        *,
        session: Session,
        run_id: str,
        tenant_id: str,
        reviewer_id: str,
        decision: str,
        version: int,
    ) -> AgentRunRecord:
        run = self.get_run(session=session, run_id=run_id, tenant_id=tenant_id, lock=True)
        if run.version != version:
            raise AgentRunConflictError("Agent run changed; refresh before reviewing")
        if run.status != AgentRunState.APPROVAL_PENDING:
            raise AgentRunConflictError("Agent run is not awaiting approval")
        if decision == ApprovalDecision.APPROVED and reviewer_id == run.actor_id:
            raise SeparationOfDutiesError("The requester cannot approve their own action")

        approval = self.get_approval(session, run.id)
        if not approval or approval.decision != ApprovalDecision.PENDING:
            raise AgentRunConflictError("Approval is no longer pending")

        now = datetime.now(UTC)
        approval.decision = decision
        approval.reviewer_id = reviewer_id
        approval.decided_at = now
        approval.version += 1
        run.status = (
            AgentRunState.APPROVED
            if decision == ApprovalDecision.APPROVED
            else AgentRunState.REJECTED
        )
        run.state_history = [*run.state_history, run.status]
        run.version += 1
        run.updated_at = now

        append_audit_event(
            session,
            actor_id=reviewer_id,
            event_type=(
                "AGENT_ACTION_APPROVED"
                if decision == ApprovalDecision.APPROVED
                else "AGENT_ACTION_REJECTED"
            ),
            resource_type="agent_run",
            resource_id=None,
            details={
                "run_id": run.id,
                "tenant_id": tenant_id,
                "tool_name": run.tool_name,
                "decision": decision,
                "requester_id": run.actor_id,
            },
        )
        session.commit()
        session.refresh(run)
        return run

    def execute_action(
        self,
        *,
        session: Session,
        run_id: str,
        tenant_id: str,
        executor_id: str,
        version: int,
    ) -> AgentRunRecord:
        run = self.get_run(session=session, run_id=run_id, tenant_id=tenant_id, lock=True)
        if run.version != version:
            raise AgentRunConflictError("Agent run changed; refresh before executing")
        if run.status not in {AgentRunState.APPROVED, AgentRunState.FAILED}:
            raise AgentRunConflictError("Agent run is not approved for execution")

        approval = self.get_approval(session, run.id)
        if not approval or approval.decision != ApprovalDecision.APPROVED:
            raise AgentRunConflictError("Persisted human approval is required")
        expected_arguments_hash = argument_integrity_hash(
            run.arguments_json,
            self.integrity_secret,
        )
        if not hmac.compare_digest(expected_arguments_hash, run.arguments_hash):
            return self._fail_closed(
                session=session,
                run=run,
                actor_id=executor_id,
                failure_code="ARGUMENT_INTEGRITY_FAILURE",
            )

        resource_authorized = self._resource_is_authorized(
            session=session,
            tool_name=run.tool_name,
            arguments=run.arguments_json,
        )
        assessment = self.policy_engine.evaluate(
            tool_name=run.tool_name,
            arguments=run.arguments_json,
            classification=run.classification,
            resource_authorized=resource_authorized,
        )
        if assessment.decision != "ALLOWED":
            return self._fail_closed(
                session=session,
                run=run,
                actor_id=executor_id,
                failure_code=assessment.reason_code,
                rejected=True,
            )

        runner_payload = self._build_runner_payload(session=session, run=run)
        run.status = AgentRunState.EXECUTING
        run.state_history = [*run.state_history, AgentRunState.EXECUTING]
        run.failure_code = None
        run.version += 1
        run.updated_at = datetime.now(UTC)
        session.commit()

        try:
            runner_result = self.tool_runner.execute(
                tool_name=run.tool_name,
                payload=runner_payload,
            )
            run.status = AgentRunState.VERIFYING
            run.state_history = [*run.state_history, AgentRunState.VERIFYING]
            persisted_result: dict[str, object] = {
                "tool": run.tool_name,
                "output": runner_result.payload["result"],
                "sandbox": runner_result.sandbox,
                "sandbox_output_hash": runner_result.output_hash,
            }
            run.result_json = persisted_result
            run.result_hash = canonical_json_hash(persisted_result)
            run.status = AgentRunState.COMPLETED
            run.state_history = [*run.state_history, AgentRunState.COMPLETED]
            run.policy_reason = "Approved action completed with verified sandbox output."
            run.version += 1
            run.updated_at = datetime.now(UTC)
            append_audit_event(
                session,
                actor_id=executor_id,
                event_type="AGENT_ACTION_COMPLETED",
                resource_type="agent_run",
                resource_id=None,
                details={
                    "run_id": run.id,
                    "tenant_id": tenant_id,
                    "tool_name": run.tool_name,
                    "result_hash": run.result_hash,
                    "sandbox_output_hash": runner_result.output_hash,
                    "network_disabled": runner_result.sandbox.get("network_disabled") is True,
                },
            )
            session.commit()
            session.refresh(run)
            return run
        except ToolRunnerError as err:
            return self._fail_closed(
                session=session,
                run=run,
                actor_id=executor_id,
                failure_code=err.code,
            )

    def _fail_closed(
        self,
        *,
        session: Session,
        run: AgentRunRecord,
        actor_id: str,
        failure_code: str,
        rejected: bool = False,
    ) -> AgentRunRecord:
        run.status = AgentRunState.REJECTED if rejected else AgentRunState.FAILED
        run.state_history = [*run.state_history, run.status]
        run.policy_decision = "REJECTED" if rejected else run.policy_decision
        run.policy_reason = (
            "Policy rejected the action during execution revalidation."
            if rejected
            else "Restricted execution failed closed; no output was accepted."
        )
        run.failure_code = failure_code
        run.result_json = None
        run.result_hash = None
        run.version += 1
        run.updated_at = datetime.now(UTC)
        append_audit_event(
            session,
            actor_id=actor_id,
            event_type="AGENT_ACTION_REJECTED" if rejected else "AGENT_ACTION_FAILED",
            resource_type="agent_run",
            resource_id=None,
            details={
                "run_id": run.id,
                "tenant_id": run.tenant_id,
                "tool_name": run.tool_name,
                "failure_code": failure_code,
            },
        )
        session.commit()
        session.refresh(run)
        return run

    @staticmethod
    def _resource_is_authorized(
        *,
        session: Session,
        tool_name: str,
        arguments: dict[str, object],
    ) -> bool:
        if tool_name != "document_report":
            return True
        document_id = arguments.get("document_id")
        if not isinstance(document_id, str):
            return False
        try:
            normalized_id = str(UUID(document_id))
        except ValueError:
            return False
        document = session.get(DocumentRecord, normalized_id)
        return bool(document and document.status == DocumentState.INDEXED)

    @staticmethod
    def _build_runner_payload(
        *,
        session: Session,
        run: AgentRunRecord,
    ) -> dict[str, object]:
        if run.tool_name == "calculator":
            return dict(run.arguments_json)

        document_id = str(run.arguments_json["document_id"])
        document = session.get(DocumentRecord, document_id)
        if not document or document.status != DocumentState.INDEXED:
            raise AgentRunConflictError("Document is no longer authorized for reporting")
        page_count = session.scalar(
            select(func.count(DocumentPageRecord.id)).where(
                DocumentPageRecord.document_id == document_id
            )
        )
        redaction_count = session.scalar(
            select(func.count(RedactionFinding.id)).where(
                RedactionFinding.document_id == document_id
            )
        )
        chunk_count = session.scalar(
            select(func.count(DocumentChunkRecord.id)).where(
                DocumentChunkRecord.document_id == document_id,
                DocumentChunkRecord.index_version == document.active_index_version,
            )
        )
        return {
            "document": {
                "document_id": document.id,
                "evidence_hash": document.sha256,
                "filename": document.filename,
                "status": document.status,
                "page_count": int(page_count or 0),
                "redaction_count": int(redaction_count or 0),
                "chunk_count": int(chunk_count or 0),
            }
        }
