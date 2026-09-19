from typing import NotRequired, TypedDict

from langgraph.graph import END, StateGraph

from koshshield.models import AgentRunState
from koshshield.services.agent.policy import AgentPolicyEngine, PolicyAssessment


class AgentPolicyState(TypedDict):
    tool_name: str | None
    arguments: dict[str, object] | None
    classification: str | None
    resource_authorized: bool
    tenant_id: NotRequired[str | None]
    roles: NotRequired[list[str] | set[str] | None]
    document_status: NotRequired[str | None]
    document_tenant_id: NotRequired[str | None]
    has_evidence: NotRequired[bool]
    runtime_ready: NotRequired[bool]
    state_history: list[str]
    assessment: NotRequired[PolicyAssessment]
    status: NotRequired[str]


class AgentPolicyWorkflow:
    def __init__(self, policy_engine: AgentPolicyEngine | None = None) -> None:
        self.policy_engine = policy_engine or AgentPolicyEngine()
        graph = StateGraph(AgentPolicyState)
        graph.add_node("evaluate_policy", self._evaluate_policy)
        graph.add_node("await_approval", self._await_approval)
        graph.add_node("reject", self._reject)
        graph.set_entry_point("evaluate_policy")
        graph.add_conditional_edges(
            "evaluate_policy",
            self._route_after_policy,
            {"allowed": "await_approval", "rejected": "reject"},
        )
        graph.add_edge("await_approval", END)
        graph.add_edge("reject", END)
        self.graph = graph.compile()

    def evaluate(
        self,
        *,
        tool_name: str | None,
        arguments: dict[str, object] | None,
        classification: str | None,
        resource_authorized: bool = True,
        tenant_id: str | None = None,
        roles: list[str] | set[str] | None = None,
        document_status: str | None = None,
        document_tenant_id: str | None = None,
        has_evidence: bool = True,
        runtime_ready: bool = True,
    ) -> AgentPolicyState:
        return self.graph.invoke(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "classification": classification,
                "resource_authorized": resource_authorized,
                "tenant_id": tenant_id,
                "roles": roles,
                "document_status": document_status,
                "document_tenant_id": document_tenant_id,
                "has_evidence": has_evidence,
                "runtime_ready": runtime_ready,
                "state_history": [AgentRunState.REQUESTED],
            }
        )

    def _evaluate_policy(self, state: AgentPolicyState) -> AgentPolicyState:
        assessment = self.policy_engine.evaluate(
            tool_name=state.get("tool_name"),
            arguments=state.get("arguments"),
            classification=state.get("classification"),
            resource_authorized=state.get("resource_authorized", True),
            tenant_id=state.get("tenant_id"),
            roles=state.get("roles"),
            document_status=state.get("document_status"),
            document_tenant_id=state.get("document_tenant_id"),
            has_evidence=state.get("has_evidence", True),
            runtime_ready=state.get("runtime_ready", True),
        )
        return {
            **state,
            "assessment": assessment,
            "state_history": [*state["state_history"], AgentRunState.POLICY_EVALUATED],
        }

    @staticmethod
    def _route_after_policy(state: AgentPolicyState) -> str:
        return "allowed" if state["assessment"].decision == "ALLOWED" else "rejected"

    @staticmethod
    def _await_approval(state: AgentPolicyState) -> AgentPolicyState:
        return {
            **state,
            "status": AgentRunState.APPROVAL_PENDING,
            "state_history": [*state["state_history"], AgentRunState.APPROVAL_PENDING],
        }

    @staticmethod
    def _reject(state: AgentPolicyState) -> AgentPolicyState:
        return {
            **state,
            "status": AgentRunState.REJECTED,
            "state_history": [*state["state_history"], AgentRunState.REJECTED],
        }
