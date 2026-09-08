from typing import NotRequired, TypedDict

from langgraph.graph import END, StateGraph

from koshshield.models import AgentRunState
from koshshield.services.agent.policy import AgentPolicyEngine, PolicyAssessment


class AgentPolicyState(TypedDict):
    tool_name: str
    arguments: dict[str, object]
    classification: str
    resource_authorized: bool
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
        tool_name: str,
        arguments: dict[str, object],
        classification: str,
        resource_authorized: bool,
    ) -> AgentPolicyState:
        return self.graph.invoke(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "classification": classification,
                "resource_authorized": resource_authorized,
                "state_history": [AgentRunState.REQUESTED],
            }
        )

    def _evaluate_policy(self, state: AgentPolicyState) -> AgentPolicyState:
        assessment = self.policy_engine.evaluate(
            tool_name=state["tool_name"],
            arguments=state["arguments"],
            classification=state["classification"],
            resource_authorized=state["resource_authorized"],
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
