from koshshield.models import AgentRunState
from koshshield.services.agent.policy import AgentPolicyEngine
from koshshield.services.agent.workflow import AgentPolicyWorkflow


def test_langgraph_routes_safe_calculation_to_persisted_approval_state() -> None:
    state = AgentPolicyWorkflow().evaluate(
        tool_name="calculator",
        arguments={"expression": "(25 * 4) + 7"},
        classification="CONFIDENTIAL",
        resource_authorized=True,
    )

    assert state["status"] == AgentRunState.APPROVAL_PENDING
    assert state["state_history"] == [
        AgentRunState.REQUESTED,
        AgentRunState.POLICY_EVALUATED,
        AgentRunState.APPROVAL_PENDING,
    ]
    assert state["assessment"].decision == "ALLOWED"
    assert state["assessment"].approval_required is True


def test_policy_rejects_prohibited_tools_and_unsafe_expressions() -> None:
    engine = AgentPolicyEngine()

    shell = engine.evaluate(
        tool_name="shell",
        arguments={"command": "curl https://example.test"},
        classification="INTERNAL",
        resource_authorized=True,
    )
    expression = engine.evaluate(
        tool_name="calculator",
        arguments={"expression": "__import__('os').system('id')"},
        classification="INTERNAL",
        resource_authorized=True,
    )

    assert shell.decision == "REJECTED"
    assert shell.reason_code == "PROHIBITED_TOOL"
    assert shell.normalized_arguments == {}
    assert expression.decision == "REJECTED"
    assert expression.reason_code == "UNSAFE_EXPRESSION"


def test_policy_bounds_exponent_and_operation_count() -> None:
    engine = AgentPolicyEngine()

    huge_power = engine.evaluate(
        tool_name="calculator",
        arguments={"expression": "2 ** 1000"},
        classification="INTERNAL",
        resource_authorized=True,
    )
    too_many_operations = engine.evaluate(
        tool_name="calculator",
        arguments={"expression": "+".join("1" for _ in range(40))},
        classification="INTERNAL",
        resource_authorized=True,
    )

    assert huge_power.decision == "REJECTED"
    assert too_many_operations.decision == "REJECTED"


def test_policy_rejects_detected_indian_pii_from_calculator_arguments() -> None:
    assessment = AgentPolicyEngine().evaluate(
        tool_name="calculator",
        arguments={"expression": "9876543210 + 25"},
        classification="CONFIDENTIAL",
        resource_authorized=True,
    )

    assert assessment.decision == "REJECTED"
    assert assessment.reason_code == "PII_IN_TOOL_ARGUMENTS"
    assert assessment.normalized_arguments == {}
