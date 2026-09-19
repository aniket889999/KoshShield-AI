from uuid import uuid4

import pytest

from koshshield.services.agent.runner import RestrictedDeterministicToolRunner
from koshshield.services.agent.tool_runner import ToolExecutionError


def test_runner_calculator_safe_execution() -> None:
    runner = RestrictedDeterministicToolRunner()
    res = runner.execute(tool_name="calculator", payload={"expression": "(50 * 2) + 25"})

    assert res.payload["ok"] is True
    assert res.payload["result"] == {"value": "125"}
    assert res.sandbox["network_disabled"] is True
    assert res.sandbox["subprocess_disabled"] is True
    assert res.output_hash


def test_runner_calculator_division_by_zero_fails_cleanly() -> None:
    runner = RestrictedDeterministicToolRunner()
    with pytest.raises(ToolExecutionError, match="Division by zero"):
        runner.execute(tool_name="calculator", payload={"expression": "100 / 0"})


def test_runner_procurement_comparison_handler() -> None:
    runner = RestrictedDeterministicToolRunner()
    res = runner.execute(
        tool_name="calculate_procurement_comparison",
        payload={
            "case_id": "TENDER-2026-X1",
            "comparison_criteria": ["flow_rate", "pressure", "delivery_lead_time"],
        },
    )

    assert res.payload["ok"] is True
    result = res.payload["result"]
    assert result["status"] == "COMPLETED"
    assert result["case_id"] == "TENDER-2026-X1"
    assert len(result["evaluated_criteria"]) == 3
    assert result["comparison_verdict"] == "COMPLIANT_SPECIFICATION"
    assert "disclaimer" in result


def test_runner_draft_approval_note_handler() -> None:
    runner = RestrictedDeterministicToolRunner()
    doc_id = str(uuid4())
    res = runner.execute(
        tool_name="draft_approval_note",
        payload={
            "document_id": doc_id,
            "note_subject": "Purchase order sanction",
            "recommended_action": "Sanction compliant bid",
            "urgency": "IMMEDIATE",
            "document": {"evidence_hash": "a" * 64},
        },
    )

    assert res.payload["ok"] is True
    result = res.payload["result"]
    assert result["status"] == "COMPLETED"
    assert result["media_type"] == "text/markdown"
    assert "# OFFICIAL MEMORANDUM" in result["content"]
    assert "IMMEDIATE" in result["content"]
    assert doc_id in result["content"]


def test_runner_draft_report_handler() -> None:
    runner = RestrictedDeterministicToolRunner()
    doc_id = str(uuid4())
    res = runner.execute(
        tool_name="draft_report",
        payload={
            "document_id": doc_id,
            "report_title": "Quarterly Technical Audit",
            "report_type": "audit_summary",
        },
    )

    assert res.payload["ok"] is True
    result = res.payload["result"]
    assert result["status"] == "COMPLETED"
    assert result["title"] == "Quarterly Technical Audit"
    assert "# Quarterly Technical Audit" in result["content"]


def test_runner_generate_verified_code_handler() -> None:
    runner = RestrictedDeterministicToolRunner()
    res = runner.execute(
        tool_name="generate_verified_code",
        payload={
            "template_id": "audit_hasher",
            "parameters": {"algorithm": "sha256"},
        },
    )

    assert res.payload["ok"] is True
    result = res.payload["result"]
    assert result["status"] == "COMPLETED"
    assert result["template_id"] == "audit_hasher"
    assert "def compute_chain_hash" in result["code"]


def test_runner_model_backed_action_returns_truthful_not_executed_when_offline() -> None:
    # When model runtime is offline
    runner = RestrictedDeterministicToolRunner(model_runtime_ready=False)
    res = runner.execute(
        tool_name="summarize_document",
        payload={"document_id": str(uuid4()), "focus": "general"},
    )

    assert res.payload["ok"] is True
    result = res.payload["result"]
    assert result["status"] == "NOT_EXECUTED"
    assert result["code"] == "MODEL_RUNTIME_UNAVAILABLE"
    assert "offline" in result["detail"]


def test_runner_rejects_unallowlisted_tool() -> None:
    runner = RestrictedDeterministicToolRunner()
    with pytest.raises(ToolExecutionError, match="not in the allowlisted"):
        runner.execute(tool_name="arbitrary_tool", payload={})
