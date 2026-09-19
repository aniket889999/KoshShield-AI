"""Deterministic agent policy evaluator for KoshShield AI.

Decides ALLOW, REQUIRE_APPROVAL, or DENY based on:
- Tenant isolation and validation
- Actor role permissions
- Data classification (INTERNAL, CONFIDENTIAL, RESTRICTED)
- Action contract and risk classification (LOW, MEDIUM, HIGH, CRITICAL)
- Document approval state and evidence availability
- Runtime readiness

Fails closed when context is incomplete.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from koshshield.models import DocumentState
from koshshield.security.pii.indian_pii import IndianPiiDetector
from koshshield.services.agent.actions import (
    ACTION_ARGUMENT_SCHEMAS,
    ActionFailureCode,
    ActionRiskLevel,
    ActionValidationError,
    get_action_evidence_requirement,
    get_action_risk_level,
    validate_action_arguments,
)


class PolicyDecision(StrEnum):
    ALLOW = "ALLOW"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    DENY = "DENY"


class PolicyReasonCode(StrEnum):
    INCOMPLETE_CONTEXT = "INCOMPLETE_CONTEXT"
    PROHIBITED_TOOL = "PROHIBITED_TOOL"
    TOOL_NOT_ALLOWLISTED = "TOOL_NOT_ALLOWLISTED"
    INVALID_CLASSIFICATION = "INVALID_CLASSIFICATION"
    INSUFFICIENT_ROLE_PERMISSIONS = "INSUFFICIENT_ROLE_PERMISSIONS"
    RESOURCE_NOT_AUTHORIZED = "RESOURCE_NOT_AUTHORIZED"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    DOCUMENT_NOT_APPROVED = "DOCUMENT_NOT_APPROVED"
    CROSS_TENANT_ACCESS_DENIED = "CROSS_TENANT_ACCESS_DENIED"
    EVIDENCE_UNAVAILABLE = "EVIDENCE_UNAVAILABLE"
    RUNTIME_UNAVAILABLE = "RUNTIME_UNAVAILABLE"
    HIGH_RISK_ACTION_APPROVAL_REQUIRED = "HIGH_RISK_ACTION_APPROVAL_REQUIRED"
    RESTRICTED_DATA_APPROVAL_REQUIRED = "RESTRICTED_DATA_APPROVAL_REQUIRED"
    ACTION_CONTRACT_VIOLATION = "ACTION_CONTRACT_VIOLATION"
    HUMAN_APPROVAL_REQUIRED = "HUMAN_APPROVAL_REQUIRED"
    UNSAFE_EXPRESSION = "UNSAFE_EXPRESSION"
    PII_IN_TOOL_ARGUMENTS = "PII_IN_TOOL_ARGUMENTS"


@dataclass(frozen=True)
class PolicyAssessment:
    decision: str  # "ALLOWED" or "REJECTED" for backward-compatibility
    reason_code: str
    reason: str
    normalized_arguments: dict[str, object]
    argument_summary: str
    approval_required: bool
    verdict: PolicyDecision = PolicyDecision.REQUIRE_APPROVAL


class AgentPolicyEngine:
    allowed_tools = frozenset(ACTION_ARGUMENT_SCHEMAS.keys())
    prohibited_tools = frozenset(
        {
            "browser",
            "filesystem",
            "network_request",
            "python",
            "shell",
            "sql",
        }
    )

    def __init__(self) -> None:
        self.pii_detector = IndianPiiDetector()

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
    ) -> PolicyAssessment:
        # 1. Fail closed if context is incomplete
        if not tool_name or arguments is None or classification is None:
            return self._reject(
                PolicyReasonCode.INCOMPLETE_CONTEXT,
                "Context incomplete: tool_name, arguments, and classification are required.",
            )

        # 2. Check prohibited tools
        if tool_name in self.prohibited_tools:
            return self._reject(
                PolicyReasonCode.PROHIBITED_TOOL,
                "The requested tool is strictly prohibited by security policy.",
            )

        # 3. Check allowlisted catalog
        if tool_name not in self.allowed_tools:
            return self._reject(
                PolicyReasonCode.TOOL_NOT_ALLOWLISTED,
                f"Tool '{tool_name}' is not in the local permitted action catalog.",
            )

        # 4. Check classification
        if classification not in {"INTERNAL", "CONFIDENTIAL", "RESTRICTED"}:
            return self._reject(
                PolicyReasonCode.INVALID_CLASSIFICATION,
                "Unsupported data classification.",
            )

        # 5. Check tenant isolation if document tenant provided
        if document_tenant_id and tenant_id and document_tenant_id != tenant_id:
            return self._reject(
                PolicyReasonCode.CROSS_TENANT_ACCESS_DENIED,
                "Cross-tenant resource access is prohibited.",
            )

        # 6. Check RBAC roles
        if roles is not None:
            role_set = set(roles)
            if not role_set:
                return self._reject(
                    PolicyReasonCode.INSUFFICIENT_ROLE_PERMISSIONS,
                    "Actor has no assigned roles.",
                )
            # Auditors have read-only access and cannot propose actions
            if role_set == {"auditor"}:
                return self._reject(
                    PolicyReasonCode.INSUFFICIENT_ROLE_PERMISSIONS,
                    "Auditor role is read-only and cannot propose actions.",
                )

        # 7. Validate typed action schema & argument contracts
        try:
            validated_args = validate_action_arguments(tool_name, arguments)
            normalized_args = validated_args.model_dump(exclude_none=True)
        except ActionValidationError as err:
            reason_code = (
                PolicyReasonCode.PII_IN_TOOL_ARGUMENTS
                if err.code == ActionFailureCode.PII_IN_ARGUMENTS
                else str(err.code)
            )
            return self._reject(
                reason_code,
                f"Action contract validation failed: {err.detail}",
            )
        except Exception as exc:
            return self._reject(
                PolicyReasonCode.ACTION_CONTRACT_VIOLATION,
                f"Validation error: {str(exc)}",
            )

        # 8. Special arithmetic AST validation for calculator
        if tool_name == "calculator":
            calc_assessment = self._evaluate_calculator_expression(
                str(normalized_args["expression"])
            )
            if calc_assessment is not None:
                return calc_assessment

        # 9. Evidence and document requirement checks
        evidence_req = get_action_evidence_requirement(tool_name)
        if evidence_req.requires_document:
            if not resource_authorized:
                return self._reject(
                    PolicyReasonCode.RESOURCE_NOT_AUTHORIZED,
                    "The requested document is unavailable or has not completed secure indexing.",
                )
            if document_status and document_status != DocumentState.INDEXED:
                return self._reject(
                    PolicyReasonCode.DOCUMENT_NOT_APPROVED,
                    f"Document state '{document_status}' is not approved for agent actions.",
                )
            if not has_evidence:
                return self._reject(
                    PolicyReasonCode.EVIDENCE_UNAVAILABLE,
                    "Required masked evidence chunks are missing for this document.",
                )

        # 10. Risk-based and classification-based policy decision
        risk_level = get_action_risk_level(tool_name)
        summary = self._build_summary(tool_name, normalized_args)

        if classification == "RESTRICTED":
            return PolicyAssessment(
                decision="ALLOWED",
                verdict=PolicyDecision.REQUIRE_APPROVAL,
                reason_code=PolicyReasonCode.RESTRICTED_DATA_APPROVAL_REQUIRED,
                reason="RESTRICTED classification requires independent human approval.",
                normalized_arguments=normalized_args,
                argument_summary=summary,
                approval_required=True,
            )

        if risk_level in {ActionRiskLevel.HIGH, ActionRiskLevel.MEDIUM}:
            return PolicyAssessment(
                decision="ALLOWED",
                verdict=PolicyDecision.REQUIRE_APPROVAL,
                reason_code=PolicyReasonCode.HIGH_RISK_ACTION_APPROVAL_REQUIRED,
                reason=f"{risk_level.value} risk action requires independent human approval.",
                normalized_arguments=normalized_args,
                argument_summary=summary,
                approval_required=True,
            )

        # Low-risk actions still require approval in government workflow
        return PolicyAssessment(
            decision="ALLOWED",
            verdict=PolicyDecision.REQUIRE_APPROVAL,
            reason_code=PolicyReasonCode.HUMAN_APPROVAL_REQUIRED,
            reason="Allowlisted government action requires independent human approval.",
            normalized_arguments=normalized_args,
            argument_summary=summary,
            approval_required=True,
        )

    def _evaluate_calculator_expression(self, expression: str) -> PolicyAssessment | None:
        normalized = expression.strip()
        if self.pii_detector.detect(normalized):
            return self._reject(
                PolicyReasonCode.PII_IN_TOOL_ARGUMENTS,
                "Potential Indian personal identifier detected in calculator input.",
            )
        try:
            tree = ast.parse(normalized, mode="eval")
        except SyntaxError:
            return self._reject(
                PolicyReasonCode.UNSAFE_EXPRESSION, "Expression is not valid arithmetic."
            )

        reason = self._validate_arithmetic_tree(tree)
        if reason:
            return self._reject(PolicyReasonCode.UNSAFE_EXPRESSION, reason)
        return None

    def _validate_arithmetic_tree(self, tree: ast.AST) -> str | None:
        allowed_nodes: tuple[type[ast.AST], ...] = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.UAdd,
            ast.USub,
        )
        nodes = list(ast.walk(tree))
        if len(nodes) > 64:
            return "Expression exceeds the operation limit."
        if any(not isinstance(node, allowed_nodes) for node in nodes):
            return "Only numeric literals and arithmetic operators are allowed."

        for node in nodes:
            if isinstance(node, ast.Constant):
                if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                    return "Only finite numeric literals are allowed."
                if abs(node.value) > 1_000_000_000_000:
                    return "Numeric literal exceeds the configured limit."
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                exponent = self._literal_number(node.right)
                if exponent is None or int(exponent) != exponent or abs(exponent) > 12:
                    return "Exponent must be an integer between -12 and 12."
        return None

    @staticmethod
    def _literal_number(node: ast.AST) -> float | int | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value if math.isfinite(node.value) else None
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
        ):
            value = node.operand.value
            if not math.isfinite(value):
                return None
            return -value if isinstance(node.op, ast.USub) else value
        return None

    @staticmethod
    def _build_summary(tool_name: str, args: dict[str, Any]) -> str:
        if tool_name == "calculator":
            return str(args.get("expression", ""))
        if "document_id" in args:
            doc_id = str(args["document_id"])
            return f"Action '{tool_name}' on document {doc_id[:8]}"
        if "case_id" in args:
            return f"Action '{tool_name}' on case {args['case_id']}"
        if "template_id" in args:
            return f"Action '{tool_name}' with template {args['template_id']}"
        return f"Action '{tool_name}' proposal"

    @staticmethod
    def _reject(reason_code: str, reason: str) -> PolicyAssessment:
        return PolicyAssessment(
            decision="REJECTED",
            verdict=PolicyDecision.DENY,
            reason_code=str(reason_code),
            reason=reason,
            normalized_arguments={},
            argument_summary="Arguments withheld by policy",
            approval_required=False,
        )
