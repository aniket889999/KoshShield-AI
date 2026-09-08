import ast
import math
from dataclasses import dataclass
from uuid import UUID

from koshshield.security.pii.indian_pii import IndianPiiDetector


@dataclass(frozen=True)
class PolicyAssessment:
    decision: str
    reason_code: str
    reason: str
    normalized_arguments: dict[str, object]
    argument_summary: str
    approval_required: bool


class AgentPolicyEngine:
    allowed_tools = frozenset({"calculator", "document_report"})
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
        tool_name: str,
        arguments: dict[str, object],
        classification: str,
        resource_authorized: bool,
    ) -> PolicyAssessment:
        if tool_name not in self.allowed_tools:
            reason_code = (
                "PROHIBITED_TOOL" if tool_name in self.prohibited_tools else "TOOL_NOT_ALLOWLISTED"
            )
            return self._reject(reason_code, "The requested tool is not in the local allowlist.")

        if classification not in {"INTERNAL", "CONFIDENTIAL", "RESTRICTED"}:
            return self._reject("INVALID_CLASSIFICATION", "Unsupported data classification.")

        if not resource_authorized:
            return self._reject(
                "RESOURCE_NOT_AUTHORIZED",
                "The requested document is unavailable or has not completed secure indexing.",
            )

        if tool_name == "calculator":
            return self._evaluate_calculator(arguments)
        return self._evaluate_document_report(arguments)

    def _evaluate_calculator(self, arguments: dict[str, object]) -> PolicyAssessment:
        if set(arguments) != {"expression"}:
            return self._reject(
                "INVALID_ARGUMENTS", "Calculator accepts only one arithmetic expression."
            )

        expression = arguments.get("expression")
        if not isinstance(expression, str) or not expression.strip() or len(expression) > 200:
            return self._reject(
                "INVALID_ARGUMENTS", "Calculator expression must contain 1 to 200 characters."
            )

        normalized = expression.strip()
        if self.pii_detector.detect(normalized):
            return self._reject(
                "PII_IN_TOOL_ARGUMENTS",
                "Potential Indian personal identifier detected in calculator input.",
            )
        try:
            tree = ast.parse(normalized, mode="eval")
        except SyntaxError:
            return self._reject("UNSAFE_EXPRESSION", "Expression is not valid arithmetic.")

        reason = self._validate_arithmetic_tree(tree)
        if reason:
            return self._reject("UNSAFE_EXPRESSION", reason)

        return PolicyAssessment(
            decision="ALLOWED",
            reason_code="HUMAN_APPROVAL_REQUIRED",
            reason="Allowlisted calculation requires independent human approval.",
            normalized_arguments={"expression": normalized},
            argument_summary=normalized,
            approval_required=True,
        )

    def _evaluate_document_report(self, arguments: dict[str, object]) -> PolicyAssessment:
        if set(arguments) != {"document_id"}:
            return self._reject(
                "INVALID_ARGUMENTS", "Document report accepts only an indexed document ID."
            )

        document_id = arguments.get("document_id")
        if not isinstance(document_id, str):
            return self._reject("INVALID_ARGUMENTS", "Document ID must be a UUID.")
        try:
            normalized_id = str(UUID(document_id))
        except ValueError:
            return self._reject("INVALID_ARGUMENTS", "Document ID must be a UUID.")

        return PolicyAssessment(
            decision="ALLOWED",
            reason_code="HUMAN_APPROVAL_REQUIRED",
            reason="Allowlisted report generation requires independent human approval.",
            normalized_arguments={"document_id": normalized_id},
            argument_summary=f"Processing report for document {normalized_id[:8]}",
            approval_required=True,
        )

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
    def _reject(reason_code: str, reason: str) -> PolicyAssessment:
        return PolicyAssessment(
            decision="REJECTED",
            reason_code=reason_code,
            reason=reason,
            normalized_arguments={},
            argument_summary="Arguments withheld by policy",
            approval_required=False,
        )
