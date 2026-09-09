import ast
import json
import math
import sys
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext
from typing import Any

MAX_ABSOLUTE_RESULT = Decimal("1e18")


class RestrictedToolError(ValueError):
    pass


def calculate(expression: str) -> dict[str, str]:
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 64:
        raise RestrictedToolError("operation limit exceeded")

    with localcontext() as context:
        context.prec = 32
        value = _evaluate_node(tree.body)
    if not value.is_finite() or abs(value) > MAX_ABSOLUTE_RESULT:
        raise RestrictedToolError("result outside allowed range")
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return {"value": normalized or "0"}


def _evaluate_node(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise RestrictedToolError("non-numeric literal")
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise RestrictedToolError("non-finite literal")
        value = Decimal(str(node.value))
        if abs(value) > Decimal("1e12"):
            raise RestrictedToolError("numeric literal outside allowed range")
        return value

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_node(node.operand)
        return -value if isinstance(node.op, ast.USub) else value

    if not isinstance(node, ast.BinOp):
        raise RestrictedToolError("unsupported expression")
    left = _evaluate_node(node.left)
    right = _evaluate_node(node.right)
    try:
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        elif isinstance(node.op, ast.Div):
            value = left / right
        elif isinstance(node.op, ast.FloorDiv):
            value = left // right
        elif isinstance(node.op, ast.Mod):
            value = left % right
        elif isinstance(node.op, ast.Pow):
            if right != right.to_integral_value() or abs(right) > 12:
                raise RestrictedToolError("invalid exponent")
            value = left ** int(right)
        else:
            raise RestrictedToolError("unsupported operator")
    except (DivisionByZero, InvalidOperation, OverflowError, ZeroDivisionError) as err:
        raise RestrictedToolError("arithmetic operation failed") from err
    if not value.is_finite() or abs(value) > MAX_ABSOLUTE_RESULT:
        raise RestrictedToolError("intermediate result outside allowed range")
    return value


def document_report(payload: dict[str, Any]) -> dict[str, str]:
    if set(payload) != {"document"} or not isinstance(payload["document"], dict):
        raise RestrictedToolError("invalid report input")
    document = payload["document"]
    required = {
        "document_id",
        "evidence_hash",
        "filename",
        "status",
        "page_count",
        "redaction_count",
        "chunk_count",
    }
    if set(document) != required:
        raise RestrictedToolError("invalid report fields")
    for key in ("document_id", "evidence_hash", "filename", "status"):
        if not isinstance(document[key], str):
            raise RestrictedToolError("invalid report field type")
    for key in ("page_count", "redaction_count", "chunk_count"):
        if not isinstance(document[key], int) or document[key] < 0:
            raise RestrictedToolError("invalid report count")

    filename = " ".join(document["filename"].split())[:255].replace("`", "'")
    content = "\n".join(
        [
            "# KoshShield Processing Report",
            "",
            f"- Document: {filename}",
            f"- Document ID: `{document['document_id']}`",
            f"- Processing state: {document['status']}",
            f"- Evidence SHA-256: `{document['evidence_hash']}`",
            f"- Extracted pages: {document['page_count']}",
            f"- Reviewed redactions: {document['redaction_count']}",
            f"- Active retrieval chunks: {document['chunk_count']}",
            "",
            "Generated inside the restricted KoshShield tool sandbox.",
        ]
    )
    return {"content": content, "media_type": "text/markdown"}


def main() -> int:
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict) or set(request) != {"tool", "payload"}:
            raise RestrictedToolError("invalid request")
        tool = request["tool"]
        payload = request["payload"]
        if not isinstance(payload, dict):
            raise RestrictedToolError("invalid payload")
        if tool == "calculator" and set(payload) == {"expression"}:
            expression = payload["expression"]
            if not isinstance(expression, str) or len(expression) > 200:
                raise RestrictedToolError("invalid expression")
            result = calculate(expression)
        elif tool == "document_report":
            result = document_report(payload)
        else:
            raise RestrictedToolError("tool not allowed")
        json.dump(
            {"ok": True, "tool": tool, "result": result},
            sys.stdout,
            separators=(",", ":"),
        )
        return 0
    except (
        RestrictedToolError,
        SyntaxError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        json.dump({"ok": False, "error": "RESTRICTED_TOOL_REJECTED"}, sys.stdout)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
