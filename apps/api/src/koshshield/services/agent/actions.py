"""Strict typed action contracts, schemas, and risk classifications for KoshShield AI agent actions.

Defines permitted local government-office actions:
1. summarize_document: Summarize approved document with specific focus.
2. extract_structured_fields: Extract key-value fields (vendor, quantities, dates).
3. calculate_procurement_comparison: Compare proposal specifications against requirements.
4. draft_approval_note: Prepare an official government memorandum note.
5. draft_report: Generate structured evaluation report.
6. generate_verified_code: Generate allowlisted verification code templates.
7. calculator: Simple bounded arithmetic evaluation (legacy).
8. document_report: Basic document summary (legacy).

Guarantees:
- Rejects unknown actions and extra fields (extra="forbid").
- Enforces strict payload bounds (max 64KB payload).
- Rejects absolute filesystem paths and prompt-injection keywords.
- Detects and rejects unredacted Indian PII in freeform arguments.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

from koshshield.security.pii.indian_pii import IndianPiiDetector

_PATH_PATTERN = re.compile(r"(?:/[a-zA-Z0-9_\.\-]+){2,}|[A-Za-z]:\\[a-zA-Z0-9_\.\-\\]+")
_INJECTION_PATTERN = re.compile(
    r"(?i)\b(ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions|system\s+prompt|disregard\s+policy|jailbreak|bypass\s+policy)\b"
)
MAX_ACTION_PAYLOAD_BYTES = 65536


class ActionRiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ActionFailureCode(StrEnum):
    UNKNOWN_ACTION = "UNKNOWN_ACTION"
    EXTRA_FIELDS_FORBIDDEN = "EXTRA_FIELDS_FORBIDDEN"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    RAW_PATH_REJECTED = "RAW_PATH_REJECTED"
    UNSAFE_PROMPT_PATTERN = "UNSAFE_PROMPT_PATTERN"
    PII_IN_ARGUMENTS = "PII_IN_ARGUMENTS"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"


class ActionValidationError(ValueError):
    def __init__(self, code: ActionFailureCode, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _validate_safe_string(v: str) -> str:
    cleaned = v.strip()
    if _PATH_PATTERN.search(cleaned):
        raise ActionValidationError(
            ActionFailureCode.RAW_PATH_REJECTED,
            "Local filesystem paths are not allowed in agent arguments.",
        )
    if _INJECTION_PATTERN.search(cleaned):
        raise ActionValidationError(
            ActionFailureCode.UNSAFE_PROMPT_PATTERN,
            "Unsafe prompt-override pattern detected in arguments.",
        )
    detector = IndianPiiDetector()
    if detector.detect(cleaned):
        raise ActionValidationError(
            ActionFailureCode.PII_IN_ARGUMENTS,
            "Personal identifier detected in tool arguments. Only masked references allowed.",
        )
    return cleaned


SafeString = Annotated[
    str,
    StringConstraints(min_length=1, max_length=500, strip_whitespace=True),
]


class BaseActionArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    document_version: int | None = Field(default=None, ge=1)


class SummarizeDocumentArguments(BaseActionArguments):
    document_id: str
    focus: Literal["general", "procurement", "compliance", "financial"] = "general"
    max_length: int = Field(default=500, ge=100, le=2000)

    @field_validator("document_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        try:
            return str(UUID(v))
        except ValueError as exc:
            raise ActionValidationError(
                ActionFailureCode.INVALID_ARGUMENTS, "document_id must be a valid UUID"
            ) from exc


class ExtractStructuredFieldsArguments(BaseActionArguments):
    document_id: str
    schema_type: Literal["vendor_proposal", "equipment_specs", "compliance_checklist"]

    @field_validator("document_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        try:
            return str(UUID(v))
        except ValueError as exc:
            raise ActionValidationError(
                ActionFailureCode.INVALID_ARGUMENTS, "document_id must be a valid UUID"
            ) from exc


class CalculateProcurementComparisonArguments(BaseActionArguments):
    case_id: Annotated[str, StringConstraints(min_length=3, max_length=64)]
    document_id: str | None = None
    comparison_criteria: list[
        Literal["flow_rate", "pressure", "delivery_lead_time", "warranty_years"]
    ] = Field(default_factory=lambda: ["flow_rate", "pressure"])

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str | None) -> str | None:
        if v is None:
            return None
        try:
            return str(UUID(v))
        except ValueError as exc:
            raise ActionValidationError(
                ActionFailureCode.INVALID_ARGUMENTS, "document_id must be a valid UUID"
            ) from exc

    @field_validator("case_id")
    @classmethod
    def check_case_id(cls, v: str) -> str:
        return _validate_safe_string(v)


class DraftApprovalNoteArguments(BaseActionArguments):
    document_id: str
    note_subject: Annotated[str, StringConstraints(min_length=5, max_length=200)]
    recommended_action: Annotated[str, StringConstraints(min_length=5, max_length=500)]
    urgency: Literal["ROUTINE", "URGENT", "IMMEDIATE"] = "ROUTINE"

    @field_validator("document_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        try:
            return str(UUID(v))
        except ValueError as exc:
            raise ActionValidationError(
                ActionFailureCode.INVALID_ARGUMENTS, "document_id must be a valid UUID"
            ) from exc

    @field_validator("note_subject", "recommended_action")
    @classmethod
    def check_strings(cls, v: str) -> str:
        return _validate_safe_string(v)


class DraftReportArguments(BaseActionArguments):
    document_id: str
    report_title: Annotated[str, StringConstraints(min_length=5, max_length=200)]
    report_type: Literal["audit_summary", "procurement_evaluation", "compliance_review"]

    @field_validator("document_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        try:
            return str(UUID(v))
        except ValueError as exc:
            raise ActionValidationError(
                ActionFailureCode.INVALID_ARGUMENTS, "document_id must be a valid UUID"
            ) from exc

    @field_validator("report_title")
    @classmethod
    def check_strings(cls, v: str) -> str:
        return _validate_safe_string(v)


class GenerateVerifiedCodeArguments(BaseActionArguments):
    template_id: Literal["audit_hasher", "procurement_checker", "csv_validator"]
    parameters: dict[str, str] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def check_parameters(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > 10:
            raise ActionValidationError(
                ActionFailureCode.INVALID_ARGUMENTS, "Too many parameters provided (max 10)."
            )
        cleaned = {}
        for key, val in v.items():
            if len(key) > 40 or len(val) > 200:
                raise ActionValidationError(
                    ActionFailureCode.INVALID_ARGUMENTS, "Parameter key or value too long."
                )
            cleaned[_validate_safe_string(key)] = _validate_safe_string(val)
        return cleaned


class CalculatorArguments(BaseActionArguments):
    expression: Annotated[str, StringConstraints(min_length=1, max_length=200)]

    @field_validator("expression")
    @classmethod
    def check_expression(cls, v: str) -> str:
        return _validate_safe_string(v)


class DocumentReportArguments(BaseActionArguments):
    document_id: str

    @field_validator("document_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        try:
            return str(UUID(v))
        except ValueError as exc:
            raise ActionValidationError(
                ActionFailureCode.INVALID_ARGUMENTS, "document_id must be a valid UUID"
            ) from exc


class ActionEvidenceRequirement(BaseModel):
    model_config = ConfigDict(frozen=True)

    requires_document: bool = False
    requires_redaction_approved: bool = False
    requires_active_index: bool = False
    allowed_classifications: frozenset[str] = frozenset({"INTERNAL", "CONFIDENTIAL", "RESTRICTED"})


ACTION_ARGUMENT_SCHEMAS: dict[str, type[BaseActionArguments]] = {
    "summarize_document": SummarizeDocumentArguments,
    "extract_structured_fields": ExtractStructuredFieldsArguments,
    "calculate_procurement_comparison": CalculateProcurementComparisonArguments,
    "draft_approval_note": DraftApprovalNoteArguments,
    "draft_report": DraftReportArguments,
    "generate_verified_code": GenerateVerifiedCodeArguments,
    "calculator": CalculatorArguments,
    "document_report": DocumentReportArguments,
}

ACTION_RISK_LEVELS: dict[str, ActionRiskLevel] = {
    "summarize_document": ActionRiskLevel.LOW,
    "extract_structured_fields": ActionRiskLevel.LOW,
    "calculate_procurement_comparison": ActionRiskLevel.MEDIUM,
    "calculator": ActionRiskLevel.LOW,
    "document_report": ActionRiskLevel.LOW,
    "draft_approval_note": ActionRiskLevel.HIGH,
    "draft_report": ActionRiskLevel.HIGH,
    "generate_verified_code": ActionRiskLevel.HIGH,
}

ACTION_EVIDENCE_REQUIREMENTS: dict[str, ActionEvidenceRequirement] = {
    "summarize_document": ActionEvidenceRequirement(
        requires_document=True,
        requires_redaction_approved=True,
        requires_active_index=True,
    ),
    "extract_structured_fields": ActionEvidenceRequirement(
        requires_document=True,
        requires_redaction_approved=True,
        requires_active_index=True,
    ),
    "calculate_procurement_comparison": ActionEvidenceRequirement(
        requires_document=False,
        requires_redaction_approved=False,
        requires_active_index=False,
    ),
    "draft_approval_note": ActionEvidenceRequirement(
        requires_document=True,
        requires_redaction_approved=True,
        requires_active_index=True,
    ),
    "draft_report": ActionEvidenceRequirement(
        requires_document=True,
        requires_redaction_approved=True,
        requires_active_index=True,
    ),
    "generate_verified_code": ActionEvidenceRequirement(
        requires_document=False,
        requires_redaction_approved=False,
        requires_active_index=False,
    ),
    "calculator": ActionEvidenceRequirement(
        requires_document=False,
        requires_redaction_approved=False,
        requires_active_index=False,
    ),
    "document_report": ActionEvidenceRequirement(
        requires_document=True,
        requires_redaction_approved=False,
        requires_active_index=True,
    ),
}


def get_action_risk_level(action_name: str) -> ActionRiskLevel:
    return ACTION_RISK_LEVELS.get(action_name, ActionRiskLevel.CRITICAL)


def get_action_evidence_requirement(action_name: str) -> ActionEvidenceRequirement:
    return ACTION_EVIDENCE_REQUIREMENTS.get(
        action_name,
        ActionEvidenceRequirement(
            requires_document=True,
            requires_redaction_approved=True,
            requires_active_index=True,
        ),
    )


def validate_action_arguments(action_name: str, arguments: dict[str, Any]) -> BaseActionArguments:
    """Strictly validates action arguments against the typed action schema.

    Rejects unknown actions, extra fields, payload size limits, local paths,
    prompt injection, and PII.
    """
    if not isinstance(arguments, dict):
        raise ActionValidationError(
            ActionFailureCode.INVALID_ARGUMENTS,
            "Action arguments must be a JSON object dictionary.",
        )

    try:
        serialized = json.dumps(arguments)
        if len(serialized.encode("utf-8")) > MAX_ACTION_PAYLOAD_BYTES:
            raise ActionValidationError(
                ActionFailureCode.PAYLOAD_TOO_LARGE,
                f"Arguments payload exceeds {MAX_ACTION_PAYLOAD_BYTES} bytes limit.",
            )
    except (TypeError, OverflowError) as exc:
        raise ActionValidationError(
            ActionFailureCode.INVALID_ARGUMENTS,
            "Arguments could not be serialized to JSON.",
        ) from exc

    schema_cls = ACTION_ARGUMENT_SCHEMAS.get(action_name)
    if not schema_cls:
        raise ActionValidationError(
            ActionFailureCode.UNKNOWN_ACTION,
            f"Action '{action_name}' is not in the permitted local action catalog.",
        )

    try:
        return schema_cls.model_validate(arguments)
    except ActionValidationError:
        raise
    except ValidationError as exc:
        for err in exc.errors():
            ctx = err.get("ctx")
            if isinstance(ctx, dict):
                inner_err = ctx.get("error")
                if isinstance(inner_err, ActionValidationError):
                    raise inner_err from exc
            if err.get("type") == "extra_forbidden":
                raise ActionValidationError(
                    ActionFailureCode.EXTRA_FIELDS_FORBIDDEN,
                    f"Extra field not permitted: {err.get('loc')}",
                ) from exc
        raise ActionValidationError(
            ActionFailureCode.SCHEMA_VALIDATION_FAILED,
            f"Validation error for action '{action_name}': {str(exc)}",
        ) from exc
    except Exception as exc:
        raise ActionValidationError(
            ActionFailureCode.SCHEMA_VALIDATION_FAILED,
            f"Validation error for action '{action_name}': {str(exc)}",
        ) from exc
