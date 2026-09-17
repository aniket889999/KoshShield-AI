"""Reference calculations for synthetic procurement fixtures, never contract awards."""

import csv
import io
import re
from decimal import Decimal
from typing import Literal, Self

from pydantic import Field, ValidationError, model_validator

from koshshield.evaluation.contracts import Number, StrictRecord
from koshshield.evaluation.fixtures import FixtureValidationError

UNITS = {"flow": "m3/h", "pressure": "bar", "delivery": "days", "warranty": "months"}


class Requirement(StrictRecord):
    id: str = Field(pattern=r"^R[1-9][0-9]{0,3}$")
    field: Literal["flow", "pressure", "delivery", "warranty"]
    operator: Literal[">=", "<="]
    value: Number = Field(ge=0, le=1_000_000_000_000)
    unit: Literal["m3/h", "bar", "days", "months"]

    @model_validator(mode="after")
    def matching_unit(self) -> Self:
        if self.unit != UNITS[self.field]:
            raise ValueError("Requirement field/unit mismatch")
        return self


class Proposal(StrictRecord):
    supplier: str = Field(min_length=1, max_length=80)
    flow: Number | None = Field(ge=0, le=1_000_000_000_000)
    pressure: Number | None = Field(ge=0, le=1_000_000_000_000)
    delivery: Number | None = Field(ge=0, le=1_000_000_000_000)
    warranty: Number | None = Field(ge=0, le=1_000_000_000_000)


class FictionalContact(StrictRecord):
    label: str = Field(min_length=1, max_length=128)
    pan: str = Field(min_length=1, max_length=64)
    phone: str = Field(min_length=1, max_length=64)
    email: str = Field(min_length=1, max_length=256)


class ProcurementCase(StrictRecord):
    case_id: str = Field(pattern=r"^[A-Z0-9-]{1,64}$")
    title: str = Field(min_length=1, max_length=256)
    synthetic: bool
    notice: str = Field(min_length=1, max_length=1000)
    contact: FictionalContact
    requirements: list[Requirement] = Field(min_length=1, max_length=50)
    proposals: list[Proposal] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def unique_records(self) -> Self:
        if not self.synthetic:
            raise ValueError("Only synthetic reference cases are supported")
        for identifiers in (
            [r.id for r in self.requirements],
            [r.field for r in self.requirements],
            [p.supplier for p in self.proposals],
        ):
            if len(set(identifiers)) != len(identifiers) or any(not i.strip() for i in identifiers):
                raise ValueError("Duplicate or blank source identity")
        return self


def parse_case(value: dict, case_id: str) -> ProcurementCase:
    try:
        case = ProcurementCase.model_validate(value)
        if case.case_id != case_id:
            raise FixtureValidationError("FIXTURE_CASE_MISMATCH")
        return case
    except ValidationError:
        raise FixtureValidationError("SOURCE_CASE_INVALID") from None


def meets_requirement(value: Decimal, requirement: Requirement) -> bool:
    threshold = Decimal(str(requirement.value))
    return value >= threshold if requirement.operator == ">=" else value <= threshold


def compare_proposals(case: ProcurementCase) -> dict[str, dict[str, str]]:
    results = {}
    for proposal in case.proposals:
        outcomes = {}
        for requirement in case.requirements:
            value = getattr(proposal, requirement.field)
            outcomes[requirement.id] = (
                "MISSING"
                if value is None
                else "SUPPORTED"
                if meets_requirement(Decimal(str(value)), requirement)
                else "CONTRADICTED"
            )
        results[proposal.supplier] = outcomes
    return results


def evaluate_measurements(content: bytes, case: ProcurementCase) -> list[dict]:
    """Read the declared CSV units exactly; no eval, automatic conversion or rounding."""
    if not content or len(content) > 1024 * 1024:
        raise FixtureValidationError("MEASUREMENT_SIZE_INVALID")
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")), strict=True)
        headers = ["equipment_id", "flow_m3_h", "pressure_bar"]
        if reader.fieldnames != headers:
            raise FixtureValidationError("MEASUREMENT_COLUMNS_INVALID")
        rules = {r.field: r for r in case.requirements}
        if not {"flow", "pressure"} <= rules.keys():
            raise FixtureValidationError("MEASUREMENT_REQUIREMENTS_MISSING")
        results = []
        seen = set()
        for row in reader:
            if len(results) >= 1000:
                raise FixtureValidationError("MEASUREMENT_ROW_LIMIT")
            if set(row) != set(headers) or any(value is None for value in row.values()):
                raise FixtureValidationError("MEASUREMENT_COLUMNS_INVALID")
            identifier = row["equipment_id"]
            if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", identifier) or identifier in seen:
                raise FixtureValidationError("MEASUREMENT_ID_INVALID")
            seen.add(identifier)
            failed = []
            for column, field_name in (("flow_m3_h", "flow"), ("pressure_bar", "pressure")):
                cell = row[column]
                if len(cell) > 32 or not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", cell):
                    raise FixtureValidationError("MEASUREMENT_NUMBER_INVALID")
                value = Decimal(cell)
                if value > Decimal("1000000000000"):
                    raise FixtureValidationError("MEASUREMENT_NUMBER_INVALID")
                if not meets_requirement(value, rules[field_name]):
                    failed.append(rules[field_name].id)
            results.append({"equipment_id": identifier, "failed_requirements": failed})
        if not results:
            raise FixtureValidationError("MEASUREMENT_ROWS_MISSING")
        return results
    except (UnicodeError, csv.Error):
        raise FixtureValidationError("MEASUREMENT_CSV_INVALID") from None
