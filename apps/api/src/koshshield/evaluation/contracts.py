"""Strict contracts for the evaluation-only PROC-001 answer key."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from koshshield.evaluation.fixtures import FixtureValidationError

Page = Annotated[int, Field(gt=0, le=500)]
Number = Annotated[int | float, Field(allow_inf_nan=False)]
Verdict = Literal["SUPPORTED", "CONTRADICTED", "MISSING"]


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Fact(StrictRecord):
    field: str = Field(min_length=1, max_length=64)
    value: Number | None = None
    proposed: Number | None = None
    required_minimum: Number | None = None
    unit: Literal["m3/h", "bar", "months", "days"] | None = None
    reason: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def valid_shape(self) -> Self:
        comparison = self.proposed is not None or self.required_minimum is not None
        if comparison and (
            self.proposed is None or self.required_minimum is None or self.value is not None
        ):
            raise ValueError("Comparison requires proposed and required_minimum only")
        known = self.value is not None or comparison
        if known and (self.unit is None or self.reason is not None):
            raise ValueError("Known facts require a unit and no unknown reason")
        if not known and (self.reason is None or self.unit is not None):
            raise ValueError("Unknown facts require a reason and no unit")
        return self


class Question(StrictRecord):
    id: str = Field(pattern=r"^Q[1-9][0-9]{0,3}$")
    query: str = Field(min_length=1, max_length=2000)
    required_evidence_pages: list[Page] = Field(min_length=1, max_length=100)
    required_facts: list[Fact] = Field(min_length=1, max_length=50)
    insufficient_evidence: bool
    visual_probe: bool = False

    @model_validator(mode="after")
    def consistent_unknowns(self) -> Self:
        if len(set(self.required_evidence_pages)) != len(self.required_evidence_pages):
            raise ValueError("Duplicate evidence page")
        unknowns = [f.value is None and f.proposed is None for f in self.required_facts]
        if any(unknown != self.insufficient_evidence for unknown in unknowns):
            raise ValueError("Expected facts disagree with insufficient_evidence")
        return self


class PiiFixture(StrictRecord):
    type: Literal["PAN", "PHONE", "EMAIL"]
    value: str = Field(min_length=1, max_length=256)
    page: Page


class MeasurementExpected(StrictRecord):
    equipment_id: str = Field(pattern=r"^[A-Za-z0-9-]{1,64}$")
    failed_requirements: list[str] = Field(max_length=50)


class GroundTruth(StrictRecord):
    schema_version: int = Field(ge=1, le=1)
    case_id: str = Field(pattern=r"^[A-Z0-9-]{1,64}$")
    synthetic: bool
    purpose: str = Field(min_length=1, max_length=1000)
    input_variants: list[str] = Field(min_length=2, max_length=8)
    isolate_variants: bool
    expected_pages: Page
    pii: list[PiiFixture] = Field(min_length=1, max_length=50)
    questions: list[Question] = Field(min_length=1, max_length=100)
    comparison_expected: dict[str, dict[str, Verdict]]
    measurement_expected: list[MeasurementExpected] = Field(min_length=1, max_length=1000)
    grading_notes: list[str] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def relationships(self) -> Self:
        if not self.synthetic or not self.isolate_variants:
            raise ValueError("Synthetic isolated variants are mandatory")
        for identities in (
            self.input_variants,
            [q.id for q in self.questions],
            [m.equipment_id for m in self.measurement_expected],
        ):
            if len(set(identities)) != len(identities):
                raise ValueError("Duplicate evaluation identity")
        if not self.comparison_expected or any(
            not rows for rows in self.comparison_expected.values()
        ):
            raise ValueError("Expected comparison must not be empty")
        pages = [p.page for p in self.pii] + [
            p for q in self.questions for p in q.required_evidence_pages
        ]
        if any(page > self.expected_pages for page in pages):
            raise ValueError("Evidence page exceeds fixture page count")
        return self


def parse_ground_truth(value: dict, case_id: str) -> GroundTruth:
    try:
        truth = GroundTruth.model_validate(value)
        if truth.case_id != case_id:
            raise FixtureValidationError("FIXTURE_CASE_MISMATCH")
        return truth
    except ValidationError:
        raise FixtureValidationError("GROUND_TRUTH_INVALID") from None
