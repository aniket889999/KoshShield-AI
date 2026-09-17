import copy

import pytest

from koshshield.evaluation.contracts import parse_ground_truth
from koshshield.evaluation.fixtures import FixtureValidationError, load_fixture_bundle
from koshshield.evaluation.procurement import compare_proposals, evaluate_measurements, parse_case
from koshshield.smoke import _find_fixture_dir


def source() -> dict:
    return copy.deepcopy(
        load_fixture_bundle(_find_fixture_dir()).evaluation_json("source_case.json")
    )


def test_reference_calculations_match_independently_stored_answer_key() -> None:
    bundle = load_fixture_bundle(_find_fixture_dir())
    case = parse_case(source(), bundle.case_id)
    truth = parse_ground_truth(
        bundle.evaluation_json("expected/proc001-ground-truth.json"), bundle.case_id
    )
    comparison = compare_proposals(case)
    assert sum(len(rows) for rows in comparison.values()) == 12
    assert comparison == truth.comparison_expected
    assert comparison["Supplier C"]["R3"] == "MISSING"
    measured = evaluate_measurements(bundle.files["inputs/proc001-measurements.csv"], case)
    assert measured == [row.model_dump() for row in truth.measurement_expected]


def test_changed_source_value_changes_prediction_without_reading_labels() -> None:
    data = source()
    data["proposals"][1]["flow"] = 85
    assert compare_proposals(parse_case(data, "PROC-001"))["Supplier B"]["R1"] == "SUPPORTED"
    data["proposals"][1]["flow"] = None
    assert compare_proposals(parse_case(data, "PROC-001"))["Supplier B"]["R1"] == "MISSING"


@pytest.mark.parametrize(("days", "expected"), [(30, "SUPPORTED"), (31, "CONTRADICTED")])
def test_delivery_upper_bound_is_inclusive(days: int, expected: str) -> None:
    data = source()
    data["proposals"][0]["delivery"] = days
    assert compare_proposals(parse_case(data, "PROC-001"))["Supplier A"]["R3"] == expected


@pytest.mark.parametrize("fault", ["units", "operator", "duplicate", "nan", "bool", "negative"])
def test_invalid_source_rules_and_values_are_rejected(fault: str) -> None:
    data = source()
    if fault == "units":
        data["requirements"][0]["unit"] = "bar"
    elif fault == "operator":
        data["requirements"][0]["operator"] = "eval"
    elif fault == "duplicate":
        data["requirements"].append(data["requirements"][0])
    else:
        data["proposals"][0]["flow"] = {"nan": float("nan"), "bool": True, "negative": -1}[fault]
    with pytest.raises(FixtureValidationError, match="SOURCE_CASE_INVALID"):
        parse_case(data, "PROC-001")


@pytest.mark.parametrize(
    "cell", ["NaN", "Infinity", "-1", "1e9", "=1+1", "", " 80", "8O", "9" * 33]
)
def test_csv_invalid_numbers_fail_closed(cell: str) -> None:
    content = f"equipment_id,flow_m3_h,pressure_bar\nP-1,{cell},12\n".encode()
    with pytest.raises(FixtureValidationError, match="MEASUREMENT_NUMBER_INVALID"):
        evaluate_measurements(content, parse_case(source(), "PROC-001"))


@pytest.mark.parametrize(
    "body",
    [
        "",
        "P-1,80\n",
        "P-1,80,12,unexpected\n",
        "P-1,80,12\nP-1,80,12\n",
        "=bad,80,12\n",
    ],
)
def test_csv_empty_ragged_and_duplicate_rows_rejected(body: str) -> None:
    with pytest.raises(FixtureValidationError):
        evaluate_measurements(
            ("equipment_id,flow_m3_h,pressure_bar\n" + body).encode(),
            parse_case(source(), "PROC-001"),
        )


def test_decimal_boundary_values_and_source_defined_thresholds() -> None:
    data = source()
    content = b"equipment_id,flow_m3_h,pressure_bar\nP-1,80,12.0000000001\n"
    assert (
        evaluate_measurements(content, parse_case(data, "PROC-001"))[0]["failed_requirements"] == []
    )
    data["requirements"][1]["value"] = 12.0000000002
    assert evaluate_measurements(content, parse_case(data, "PROC-001"))[0][
        "failed_requirements"
    ] == ["R2"]
