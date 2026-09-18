"""Read-only known-answer fixture consistency report; never invokes an AI model."""

import argparse
import json
from pathlib import Path

from koshshield.evaluation.contracts import parse_ground_truth
from koshshield.evaluation.fixtures import (
    FixtureValidationError,
    default_fixture_dir,
    load_fixture_bundle,
)
from koshshield.evaluation.procurement import compare_proposals, evaluate_measurements, parse_case


def run_reference_evaluation(root: Path | None = None) -> dict:
    report = {
        "report_type": "synthetic_fixture_consistency",
        "status": "FAILED",
        "failure_code": None,
        "fixture_integrity_verified": False,
        "model_evaluation_executed": False,
        "pdf_content_semantics_checked": False,
        "disclaimer": (
            "Known-answer reference calculations only; not held-out AI or industrial accuracy."
        ),
    }
    try:
        bundle = load_fixture_bundle(root if root is not None else default_fixture_dir())
        report["fixture_integrity_verified"] = True
        truth = parse_ground_truth(
            bundle.evaluation_json("expected/proc001-ground-truth.json"), bundle.case_id
        )
        case = parse_case(bundle.evaluation_json("source_case.json"), bundle.case_id)
        for name in truth.input_variants:
            bundle.input_pdf(name)
        csv_path = "inputs/proc001-measurements.csv"
        if bundle.roles.get(csv_path) != "input":
            raise FixtureValidationError("MEASUREMENT_INPUT_MISSING")
        predictions = compare_proposals(case)
        expected = {
            (s, r): v for s, rows in truth.comparison_expected.items() for r, v in rows.items()
        }
        observed = {(s, r): v for s, rows in predictions.items() for r, v in rows.items()}
        comparison_mismatches = sum(
            expected.get(key) != observed.get(key) for key in expected.keys() | observed.keys()
        )
        measurements = evaluate_measurements(bundle.files[csv_path], case)
        actual_rows = {
            row["equipment_id"]: sorted(row["failed_requirements"]) for row in measurements
        }
        expected_rows = {
            row.equipment_id: sorted(row.failed_requirements) for row in truth.measurement_expected
        }
        measurement_mismatches = sum(
            expected_rows.get(key) != actual_rows.get(key)
            for key in expected_rows.keys() | actual_rows.keys()
        )
        report.update(
            {
                "case_id": case.case_id,
                "input_variant_count": len(truth.input_variants),
                "comparison_checks": len(observed),
                "comparison_mismatches": comparison_mismatches,
                "measurement_checks": len(actual_rows),
                "measurement_mismatches": measurement_mismatches,
                "missing_proposal_values": sum(value == "MISSING" for value in observed.values()),
            }
        )
        if comparison_mismatches or measurement_mismatches:
            report["failure_code"] = "REFERENCE_RESULT_MISMATCH"
        else:
            report["status"] = "PASSED"
    except FixtureValidationError as exc:
        report["failure_code"] = exc.code
    except Exception:
        report["failure_code"] = "REFERENCE_EVALUATION_FAILED"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check synthetic procurement reference calculations"
    )
    parser.add_argument("--fixture-dir", type=Path, help="Local synthetic fixture package")
    args = parser.parse_args(argv)
    report = run_reference_evaluation(args.fixture_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
