import copy

import pytest

from koshshield.evaluation.contracts import parse_ground_truth
from koshshield.evaluation.fixtures import FixtureValidationError, load_fixture_bundle
from koshshield.smoke import (
    ALL_STAGES_IN_ORDER,
    StageResult,
    StageStatus,
    _compile_variant_report,
    _find_fixture_dir,
)


def truth_dict() -> dict:
    return copy.deepcopy(
        load_fixture_bundle(_find_fixture_dir()).evaluation_json(
            "expected/proc001-ground-truth.json"
        )
    )


def test_existing_answer_key_satisfies_strict_contract() -> None:
    truth = parse_ground_truth(truth_dict(), "PROC-001")
    assert len(truth.questions) == 5
    assert truth.questions[3].visual_probe is True


@pytest.mark.parametrize(
    "fault",
    [
        "empty_questions",
        "empty_variants",
        "duplicate_question",
        "duplicate_variant",
        "page",
        "schema_bool",
        "synthetic",
        "isolated",
        "unit",
        "infinity",
        "bool_value",
        "unknown",
        "comparison",
        "extra",
    ],
)
def test_bad_answer_key_cannot_silently_weaken_evaluation(fault: str) -> None:
    data = truth_dict()
    if fault == "empty_questions":
        data["questions"] = []
    elif fault == "empty_variants":
        data["input_variants"] = []
    elif fault == "duplicate_question":
        data["questions"].append(data["questions"][0])
    elif fault == "duplicate_variant":
        data["input_variants"][1] = data["input_variants"][0]
    elif fault == "page":
        data["questions"][0]["required_evidence_pages"] = [999]
    elif fault == "schema_bool":
        data["schema_version"] = True
    elif fault == "synthetic":
        data["synthetic"] = False
    elif fault == "isolated":
        data["isolate_variants"] = False
    elif fault == "unit":
        data["questions"][0]["required_facts"][0]["unit"] = "kg"
    elif fault in {"infinity", "bool_value"}:
        data["questions"][0]["required_facts"][0]["value"] = (
            float("inf") if fault == "infinity" else True
        )
    elif fault == "unknown":
        data["questions"][2]["insufficient_evidence"] = False
    elif fault == "comparison":
        del data["questions"][1]["required_facts"][0]["required_minimum"]
    else:
        data["ignored_new_field"] = "secret"
    with pytest.raises(FixtureValidationError, match="GROUND_TRUTH_INVALID"):
        parse_ground_truth(data, "PROC-001")


def test_case_mismatch_is_rejected() -> None:
    with pytest.raises(FixtureValidationError, match="FIXTURE_CASE_MISMATCH"):
        parse_ground_truth(truth_dict(), "ANOTHER")


def test_missing_stages_cannot_produce_vacuous_success() -> None:
    report = _compile_variant_report("demo.pdf", {}, {}, None)
    assert report.status == StageStatus.NOT_EXECUTED
    assert len(report.stages) == len(ALL_STAGES_IN_ORDER)


def test_empty_question_results_cannot_pass_answer_validation() -> None:
    stages = {name: StageResult(StageStatus.PASSED) for name in ALL_STAGES_IN_ORDER}
    report = _compile_variant_report("demo.pdf", stages, {}, True)
    assert report.status == StageStatus.FAILED
    assert report.stages["answer_validation"]["failure_code"] == "QUESTION_RESULTS_INCOMPLETE"
