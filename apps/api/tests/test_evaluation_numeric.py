import pytest

from koshshield.evaluation.contracts import Fact
from koshshield.evaluation.numeric_checks import contains_synthetic_pii, numeric_facts_present


@pytest.mark.parametrize("answer", ["80 m3/h", "80.0 m3/h", "80m^3/hr", "80 m\u00b3/h"])
def test_equal_quantities_and_common_unit_spellings(answer: str) -> None:
    assert numeric_facts_present(answer, [Fact(field="flow", value=80, unit="m3/h")]) is True


@pytest.mark.parametrize(
    "answer",
    [
        "180 m3/h",
        "800 m3/h",
        "0.80 m3/h",
        "-80 m3/h",
        "80 bar",
        "80",
        "80 m3/s",
        "X80 m3/h",
        "1,080 m3/h",
    ],
)
def test_digit_substrings_wrong_units_and_wrong_sign_do_not_pass(answer: str) -> None:
    assert numeric_facts_present(answer, [Fact(field="flow", value=80, unit="m3/h")]) is False


def test_both_proposed_and_required_quantities_are_checked() -> None:
    facts = [
        Fact(field="flow", proposed=75, required_minimum=80, unit="m3/h"),
        Fact(field="warranty", proposed=12, required_minimum=24, unit="months"),
    ]
    assert numeric_facts_present("75 m3/h, 12 months", facts) is False
    assert numeric_facts_present("75 m3/h against 80 m3/h; 12 months against 24 months", facts)


def test_unknown_facts_are_ungraded_not_silently_verified() -> None:
    facts = [Fact(field="delivery", value=None, reason="Not provided")]
    assert numeric_facts_present("Delivery is unknown", facts) is None
    assert numeric_facts_present("Delivery is 10 days", facts) is None


def test_presence_check_does_not_claim_to_understand_negation() -> None:
    # A numeric pass is intentionally not a semantic support result.
    assert numeric_facts_present("This is NOT 80 m3/h", [Fact(field="flow", value=80, unit="m3/h")])


def test_synthetic_identifier_check_normalizes_case_and_width() -> None:
    assert contains_synthetic_pii("testp0000a", ["TESTP0000A"])
    assert contains_synthetic_pii(
        "\uff19\uff10\uff10\uff10\uff10\uff10\uff10\uff10\uff10\uff10", ["9000000000"]
    )
    assert not contains_synthetic_pii("[REDACTED_PAN]", ["TESTP0000A"])
