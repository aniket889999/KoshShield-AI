"""Make targets keep inspection, preparation and frontend/backend gates distinct."""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("target", "expected", "forbidden"),
    [
        ("dependency-inventory", "--wheelhouse", "--prepare"),
        ("prepare-dependencies", "--prepare", "pip install"),
        ("test-api", "pytest apps/api/tests", "pnpm"),
        ("lint-api", "ruff check apps/api scripts", "pnpm"),
        ("runtime-preflight", "python -B -m koshshield.runtime_preflight", "--probe-"),
        ("runtime-probes", "--probe-services --probe-storage", "docker compose up"),
        ("evaluate-fixtures", "python -B -m koshshield.evaluation.reference", "koshshield.smoke"),
    ],
)
def test_make_target_contract(target: str, expected: str, forbidden: str) -> None:
    result = subprocess.run(
        ["make", "--dry-run", target, "WHEELHOUSE=.temp_requirements"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert expected in result.stdout
    assert forbidden not in result.stdout
