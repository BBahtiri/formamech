"""S26 anti-scope (ADR 0108 decision 10): in-plane fibres only, S25 untouched."""

from __future__ import annotations

import pytest

from piginot.contracts.energy_formula import (
    BUILTIN_FORMULAS,
    MAX_FIBRE_FAMILIES,
    EnergyFormula,
)
from piginot.materials.values import fibre_directions

# Identities of the S25 formulas, as registered at G25 (14 Oct 2026).
S25_IDENTITIES = {
    "mooney_rivlin": "93ecc32fef55e97d",
    "yeoh": "a6b88af23cec1dca",
    "gent": "4d80109e586e26cf",
}
S25_USER_MODEL = "energy-fcc67c3a3825f6cd"


def test_the_isotropic_identities_are_unchanged() -> None:
    for model, prefix in S25_IDENTITIES.items():
        assert BUILTIN_FORMULAS[model].formula_hash[:16] == prefix
    user = EnergyFormula.from_document(
        {
            "energy": "c / 2 * (I1bar - 3) + k / 2 * (J - 1) ** 2",
            "parameters": {"c": "MPa", "k": "MPa"},
            "positive": ["c", "k"],
            "small_strain": {"shear_modulus": "c", "bulk_modulus": "k"},
        }
    )
    assert user.model_name == S25_USER_MODEL


@pytest.mark.parametrize("angle", [-120.0, -30.0, 0.0, 17.5, 90.0, 400.0])
def test_fibres_stay_in_the_plane(angle: float) -> None:
    for model in ("hgo", "hgo_single", "goh"):
        directions = fibre_directions(BUILTIN_FORMULAS[model], {"fibre_angle": angle})
        for x, y, z in directions.values():
            assert z == 0.0
            assert x * x + y * y == pytest.approx(1.0)


def test_at_most_two_families_and_no_fibre_field() -> None:
    assert MAX_FIBRE_FAMILIES == 2
    for formula in BUILTIN_FORMULAS.values():
        assert len(formula.fibres) <= MAX_FIBRE_FAMILIES
        for reference in formula.fibres.values():
            assert reference.lstrip("-").isidentifier()
