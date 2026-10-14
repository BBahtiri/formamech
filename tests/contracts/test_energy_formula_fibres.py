"""S26 fibre families in the formula language (ADR 0108 decisions 1-3)."""

from __future__ import annotations

from typing import Any

import pytest

from piginot.contracts import CanonicalQuantity, Unit
from piginot.contracts.energy_formula import (
    BUILTIN_FORMULAS,
    EnergyFormula,
    EnergyFormulaError,
    fibre_angle,
    fibre_invariant_names,
    model_dimensions,
    parse,
)
from piginot.contracts.units import Dimension, Quantity
from piginot.materials.values import fibre_directions, formula_value


def _document(**changes: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "energy": "mu / 2 * (I1bar - 3) + k / 2 * (J - 1) ** 2"
        " + k1 * ramp(I4a - 1) ** 2",
        "parameters": {"mu": "MPa", "k": "MPa", "k1": "MPa", "theta": "deg"},
        "positive": ["mu", "k"],
        "small_strain": {"shear_modulus": "mu", "bulk_modulus": "k"},
        "fibres": {"a": "theta"},
    }
    document.update(changes)
    return document


def test_a_fibre_formula_round_trips_and_is_fibre_reinforced() -> None:
    formula = EnergyFormula.from_document(_document())
    assert formula.symmetry == "fibre-reinforced"
    assert formula.fibre_invariants == ("I4a", "I5a")
    assert EnergyFormula.from_document(formula.document()).formula_hash == (
        formula.formula_hash
    )
    assert formula.document()["fibres"] == {"a": "theta"}
    assert formula.parameters["theta"] is Unit.DEG


def test_fibres_and_bounds_leave_isotropic_identities_alone() -> None:
    isotropic = EnergyFormula.from_document(
        {
            "energy": "mu / 2 * (I1bar - 3) + k / 2 * (J - 1) ** 2",
            "parameters": {"mu": "MPa", "k": "MPa"},
            "small_strain": {"shear_modulus": "mu", "bulk_modulus": "k"},
        }
    )
    document = isotropic.document(canonical=True)
    assert "fibres" not in document and "bounds" not in document
    assert document["symmetry"] == "isotropic"


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"fibres": {"a": "mu"}}, "not a parameter in deg"),
        ({"fibres": {"a": "phi"}}, "not a parameter in deg"),
        ({"fibres": {"a": "2 * theta"}}, "parameter or its negative"),
        ({"fibres": {"A": "theta"}}, "lowercase"),
        ({"fibres": {"toolongname": "theta"}}, "lowercase"),
        ({"fibres": {"a": "theta", "b": "theta", "c": "theta"}}, "at most 2"),
        ({"fibres": []}, "map labels"),
        ({"energy": "mu / 2 * (I1bar - 3) + k * (J - 1) ** 2 + k1 * I4b"}, "unknown"),
        ({"energy": "mu * (I1 - 3) + k * (J - 1) ** 2 + k1 * ramp(k1)"}, "dimension"),
        ({"energy": "mu * (I1 - 3) + k * (J-1)**2 + k1 * I4a * theta"}, "unknown"),
        (
            {"small_strain": {"shear_modulus": "mu * I4a", "bulk_modulus": "k"}},
            "parameters only",
        ),
        ({"symmetry": "isotropic"}, "exactly when fibres"),
        ({"fibres": {}}, "deg parameters must be fibre angles"),
        (
            {"parameters": {"mu": "MPa", "k": "MPa", "k1": "MPa", "I4a": "deg"}},
            "not allowed",
        ),
        ({"bounds": {"theta": [0, 1]}}, "unknown parameter"),
        ({"bounds": {"k1": [2, 1]}}, "lo <= hi"),
        ({"bounds": {"k1": "1"}}, r"\[lo, hi\]"),
        ({"bounds": []}, r"\[lo, hi\]"),
    ],
)
def test_malformed_fibre_formulas_are_refused(
    changes: dict[str, Any], match: str
) -> None:
    with pytest.raises(EnergyFormulaError, match=match):
        EnergyFormula.from_document(_document(**changes))


def test_bounds_are_checked_on_values() -> None:
    formula = BUILTIN_FORMULAS["goh"]
    values = {
        "shear_modulus": 300.0,
        "bulk_modulus": 500.0,
        "k1": 50.0,
        "k2": 2.0,
        "fibre_angle": 30.0,
        "kappa_d": 0.1,
    }
    formula.check_values(values)
    with pytest.raises(EnergyFormulaError, match="must lie in"):
        formula.check_values({**values, "kappa_d": 0.4})


def test_angle_references_and_invariant_names() -> None:
    assert fibre_angle("beta") == (1, "beta")
    assert fibre_angle("-beta") == (-1, "beta")
    for bad in ("--beta", "2beta", "", 3):
        with pytest.raises(EnergyFormulaError):
            fibre_angle(bad)  # type: ignore[arg-type]
    assert fibre_invariant_names(("a", "b")) == ("I4a", "I5a", "I4b", "I5b")


def test_the_fibre_builtins() -> None:
    hgo = BUILTIN_FORMULAS["hgo"]
    assert dict(hgo.fibres) == {"a": "fibre_angle", "b": "-fibre_angle"}
    assert dict(BUILTIN_FORMULAS["hgo_single"].fibres) == {"a": "fibre_angle"}
    assert BUILTIN_FORMULAS["goh"].bounds == {"kappa_d": (0.0, 1.0 / 3.0)}
    assert model_dimensions("hgo")["fibre_angle"] is Dimension.ANGLE
    assert "k1" not in hgo.positive and "k2" in hgo.positive
    directions = fibre_directions(hgo, {"fibre_angle": 30.0})
    assert directions["a"][0] == pytest.approx(3**0.5 / 2)
    assert directions["b"][1] == pytest.approx(-0.5)


def test_ramp_is_the_tension_switch() -> None:
    assert formula_value(parse("ramp(x - 1)"), {"x": 1.5}) == 0.5
    assert formula_value(parse("ramp(x - 1)"), {"x": 0.5}) == 0.0


def test_the_deg_unit_is_a_canonical_angle() -> None:
    assert Unit.DEG.dimension is Dimension.ANGLE and Unit.DEG.is_canonical
    canonical = Quantity(30.0, Unit.DEG).to_canonical(
        expected_dimension=Dimension.ANGLE
    )
    assert canonical == CanonicalQuantity(30.0, Unit.DEG, Dimension.ANGLE)
