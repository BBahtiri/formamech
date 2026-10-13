"""S25 strain-energy formula language: whitelist, units, identity."""

from __future__ import annotations

import math
from typing import Any

import pytest

from piginot.contracts.energy_formula import (
    BUILTIN_FORMULAS,
    CLOSED_FORM_MODELS,
    EnergyFormula,
    EnergyFormulaError,
    Formula,
    evaluate,
    model_dimensions,
    names,
    parse,
)
from piginot.contracts.units import Dimension, Unit


def _document(**changes: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "energy": "mu / 2 * (I1 - 3) - mu * log(J) + lam / 2 * log(J) ** 2",
        "parameters": {"mu": "MPa", "lam": "MPa"},
        "positive": ["mu"],
        "small_strain": {"shear_modulus": "mu", "bulk_modulus": "lam + 2 / 3 * mu"},
    }
    document.update(changes)
    return document


def test_a_formula_round_trips_and_names_itself() -> None:
    formula = EnergyFormula.from_document(_document())
    assert EnergyFormula.from_document(formula.document()) == formula
    assert formula.model_name == f"energy-{formula.formula_hash[:16]}"
    assert formula.moduli({"mu": 3.0, "lam": 2.0}) == (3.0, 4.0)
    assert formula.parameters == {"mu": Unit.MPA, "lam": Unit.MPA}
    assert formula.document()["schema"] == "piginot.energy-formula-v1"


def test_the_identity_ignores_spelling_but_not_meaning() -> None:
    base = EnergyFormula.from_document(_document())
    respelled = EnergyFormula.from_document(
        _document(energy="(mu/2)*(I1-3) - mu*log(J) + lam/2*log(J)^2")
    )
    assert respelled.formula_hash != base.formula_hash or respelled == base
    same = EnergyFormula.from_document(
        _document(energy="mu / 2 * (I1 - 3) - mu * log(J) + lam / 2 * log(J)**2")
    )
    assert same.formula_hash == base.formula_hash
    changed = EnergyFormula.from_document(
        _document(energy="mu / 2 * (I1 - 3) - mu * log(J) + lam / 3 * log(J) ** 2")
    )
    assert changed.formula_hash != base.formula_hash


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('ls')",
        "mu.real",
        "I1[0]",
        "max(I1, 3)",
        "exp(I1, 2)",
        "exp(x=I1)",
        "I1 ** mu",
        "lambda: 1",
        "I1 if J else 3",
        "I1 < 3",
        "'text'",
        "I1 // 2",
        "",
        "1e400 * mu",
        7,
    ],
)
def test_the_whitelist_refuses_everything_else(source: object) -> None:
    with pytest.raises(EnergyFormulaError):
        parse(source)


def test_the_tree_and_its_evaluation() -> None:
    tree = parse("-a + sqrt(b) * exp(0) / 2 - log(c) + a ^ (-2/3)")
    assert names(tree) == {"a", "b", "c"}
    value = evaluate(tree, {"a": 8.0, "b": 16.0, "c": math.e})
    assert value == pytest.approx(-8.0 + 2.0 - 1.0 + 8.0 ** (-2 / 3))
    # As in Python, an unparenthesised exponent binds first: a ** -2 / 3.
    assert evaluate(parse("a ** -2/3"), {"a": 8.0}) == pytest.approx(8.0**-2 / 3)
    with pytest.raises(EnergyFormulaError, match="number"):
        parse("a ** (b - 1)")
    assert Formula.parse("x").canonical() == ["name", "x"]


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"energy": "mu * I1 * lam"}, "stress"),
        ({"energy": "mu * exp(mu)"}, "dimensionless"),
        ({"energy": "mu + I1"}, "same unit"),
        ({"energy": "mu * I3"}, "unknown name"),
        ({"energy": "mu * (I1 - 3)"}, "not used"),
        ({"parameters": {"mu": "GPa", "lam": "MPa"}}, "'MPa' or '1'"),
        ({"parameters": {"I1": "MPa", "lam": "MPa"}}, "not allowed"),
        ({"parameters": []}, "objects"),
        ({"positive": ["nu"]}, "unknown parameters"),
        ({"positive": "mu"}, "list"),
        ({"small_strain": {"shear_modulus": "mu"}}, "exactly"),
        (
            {"small_strain": {"shear_modulus": "mu * I1", "bulk_modulus": "lam"}},
            "parameters only",
        ),
        (
            {"small_strain": {"shear_modulus": "mu * mu", "bulk_modulus": "lam"}},
            "stress",
        ),
        ({"symmetry": "transversely_isotropic"}, "symmetry"),
        ({"schema": "piginot.energy-formula-v0"}, "schema"),
        ({"extra": 1}, "unknown key"),
    ],
)
def test_invalid_formulas_are_refused(changes: dict[str, Any], match: str) -> None:
    with pytest.raises(EnergyFormulaError, match=match):
        EnergyFormula.from_document(_document(**changes))


def test_missing_parts_and_non_objects_are_refused() -> None:
    with pytest.raises(EnergyFormulaError, match="object"):
        EnergyFormula.from_document([])
    document = _document()
    del document["energy"]
    with pytest.raises(EnergyFormulaError, match="missing"):
        EnergyFormula.from_document(document)


def test_parameter_values_are_checked() -> None:
    formula = EnergyFormula.from_document(_document())
    formula.check_values({"mu": 1.0, "lam": -1.0})
    with pytest.raises(EnergyFormulaError, match="exactly"):
        formula.check_values({"mu": 1.0})
    with pytest.raises(EnergyFormulaError, match="finite"):
        formula.check_values({"mu": math.nan, "lam": 1.0})
    with pytest.raises(EnergyFormulaError, match="positive"):
        formula.check_values({"mu": 0.0, "lam": 1.0})


def test_the_builtin_library() -> None:
    assert set(BUILTIN_FORMULAS) == {"mooney_rivlin", "yeoh", "gent"}
    assert set(CLOSED_FORM_MODELS) == {"compressible_neo_hookean"}
    assert model_dimensions("gent") == {
        "shear_modulus": Dimension.STRESS,
        "jm": Dimension.DIMENSIONLESS,
        "bulk_modulus": Dimension.STRESS,
    }
    assert model_dimensions("compressible_neo_hookean")["poisson_ratio"] is (
        Dimension.DIMENSIONLESS
    )
    mooney = BUILTIN_FORMULAS["mooney_rivlin"]
    assert mooney.moduli({"c10": 1.0, "c01": 2.0, "bulk_modulus": 5.0}) == (6.0, 5.0)
    assert BUILTIN_FORMULAS["yeoh"].positive == ("c10", "bulk_modulus")
    with pytest.raises(KeyError):
        model_dimensions("ogden")
