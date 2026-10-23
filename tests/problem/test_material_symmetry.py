"""S26: fibre materials in a ProblemSpec and the material-symmetry rule."""

from __future__ import annotations

from typing import Any

import pytest

from piginot.contracts import CanonicalQuantity, ProblemSpec, ProblemSpecError
from piginot.contracts.units import Dimension, Unit
from piginot.problem import admit_material, fem_request, material_spec
from tests.problem.test_problem import PROFILE
from tests.problem_helpers import plate_document, plate_mesh

_STRESS = {"shear": 300.0, "bulk": 500.0, "k1": 50.0}


def _parameter(name: str, lo: float, hi: float, unit: str) -> dict[str, Any]:
    return {
        "name": name,
        "role": "material",
        "lo": lo,
        "hi": hi,
        "unit": unit,
        "distribution": "fixed" if lo == hi else "uniform",
    }


def _document(
    model: str,
    angle: tuple[float, float],
    *,
    symmetry: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    document = plate_document()
    parameters = [p for p in document["parameters"] if p["role"] != "material"]
    parameters += [_parameter(n, v, v, "MPa") for n, v in _STRESS.items()]
    parameters += [_parameter("k2", 2.0, 2.0, "1"), _parameter("beta", *angle, "deg")]
    document["parameters"] = parameters
    document["material"] = {
        "model": model,
        "parameters": {
            "shear_modulus": "shear",
            "bulk_modulus": "bulk",
            "k1": "k1",
            "k2": "k2",
            "fibre_angle": "beta",
        },
        **(extra or {}),
    }
    if not symmetry:
        for condition in document["boundary_conditions"]:
            if condition["kind"] == "symmetry":
                condition["kind"] = "dirichlet"
                condition["values"] = [{"expression": "0", "unit": "mm"}]
    return document


@pytest.mark.parametrize(
    ("model", "angle"),
    [("hgo", (30.0, 30.0)), ("hgo", (10.0, 50.0)), ("hgo_single", (0.0, 0.0))],
)
def test_symmetric_fibre_materials_suit_a_quarter_model(
    model: str, angle: tuple[float, float]
) -> None:
    spec = ProblemSpec.from_document(_document(model, angle))
    assert spec.material.dimensions["fibre_angle"] is Dimension.ANGLE
    assert spec.parameter("beta").unit is Unit.DEG


@pytest.mark.parametrize(
    ("model", "angle"),
    [("hgo_single", (30.0, 30.0)), ("hgo_single", (0.0, 10.0))],
)
def test_an_off_axis_single_family_is_refused_by_name(
    model: str, angle: tuple[float, float]
) -> None:
    with pytest.raises(ProblemSpecError) as error:
        ProblemSpec.from_document(_document(model, angle))
    assert error.value.rule == "material-symmetry"


def test_without_symmetry_conditions_a_single_family_is_accepted() -> None:
    spec = ProblemSpec.from_document(
        _document("hgo_single", (30.0, 30.0), symmetry=False)
    )
    assert spec.material.model == "hgo_single"


def test_user_fibre_formulas_follow_the_rule() -> None:
    energy = {
        "energy": "mu / 2 * (I1bar - 3) + k / 2 * (J - 1) ** 2"
        " + c * ramp(I4a - 1) ** 2 + c * ramp(I4b - 1) ** 2",
        "parameters": {"mu": "MPa", "k": "MPa", "c": "MPa", "ta": "deg", "tb": "deg"},
        "small_strain": {"shear_modulus": "mu", "bulk_modulus": "k"},
        "fibres": {"a": "ta", "b": "tb"},
    }

    def spec(ta: float, tb: float) -> dict[str, Any]:
        document = plate_document()
        parameters = [p for p in document["parameters"] if p["role"] != "material"]
        parameters += [
            _parameter("mu", 300.0, 300.0, "MPa"),
            _parameter("k", 500.0, 500.0, "MPa"),
            _parameter("c", 50.0, 50.0, "MPa"),
            _parameter("ta", ta, ta, "deg"),
            _parameter("tb", tb, tb, "deg"),
        ]
        document["parameters"] = parameters
        document["material"] = {
            "energy": energy,
            "parameters": {"mu": "mu", "k": "k", "c": "c", "ta": "ta", "tb": "tb"},
        }
        return document

    assert ProblemSpec.from_document(spec(25.0, -25.0)).material.energy is not None
    assert ProblemSpec.from_document(spec(0.0, 90.0)).material.energy is not None
    with pytest.raises(ProblemSpecError, match="mirror"):
        ProblemSpec.from_document(spec(25.0, 40.0))


def test_a_fibre_spec_is_admitted_and_builds_a_fem_request() -> None:
    spec = ProblemSpec.from_document(_document("hgo", (30.0, 30.0)))
    assert admit_material(spec).passed
    values = {
        name: spec.parameter(name).lo for name in spec.material.parameters.values()
    }
    material = material_spec(spec, values)
    assert material.properties["fibre_angle"] == CanonicalQuantity(
        30.0, Unit.DEG, Dimension.ANGLE
    )
    inputs = {
        "width": CanonicalQuantity(16.0, Unit.MM, Dimension.LENGTH),
        "height": CanonicalQuantity(6.0, Unit.MM, Dimension.LENGTH),
        "strain": CanonicalQuantity(0.08, Unit.ONE, Dimension.DIMENSIONLESS),
        **{
            name: material.properties[prop]
            for prop, name in spec.material.parameters.items()
        },
    }
    request = fem_request(spec, inputs, plate_mesh(), "case", PROFILE)
    assert request.material.model == "hgo"
