"""S25: a ProblemSpec's material — built-in models, user formulas, admission."""

from __future__ import annotations

from typing import Any

import pytest

from piginot.contracts import (
    CanonicalQuantity,
    ProblemSpec,
    ProblemSpecError,
)
from piginot.contracts.units import Dimension, Unit
from piginot.materials import formula_for
from piginot.problem import admit_material, fem_request, material_points, material_spec
from tests.problem.test_problem import PROFILE
from tests.problem_helpers import plate_document, plate_mesh

_USER = {
    "energy": "c / 2 * (I1bar - 3) + k / 2 * (J - 1) ** 2",
    "parameters": {"c": "MPa", "k": "MPa"},
    "positive": ["c", "k"],
    "small_strain": {"shear_modulus": "c", "bulk_modulus": "k"},
}


def _material_parameters(
    names: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "role": "material",
            "lo": lo,
            "hi": hi,
            "unit": "MPa",
            "distribution": "fixed" if lo == hi else "uniform",
        }
        for name, (lo, hi) in names.items()
    ]


def _document(material: dict[str, Any], ranges: dict[str, tuple[float, float]]) -> Any:
    document = plate_document()
    document["parameters"] = [
        p for p in document["parameters"] if p["role"] != "material"
    ] + _material_parameters(ranges)
    document["material"] = material
    return document


def _mooney(**changes: Any) -> ProblemSpec:
    material = {
        "model": "mooney_rivlin",
        "parameters": {"c10": "c10", "c01": "c01", "bulk_modulus": "kappa"},
        **changes,
    }
    ranges = {"c10": (100.0, 140.0), "c01": (20.0, 20.0), "kappa": (500.0, 500.0)}
    return ProblemSpec.from_document(_document(material, ranges))


def _user_spec() -> ProblemSpec:
    material = {"energy": _USER, "parameters": {"c": "shear", "k": "bulk"}}
    ranges = {"shear": (200.0, 200.0), "bulk": (600.0, 600.0)}
    return ProblemSpec.from_document(_document(material, ranges))


def test_a_builtin_model_spec_round_trips() -> None:
    spec = _mooney(stretch_range=[0.7, 1.4])
    assert spec.material.stretch_range == (0.7, 1.4)
    assert ProblemSpec.from_document(spec.document()) == spec
    assert spec.document()["material"]["stretch_range"] == [0.7, 1.4]
    assert "stretch_range" not in _mooney().document()["material"]


def test_a_user_formula_is_named_by_its_content() -> None:
    spec = _user_spec()
    assert spec.material.model.startswith("energy-")
    again = ProblemSpec.from_document(spec.document())
    assert again == spec and again.problem_hash == spec.problem_hash
    canonical = spec.canonical_form()["material"]["energy"]["energy"]
    assert isinstance(canonical, list)


@pytest.mark.parametrize(
    ("material", "rule"),
    [
        ({"model": "ogden", "parameters": {}}, "material"),
        ({"parameters": {"c": "shear", "k": "bulk"}}, "missing-key"),
        (
            {
                "model": "mooney_rivlin",
                "energy": _USER,
                "parameters": {"c": "shear", "k": "bulk"},
            },
            "material",
        ),
        ({"energy": {**_USER, "energy": "c * I1"}, "parameters": {}}, "energy"),
        (
            {
                "energy": _USER,
                "parameters": {"c": "shear", "k": "bulk"},
                "stretch_range": [1.1, 2.0],
            },
            "range",
        ),
        (
            {
                "energy": _USER,
                "parameters": {"c": "shear", "k": "bulk"},
                "stretch_range": [0.5],
            },
            "type",
        ),
        ({"energy": _USER, "parameters": {"c": "shear"}}, "material"),
        ({"energy": _USER, "parameters": {"c": "shear", "k": "width"}}, "material"),
    ],
)
def test_invalid_materials_are_refused(material: dict[str, Any], rule: str) -> None:
    ranges = {"shear": (200.0, 200.0), "bulk": (600.0, 600.0)}
    with pytest.raises(ProblemSpecError) as error:
        ProblemSpec.from_document(_document(material, ranges))
    assert error.value.rule == rule


def test_material_points_and_contracts() -> None:
    spec = _mooney()
    points = material_points(spec)
    assert {p["c10"] for p in points} == {100.0, 120.0, 140.0}
    assert len(points) == 3
    contract = material_spec(spec, points[0])
    assert contract.model == "mooney_rivlin"
    assert contract.properties["bulk_modulus"].value == 500.0
    user = _user_spec()
    assert (
        material_spec(user, {"shear": 200.0, "bulk": 600.0}).model
        == user.material.model
    )
    assert formula_for(user.material.model) == user.material.energy


def test_the_admission_of_a_spec_material() -> None:
    admission = admit_material(_mooney())
    assert admission.passed
    assert len(admission.reports) == 3 and admission.fem == ()
    assert admission.document()["model"] == "mooney_rivlin"
    assert admit_material(_user_spec()).passed


def test_a_fem_request_with_a_formula_material() -> None:
    spec = _user_spec()
    inputs = {
        "width": CanonicalQuantity(16.0, Unit.MM, Dimension.LENGTH),
        "height": CanonicalQuantity(6.0, Unit.MM, Dimension.LENGTH),
        "strain": CanonicalQuantity(0.08, Unit.ONE, Dimension.DIMENSIONLESS),
        "shear": CanonicalQuantity(200.0, Unit.MPA, Dimension.STRESS),
        "bulk": CanonicalQuantity(600.0, Unit.MPA, Dimension.STRESS),
    }
    request = fem_request(spec, inputs, plate_mesh(), "case", PROFILE)
    assert request.material.model == spec.material.model
    assert set(request.material.properties) == {"c", "k"}
