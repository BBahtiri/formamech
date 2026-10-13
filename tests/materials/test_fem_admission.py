"""Admission check 8: FEM convergence on an autograd manufactured solution."""

from __future__ import annotations

import numpy as np
import pytest

from piginot.fem.manufactured import build_manufactured_solution
from piginot.materials import fem_admission
from piginot.materials.fem_admission import AutogradManufacturedSolution, check_fem
from tests.materials.helpers import BUILTIN_VALUES, material, neo_hookean


def test_the_autograd_fields_equal_the_symbolic_s8_fields() -> None:
    from piginot.contracts import CanonicalQuantity, MaterialSpec, Unit

    spec = MaterialSpec(
        "s8",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(100.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
        },
    )
    symbolic = build_manufactured_solution("strongly_varying_f33", spec)
    autograd = AutogradManufacturedSolution(spec)
    points = np.random.default_rng(0).uniform(0.0, 1.0, size=(2, 5, 3))
    for name in ("displacement", "gradient", "first_piola", "body_force"):
        expected = getattr(symbolic, name)(points)
        actual = getattr(autograd, name)(points)
        assert actual.shape == expected.shape, name
        assert np.allclose(actual, expected, rtol=1e-10, atol=1e-12), name
    assert np.allclose(
        autograd.thickness_stretch(points), symbolic.thickness_stretch(points)
    )
    with pytest.raises(ValueError, match="shape"):
        autograd.displacement(np.zeros((3, 2)))


def test_a_registry_material_converges_at_the_optimal_orders() -> None:
    result = check_fem(material("mooney_rivlin", BUILTIN_VALUES["mooney_rivlin"]))
    assert result.passed, result.document()
    assert abs(result.detail["displacement_l2_order"] - 3.0) <= 0.15
    assert abs(result.detail["first_piola_l2_order"] - 2.0) <= 0.2


def test_a_failed_solve_fails_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object) -> tuple[float, float, float]:
        raise RuntimeError("manufactured solve failed: diverged")

    monkeypatch.setattr(fem_admission, "solve_level", fail)
    result = check_fem(neo_hookean())
    assert not result.passed and "diverged" in result.detail["error"]
