"""Formula materials against the closed-form kernel, and their failure modes."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from piginot import physics
from piginot.contracts.energy_formula import BUILTIN_FORMULAS, EnergyFormula
from piginot.materials import (
    embed,
    formula_for,
    is_formula_model,
    known_models,
    register,
    resolve,
)
from piginot.physics import (
    InadmissibleDeformationError,
    InvalidMaterialError,
    NewtonNonConvergenceError,
    PlaneStressClosureError,
)
from tests.materials.helpers import (
    BUILTIN_VALUES,
    NEO_HOOKEAN_FORMULA,
    formula_neo_hookean,
    material,
    neo_hookean,
)

F64 = torch.float64


def _planar(count: int = 500, seed: int = 0, spread: float = 0.2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn((count, 2, 2), generator=generator, dtype=F64)
    return torch.eye(2, dtype=F64) + spread * noise


def test_a_neo_hookean_formula_equals_the_closed_form() -> None:
    closed, formula = neo_hookean(), formula_neo_hookean()
    planar = _planar()
    f33 = physics.plane_stress_closure(planar, closed)
    assert torch.allclose(
        physics.plane_stress_closure(planar, formula), f33, rtol=1e-13, atol=0.0
    )
    deformation = embed(planar, f33)
    for name in ("energy_density", "first_piola", "consistent_tangent"):
        expected = getattr(physics, name)(deformation, closed)
        actual = getattr(physics, name)(deformation, formula)
        scale = float(expected.abs().max())
        assert float((actual - expected).abs().max()) <= 1e-12 * scale, name
    assert physics.admissibility(deformation, formula)
    leaf = planar[:7].clone().requires_grad_(True)
    expected_grad = torch.autograd.grad(
        physics.plane_stress_closure(leaf, closed).sum(), leaf
    )[0]
    actual_grad = torch.autograd.grad(
        physics.plane_stress_closure(leaf, formula).sum(), leaf
    )[0]
    assert torch.allclose(actual_grad, expected_grad, rtol=1e-11, atol=1e-14)
    assert physics.reference_modulus(closed) == 760.0
    assert physics.reference_modulus(formula) == pytest.approx(760.0, rel=1e-12)


def test_stress_is_differentiable_for_training() -> None:
    formula = formula_neo_hookean()
    deformation = embed(_planar(4), torch.ones(4, dtype=F64)).requires_grad_(True)
    stress = physics.first_piola(deformation, formula)
    assert stress.requires_grad
    (gradient,) = torch.autograd.grad(stress.square().sum(), deformation)
    assert bool(torch.isfinite(gradient).all())


@pytest.mark.parametrize("model", sorted(BUILTIN_FORMULAS))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_builtin_closures_zero_the_thickness_stress(
    model: str, dtype: torch.dtype
) -> None:
    spec = material(model, BUILTIN_VALUES[model])
    planar = _planar(200, seed=1).to(dtype)
    f33 = physics.plane_stress_closure(planar, spec)
    stress = physics.first_piola(embed(planar, f33), spec)
    tolerance = 1e-10 if dtype is torch.float64 else 1e-3
    assert float(stress[:, 2, 2].abs().max()) <= tolerance * 300.0


def test_the_registry() -> None:
    name = register(NEO_HOOKEAN_FORMULA)
    assert register(NEO_HOOKEAN_FORMULA) == name
    assert formula_for(name) == NEO_HOOKEAN_FORMULA
    assert formula_for("compressible_neo_hookean") is None
    assert formula_for("ogden") is None
    assert is_formula_model("gent") and is_formula_model(name)
    assert not is_formula_model("compressible_neo_hookean")
    assert not is_formula_model(3)
    models = known_models()
    assert models[0] == "compressible_neo_hookean" and name in models
    first = resolve(formula_neo_hookean())
    assert resolve(formula_neo_hookean()) is first
    with pytest.raises(TypeError):
        register("mu * I1")  # type: ignore[arg-type]


def test_invalid_materials_are_refused() -> None:
    with pytest.raises(InvalidMaterialError, match="unknown"):
        resolve(material("ogden", {"mu": 1.0}))
    with pytest.raises(InvalidMaterialError, match="MaterialSpec"):
        resolve("gent")  # type: ignore[arg-type]
    with pytest.raises(InvalidMaterialError, match="exactly"):
        resolve(material("gent", {"shear_modulus": 1.0}))
    wrong_unit = material("gent", BUILTIN_VALUES["gent"])
    properties = dict(wrong_unit.properties)
    properties["jm"] = properties["shear_modulus"]
    with pytest.raises(InvalidMaterialError, match="unit"):
        resolve(replace(wrong_unit, properties=properties))
    with pytest.raises(InvalidMaterialError, match="positive"):
        resolve(material("gent", {**BUILTIN_VALUES["gent"], "jm": 0.0}))
    with pytest.raises(InvalidMaterialError):
        physics.energy_density(torch.eye(3, dtype=F64)[None], material("ogden", {}))


def test_an_undefined_energy_and_bad_closure_arguments_are_refused() -> None:
    gent = material("gent", {**BUILTIN_VALUES["gent"], "jm": 0.5})
    stretched = torch.diag(torch.tensor([3.0, 1.0, 1.0], dtype=F64))[None]
    with pytest.raises(InadmissibleDeformationError, match="not defined"):
        physics.energy_density(stretched, gent)
    with pytest.raises(InadmissibleDeformationError, match="not defined"):
        physics.first_piola(stretched, gent)
    with pytest.raises(InadmissibleDeformationError, match="not defined"):
        physics.first_piola(stretched.clone().requires_grad_(True), gent)
    assert not physics.admissibility(stretched, gent)
    assert not physics.admissibility(-torch.eye(3, dtype=F64)[None], gent)
    spec = material("mooney_rivlin", BUILTIN_VALUES["mooney_rivlin"])
    planar = _planar(3)
    with pytest.raises(PlaneStressClosureError, match="scalar"):
        physics.plane_stress_closure(planar, spec, initial_f33=torch.ones(3))
    with pytest.raises(PlaneStressClosureError, match="integer"):
        physics.plane_stress_closure(planar, spec, max_iterations=2.0)  # type: ignore[arg-type]
    with pytest.raises(NewtonNonConvergenceError, match="at least one"):
        physics.plane_stress_closure(planar, spec, max_iterations=0)
    with pytest.raises(NewtonNonConvergenceError, match="did not converge"):
        physics.plane_stress_closure(planar, spec, max_iterations=1)
    with pytest.raises(PlaneStressClosureError, match="positive"):
        physics.plane_stress_closure(planar, spec, initial_f33=-1.0)


def test_a_material_without_a_thickness_root_is_refused() -> None:
    lonely = EnergyFormula.from_document(
        {
            "energy": "c * (I1 - 3) + c * (I2 - 3)",
            "parameters": {"c": "MPa"},
            "small_strain": {"shear_modulus": "c", "bulk_modulus": "c"},
        }
    )
    spec = material(register(lonely), {"c": 1.0})
    with pytest.raises(PlaneStressClosureError, match="bracket"):
        physics.plane_stress_closure(torch.eye(2, dtype=F64)[None] * 1.3, spec)


def test_a_constant_energy_still_has_a_shape() -> None:
    constant = EnergyFormula.from_document(
        {
            "energy": "c * (J - J) + c",
            "parameters": {"c": "MPa"},
            "small_strain": {"shear_modulus": "c", "bulk_modulus": "c"},
        }
    )
    spec = material(register(constant), {"c": 2.0})
    energy = physics.energy_density(torch.eye(3, dtype=F64).expand(4, 3, 3), spec)
    assert energy.shape == (4,) and float(energy[0]) == 2.0
