"""Public contract and validation tests for the S4 physics package."""

from __future__ import annotations

import pytest
import torch

from piginot.contracts import CanonicalQuantity, MaterialSpec, Unit
from piginot.physics import (
    InadmissibleDeformationError,
    InvalidMaterialError,
    InvalidTensorError,
    InvalidTensorShapeError,
    NewtonNonConvergenceError,
    NonFiniteTensorError,
    PhysicsError,
    PhysicsValidationError,
    PlaneStressClosureError,
    admissibility,
    consistent_tangent,
    energy_density,
    first_piola,
    plane_stress_closure,
)


def _material(
    elastic_modulus: float = 760.0,
    poisson_ratio: float = 0.3,
    *,
    model: str = "compressible_neo_hookean",
) -> MaterialSpec:
    return MaterialSpec(
        name="neo_hookean",
        model=model,
        properties={
            "elastic_modulus": CanonicalQuantity(elastic_modulus, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(poisson_ratio, Unit.ONE),
        },
    )


def test_public_errors_have_the_frozen_hierarchy() -> None:
    assert issubclass(PhysicsValidationError, PhysicsError)
    assert issubclass(InvalidMaterialError, PhysicsValidationError)
    assert issubclass(InvalidTensorError, PhysicsValidationError)
    assert issubclass(InvalidTensorShapeError, InvalidTensorError)
    assert issubclass(NonFiniteTensorError, InvalidTensorError)
    assert issubclass(InadmissibleDeformationError, PhysicsValidationError)
    assert issubclass(PlaneStressClosureError, PhysicsError)
    assert issubclass(NewtonNonConvergenceError, PlaneStressClosureError)


@pytest.mark.parametrize(
    "material",
    [
        _material(model="finite_strain"),
        _material(elastic_modulus=-1.0),
        _material(elastic_modulus=0.0),
        _material(elastic_modulus=0.999_999),
        _material(elastic_modulus=10_000.001),
        _material(poisson_ratio=-0.01),
        _material(poisson_ratio=0.490_001),
        _material(poisson_ratio=0.5),
    ],
)
def test_kernel_rejects_materials_outside_the_frozen_domain(
    material: MaterialSpec,
) -> None:
    with pytest.raises(InvalidMaterialError):
        energy_density(torch.eye(3, dtype=torch.float64), material)


@pytest.mark.parametrize(
    "material",
    [
        _material(elastic_modulus=1.0),
        _material(elastic_modulus=10_000.0),
        _material(poisson_ratio=0.0),
        _material(poisson_ratio=0.49),
    ],
)
def test_kernel_accepts_the_inclusive_material_boundaries(
    material: MaterialSpec,
) -> None:
    assert energy_density(torch.eye(3, dtype=torch.float64), material).item() == 0.0


def test_kernel_rejects_missing_and_dimensionally_invalid_properties() -> None:
    missing = MaterialSpec(
        "missing_nu",
        "compressible_neo_hookean",
        {"elastic_modulus": CanonicalQuantity(1.0, Unit.MPA)},
    )
    wrong_unit = MaterialSpec(
        "wrong_unit",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(1.0, Unit.ONE),
            "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
        },
    )
    unexpected = MaterialSpec(
        "unexpected_property",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(1.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
            "density": CanonicalQuantity(1.0, Unit.ONE),
        },
    )
    wrong_poisson_unit = MaterialSpec(
        "wrong_poisson_unit",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(1.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.3, Unit.MPA),
        },
    )

    for material in (missing, wrong_unit, wrong_poisson_unit, unexpected):
        with pytest.raises(InvalidMaterialError):
            first_piola(torch.eye(3, dtype=torch.float64), material)


def test_kernel_rejects_a_non_material_object() -> None:
    with pytest.raises(InvalidMaterialError):
        energy_density(
            torch.eye(3, dtype=torch.float64),
            object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("property_name", ["elastic_modulus", "poisson_ratio"])
def test_kernel_rejects_a_corrupted_nonfinite_material_property(
    property_name: str,
) -> None:
    material = _material()
    quantity = material.properties[property_name]
    # Simulate an in-memory contract corrupted after its canonical DTO boundary.
    object.__setattr__(quantity, "value", float("nan"))

    with pytest.raises(InvalidMaterialError):
        energy_density(torch.eye(3, dtype=torch.float64), material)


@pytest.mark.parametrize(
    "function",
    [energy_density, first_piola, consistent_tangent, admissibility],
)
def test_three_dimensional_apis_reject_the_wrong_shape(function: object) -> None:
    with pytest.raises(InvalidTensorShapeError):
        function(torch.eye(2, dtype=torch.float64), _material())  # type: ignore[operator]


def test_tensor_validation_is_typed() -> None:
    with pytest.raises(InvalidTensorError):
        energy_density([[1.0, 0.0], [0.0, 1.0]], _material())  # type: ignore[arg-type]
    with pytest.raises(InvalidTensorError):
        energy_density(torch.eye(3, dtype=torch.int64), _material())
    with pytest.raises(NonFiniteTensorError):
        energy_density(
            torch.diag(torch.tensor([1.0, 1.0, float("nan")], dtype=torch.float64)),
            _material(),
        )


@pytest.mark.parametrize(
    "invalid_tensor",
    [
        torch.eye(3, dtype=torch.float16),
        torch.eye(3, dtype=torch.complex128),
        torch.eye(3, dtype=torch.float64).to_sparse(),
        torch.empty((0, 3, 3), dtype=torch.float64),
    ],
)
def test_kernel_rejects_unsupported_tensor_representations(
    invalid_tensor: torch.Tensor,
) -> None:
    with pytest.raises(InvalidTensorError):
        energy_density(invalid_tensor, _material())


def test_admissibility_uses_the_frozen_strict_determinant_margin() -> None:
    material = _material()
    at_margin = torch.diag(torch.tensor([1.0e-12, 1.0, 1.0], dtype=torch.float64))
    above_margin = torch.diag(torch.tensor([1.0001e-12, 1.0, 1.0], dtype=torch.float64))

    assert admissibility(at_margin, material) is False
    assert admissibility(above_margin, material) is True


@pytest.mark.parametrize(
    "function",
    [energy_density, first_piola, consistent_tangent],
)
def test_constitutive_apis_fail_closed_for_inadmissible_deformations(
    function: object,
) -> None:
    singular = torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64))

    with pytest.raises(InadmissibleDeformationError):
        function(singular, _material())  # type: ignore[operator]


def test_identity_state_and_plane_stress_contract() -> None:
    material = _material()
    identity_3d = torch.eye(3, dtype=torch.float64)
    identity_2d = torch.eye(2, dtype=torch.float64)

    assert energy_density(identity_3d, material).item() == 0.0
    assert torch.equal(
        first_piola(identity_3d, material), torch.zeros_like(identity_3d)
    )
    assert consistent_tangent(identity_3d, material).shape == (3, 3, 3, 3)
    assert plane_stress_closure(identity_2d, material).item() == 1.0


def test_plane_stress_exposes_initial_guess_and_typed_nonconvergence() -> None:
    deformation = torch.tensor([[1.2, 0.1], [0.0, 0.9]], dtype=torch.float64)

    converged = plane_stress_closure(
        deformation,
        _material(),
        initial_f33=0.25,
    )
    assert converged > 0.0

    with pytest.raises(NewtonNonConvergenceError):
        plane_stress_closure(deformation, _material(), max_iterations=0)
