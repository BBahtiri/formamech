"""Residual and implicit-gradient tests for plane-stress closure."""

from __future__ import annotations

import pytest
import torch

from piginot.physics import (
    InadmissibleDeformationError,
    InvalidTensorShapeError,
    NewtonNonConvergenceError,
    PlaneStressClosureError,
    first_piola,
    plane_stress_closure,
)

from .reference import (
    _closure_unrolled,
    logarithmic_material_grid,
    make_material,
    max_relative_error,
    random_admissible_deformations,
    relative_l2,
)


def _embed_plane_stress_deformation(
    deformation_2d: torch.Tensor,
    f33: torch.Tensor,
) -> torch.Tensor:
    leading_shape = deformation_2d.shape[:-2]
    zero_column = torch.zeros(
        (*leading_shape, 2, 1),
        dtype=deformation_2d.dtype,
        device=deformation_2d.device,
    )
    zero_row = torch.zeros(
        (*leading_shape, 1, 2),
        dtype=deformation_2d.dtype,
        device=deformation_2d.device,
    )
    top = torch.cat((deformation_2d, zero_column), dim=-1)
    bottom = torch.cat((zero_row, f33[..., None, None]), dim=-1)
    return torch.cat((top, bottom), dim=-2)


def test_closure_drives_p33_below_tolerance_for_1000_cases() -> None:
    deformation_2d = random_admissible_deformations(1_000, 2, seed=4201, scale=0.25)
    residuals = []

    for index, (elastic_modulus, poisson_ratio) in enumerate(
        logarithmic_material_grid(10)
    ):
        material = make_material(elastic_modulus, poisson_ratio)
        batch = deformation_2d[index * 100 : (index + 1) * 100]
        f33 = plane_stress_closure(batch, material)
        deformation_3d = _embed_plane_stress_deformation(batch, f33)
        residuals.append(first_piola(deformation_3d, material)[..., 2, 2])

    p33 = torch.cat(residuals)

    assert p33.shape == (1_000,)
    assert torch.max(torch.abs(p33)).item() < 1.0e-10


def test_newton_converges_to_one_root_from_three_distinct_starts() -> None:
    deformation_2d = random_admissible_deformations(128, 2, seed=4202, scale=0.3)
    material = make_material(elastic_modulus=10_000.0, poisson_ratio=0.49)

    roots = torch.stack(
        tuple(
            plane_stress_closure(
                deformation_2d,
                material,
                initial_f33=initial_f33,
            )
            for initial_f33 in (0.2, 1.0, 5.0)
        )
    )

    spread = torch.max(roots, dim=0).values - torch.min(roots, dim=0).values
    assert torch.max(spread).item() < 1.0e-12


def test_custom_backward_matches_central_differences() -> None:
    base = random_admissible_deformations(8, 2, seed=4203, scale=0.15)
    material = make_material(elastic_modulus=7_500.0, poisson_ratio=0.32)
    weights = torch.linspace(0.7, 1.3, base.shape[0], dtype=torch.float64)
    deformation_2d = base.detach().requires_grad_(True)
    custom_output = plane_stress_closure(deformation_2d, material)
    custom_gradient = torch.autograd.grad(
        torch.sum(weights * custom_output), deformation_2d
    )[0]

    flattened_size = base.numel()
    perturbations = torch.eye(flattened_size, dtype=torch.float64).reshape(
        flattened_size, *base.shape
    )
    step = 1.0e-6
    outputs_plus = plane_stress_closure(
        base.unsqueeze(0) + step * perturbations,
        material,
    )
    outputs_minus = plane_stress_closure(
        base.unsqueeze(0) - step * perturbations,
        material,
    )
    objectives_plus = torch.sum(outputs_plus * weights, dim=-1)
    objectives_minus = torch.sum(outputs_minus * weights, dim=-1)
    finite_difference = ((objectives_plus - objectives_minus) / (2.0 * step)).reshape(
        base.shape
    )

    assert max_relative_error(custom_gradient, finite_difference) < 1.0e-6


def test_custom_backward_matches_unrolled_newton_autodiff() -> None:
    base = random_admissible_deformations(64, 2, seed=4204, scale=0.2)
    material = make_material(elastic_modulus=8_000.0, poisson_ratio=0.37)
    weights = torch.linspace(0.5, 1.5, base.shape[0], dtype=torch.float64)
    custom_input = base.detach().requires_grad_(True)
    oracle_input = base.detach().requires_grad_(True)

    custom_gradient = torch.autograd.grad(
        torch.sum(weights * plane_stress_closure(custom_input, material)),
        custom_input,
    )[0]
    oracle_gradient = torch.autograd.grad(
        torch.sum(weights * _closure_unrolled(oracle_input, material)),
        oracle_input,
    )[0]

    assert relative_l2(custom_gradient, oracle_gradient) < 1.0e-8


def test_nu_zero_has_exact_unit_root_and_zero_derivative() -> None:
    deformation_2d = random_admissible_deformations(
        64, 2, seed=4205, scale=0.35
    ).requires_grad_(True)
    material = make_material(elastic_modulus=5_000.0, poisson_ratio=0.0)

    f33 = plane_stress_closure(deformation_2d, material)
    gradient = torch.autograd.grad(f33.sum(), deformation_2d)[0]

    assert torch.equal(f33, torch.ones_like(f33))
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_closure_validates_shape_and_in_plane_admissibility() -> None:
    material = make_material()
    reflection = torch.diag(torch.tensor([-1.0, 1.0], dtype=torch.float64))

    with pytest.raises(InvalidTensorShapeError):
        plane_stress_closure(torch.eye(3, dtype=torch.float64), material)
    with pytest.raises(InadmissibleDeformationError):
        plane_stress_closure(reflection, material)


def test_tensor_initial_guess_broadcasts_over_multiple_batch_axes() -> None:
    deformation_2d = random_admissible_deformations(6, 2, seed=4206, scale=0.2).reshape(
        2, 3, 2, 2
    )
    material = make_material()
    tensor_initial = torch.ones(3, dtype=torch.float64)

    tensor_root = plane_stress_closure(
        deformation_2d,
        material,
        initial_f33=tensor_initial,
    )
    scalar_root = plane_stress_closure(deformation_2d, material, initial_f33=1.0)

    assert tensor_root.shape == (2, 3)
    assert torch.equal(tensor_root, scalar_root)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.complex128])
def test_closure_rejects_an_initial_tensor_with_unsupported_dtype(
    dtype: torch.dtype,
) -> None:
    with pytest.raises(PlaneStressClosureError):
        plane_stress_closure(
            torch.eye(2, dtype=torch.float64),
            make_material(),
            initial_f33=torch.ones((), dtype=dtype),
        )


def test_closure_rejects_a_sparse_initial_tensor() -> None:
    sparse_initial = torch.ones(1, dtype=torch.float64).to_sparse()

    with pytest.raises(PlaneStressClosureError):
        plane_stress_closure(
            torch.eye(2, dtype=torch.float64),
            make_material(),
            initial_f33=sparse_initial,
        )


def test_closure_rejects_an_initial_tensor_on_another_device() -> None:
    meta_initial = torch.ones((), dtype=torch.float64, device="meta")

    with pytest.raises(PlaneStressClosureError):
        plane_stress_closure(
            torch.eye(2, dtype=torch.float64),
            make_material(),
            initial_f33=meta_initial,
        )


def test_closure_rejects_an_initial_tensor_that_cannot_broadcast() -> None:
    deformation_2d = torch.eye(2, dtype=torch.float64).expand(2, 2, 2)

    with pytest.raises(PlaneStressClosureError):
        plane_stress_closure(
            deformation_2d,
            make_material(),
            initial_f33=torch.ones(3, dtype=torch.float64),
        )


@pytest.mark.parametrize("initial_f33", [True, "one", None])
def test_closure_rejects_a_nonreal_scalar_initial_guess(initial_f33: object) -> None:
    with pytest.raises(PlaneStressClosureError):
        plane_stress_closure(
            torch.eye(2, dtype=torch.float64),
            make_material(),
            initial_f33=initial_f33,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("initial_f33", [float("nan"), float("inf"), -float("inf")])
def test_closure_rejects_a_nonfinite_scalar_initial_guess(
    initial_f33: float,
) -> None:
    with pytest.raises(PlaneStressClosureError):
        plane_stress_closure(
            torch.eye(2, dtype=torch.float64),
            make_material(),
            initial_f33=initial_f33,
        )


def test_closure_rejects_a_nonfinite_tensor_initial_guess() -> None:
    deformation_2d = torch.eye(2, dtype=torch.float64).expand(2, 2, 2)

    with pytest.raises(PlaneStressClosureError):
        plane_stress_closure(
            deformation_2d,
            make_material(),
            initial_f33=torch.tensor([1.0, float("nan")], dtype=torch.float64),
        )


@pytest.mark.parametrize("initial_f33", [0.0, -1.0])
def test_closure_rejects_a_nonpositive_initial_guess(initial_f33: float) -> None:
    with pytest.raises(PlaneStressClosureError):
        plane_stress_closure(
            torch.eye(2, dtype=torch.float64),
            make_material(),
            initial_f33=initial_f33,
        )


@pytest.mark.parametrize("max_iterations", [True, 1.5])
def test_closure_rejects_a_noninteger_iteration_limit(
    max_iterations: object,
) -> None:
    with pytest.raises(PlaneStressClosureError):
        plane_stress_closure(
            torch.eye(2, dtype=torch.float64),
            make_material(),
            max_iterations=max_iterations,  # type: ignore[arg-type]
        )


def test_closure_reports_bounded_newton_nonconvergence() -> None:
    deformation_2d = torch.diag(torch.tensor([3.0, 0.2], dtype=torch.float64))

    with pytest.raises(NewtonNonConvergenceError):
        plane_stress_closure(
            deformation_2d,
            make_material(poisson_ratio=0.49),
            initial_f33=10.0,
            max_iterations=1,
        )


def test_nu_zero_still_validates_iteration_controls() -> None:
    material = make_material(poisson_ratio=0.0)
    deformation_2d = torch.eye(2, dtype=torch.float64)

    with pytest.raises(PlaneStressClosureError):
        plane_stress_closure(deformation_2d, material, max_iterations=True)
    with pytest.raises(NewtonNonConvergenceError):
        plane_stress_closure(deformation_2d, material, max_iterations=0)
