"""Independent identities for the S8 symbolic manufactured solutions."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
import torch

from piginot.contracts import CanonicalQuantity, MaterialSpec, Unit
from piginot.fem.manufactured import (
    MANUFACTURED_CASE_NAMES,
    BoundaryKind,
    build_manufactured_solution,
)
from piginot.physics import first_piola


def _material(*, poisson_ratio: float = 0.3) -> MaterialSpec:
    return MaterialSpec(
        name="s8_mms_material",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": CanonicalQuantity(100.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(poisson_ratio, Unit.ONE),
        },
    )


def _interior_points() -> np.ndarray:
    return np.asarray(
        (
            (0.17, 0.31, 0.48, 0.64, 0.83),
            (0.23, 0.72, 0.44, 0.81, 0.36),
        ),
        dtype=np.float64,
    )


def _unit_square_grid(count: int = 17) -> np.ndarray:
    coordinates = np.linspace(0.0, 1.0, count, dtype=np.float64)
    x_coordinates, y_coordinates = np.meshgrid(coordinates, coordinates, indexing="ij")
    return np.stack((x_coordinates, y_coordinates))


@pytest.mark.parametrize("case_name", MANUFACTURED_CASE_NAMES)
def test_symbolic_plane_stress_closure_is_exact(case_name: str) -> None:
    solution = build_manufactured_solution(case_name, _material())

    assert solution.thickness_stretch_symbolic.has(sp.LambertW)
    assert np.max(np.abs(solution.plane_stress_residual(_interior_points()))) < 2.0e-12


@pytest.mark.parametrize("case_name", MANUFACTURED_CASE_NAMES)
def test_vectorized_fields_are_finite_admissible_and_float64(case_name: str) -> None:
    solution = build_manufactured_solution(case_name, _material())
    points = _unit_square_grid()

    displacement = solution.displacement(points)
    gradient = solution.gradient(points)
    deformation = solution.deformation_gradient(points)
    thickness_stretch = solution.thickness_stretch(points)
    stress = solution.first_piola(points)
    body_force = solution.body_force(points)

    assert displacement.shape == body_force.shape == points.shape
    assert gradient.shape == deformation.shape == (2, 2, *points.shape[1:])
    assert thickness_stretch.shape == points.shape[1:]
    assert stress.shape == (3, 3, *points.shape[1:])
    assert all(
        field.dtype == np.float64
        for field in (
            displacement,
            gradient,
            deformation,
            thickness_stretch,
            stress,
            body_force,
        )
    )
    assert all(
        np.isfinite(field).all()
        for field in (
            displacement,
            gradient,
            deformation,
            thickness_stretch,
            stress,
            body_force,
        )
    )
    assert np.min(np.linalg.det(np.moveaxis(deformation, (0, 1), (-2, -1)))) > 0.0
    assert np.min(thickness_stretch) > 0.0
    assert np.max(np.abs(stress[2, 2])) < 5.0e-12


@pytest.mark.parametrize("case_name", MANUFACTURED_CASE_NAMES)
def test_body_force_is_negative_reference_divergence_of_nominal_stress(
    case_name: str,
) -> None:
    solution = build_manufactured_solution(case_name, _material())
    points = _interior_points()
    step = 2.0e-6
    x_shift = np.asarray(((step,), (0.0,)), dtype=np.float64)
    y_shift = np.asarray(((0.0,), (step,)), dtype=np.float64)

    derivative_x = (
        solution.first_piola(points + x_shift) - solution.first_piola(points - x_shift)
    ) / (2.0 * step)
    derivative_y = (
        solution.first_piola(points + y_shift) - solution.first_piola(points - y_shift)
    ) / (2.0 * step)
    numerical_body = -(derivative_x[:2, 0] + derivative_y[:2, 1])

    np.testing.assert_allclose(
        solution.body_force(points), numerical_body, rtol=2.0e-7, atol=2.0e-8
    )


@pytest.mark.parametrize("case_name", MANUFACTURED_CASE_NAMES)
def test_nominal_traction_is_first_piola_times_reference_normal(
    case_name: str,
) -> None:
    solution = build_manufactured_solution(case_name, _material())
    points = _interior_points()
    normals = np.asarray(
        ((1.0, 0.0, -1.0, 0.0, 2.0**-0.5), (0.0, 1.0, 0.0, -1.0, 2.0**-0.5)),
        dtype=np.float64,
    )

    expected = np.einsum(
        "ij...,j...->i...", solution.first_piola(points)[:2, :2], normals
    )

    np.testing.assert_allclose(
        solution.traction(points, normals), expected, atol=1.0e-13
    )


@pytest.mark.parametrize("case_name", MANUFACTURED_CASE_NAMES)
def test_symbolic_stress_matches_the_production_material_kernel(case_name: str) -> None:
    material = _material()
    solution = build_manufactured_solution(case_name, material)
    points = _interior_points()
    deformation_2d = solution.deformation_gradient(points)
    deformation_3d = np.zeros((points.shape[1], 3, 3), dtype=np.float64)
    deformation_3d[:, :2, :2] = np.moveaxis(deformation_2d, (0, 1), (-2, -1))
    deformation_3d[:, 2, 2] = solution.thickness_stretch(points)

    expected = first_piola(torch.from_numpy(deformation_3d), material).numpy()
    actual = np.moveaxis(solution.first_piola(points), (0, 1), (-2, -1))

    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=5.0e-13)


def test_pure_traction_metadata_distinguishes_physics_from_global_constraints() -> None:
    solution = build_manufactured_solution("pure_traction_body_force", _material())
    x_coordinate, y_coordinate = sp.symbols("X Y", real=True)

    assert set(solution.boundary_types) == {"left", "right", "top", "bottom"}
    assert set(solution.boundary_types.values()) == {BoundaryKind.TRACTION}
    assert solution.rigid_mode_constraints == (
        "mean_x",
        "mean_y",
        "mean_rotation",
    )
    assert solution.boundary_types["left"] is not BoundaryKind.DIRICHLET
    ux, uy = solution.displacement_symbolic
    integrals = (
        sp.integrate(ux, (x_coordinate, 0, 1), (y_coordinate, 0, 1)),
        sp.integrate(uy, (x_coordinate, 0, 1), (y_coordinate, 0, 1)),
        sp.integrate(
            x_coordinate * uy - y_coordinate * ux,
            (x_coordinate, 0, 1),
            (y_coordinate, 0, 1),
        ),
    )
    assert integrals == (0, 0, 0)


def test_pure_traction_field_is_not_exactly_representable_by_p2() -> None:
    solution = build_manufactured_solution("pure_traction_body_force", _material())
    x_coordinate, y_coordinate = sp.symbols("X Y", real=True)

    third_derivatives = tuple(
        sp.diff(component, x_coordinate, 2, y_coordinate)
        for component in solution.displacement_symbolic
    ) + tuple(
        sp.diff(component, x_coordinate, y_coordinate, 2)
        for component in solution.displacement_symbolic
    )

    assert any(value != 0 for value in third_derivatives)


def test_declared_boundary_families_match_each_case_name() -> None:
    material = _material()

    pure_dirichlet = build_manufactured_solution("pure_dirichlet", material)
    mixed = build_manufactured_solution("mixed_dirichlet_traction", material)
    varying = build_manufactured_solution("strongly_varying_f33", material)

    assert set(pure_dirichlet.boundary_types.values()) == {BoundaryKind.DIRICHLET}
    assert set(varying.boundary_types.values()) == {BoundaryKind.DIRICHLET}
    assert BoundaryKind.DIRICHLET in set(mixed.boundary_types.values())
    assert BoundaryKind.TRACTION in set(mixed.boundary_types.values())
    assert pure_dirichlet.rigid_mode_constraints == mixed.rigid_mode_constraints == ()


def test_strong_case_has_materially_larger_f33_range_than_ordinary_cases() -> None:
    material = _material()
    points = _unit_square_grid(31)
    ranges = {
        case_name: float(
            np.ptp(
                build_manufactured_solution(case_name, material).thickness_stretch(
                    points
                )
            )
        )
        for case_name in MANUFACTURED_CASE_NAMES
    }

    ordinary_maximum = max(
        value for name, value in ranges.items() if name != "strongly_varying_f33"
    )

    assert ranges["strongly_varying_f33"] > 3.0 * ordinary_maximum
    assert ranges["strongly_varying_f33"] > 0.08


def test_zero_first_lame_limit_has_unit_thickness_stretch() -> None:
    solution = build_manufactured_solution(
        "strongly_varying_f33", _material(poisson_ratio=0.0)
    )

    np.testing.assert_array_equal(
        solution.thickness_stretch(_interior_points()), np.ones(5, dtype=np.float64)
    )
    assert solution.thickness_stretch_symbolic == sp.Integer(1)
    assert solution.plane_stress_residual_symbolic == sp.Integer(0)


def test_manufactured_evaluators_are_deterministic_and_reject_bad_shapes() -> None:
    first = build_manufactured_solution("mixed_dirichlet_traction", _material())
    second = build_manufactured_solution("mixed_dirichlet_traction", _material())
    points = _interior_points()

    np.testing.assert_array_equal(first.body_force(points), second.body_force(points))
    np.testing.assert_array_equal(first.first_piola(points), second.first_piola(points))
    with pytest.raises(ValueError, match=r"shape \(2, \.\.\.\)"):
        first.displacement(np.zeros((3, 4), dtype=np.float64))
    with pytest.raises(ValueError, match="same shape"):
        first.traction(points, np.zeros((2, 4), dtype=np.float64))
    with pytest.raises(ValueError, match="unknown manufactured case"):
        build_manufactured_solution("not_a_case", _material())
