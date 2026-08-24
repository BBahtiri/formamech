"""True solved-field norms used by the S8 manufactured-solution studies."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import sympy as sp
from skfem import Basis, ElementTriP2, ElementVector, MeshTri

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    ExecutionProfile,
    LoadSpec,
    MaterialSpec,
    Unit,
)
from piginot.fem.backend import ScikitFemBackend
from piginot.fem.manufactured import BoundaryKind, ManufacturedSolution
from piginot.fem.models import FemRequest, FemResult, FemSolverConfig
from piginot.fem.verification_norms import (
    FemVerificationNorms,
    compute_verification_norms,
)

_LATERAL_STRETCH = 0.945632624291129
_TRACTION = 17.492914739624112


def _quantity(value: float, unit: Unit) -> CanonicalQuantity:
    return CanonicalQuantity(value, unit)


def _mesh() -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id="verification-norm-square",
        nodes=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        cells=((0, 1, 2), (0, 2, 3)),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 3), (3, 0)),
        semantic_tags={
            "bottom": (0,),
            "right": (1,),
            "top": (2,),
            "left": (3,),
        },
        region_ids=(1, 1),
        unit=Unit.MM,
        orientation="ccw",
        topology_signature="verification-norm-square-v1",
        quality_metrics={"min_scaled_jacobian": 1.0},
        source_provenance={"fixture": "s8-verification-norms"},
    )


def _material() -> MaterialSpec:
    return MaterialSpec(
        "verification-rubber",
        "compressible_neo_hookean",
        {
            "elastic_modulus": _quantity(100.0, Unit.MPA),
            "poisson_ratio": _quantity(0.3, Unit.ONE),
        },
    )


def _profile() -> ExecutionProfile:
    return ExecutionProfile(
        python_version="3.12.0",
        library_versions={
            "numpy": "2.2.6",
            "scikit-fem": "12.0.2",
            "scipy": "1.18.0",
            "torch": "2.13.0",
        },
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )


def _request(*, element_order: int) -> FemRequest:
    return FemRequest(
        request_id=f"verification-norm-p{element_order}",
        mesh=_mesh(),
        material=_material(),
        boundary_conditions=(
            BoundaryConditionSpec(
                "left_x",
                "left",
                "dirichlet",
                ("x",),
                (_quantity(0.0, Unit.MM),),
            ),
            BoundaryConditionSpec(
                "bottom_y",
                "bottom",
                "symmetry",
                ("y",),
                (_quantity(0.0, Unit.MM),),
            ),
        ),
        loads=(
            LoadSpec(
                "right_pull",
                "traction",
                "right",
                ("x",),
                (_quantity(_TRACTION, Unit.MPA),),
            ),
        ),
        qois=(),
        config=FemSolverConfig(
            element_order=element_order,
            solver_tolerance=1.0e-11,
            max_newton_iterations=20,
            load_steps=4,
        ),
        execution_profile=_profile(),
    )


def _affine_solution(material: MaterialSpec) -> ManufacturedSolution:
    x_coordinate, y_coordinate = sp.symbols("X Y", real=True)
    lateral_displacement = sp.Float(_LATERAL_STRETCH - 1.0, 17)
    traction = sp.Float(_TRACTION, 17)
    zero = sp.Integer(0)
    return ManufacturedSolution(
        name="affine-uniaxial-verification",
        material=material,
        boundary_types={
            "left": BoundaryKind.DIRICHLET,
            "right": BoundaryKind.TRACTION,
            "top": BoundaryKind.TRACTION,
            "bottom": BoundaryKind.DIRICHLET,
        },
        rigid_mode_constraints=(),
        displacement_symbolic=(
            sp.Rational(1, 5) * x_coordinate,
            lateral_displacement * y_coordinate,
        ),
        gradient_symbolic=(
            (sp.Rational(1, 5), zero),
            (zero, lateral_displacement),
        ),
        deformation_gradient_symbolic=(
            (sp.Rational(6, 5), zero),
            (zero, sp.Float(_LATERAL_STRETCH, 17)),
        ),
        thickness_stretch_symbolic=sp.Float(_LATERAL_STRETCH, 17),
        first_piola_symbolic=(
            (traction, zero, zero),
            (zero, zero, zero),
            (zero, zero, zero),
        ),
        body_force_symbolic=(zero, zero),
        plane_stress_residual_symbolic=zero,
    )


def _synthetic_exact_result(request: FemRequest) -> FemResult:
    points = np.asarray(request.mesh.nodes, dtype=np.float64).T
    cells = np.asarray(request.mesh.cells, dtype=np.int32).T
    mesh = MeshTri(points, cells, sort_t=False)
    scalar_basis = Basis(mesh, ElementTriP2(), intorder=5)
    vector_basis = Basis(mesh, ElementVector(ElementTriP2()), intorder=5)
    solution = _affine_solution(request.material)

    dof_coordinates = np.asarray(scalar_basis.doflocs, dtype=np.float64)
    displacement = solution.displacement(dof_coordinates).T
    quadrature_points = np.asarray(vector_basis.global_coordinates(), dtype=np.float64)
    exact_stress = np.moveaxis(
        solution.first_piola(quadrature_points),
        (0, 1),
        (-2, -1),
    )
    recovered_stress = np.moveaxis(
        solution.first_piola(dof_coordinates),
        (0, 1),
        (-2, -1),
    )
    deformation = np.zeros((*exact_stress.shape[:2], 3, 3), dtype=np.float64)
    deformation[..., 0, 0] = 1.2
    deformation[..., 1, 1] = _LATERAL_STRETCH
    deformation[..., 2, 2] = _LATERAL_STRETCH
    return FemResult(
        request_id=request.request_id,
        mesh_id=request.mesh.mesh_id,
        element_order=request.config.element_order,
        displacement_dof_coordinates=tuple(map(tuple, dof_coordinates.T)),
        displacement=tuple(map(tuple, displacement)),
        quadrature_points=tuple(
            tuple(map(tuple, cell))
            for cell in np.transpose(quadrature_points, (1, 2, 0))
        ),
        quadrature_weights=tuple(map(tuple, np.asarray(vector_basis.dx))),
        first_piola_stress=tuple(
            tuple(tuple(tuple(row) for row in point) for point in cell)
            for cell in exact_stress
        ),
        deformation_gradient=tuple(
            tuple(tuple(tuple(row) for row in point) for point in cell)
            for cell in deformation
        ),
        recovered_first_piola_stress=tuple(
            tuple(tuple(row) for row in point) for point in recovered_stress
        ),
        reactions={},
        rigid_mode_multipliers=(),
        total_strain_energy=0.0,
        external_work=0.0,
        total_potential_energy=0.0,
        convergence_history=(),
        qois={},
    )


def test_exact_p2_result_has_machine_zero_true_errors_and_recovery_estimate() -> None:
    request = _request(element_order=2)
    result = _synthetic_exact_result(request)

    norms = compute_verification_norms(
        request,
        result,
        _affine_solution(request.material),
    )

    assert isinstance(norms, FemVerificationNorms)
    assert norms.displacement_l2_error < 1.0e-14
    assert norms.displacement_h1_seminorm_error < 1.0e-14
    assert norms.first_piola_l2_error < 1.0e-14
    assert norms.recovery_estimator_l2 < 1.0e-14
    assert norms.integrated_weight == pytest.approx(1.0)
    assert norms.quadrature_point_count == 14


def test_perturbations_strictly_increase_each_reported_error_norm() -> None:
    request = _request(element_order=2)
    exact = _synthetic_exact_result(request)
    solution = _affine_solution(request.material)
    baseline = compute_verification_norms(request, exact, solution)

    perturbed_displacement = np.asarray(exact.displacement, dtype=np.float64).copy()
    perturbed_displacement[:, 0] += 0.01 * np.square(
        np.asarray(exact.displacement_dof_coordinates)[:, 0]
    )
    displacement_result = replace(
        exact,
        displacement=tuple(map(tuple, perturbed_displacement)),
    )
    displacement_norms = compute_verification_norms(
        request,
        displacement_result,
        solution,
    )

    perturbed_stress = np.asarray(exact.first_piola_stress, dtype=np.float64).copy()
    perturbed_stress[..., 0, 1] += 2.0
    stress_result = replace(
        exact,
        first_piola_stress=tuple(
            tuple(tuple(tuple(row) for row in point) for point in cell)
            for cell in perturbed_stress
        ),
    )
    stress_norms = compute_verification_norms(request, stress_result, solution)

    assert displacement_norms.displacement_l2_error > baseline.displacement_l2_error
    assert (
        displacement_norms.displacement_h1_seminorm_error
        > baseline.displacement_h1_seminorm_error
    )
    assert stress_norms.first_piola_l2_error > baseline.first_piola_l2_error
    assert stress_norms.recovery_estimator_l2 > baseline.recovery_estimator_l2


def test_actual_backend_affine_solve_is_measured_against_exact_fields() -> None:
    request = _request(element_order=1)
    attempt = ScikitFemBackend().solve_many((request,))[0]

    assert attempt.success, attempt.message
    assert attempt.result is not None
    norms = compute_verification_norms(
        request,
        attempt.result,
        _affine_solution(request.material),
    )

    assert norms.displacement_l2_error < 1.0e-9
    assert norms.displacement_h1_seminorm_error < 1.0e-9
    assert norms.first_piola_l2_error < 1.0e-8
    assert norms.recovery_estimator_l2 < 1.0e-8


@pytest.mark.parametrize(
    ("result_change", "solution_change", "match"),
    (
        ({"request_id": "other-request"}, False, "request_id"),
        ({"mesh_id": "other-mesh"}, False, "mesh_id"),
        ({"element_order": 1}, False, "element_order"),
        ({}, True, "material"),
    ),
)
def test_request_result_and_solution_identities_must_match(
    result_change: dict[str, object],
    solution_change: bool,
    match: str,
) -> None:
    request = _request(element_order=2)
    result = replace(_synthetic_exact_result(request), **result_change)
    solution = _affine_solution(request.material)
    if solution_change:
        solution = _affine_solution(
            replace(
                request.material,
                name="different-material",
            )
        )

    with pytest.raises(ValueError, match=match):
        compute_verification_norms(request, result, solution)


def test_dof_and_quadrature_layout_must_match_the_declared_mesh() -> None:
    request = _request(element_order=2)
    result = _synthetic_exact_result(request)
    reversed_coordinates = tuple(reversed(result.displacement_dof_coordinates))
    reordered = replace(result, displacement_dof_coordinates=reversed_coordinates)

    with pytest.raises(ValueError, match="DOF coordinate layout"):
        compute_verification_norms(
            request,
            reordered,
            _affine_solution(request.material),
        )

    points = np.asarray(result.quadrature_points, dtype=np.float64).copy()
    points[1, 0, 0] += 0.01
    wrong_quadrature = replace(
        result,
        quadrature_points=tuple(tuple(map(tuple, cell)) for cell in points),
    )
    with pytest.raises(ValueError, match="quadrature"):
        compute_verification_norms(
            request,
            wrong_quadrature,
            _affine_solution(request.material),
        )


@pytest.mark.parametrize(
    ("invalid_case", "match"),
    (
        ("point_shape", "quadrature_points must have shape"),
        ("empty_points", "at least one point"),
        ("weight_shape", "quadrature_weights must match"),
        ("negative_weight", "quadrature_weights must be positive"),
        ("coordinate_shape", "displacement_dof_coordinates must have shape"),
        ("displacement_shape", "displacement values must match"),
        ("stress_shape", "first_piola_stress must have shape"),
        ("recovered_shape", "recovered_first_piola_stress must have shape"),
    ),
)
def test_result_field_shapes_and_positive_weights_are_validated(
    invalid_case: str,
    match: str,
) -> None:
    request = _request(element_order=2)
    result = _synthetic_exact_result(request)
    if invalid_case == "point_shape":
        result = replace(result, quadrature_points=result.quadrature_points[:1])
    elif invalid_case == "empty_points":
        result = replace(
            result,
            quadrature_points=np.empty((2, 0, 2), dtype=np.float64),  # type: ignore[arg-type]
            quadrature_weights=np.empty((2, 0), dtype=np.float64),  # type: ignore[arg-type]
        )
    elif invalid_case == "weight_shape":
        result = replace(result, quadrature_weights=result.quadrature_weights[:1])
    elif invalid_case == "negative_weight":
        weights = np.asarray(result.quadrature_weights, dtype=np.float64).copy()
        weights[0, 0] = -weights[0, 0]
        result = replace(result, quadrature_weights=weights)  # type: ignore[arg-type]
    elif invalid_case == "coordinate_shape":
        result = replace(
            result,
            displacement_dof_coordinates=((0.0,),),  # type: ignore[arg-type]
        )
    elif invalid_case == "displacement_shape":
        result = replace(result, displacement=result.displacement[:-1])
    elif invalid_case == "stress_shape":
        result = replace(result, first_piola_stress=())
    else:
        result = replace(result, recovered_first_piola_stress=())

    with pytest.raises(ValueError, match=match):
        compute_verification_norms(
            request,
            result,
            _affine_solution(request.material),
        )


def test_nonfinite_and_nonrectangular_payloads_are_rejected() -> None:
    request = _request(element_order=2)
    result = _synthetic_exact_result(request)
    displacement = np.asarray(result.displacement, dtype=np.float64).copy()
    displacement[0, 0] = np.nan
    nonfinite = replace(result, displacement=displacement)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite values"):
        compute_verification_norms(
            request,
            nonfinite,
            _affine_solution(request.material),
        )

    ragged = replace(
        result,
        quadrature_points=(result.quadrature_points[0], ((0.0, 0.0),)),
    )
    with pytest.raises(ValueError, match="rectangular array"):
        compute_verification_norms(
            request,
            ragged,
            _affine_solution(request.material),
        )


def test_quadrature_points_stay_inside_cells_and_weights_share_one_rule() -> None:
    request = _request(element_order=2)
    result = _synthetic_exact_result(request)
    solution = _affine_solution(request.material)
    points = np.asarray(result.quadrature_points, dtype=np.float64).copy()
    points[0, 0] = (2.0, 2.0)
    outside = replace(result, quadrature_points=points)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outside"):
        compute_verification_norms(request, outside, solution)

    weights = np.asarray(result.quadrature_weights, dtype=np.float64).copy()
    weights[1, 0] *= 1.1
    inconsistent = replace(result, quadrature_weights=weights)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reference-cell rule"):
        compute_verification_norms(request, inconsistent, solution)
