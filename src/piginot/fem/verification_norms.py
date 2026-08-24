"""Solved-field error norms for manufactured FEM verification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skfem import (  # type: ignore[import-untyped]
    Basis,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    MeshTri,
)

from piginot.fem.error_estimation import recovery_stress_estimate
from piginot.fem.manufactured import ManufacturedSolution
from piginot.fem.models import FemRequest, FemResult

FloatArray = NDArray[np.float64]
_LAYOUT_ATOL = 5.0e-13


@dataclass(frozen=True, slots=True)
class FemVerificationNorms:
    """True MMS errors and the stress-recovery estimate for one solved mesh."""

    displacement_l2_error: float
    displacement_h1_seminorm_error: float
    first_piola_l2_error: float
    recovery_estimator_l2: float
    integrated_weight: float
    quadrature_point_count: int


def compute_verification_norms(
    request: FemRequest,
    result: FemResult,
    manufactured_solution: ManufacturedSolution,
) -> FemVerificationNorms:
    """Evaluate solved P1/P2 fields at the result's stored quadrature rule.

    The public result payload stores vector displacement as paired scalar-field
    DOFs.  This function rebuilds that precise finite-element layout, verifies
    it against the request and result metadata, and only then interpolates the
    approximate fields.  The full 3x3 first-Piola tensor participates in both
    the true error and recovery estimate.
    """

    _validate_identity(request, result, manufactured_solution)
    points = _finite_array(result.quadrature_points, "quadrature_points")
    weights = _finite_array(result.quadrature_weights, "quadrature_weights")
    displacement_coordinates = _finite_array(
        result.displacement_dof_coordinates,
        "displacement_dof_coordinates",
    )
    displacement_values = _finite_array(result.displacement, "displacement")
    raw_stress = _finite_array(result.first_piola_stress, "first_piola_stress")
    recovered_dofs = _finite_array(
        result.recovered_first_piola_stress,
        "recovered_first_piola_stress",
    )
    _validate_result_shapes(
        request,
        points,
        weights,
        displacement_coordinates,
        displacement_values,
        raw_stress,
        recovered_dofs,
    )

    skmesh = MeshTri(
        np.asarray(request.mesh.nodes, dtype=np.float64).T,
        np.asarray(request.mesh.cells, dtype=np.int32).T,
        sort_t=False,
    )
    reference_points, reference_weights = _reference_quadrature(
        request,
        points,
        weights,
    )
    scalar_element = ElementTriP1() if result.element_order == 1 else ElementTriP2()
    quadrature = (reference_points, reference_weights)
    scalar_basis = Basis(skmesh, scalar_element, quadrature=quadrature)
    vector_basis = Basis(
        skmesh,
        ElementVector(scalar_element),
        quadrature=quadrature,
    )
    _validate_rebuilt_layout(
        scalar_basis,
        vector_basis,
        points,
        weights,
        displacement_coordinates,
        displacement_values,
        recovered_dofs,
    )

    vector_dofs = _pack_vector_dofs(vector_basis, displacement_values)
    discrete_displacement = vector_basis.interpolate(vector_dofs)
    approximate_displacement = np.asarray(
        discrete_displacement,
        dtype=np.float64,
    )
    approximate_gradient = np.asarray(
        discrete_displacement.grad,
        dtype=np.float64,
    )
    component_leading_points = np.moveaxis(points, -1, 0)
    exact_displacement = manufactured_solution.displacement(component_leading_points)
    exact_gradient = manufactured_solution.gradient(component_leading_points)
    exact_stress = np.moveaxis(
        manufactured_solution.first_piola(component_leading_points),
        (0, 1),
        (-2, -1),
    )

    displacement_l2_error = _weighted_component_norm(
        approximate_displacement - exact_displacement,
        weights,
        component_axes=(0,),
        name="displacement L2 error",
    )
    displacement_h1_error = _weighted_component_norm(
        approximate_gradient - exact_gradient,
        weights,
        component_axes=(0, 1),
        name="displacement H1-seminorm error",
    )
    first_piola_l2_error = _weighted_component_norm(
        raw_stress - exact_stress,
        weights,
        component_axes=(2, 3),
        name="first-Piola L2 error",
    )

    recovered_stress = _interpolate_recovered_stress(
        scalar_basis,
        recovered_dofs,
        (points.shape[0], points.shape[1]),
    )
    recovery = recovery_stress_estimate(raw_stress, recovered_stress, weights)
    return FemVerificationNorms(
        displacement_l2_error=displacement_l2_error,
        displacement_h1_seminorm_error=displacement_h1_error,
        first_piola_l2_error=first_piola_l2_error,
        recovery_estimator_l2=recovery.estimate_l2,
        integrated_weight=recovery.integrated_weight,
        quadrature_point_count=recovery.quadrature_point_count,
    )


def _validate_identity(
    request: FemRequest,
    result: FemResult,
    manufactured_solution: ManufacturedSolution,
) -> None:
    if result.request_id != request.request_id:
        raise ValueError("result request_id does not match the FEM request")
    if result.mesh_id != request.mesh.mesh_id:
        raise ValueError("result mesh_id does not match the FEM request")
    if result.element_order != request.config.element_order:
        raise ValueError("result element_order does not match the FEM request")
    if manufactured_solution.material != request.material:
        raise ValueError(
            "manufactured-solution material does not match the FEM request"
        )


def _validate_result_shapes(
    request: FemRequest,
    points: FloatArray,
    weights: FloatArray,
    displacement_coordinates: FloatArray,
    displacement_values: FloatArray,
    raw_stress: FloatArray,
    recovered_dofs: FloatArray,
) -> None:
    cell_count = len(request.mesh.cells)
    if points.ndim != 3 or points.shape[0] != cell_count or points.shape[2] != 2:
        raise ValueError(
            "quadrature_points must have shape (mesh cells, quadrature points, 2)"
        )
    if points.shape[1] == 0:
        raise ValueError("quadrature_points must contain at least one point per cell")
    if weights.shape != points.shape[:2]:
        raise ValueError("quadrature_weights must match the quadrature point layout")
    if np.any(weights <= 0.0):
        raise ValueError("quadrature_weights must be positive")
    if displacement_coordinates.ndim != 2 or displacement_coordinates.shape[1] != 2:
        raise ValueError("displacement_dof_coordinates must have shape (DOFs, 2)")
    if displacement_values.shape != displacement_coordinates.shape:
        raise ValueError(
            "displacement values must match the displacement DOF coordinate layout"
        )
    if raw_stress.shape != (*points.shape[:2], 3, 3):
        raise ValueError(
            "first_piola_stress must have shape (mesh cells, quadrature points, 3, 3)"
        )
    if recovered_dofs.shape != (displacement_coordinates.shape[0], 3, 3):
        raise ValueError(
            "recovered_first_piola_stress must have shape (scalar-field DOFs, 3, 3)"
        )


def _reference_quadrature(
    request: FemRequest,
    points: FloatArray,
    weights: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    nodes = np.asarray(request.mesh.nodes, dtype=np.float64)
    cells = np.asarray(request.mesh.cells, dtype=np.int64)
    common_points: FloatArray | None = None
    common_weights: FloatArray | None = None
    for cell_index, cell in enumerate(cells):
        vertices = nodes[cell]
        affine = np.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0]))
        determinant = float(np.linalg.det(affine))
        if not math.isfinite(determinant) or determinant == 0.0:
            raise ValueError("request mesh contains a singular triangle")
        local_points = np.linalg.solve(
            affine,
            (points[cell_index] - vertices[0]).T,
        )
        if np.any(local_points < -_LAYOUT_ATOL) or np.any(
            np.sum(local_points, axis=0) > 1.0 + _LAYOUT_ATOL
        ):
            raise ValueError("quadrature point lies outside its declared mesh cell")
        local_weights = weights[cell_index] / abs(determinant)
        if common_points is None:
            common_points = np.asarray(local_points, dtype=np.float64)
            common_weights = np.asarray(local_weights, dtype=np.float64)
            continue
        assert common_weights is not None
        if not np.allclose(
            local_points,
            common_points,
            rtol=0.0,
            atol=_LAYOUT_ATOL,
        ):
            raise ValueError(
                "quadrature points do not use one common reference-cell layout"
            )
        if not np.allclose(
            local_weights,
            common_weights,
            rtol=0.0,
            atol=_LAYOUT_ATOL,
        ):
            raise ValueError(
                "quadrature weights do not use one common reference-cell rule"
            )
    if common_points is None or common_weights is None:
        raise ValueError("request mesh must contain at least one cell")
    return common_points, common_weights


def _validate_rebuilt_layout(
    scalar_basis: Any,
    vector_basis: Any,
    points: FloatArray,
    weights: FloatArray,
    displacement_coordinates: FloatArray,
    displacement_values: FloatArray,
    recovered_dofs: FloatArray,
) -> None:
    expected_coordinates = np.asarray(scalar_basis.doflocs, dtype=np.float64).T
    if expected_coordinates.shape != displacement_coordinates.shape or not np.allclose(
        expected_coordinates,
        displacement_coordinates,
        rtol=0.0,
        atol=_LAYOUT_ATOL,
    ):
        raise ValueError(
            "result displacement DOF coordinate layout does not match the rebuilt basis"
        )
    rebuilt_points = np.transpose(
        np.asarray(vector_basis.global_coordinates(), dtype=np.float64),
        (1, 2, 0),
    )
    if rebuilt_points.shape != points.shape or not np.allclose(
        rebuilt_points,
        points,
        rtol=0.0,
        atol=_LAYOUT_ATOL,
    ):
        raise ValueError(
            "result quadrature point layout does not match the rebuilt basis"
        )
    rebuilt_weights = np.asarray(vector_basis.dx, dtype=np.float64)
    if rebuilt_weights.shape != weights.shape or not np.allclose(
        rebuilt_weights,
        weights,
        rtol=0.0,
        atol=_LAYOUT_ATOL,
    ):
        raise ValueError(
            "result quadrature weight layout does not match the rebuilt basis"
        )


def _pack_vector_dofs(vector_basis: Any, displacement: FloatArray) -> FloatArray:
    x_dofs, y_dofs = tuple(
        np.asarray(dofs, dtype=np.int64) for dofs in vector_basis.split_indices()
    )
    packed = np.empty(vector_basis.N, dtype=np.float64)
    packed[x_dofs] = displacement[:, 0]
    packed[y_dofs] = displacement[:, 1]
    return packed


def _interpolate_recovered_stress(
    scalar_basis: Any,
    recovered_dofs: FloatArray,
    quadrature_shape: tuple[int, int],
) -> FloatArray:
    recovered = np.empty((*quadrature_shape, 3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            component = np.asarray(
                scalar_basis.interpolate(recovered_dofs[:, row, column]),
                dtype=np.float64,
            )
            recovered[..., row, column] = component
    return recovered


def _weighted_component_norm(
    difference: FloatArray,
    weights: FloatArray,
    *,
    component_axes: tuple[int, ...],
    name: str,
) -> float:
    try:
        with np.errstate(over="raise", invalid="raise"):
            squared_components = np.sum(
                np.square(np.asarray(difference, dtype=np.float64)),
                axis=component_axes,
            )
            squared_norm = float(np.sum(weights * squared_components))
    except FloatingPointError as error:
        raise ValueError(f"{name} is not finite") from error
    if not math.isfinite(squared_norm) or squared_norm < 0.0:
        raise ValueError(f"{name} is not finite")
    return math.sqrt(squared_norm)


def _finite_array(values: ArrayLike, name: str) -> FloatArray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite rectangular array") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


__all__ = ("FemVerificationNorms", "compute_verification_norms")
