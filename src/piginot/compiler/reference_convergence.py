"""Measured FEM field changes on one fixed polygon; never an error bound.

The coarse displacement and its gradient are evaluated with their actual
finite-element basis at fine quadrature points. Stress is recomputed from
that gradient with the request's material and plane-stress closure. Boundary
peaks use the same recovered vertex fields and exclusions as model screening.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from piginot.contracts import ProblemSpec
    from piginot.fem.models import FemRequest, FemResult

_QUANTITIES = frozenset(("u_relative_l2", "p_relative_l2", "absolute_peak_error"))


@dataclass(frozen=True)
class _Field:
    request: FemRequest
    result: FemResult
    scalar: Any
    points: Any
    weights: Any
    displacement: Any
    stress: Any
    recovered: Any


def _array(value: Any, shape: tuple[int, ...], name: str) -> Any:
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name}: invalid shape or nonfinite data")
    return array


def _validated(request: FemRequest, result: FemResult) -> _Field:
    import numpy as np

    from piginot.fem.backend import _build_basis

    if (
        result.request_id != request.request_id
        or result.mesh_id != request.mesh.mesh_id
        or result.element_order != request.config.element_order
    ):
        raise ValueError("FEM result identity differs from its request")
    _, basis, scalar = _build_basis(request)
    points = np.moveaxis(np.asarray(basis.global_coordinates()), 0, -1)
    shape = points.shape[:2]
    observed = _array(result.quadrature_points, points.shape, "quadrature points")
    weights = _array(result.quadrature_weights, shape, "quadrature weights")
    coordinates = _array(
        result.displacement_dof_coordinates, (scalar.N, 2), "DOF coordinates"
    )
    if (
        not np.array_equal(coordinates, scalar.doflocs.T)
        or not np.allclose(observed, points, rtol=0, atol=5e-13)
        or not np.allclose(weights, basis.dx, rtol=0, atol=5e-13)
        or np.any(weights <= 0)
    ):
        raise ValueError("stored FEM layout differs from the rebuilt basis")
    return _Field(
        request,
        result,
        scalar,
        points,
        weights,
        _array(result.displacement, (scalar.N, 2), "displacement"),
        _array(result.first_piola_stress, (*shape, 3, 3), "stress"),
        _array(
            result.recovered_first_piola_stress, (scalar.N, 3, 3), "recovered stress"
        ),
    )


def _same_polygon(coarse: FemRequest, fine: FemRequest) -> None:
    from piginot.cad.refinement import refine_fixed_mesh

    if replace(fine, request_id=coarse.request_id, mesh=coarse.mesh) != coarse:
        raise ValueError("FEM convergence requires identical physical/solver settings")
    refine_fixed_mesh(fine.mesh, 1)  # Validate the supplied fine identity too.
    levels = {
        4 * len(coarse.mesh.cells): 2,
        16 * len(coarse.mesh.cells): 4,
    }
    level = levels.get(len(fine.mesh.cells))
    if level is None:
        raise ValueError("fine mesh must be an exact tagged midpoint refinement")
    expected = refine_fixed_mesh(coarse.mesh, level)
    attributes = (
        "nodes",
        "cells",
        "boundary_facets",
        "semantic_tags",
        "region_ids",
        "cell_type",
    )
    if any(getattr(expected, name) != getattr(fine.mesh, name) for name in attributes):
        raise ValueError("fine mesh must be an exact tagged midpoint refinement")


def _evaluate(field: _Field, points: Any) -> tuple[Any, Any]:
    """Evaluate FE values and physical gradients without nodal approximation."""
    import numpy as np

    scalar = field.scalar
    finder = scalar.mesh.element_finder(mapping=scalar.mapping)
    values, gradients = [], []
    for start in range(0, len(points), 1024):
        x = points[start : start + 1024].T
        cells = finder(*x)
        local = scalar.mapping.invF(x[:, :, None], tind=cells)
        shapes = [
            scalar.elem.gbasis(scalar.mapping, local, k, tind=cells)[0]
            for k in range(scalar.Nbfun)
        ]
        coefficients = field.displacement[scalar.element_dofs[:, cells]]
        phi = np.stack([np.asarray(shape)[:, 0] for shape in shapes])
        derivative = np.stack([shape.grad[:, :, 0] for shape in shapes])
        values.append(np.einsum("knc,kn->nc", coefficients, phi))
        gradients.append(np.einsum("knc,kdn->ncd", coefficients, derivative))
    return np.concatenate(values), np.concatenate(gradients)


def _stress(gradient: Any, request: FemRequest) -> Any:
    import numpy as np
    import torch

    from piginot.fem.backend import _kernel_material
    from piginot.physics import first_piola, plane_stress_closure
    from piginot.physics.errors import PhysicsError

    material = _kernel_material(request.material)
    values = []
    for start in range(0, len(gradient), 1024):
        deformation = torch.from_numpy(gradient[start : start + 1024] + np.eye(2))
        try:
            closure = plane_stress_closure(
                deformation,
                material,
                max_iterations=request.config.plane_stress_max_iterations,
            )
            full = torch.zeros((len(deformation), 3, 3), dtype=torch.float64)
            full[:, :2, :2] = deformation
            full[:, 2, 2] = closure
            values.append(first_piola(full, material).detach().numpy()[:, :2, :2])
        except PhysicsError as error:
            raise ValueError("coarse FE constitutive evaluation failed") from error
    return np.concatenate(values)


def _log_squared_norm(values: Any, log_weights: Any, log_scale: float) -> float:
    import numpy as np

    # Shared field/weight scales cancel in the ratio. Log-sum-exp preserves
    # contributions when squaring either the raw or scaled values underflows.
    with np.errstate(divide="ignore"):
        terms = 2 * (np.log(np.abs(values)) - log_scale) + log_weights[:, None]
    largest = float(np.max(terms))
    return largest + math.log(float(np.exp(terms - largest).sum()))


def _relative(difference: Any, reference: Any, weights: Any) -> float:
    import numpy as np

    if not all(
        np.isfinite(v).all() for v in (difference, reference, weights)
    ) or np.any(weights <= 0):
        raise ValueError("FEM field-change norm is nonfinite or has invalid weights")
    if not np.any(reference):
        if np.any(difference):
            raise ValueError("relative field change has a zero fine-reference norm")
        return 0.0
    if not np.any(difference):
        return 0.0
    scale = max(float(np.abs(difference).max()), float(np.abs(reference).max()))
    log_weights = np.log(weights)
    log_weights = log_weights - log_weights.max()
    numerator = _log_squared_norm(difference, log_weights, math.log(scale))
    denominator = _log_squared_norm(reference, log_weights, math.log(scale))
    try:
        return math.exp((numerator - denominator) / 2)
    except OverflowError as error:
        raise ValueError("FEM field-change norm is nonfinite") from error


def _vertices(field: _Field, name: str) -> Any:
    import numpy as np

    recovered = field.recovered
    if name == "sigma":
        from piginot.fem.assembly import recover_stress

        deformation = _array(
            field.result.deformation_gradient, field.stress.shape, "deformation"
        )
        determinant = np.linalg.det(deformation)
        if np.any(determinant <= 0):
            raise ValueError("Cauchy stress requires positive reference determinant")
        sigma = (
            field.stress @ deformation.swapaxes(-2, -1) / determinant[..., None, None]
        )
        recovered = recover_stress(field.scalar, sigma)
    return recovered[np.asarray(field.scalar.nodal_dofs[0]), :2, :2]


def _peaks(spec: ProblemSpec, coarse: _Field, fine: _Field) -> dict[str, Any]:
    import numpy as np

    from piginot.contracts.problem_spec import region_nodes

    qois = {}
    for qoi in spec.qois:
        peaks = []
        for field in (coarse, fine):
            indices = set(region_nodes(field.request.mesh, qoi.region))
            for excluded in qoi.exclude_regions:
                indices.difference_update(region_nodes(field.request.mesh, excluded))
            if not indices:
                raise ValueError(f"{qoi.name}: the declared QoI has no vertices")
            stress = _vertices(field, qoi.field)[sorted(indices)]
            peaks.append(
                float(np.hypot.reduce(stress.reshape(len(stress), -1), axis=1).max())
            )
        difference = np.array([[peaks[0] - peaks[1]]])
        relative = _relative(difference, np.array([[peaks[1]]]), np.ones(1))
        qois[qoi.name] = {
            "field": qoi.field,
            "coarse": peaks[0],
            "fine": peaks[1],
            "relative_change": relative,
            "unit": "MPa",
        }
    if not qois:
        raise ValueError("FEM convergence requires declared selected-peak QoIs")
    return qois


def compare_fem_pair(
    spec: ProblemSpec,
    coarse_request: FemRequest,
    coarse_result: FemResult,
    fine_request: FemRequest,
    fine_result: FemResult,
    *,
    tolerances: Mapping[str, float] | None,
) -> dict[str, Any]:
    """Compare supported midpoint refinements using explicit tolerances.

    PASS means only that the measured successive changes meet these targets.
    It supplies neither a continuum error bound nor proof of peak convergence.
    Explicit ``None`` measures changes without introducing acceptance targets.
    """
    import numpy as np

    if tolerances is not None and (
        set(tolerances) != _QUANTITIES
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v < 0
            for v in tolerances.values()
        )
    ):
        raise ValueError("tolerances must specify three finite nonnegative targets")
    _same_polygon(coarse_request, fine_request)
    coarse = _validated(coarse_request, coarse_result)
    fine = _validated(fine_request, fine_result)
    points, weights = fine.points.reshape(-1, 2), fine.weights.reshape(-1)
    coarse_u, coarse_gradient = _evaluate(coarse, points)
    fine_u = np.stack(
        [fine.scalar.interpolate(fine.displacement[:, axis]) for axis in (0, 1)], -1
    ).reshape(-1, 2)
    fine_p = fine.stress[..., :2, :2].reshape(-1, 4)
    coarse_p = _stress(coarse_gradient, coarse_request).reshape(-1, 4)
    qois = _peaks(spec, coarse, fine)
    quantities = {
        "u_relative_l2": _relative(coarse_u - fine_u, fine_u, weights),
        "p_relative_l2": _relative(coarse_p - fine_p, fine_p, weights),
        "absolute_peak_error": max(qoi["relative_change"] for qoi in qois.values()),
    }
    return {
        "status": "MEASURED_CHANGES_ONLY"
        if tolerances is None
        else (
            "PASS"
            if all(quantities[k] <= tolerances[k] for k in quantities)
            else "FAIL"
        ),
        "qualification": "UNCALIBRATED",
        "geometry_status": "SAME_POLYGON",
        "method": "nested-fe-field-and-recovered-region-vertex-changes",
        "coarse_mesh_id": coarse_request.mesh.mesh_id,
        "fine_mesh_id": fine_request.mesh.mesh_id,
        "quantities": quantities,
        "tolerances": None if tolerances is None else dict(tolerances),
        "qois": qois,
        "total_error_bound": None,
        "reference_terms_qualified": False,
    }


__all__ = ("compare_fem_pair",)
