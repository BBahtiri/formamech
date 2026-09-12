"""Non-vacuous weak/strong comparison on one converged P2 FEM solution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from skfem import Basis, MeshTri  # type: ignore[import-untyped]
from skfem.element import ElementTriP2  # type: ignore[import-untyped]

from piginot.fem.mms_runner import MmsLevelResult, solve_mms_level
from piginot.objectives.equilibrium import (
    strong_equilibrium_residual,
    weak_equilibrium_residual,
)
from piginot.objectives.models import CoordinateConversion, NondimensionalScales


@dataclass(frozen=True, slots=True)
class WeakStrongEvidence:
    fem_solve_success: bool
    difference: float
    weak_residual_max: float
    projected_strong_residual_max: float
    reconstructed_stress_variation: float
    request_id: str
    quadrature_point_count: int


def run_weak_strong_fem_check() -> WeakStrongEvidence:
    """Compare both forms on the same nonconstant recovered-stress field."""

    request_id = "s8-pure_dirichlet-structured-l2-p2"
    try:
        solved = solve_mms_level("pure_dirichlet", level=2, element_order=2)
    except RuntimeError:
        return WeakStrongEvidence(False, math.inf, 0.0, 0.0, 0.0, request_id, 0)
    result = solved.result
    point_array = np.asarray(result.quadrature_points, dtype=np.float64)
    recovered_stress = _recovered_stress_at_quadrature(solved)[..., :2, :2]
    fit_points = torch.tensor(point_array, dtype=torch.float64)
    centers = fit_points.mean(dim=1, keepdim=True)
    scales = fit_points.amax(dim=1, keepdim=True) - fit_points.amin(dim=1, keepdim=True)
    fit_basis = _quadratic_basis(fit_points, centers, scales)
    coefficients = torch.linalg.lstsq(
        fit_basis,
        torch.tensor(
            recovered_stress.reshape(*recovered_stress.shape[:2], 4),
            dtype=torch.float64,
        ),
    ).solution
    points = fit_points.reshape(1, -1, 2).detach().clone().requires_grad_(True)
    live_points = points.reshape_as(fit_points)
    reconstructed = torch.einsum(
        "cqk,ckm->cqm",
        _quadratic_basis(live_points, centers, scales),
        coefficients,
    ).reshape(*live_points.shape[:2], 2, 2)
    stress = reconstructed.reshape(1, -1, 2, 2)
    weights = torch.tensor(
        np.asarray(result.quadrature_weights).reshape(1, -1), dtype=torch.float64
    )
    body_force = torch.tensor(
        np.moveaxis(
            solved.manufactured_solution.body_force(np.moveaxis(point_array, -1, 0)),
            0,
            -1,
        ).reshape(1, -1, 2),
        dtype=torch.float64,
    )
    x_coordinate = points[..., 0]
    y_coordinate = points[..., 1]
    test_values = (
        x_coordinate * (1.0 - x_coordinate) * y_coordinate * (1.0 - y_coordinate)
    ).unsqueeze(-1)
    gradients = torch.stack(
        (
            (1.0 - 2.0 * x_coordinate) * y_coordinate * (1.0 - y_coordinate),
            x_coordinate * (1.0 - x_coordinate) * (1.0 - 2.0 * y_coordinate),
        ),
        dim=-1,
    ).unsqueeze(-2)
    fem_scales = NondimensionalScales(1.0, 0.08, 100.0)
    weak = weak_equilibrium_residual(
        stress=stress,
        body_force=body_force,
        test_values=test_values,
        test_gradients=gradients,
        volume_weights=weights,
        scales=fem_scales,
    )
    strong = strong_equilibrium_residual(
        stress,
        points,
        body_force,
        fem_scales,
        CoordinateConversion((1.0, 1.0)),
    )
    projected_strong = (
        torch.einsum("bqi,bqa,bq->bai", strong, test_values, weights)
        / fem_scales.length_mm**2
    )
    centered_stress = stress - stress.mean(dim=1, keepdim=True)
    return WeakStrongEvidence(
        True,
        float((weak + projected_strong).abs().max().item()),
        float(weak.abs().max().item()),
        float(projected_strong.abs().max().item()),
        float(centered_stress.abs().max().item()),
        solved.request.request_id,
        int(points.shape[1]),
    )


def _quadratic_basis(
    points: torch.Tensor, centers: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    local = (points - centers) / scales
    x_coordinate = local[..., 0]
    y_coordinate = local[..., 1]
    return torch.stack(
        (
            torch.ones_like(x_coordinate),
            x_coordinate,
            y_coordinate,
            x_coordinate.square(),
            x_coordinate * y_coordinate,
            y_coordinate.square(),
        ),
        dim=-1,
    )


def _recovered_stress_at_quadrature(
    solved: MmsLevelResult,
) -> NDArray[np.float64]:
    request = solved.request
    result = solved.result
    mesh = MeshTri(
        np.asarray(request.mesh.nodes, dtype=np.float64).T,
        np.asarray(request.mesh.cells, dtype=np.int32).T,
        sort_t=False,
    )
    basis = Basis(mesh, ElementTriP2(), intorder=5)
    recovered_dofs = np.asarray(result.recovered_first_piola_stress, dtype=np.float64)
    point_shape = np.asarray(result.quadrature_points).shape[:2]
    recovered = np.empty((*point_shape, 3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            recovered[..., row, column] = np.asarray(
                basis.interpolate(recovered_dofs[:, row, column]), dtype=np.float64
            )
    return recovered


__all__ = ("WeakStrongEvidence", "run_weak_strong_fem_check")
