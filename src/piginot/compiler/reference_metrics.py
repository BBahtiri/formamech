"""Discrete FEM comparisons and explicitly uncalibrated refinement diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from piginot.contracts import ProblemSpec
from piginot.contracts.problem_spec import region_nodes
from piginot.fem.models import FemRequest, FemResult
from piginot.operators.architecture import OperatorOutput


def _finite(value: object, shape: tuple[int, ...], name: str) -> Any:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name}: invalid shape or nonfinite data")
    return array


def _relative(difference: Any, reference: Any, weights: Any) -> float:
    numerator = float(np.sum(weights * np.sum(difference**2, axis=-1)))
    denominator = float(np.sum(weights * np.sum(reference**2, axis=-1)))
    if denominator == 0.0:
        if numerator != 0.0:
            raise ValueError("relative error has a zero reference norm")
        return 0.0
    return float(np.sqrt(numerator / denominator))


def _predict(
    predict: Callable[[torch.Tensor], OperatorOutput], points: Any
) -> OperatorOutput:
    output = predict(torch.tensor(points, dtype=torch.float64))
    if any(
        field.dtype != torch.float64 or field.device.type != "cpu"
        for field in (output.u, output.P)
    ):
        raise ValueError("reference comparison requires float64 CPU inference")
    return output


def _cauchy_vertices(
    request: FemRequest, result: FemResult, scalar: Any, prediction: OperatorOutput
) -> tuple[Any, Any]:
    from piginot.fem.assembly import recover_stress
    from piginot.fem.backend import _kernel_material
    from piginot.physics import plane_stress_closure

    shape = np.asarray(result.first_piola_stress).shape
    deformation = _finite(result.deformation_gradient, shape, "reference deformation")
    determinant = np.linalg.det(deformation)
    if np.any(determinant <= 0.0):
        raise ValueError("Cauchy stress requires positive reference determinant")
    stress = np.asarray(result.first_piola_stress, dtype=np.float64)
    sigma = stress @ deformation.swapaxes(-2, -1) / determinant[..., None, None]
    recovered = recover_stress(scalar, sigma)
    vertices = recovered[np.asarray(scalar.nodal_dofs[0]), :2, :2]
    gradient = getattr(prediction, "gradient", None)
    if (
        not isinstance(gradient, torch.Tensor)
        or gradient.dtype != torch.float64
        or gradient.device.type != "cpu"
    ):
        raise ValueError("Cauchy stress requires a float64 CPU operator gradient")
    values = _finite(
        gradient.detach().numpy(),
        (1, len(request.mesh.nodes), 2, 2),
        "operator gradient",
    )[0]
    deformation_2d = torch.eye(2, dtype=torch.float64) + torch.tensor(values)
    determinant_2d = torch.linalg.det(deformation_2d)
    if bool((determinant_2d <= 0.0).any()):
        raise ValueError("Cauchy stress requires positive operator determinant")
    closure = plane_stress_closure(deformation_2d, _kernel_material(request.material))
    determinant_3d = determinant_2d * closure
    predicted = (
        prediction.P[0]
        @ deformation_2d.transpose(-2, -1)
        / determinant_3d[:, None, None]
    )
    return vertices, predicted.detach().numpy()


def compare_prediction(
    spec: ProblemSpec,
    request: FemRequest,
    result: FemResult,
    predict: Callable[[torch.Tensor], OperatorOutput],
) -> dict[str, Any]:
    """Compare fields on FEM quadrature and declared QoIs on region vertices.

    Stress uses the in-plane first-Piola tensor. Displacement is interpolated
    with the exact FEM element and layout; no nearest-node transfer is used.
    """
    from piginot.fem.backend import _build_basis

    if (
        result.request_id != request.request_id
        or result.mesh_id != request.mesh.mesh_id
        or result.element_order != request.config.element_order
    ):
        raise ValueError("reference identity differs from its request")
    _, basis, scalar = _build_basis(request)
    nodes = np.asarray(request.mesh.nodes, dtype=np.float64)
    coordinates = _finite(
        result.displacement_dof_coordinates, (scalar.N, 2), "DOF coordinates"
    )
    if not np.array_equal(coordinates, scalar.doflocs.T):
        raise ValueError("reference DOF coordinates differ from the FEM layout")
    points = np.moveaxis(np.asarray(basis.global_coordinates()), 0, -1)
    ncell, nquad, _ = points.shape
    observed = _finite(result.quadrature_points, points.shape, "quadrature points")
    weights = _finite(result.quadrature_weights, (ncell, nquad), "quadrature weights")
    if (
        not np.allclose(observed, points, rtol=0.0, atol=5e-13)
        or not np.allclose(weights, basis.dx, rtol=0.0, atol=5e-13)
        or np.any(weights <= 0.0)
    ):
        raise ValueError("reference quadrature differs from the FEM layout")
    displacement = _finite(result.displacement, (scalar.N, 2), "displacement")
    target_u = np.stack(
        [scalar.interpolate(displacement[:, axis]) for axis in (0, 1)], -1
    )
    target_p = _finite(result.first_piola_stress, (ncell, nquad, 3, 3), "stress")[
        ..., :2, :2
    ]
    output = _predict(predict, points.reshape(-1, 2))
    pred_u = _finite(
        output.u.detach().cpu().numpy(), (1, ncell * nquad, 2), "predicted u"
    )[0]
    pred_p = _finite(
        output.P.detach().cpu().numpy(), (1, ncell * nquad, 2, 2), "predicted P"
    )[0]
    recovered = _finite(
        result.recovered_first_piola_stress, (scalar.N, 3, 3), "recovered stress"
    )
    vertex_stress = recovered[np.asarray(scalar.nodal_dofs[0]), :2, :2]
    vertex_output = _predict(predict, nodes)
    predicted_vertices = _finite(
        vertex_output.P.detach().cpu().numpy(),
        (1, len(nodes), 2, 2),
        "predicted vertex stress",
    )[0]
    cauchy = (
        _cauchy_vertices(request, result, scalar, vertex_output)
        if any(qoi.field == "sigma" for qoi in spec.qois)
        else None
    )
    qois: dict[str, Any] = {}
    for qoi in spec.qois:
        indices = set(region_nodes(request.mesh, qoi.region))
        for region in qoi.exclude_regions:
            indices.difference_update(region_nodes(request.mesh, region))
        if not indices:
            raise ValueError(f"{qoi.name}: the declared QoI has no vertices")
        index = sorted(indices)
        reference_field, predicted_field = (
            cauchy
            if qoi.field == "sigma" and cauchy is not None
            else (vertex_stress, predicted_vertices)
        )
        reference = float(np.linalg.norm(reference_field[index], axis=(-2, -1)).max())
        predicted = float(np.linalg.norm(predicted_field[index], axis=(-2, -1)).max())
        relative = _relative(
            np.array([[predicted - reference]]), np.array([[reference]]), np.ones(1)
        )
        qois[qoi.name] = {
            "field": qoi.field,
            "reference": reference,
            "predicted": predicted,
            "absolute_peak_error": relative,
            "unit": "MPa",
        }
    return {
        "u_relative_l2": _relative(
            pred_u - target_u.reshape(-1, 2),
            target_u.reshape(-1, 2),
            weights.reshape(-1),
        ),
        "p_relative_l2": _relative(
            pred_p.reshape(-1, 4) - target_p.reshape(-1, 4),
            target_p.reshape(-1, 4),
            weights.reshape(-1),
        ),
        "absolute_peak_error": max(
            item["absolute_peak_error"] for item in qois.values()
        ),
        "qois": qois,
        "comparison": "discrete-fem-quadrature-and-region-vertices",
    }


def refinement_diagnostics(results: tuple[FemResult, ...]) -> dict[str, Any]:
    """Physical-domain integral deltas; these are not calibrated error terms."""
    integrals = []
    for result in results:
        stress = np.asarray(result.first_piola_stress, dtype=np.float64)[..., :2, :2]
        weights = np.asarray(result.quadrature_weights, dtype=np.float64)
        integrals.append(np.sum(stress * weights[..., None, None], axis=(0, 1)))
    return {
        "status": "UNCALIBRATED",
        "total_error_bound": None,
        "method": "physical-domain-stress-integral-delta",
        "successive_integral_deltas": [
            float(np.linalg.norm(right - left))
            for left, right in zip(integrals, integrals[1:], strict=False)
        ],
        "integral_unit": "N",
        "reference_terms_qualified": False,
    }


__all__ = ("compare_prediction", "refinement_diagnostics")
