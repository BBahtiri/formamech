"""Pure area-weighted diagnostics of physical S13 and FEM quadrature fields."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch

from piginot.contracts import MaterialSpec
from piginot.objectives import NondimensionalScales, constitutive_residual
from piginot.physics._validation import DETERMINANT_MARGIN
from piginot.physics.errors import PhysicsError

Matrix2 = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True, slots=True)
class InvalidQuadraturePoint:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    cell_index: int
    quadrature_index: int
    xy_mm: tuple[float, float]
    determinant: float
    area_weight_mm2: float


@dataclass(frozen=True, slots=True)
class ConstitutiveDecomposition:
    """A=P-M(model), B=M(model)-M(FEM), C=M(FEM)-P_FEM, valid model only.

    Energies and cross terms are area-weighted means per in-plane component,
    normalized by stress_mpa squared. Signed cross terms are not clamped;
    the three RMS norms do not add to the total RMS.
    """

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    admissible_count: int
    admissible_area_mm2: float
    a_normalized_component_rms: float
    b_normalized_component_rms: float
    c_normalized_component_rms: float
    material_fem_relative_l2: float
    material_model_relative_l2: float
    cross_ab_normalized_mean: float
    cross_ac_normalized_mean: float
    cross_bc_normalized_mean: float
    total_normalized_mean_squared: float
    reconstructed_normalized_mean_squared: float
    identity_absolute_error: float
    reconstruction_max_abs_mpa: float


@dataclass(frozen=True, slots=True)
class QuadraturePopulationMetrics:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    cell_count: int
    point_count: int
    area_mm2: float
    gradient_relative_h1: float
    gradient_component_rms: float
    gradient_bias: Matrix2
    p_relative_l2: float
    p_normalized_component_rms: float
    u_relative_l2: float | None
    u_component_rms_mm: float | None
    model_determinant_min: float
    model_determinant_max: float
    fem_determinant_min: float
    fem_determinant_max: float
    material_fem_relative_l2: float
    material_fem_normalized_component_rms: float
    invalid_count: int
    invalid_points: tuple[InvalidQuadraturePoint, ...]
    admissible_count: int
    admissible_area_fraction: float
    decomposition: ConstitutiveDecomposition | None


@dataclass(frozen=True, slots=True)
class QuadratureCaseMetrics:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    all_points: QuadraturePopulationMetrics
    root_incident: QuadraturePopulationMetrics
    nonroot: QuadraturePopulationMetrics


def _field(value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError("quadrature fields must have aligned cell/point shapes")
    if (
        value.dtype is not torch.float64
        or value.device.type != "cpu"
        or value.layout is not torch.strided
    ):
        raise ValueError("quadrature fields must be dense CPU float64")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("quadrature fields must be finite")


def _weighted_inner(
    left: torch.Tensor, right: torch.Tensor, weights: torch.Tensor
) -> float:
    return float((weights * (left * right).flatten(1).sum(dim=1)).sum())


def _relative(
    error: torch.Tensor, reference: torch.Tensor, weights: torch.Tensor, label: str
) -> float:
    denominator = _weighted_inner(reference, reference, weights)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"each FEM population needs a finite positive {label} norm")
    numerator = _weighted_inner(error, error, weights)
    if not math.isfinite(numerator):
        raise ValueError(f"{label} error energy must be finite")
    return math.sqrt(numerator / denominator)


def _rms(error: torch.Tensor, weights: torch.Tensor) -> float:
    components = math.prod(error.shape[1:])
    return math.sqrt(
        _weighted_inner(error, error, weights) / (components * float(weights.sum()))
    )


def _material_stress(
    gradient: torch.Tensor, material: MaterialSpec, scales: NondimensionalScales
) -> torch.Tensor:
    return (
        -constitutive_residual(torch.zeros_like(gradient), gradient, material, scales)
        * scales.stress_mpa
    )


def _checked_material_stress(
    gradient: torch.Tensor,
    material: MaterialSpec,
    scales: NondimensionalScales,
    label: str,
) -> torch.Tensor:
    try:
        stress = _material_stress(gradient, material, scales)
    except (PhysicsError, ValueError, RuntimeError) as error:
        raise ValueError(
            f"{label} constitutive evaluation failed; no points dropped"
        ) from error
    _field(stress, tuple(gradient.shape))
    return stress


def _decompose(
    served: torch.Tensor,
    raw: torch.Tensor,
    model: torch.Tensor,
    fem: torch.Tensor,
    weights: torch.Tensor,
    stress_scale: float,
) -> ConstitutiveDecomposition:
    a, b, c = (
        (served - model) / stress_scale,
        (model - fem) / stress_scale,
        (fem - raw) / stress_scale,
    )
    divisor = 4.0 * float(weights.sum())
    energies = tuple(
        _weighted_inner(term, term, weights) / divisor for term in (a, b, c)
    )
    cross = tuple(
        2.0 * _weighted_inner(left, right, weights) / divisor
        for left, right in ((a, b), (a, c), (b, c))
    )
    total = (served - raw) / stress_scale
    total_energy = _weighted_inner(total, total, weights) / divisor
    reconstructed = sum(energies) + sum(cross)
    return ConstitutiveDecomposition(
        admissible_count=len(weights),
        admissible_area_mm2=float(weights.sum()),
        a_normalized_component_rms=math.sqrt(energies[0]),
        b_normalized_component_rms=math.sqrt(energies[1]),
        c_normalized_component_rms=math.sqrt(energies[2]),
        material_fem_relative_l2=_relative(
            fem - raw, raw, weights, "conditional stress"
        ),
        material_model_relative_l2=_relative(
            model - raw, raw, weights, "conditional stress"
        ),
        cross_ab_normalized_mean=cross[0],
        cross_ac_normalized_mean=cross[1],
        cross_bc_normalized_mean=cross[2],
        total_normalized_mean_squared=total_energy,
        reconstructed_normalized_mean_squared=reconstructed,
        identity_absolute_error=abs(total_energy - reconstructed),
        reconstruction_max_abs_mpa=float(
            ((a + b + c - total) * stress_scale).abs().max()
        ),
    )


def _population(
    *,
    mask: torch.Tensor,
    grad_model: torch.Tensor,
    grad_fem: torch.Tensor,
    served_p: torch.Tensor,
    raw_fem_p: torch.Tensor,
    area_weights: torch.Tensor,
    xy_mm: torch.Tensor,
    model_det: torch.Tensor,
    fem_det: torch.Tensor,
    model_stress: torch.Tensor,
    fem_stress: torch.Tensor,
    u_model: torch.Tensor | None,
    u_fem: torch.Tensor | None,
    stress_scale: float,
) -> QuadraturePopulationMetrics:
    weights = area_weights[mask].flatten()
    area = float(weights.sum())
    gradient_delta = (grad_model - grad_fem)[mask].reshape(-1, 2, 2)
    p_delta = (served_p - raw_fem_p)[mask].reshape(-1, 2, 2)
    bias = (weights[:, None, None] * gradient_delta).sum(dim=0) / area
    gradient_bias = (
        (float(bias[0, 0]), float(bias[0, 1])),
        (float(bias[1, 0]), float(bias[1, 1])),
    )
    selected_model_det, selected_fem_det = model_det[mask], fem_det[mask]
    valid = selected_model_det > DETERMINANT_MARGIN
    invalid_indices = (mask[:, None] & (model_det <= DETERMINANT_MARGIN)).nonzero()
    invalid_points = tuple(
        InvalidQuadraturePoint(
            int(cell),
            int(point),
            (float(xy_mm[cell, point, 0]), float(xy_mm[cell, point, 1])),
            float(model_det[cell, point]),
            float(area_weights[cell, point]),
        )
        for cell, point in invalid_indices
    )
    conditional = None
    if bool(valid.any()):
        conditional = _decompose(
            served_p[mask][valid],
            raw_fem_p[mask][valid],
            model_stress[mask][valid],
            fem_stress[mask][valid],
            area_weights[mask][valid],
            stress_scale,
        )
    fem_delta = (fem_stress - raw_fem_p)[mask].reshape(-1, 2, 2)
    u_relative, u_rms = None, None
    if u_model is not None and u_fem is not None:
        u_delta = (u_model - u_fem)[mask].reshape(-1, 2)
        u_relative = _relative(
            u_delta, u_fem[mask].reshape(-1, 2), weights, "displacement"
        )
        u_rms = _rms(u_delta, weights)
    return QuadraturePopulationMetrics(
        cell_count=int(mask.sum()),
        point_count=len(weights),
        area_mm2=area,
        gradient_relative_h1=_relative(
            gradient_delta, grad_fem[mask].reshape(-1, 2, 2), weights, "gradient"
        ),
        gradient_component_rms=_rms(gradient_delta, weights),
        gradient_bias=gradient_bias,
        p_relative_l2=_relative(
            p_delta, raw_fem_p[mask].reshape(-1, 2, 2), weights, "stress"
        ),
        p_normalized_component_rms=_rms(p_delta / stress_scale, weights),
        u_relative_l2=u_relative,
        u_component_rms_mm=u_rms,
        model_determinant_min=float(selected_model_det.min()),
        model_determinant_max=float(selected_model_det.max()),
        fem_determinant_min=float(selected_fem_det.min()),
        fem_determinant_max=float(selected_fem_det.max()),
        material_fem_relative_l2=_relative(
            fem_delta, raw_fem_p[mask].reshape(-1, 2, 2), weights, "stress"
        ),
        material_fem_normalized_component_rms=_rms(fem_delta / stress_scale, weights),
        invalid_count=len(invalid_points),
        invalid_points=invalid_points,
        admissible_count=int(valid.sum()),
        admissible_area_fraction=float(area_weights[mask][valid].sum()) / area,
        decomposition=conditional,
    )


def summarize_quadrature_case(
    *,
    grad_model: torch.Tensor,
    grad_fem: torch.Tensor,
    served_p: torch.Tensor,
    raw_fem_p: torch.Tensor,
    area_weights: torch.Tensor,
    root_cell_mask: torch.Tensor,
    xy_mm: torch.Tensor,
    material: MaterialSpec,
    scales: NondimensionalScales,
    u_model: torch.Tensor | None = None,
    u_fem: torch.Tensor | None = None,
) -> QuadratureCaseMetrics:
    """Compare every physical point; material decomposition is explicitly conditional.

    Gradients are physical du_i/dx_j; P is in MPa, xy/u are in mm, and positive
    area_weights are physical reference-domain quadrature areas in mm squared.
    H1 is the displacement-gradient seminorm, not a deformation-gradient norm.
    """

    if (
        not isinstance(grad_model, torch.Tensor)
        or grad_model.ndim != 4
        or grad_model.shape[-2:] != (2, 2)
        or min(grad_model.shape[:2]) <= 0
    ):
        raise ValueError(
            "quadrature fields must have aligned nonempty (cells,q,2,2) shapes"
        )
    cells, points = grad_model.shape[:2]
    for value in (grad_model, grad_fem, served_p, raw_fem_p):
        _field(value, (cells, points, 2, 2))
    _field(area_weights, (cells, points))
    _field(xy_mm, (cells, points, 2))
    if not bool((area_weights > 0.0).all()) or not math.isfinite(
        float(area_weights.sum())
    ):
        raise ValueError(
            "physical quadrature area weights must be positive with finite total"
        )
    if (
        not isinstance(root_cell_mask, torch.Tensor)
        or root_cell_mask.shape != (cells,)
        or root_cell_mask.dtype is not torch.bool
        or root_cell_mask.device.type != "cpu"
        or root_cell_mask.layout is not torch.strided
    ):
        raise ValueError("root_cell_mask must be an aligned dense CPU boolean field")
    if not bool(root_cell_mask.any()) or bool(root_cell_mask.all()):
        raise ValueError("root_cell_mask needs a nonempty root and complement")
    if type(material) is not MaterialSpec or type(scales) is not NondimensionalScales:
        raise TypeError("material and scales must be exact S13 contract types")
    if (u_model is None) != (u_fem is None):
        raise ValueError("u_model and u_fem must be supplied together")
    if u_model is not None and u_fem is not None:
        _field(u_model, (cells, points, 2))
        _field(u_fem, (cells, points, 2))
    identity = torch.eye(2, dtype=torch.float64)
    model_det, fem_det = (
        torch.linalg.det(identity + grad_model),
        torch.linalg.det(identity + grad_fem),
    )
    if not bool(torch.isfinite(model_det).all() & torch.isfinite(fem_det).all()):
        raise ValueError("deformation determinants must be finite")
    if not bool((fem_det > DETERMINANT_MARGIN).all()):
        raise ValueError("FEM deformation must be admissible at every point")
    with torch.no_grad():
        fem_stress = _checked_material_stress(grad_fem, material, scales, "FEM")
        admissible = model_det > DETERMINANT_MARGIN
        model_stress = torch.zeros_like(grad_model)
        if bool(admissible.any()):
            indices = admissible.flatten().nonzero().flatten()
            stress = _checked_material_stress(
                grad_model[admissible], material, scales, "model"
            )
            model_stress = (
                model_stress.reshape(-1, 2, 2)
                .index_copy(0, indices, stress)
                .reshape(cells, points, 2, 2)
            )
        populations = tuple(
            _population(
                mask=mask,
                grad_model=grad_model,
                grad_fem=grad_fem,
                served_p=served_p,
                raw_fem_p=raw_fem_p,
                area_weights=area_weights,
                xy_mm=xy_mm,
                model_det=model_det,
                fem_det=fem_det,
                model_stress=model_stress,
                fem_stress=fem_stress,
                u_model=u_model,
                u_fem=u_fem,
                stress_scale=scales.stress_mpa,
            )
            for mask in (
                torch.ones(cells, dtype=torch.bool),
                root_cell_mask,
                ~root_cell_mask,
            )
        )
    return QuadratureCaseMetrics(*populations)


__all__ = (
    "ConstitutiveDecomposition",
    "InvalidQuadraturePoint",
    "QuadratureCaseMetrics",
    "QuadraturePopulationMetrics",
    "summarize_quadrature_case",
)
