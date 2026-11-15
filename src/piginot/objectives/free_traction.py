"""Reference-surface quadrature and dimensionless free-traction consistency."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from piginot.contracts import CanonicalMesh

FREE_TRACTION_POLICY_ID = "free-traction-reference-facets-v1"


@dataclass(frozen=True, slots=True)
class BoundaryQuadrature:
    points: torch.Tensor
    normals: torch.Tensor
    weights: torch.Tensor
    region_indices: torch.Tensor
    regions: tuple[str, ...]


def _facet_samples(
    mesh: CanonicalMesh, region: str
) -> tuple[list[object], list[object], list[float]]:
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    owners: dict[tuple[int, ...], list[int]] = {}
    for index, cell in enumerate(mesh.cells):
        if len(cell) != 3:
            raise ValueError("boundary integration requires triangle cells")
        for side in range(3):
            edge = tuple(sorted((cell[side], cell[(side + 1) % 3])))
            owners.setdefault(edge, []).append(index)
    points, normals, weights = [], [], []
    fractions = (0.5 - math.sqrt(3) / 6, 0.5 + math.sqrt(3) / 6)
    for index in mesh.semantic_tags[region]:
        edge = mesh.boundary_facets[index]
        incident = owners.get(tuple(sorted(edge)), [])
        a, b = nodes[list(edge)]
        tangent = b - a
        length = float(np.linalg.norm(tangent))
        if len(incident) != 1 or length <= 0:
            raise ValueError("free facets must be non-degenerate exterior edges")
        normal = np.array((tangent[1], -tangent[0])) / length
        center = nodes[list(mesh.cells[incident[0]])].mean(axis=0)
        if float(normal @ (center - (a + b) / 2)) > 0:
            normal = -normal
        for fraction in fractions:
            points.append((a + fraction * tangent).tolist())
            normals.append(normal.tolist())
            weights.append(length / 2)
    return points, normals, weights


def boundary_quadrature(
    mesh: CanonicalMesh,
    regions: tuple[str, ...],
    *,
    dtype: torch.dtype = torch.float64,
) -> BoundaryQuadrature:
    """Use actual polygon normals and two interior Gauss points per facet."""
    if (
        not regions
        or len(set(regions)) != len(regions)
        or any(not mesh.semantic_tags.get(region) for region in regions)
        or dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("distinct nonempty mesh regions and floating dtype required")
    points, normals, weights, indices = [], [], [], []
    for index, region in enumerate(regions):
        p, n, w = _facet_samples(mesh, region)
        points.extend(p)
        normals.extend(n)
        weights.extend(w)
        indices.extend([index] * len(w))
    return BoundaryQuadrature(
        torch.tensor(points, dtype=dtype),
        torch.tensor(normals, dtype=dtype),
        torch.tensor(weights, dtype=dtype),
        torch.tensor(indices, dtype=torch.int64),
        regions,
    )


def _validate(values: torch.Tensor, rule: BoundaryQuadrature) -> None:
    if (
        values.ndim < 2
        or values.shape[0] != len(rule.weights)
        or values.dtype != rule.points.dtype
        or values.device != rule.points.device
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("boundary values must align with finite quadrature values")


def regional_rms(values: torch.Tensor, rule: BoundaryQuadrature) -> torch.Tensor:
    """Length-weighted RMS of a vector or tensor, separately for every region."""
    _validate(values, rule)
    squared = values.reshape(len(values), -1).square().sum(dim=1)
    total = values.new_zeros(len(rule.regions))
    lengths = total.scatter_add(0, rule.region_indices, rule.weights)
    integrals = total.scatter_add(0, rule.region_indices, squared * rule.weights)
    return torch.sqrt(integrals / lengths)


def boundary_stress_scales(
    stress: torch.Tensor, rule: BoundaryQuadrature, *, minimum: float
) -> torch.Tensor:
    """Freeze FIT-only stress scales; the positive floor handles quiet loads."""
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("the boundary stress floor must be finite and positive")
    return regional_rms(stress, rule).detach().clamp_min(minimum)


def free_traction_loss(
    stress: torch.Tensor, rule: BoundaryQuadrature, scales: torch.Tensor
) -> torch.Tensor:
    """Mean normalized ``||P N||²`` over declared free reference surfaces."""
    _validate(stress, rule)
    if stress.shape != (len(rule.weights), 2, 2):
        raise ValueError("stress must have shape (boundary points, 2, 2)")
    weights = traction_point_weights(rule, scales)
    traction = (stress @ rule.normals.unsqueeze(-1)).squeeze(-1)
    return (traction.square().sum(dim=1) * weights).sum()


def traction_point_weights(
    rule: BoundaryQuadrature, scales: torch.Tensor
) -> torch.Tensor:
    """Global surface weights retained unchanged when a gradient is chunked."""
    if (
        scales.shape != (len(rule.regions),)
        or scales.dtype != rule.points.dtype
        or scales.device != rule.points.device
        or scales.requires_grad
        or not bool(torch.isfinite(scales).all())
        or bool((scales <= 0).any())
    ):
        raise ValueError("stress and fixed positive regional scales must align")
    total = rule.points.new_zeros(len(rule.regions))
    lengths = total.scatter_add(0, rule.region_indices, rule.weights)
    return rule.weights / (
        lengths[rule.region_indices]
        * scales[rule.region_indices].square()
        * len(rule.regions)
    )
