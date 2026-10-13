"""Per-query confidence from the equilibrium residual (``G14.confidence``).

The served stress of the energy-form operator is constitutive, so its
in-plane constitutive and angular-momentum residuals vanish identically. The
remaining indicator is the strong-form equilibrium residual
``r = |Div P| * length_scale / E`` evaluated by automatic differentiation of
the predicted displacement, together with the natural-boundary residual
``t = |(P N)_free| / E``: the traction components that no boundary condition
prescribes, which must vanish (both components on a traction-free edge, the
tangential one on a symmetry or single-component Dirichlet edge). Three
preregistered variants combine them at the vertices:

- ``pointwise``: ``r(v) + t(v)``;
- ``patch-mean``: the mean of ``r`` over the incident cells' quadrature
  points, plus ``t(v)``;
- ``patch-mean-times-h``: that mean times ``h(v) / length_scale``, plus
  ``t(v)`` (the pointwise analogue of a residual error estimator).

``t`` is zero at interior vertices. Spearman's rank correlation and an
isotonic calibration of an error quantile complete the module.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from piginot.contracts import MaterialSpec
from piginot.operators.energy_operator import (
    DirichletLifting,
    EnergyFormOperator,
    QueryFrame,
    constitutive_in_plane_stress,
)

INDICATOR_VARIANTS = ("pointwise", "patch-mean", "patch-mean-times-h")
_CHUNK = 1_024
# Symmetric three-point rule with degree-2 exactness (barycentric weights).
_QUADRATURE = torch.tensor(
    [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]],
    dtype=torch.float64,
)


def _elastic_modulus(material: MaterialSpec) -> float:
    if "elastic_modulus" in material.properties:
        value = float(material.properties["elastic_modulus"].value)
    else:
        from piginot.physics import reference_modulus

        value = reference_modulus(material)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("elastic modulus must be finite and positive")
    return value


def equilibrium_residual(
    model: EnergyFormOperator,
    query_xy: torch.Tensor,
    *,
    parameters: torch.Tensor,
    frame: QueryFrame,
    lifting: DirichletLifting,
    material: MaterialSpec,
    length_scale: float,
) -> torch.Tensor:
    """Nondimensional ``|Div P|`` at ``query_xy`` with shape ``(points,)``."""

    if query_xy.ndim != 2 or query_xy.shape[-1] != 2:
        raise ValueError("query_xy must have shape (points, 2)")
    if not math.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_scale must be finite and positive")
    scale = length_scale / _elastic_modulus(material)
    parts: list[torch.Tensor] = []
    for start in range(0, query_xy.shape[0], _CHUNK):
        query = query_xy[start : start + _CHUNK].detach().clone().unsqueeze(0)
        query.requires_grad_(True)
        _, gradient = model.displacement_and_gradient(
            query_xy=query,
            parameters=parameters,
            frame=frame,
            lifting=lifting,
            create_graph=True,
        )
        stress = constitutive_in_plane_stress(gradient[0], material)
        divergence = []
        for row in range(2):
            total = torch.zeros(query.shape[1], dtype=query.dtype)
            for column in range(2):
                (derivative,) = torch.autograd.grad(
                    stress[:, row, column].sum(), query, retain_graph=True
                )
                total = total + derivative[0, :, column]
            divergence.append(total)
        magnitude = torch.sqrt(divergence[0] ** 2 + divergence[1] ** 2)
        parts.append(magnitude.detach() * scale)
    return torch.cat(parts)


def cell_quadrature_points(vertices: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Three interior points per cell, shape ``(cells, 3, 2)``."""

    corners = vertices[cells]
    return torch.einsum("qk,ckd->cqd", _QUADRATURE.to(vertices.dtype), corners)


def vertex_patch_mean(
    vertex_count: int, cells: torch.Tensor, cell_values: torch.Tensor
) -> torch.Tensor:
    """Mean of per-cell values over the cells incident to each vertex."""

    totals = torch.zeros(vertex_count, dtype=cell_values.dtype)
    counts = torch.zeros(vertex_count, dtype=cell_values.dtype)
    for corner in range(3):
        totals.index_add_(0, cells[:, corner], cell_values)
        counts.index_add_(0, cells[:, corner], torch.ones_like(cell_values))
    if bool((counts == 0).any()):
        raise ValueError("every vertex must belong to at least one cell")
    return totals / counts


def vertex_mesh_size(vertices: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Mean length of the cell edges incident to each vertex."""

    totals = torch.zeros(vertices.shape[0], dtype=vertices.dtype)
    counts = torch.zeros(vertices.shape[0], dtype=vertices.dtype)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        length = torch.linalg.vector_norm(
            vertices[cells[:, first]] - vertices[cells[:, second]], dim=1
        )
        for corner in (first, second):
            totals.index_add_(0, cells[:, corner], length)
            counts.index_add_(0, cells[:, corner], torch.ones_like(length))
    return totals / counts


def boundary_traction_residual(
    vertex_stress: torch.Tensor,
    vertices: torch.Tensor,
    cells: torch.Tensor,
    facets: tuple[tuple[int, int], ...],
    free_components: tuple[tuple[int, ...], ...],
    *,
    elastic_modulus: float,
) -> torch.Tensor:
    """Mean ``|(P N)_free| / E`` over the boundary facets incident to a vertex.

    ``free_components[f]`` names the traction components of facet ``f`` that
    must vanish; ``N`` is the outward unit normal of the facet. Interior
    vertices get zero.
    """

    count = vertices.shape[0]
    if vertex_stress.shape != (count, 2, 2):
        raise ValueError("vertex_stress must have shape (vertices, 2, 2)")
    if len(facets) != len(free_components):
        raise ValueError("every facet needs its free traction components")
    if not math.isfinite(elastic_modulus) or elastic_modulus <= 0.0:
        raise ValueError("elastic modulus must be finite and positive")
    opposite: dict[frozenset[int], int] = {}
    for cell in cells.tolist():
        for first, second, third in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            opposite[frozenset((cell[first], cell[second]))] = cell[third]
    totals = torch.zeros(count, dtype=vertex_stress.dtype)
    counts = torch.zeros(count, dtype=vertex_stress.dtype)
    for (first, second), components in zip(facets, free_components, strict=True):
        if any(component not in (0, 1) for component in components):
            raise ValueError("traction components must be 0 (x) or 1 (y)")
        try:
            inner = opposite[frozenset((first, second))]
        except KeyError as exc:
            raise ValueError("boundary facet is not an edge of the mesh") from exc
        edge = vertices[second] - vertices[first]
        normal = torch.stack((edge[1], -edge[0])) / torch.linalg.vector_norm(edge)
        if float(torch.dot(vertices[inner] - vertices[first], normal)) > 0.0:
            normal = -normal
        for vertex in (first, second):
            counts[vertex] += 1.0
            if components:
                traction = vertex_stress[vertex] @ normal
                totals[vertex] += (
                    torch.linalg.vector_norm(traction[list(components)])
                    / elastic_modulus
                )
    return torch.where(counts > 0, totals / counts.clamp(min=1.0), totals)


def indicator_variants(
    residual: Callable[[torch.Tensor], torch.Tensor],
    vertices: torch.Tensor,
    cells: torch.Tensor,
    *,
    length_scale: float,
    traction: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate the three preregistered variants at the mesh vertices.

    ``traction`` is the vertex boundary residual of
    :func:`boundary_traction_residual`; ``None`` evaluates the interior
    residual alone.
    """

    pointwise = residual(vertices)
    points = cell_quadrature_points(vertices, cells)
    cell_mean = residual(points.reshape(-1, 2)).reshape(points.shape[:2]).mean(dim=1)
    patch = vertex_patch_mean(vertices.shape[0], cells, cell_mean)
    size = vertex_mesh_size(vertices, cells) / length_scale
    boundary = torch.zeros_like(pointwise) if traction is None else traction
    if boundary.shape != pointwise.shape:
        raise ValueError("traction must hold one value per vertex")
    return {
        "pointwise": pointwise + boundary,
        "patch-mean": patch + boundary,
        "patch-mean-times-h": patch * size + boundary,
    }


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float64)
    start = 0
    count = values.shape[0]
    while start < count:
        end = start
        while end + 1 < count and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = 0.5 * (start + end) + 1.0
        start = end + 1
    return ranks


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman's rank correlation with average ranks for ties."""

    if x.ndim != 1 or x.shape != y.shape or x.shape[0] < 2:
        raise ValueError("x and y must be equal-length vectors of two or more")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("x and y must be finite")
    rx = _average_ranks(x.to(torch.float64))
    ry = _average_ranks(y.to(torch.float64))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = torch.sqrt((rx * rx).sum() * (ry * ry).sum())
    if float(denominator) == 0.0:
        raise ValueError("a constant vector has no rank correlation")
    return float((rx * ry).sum() / denominator)


@dataclass(frozen=True, slots=True)
class QuantileCurve:
    """Monotone step function from indicator to an error quantile."""

    quantile: float
    upper_edges: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.upper_edges) != len(self.values) or not self.values:
            raise ValueError("edges and values must be non-empty and aligned")
        if any(b < a for a, b in zip(self.values, self.values[1:], strict=False)):
            raise ValueError("calibrated values must be non-decreasing")

    def predict(self, indicator: torch.Tensor) -> torch.Tensor:
        edges = torch.tensor(self.upper_edges, dtype=torch.float64)
        values = torch.tensor(self.values, dtype=torch.float64)
        index = torch.searchsorted(edges, indicator.to(torch.float64), right=False)
        index = torch.clamp(index, max=len(self.values) - 1)
        return values[index]

    def coverage(self, indicator: torch.Tensor, error: torch.Tensor) -> float:
        bound = self.predict(indicator)
        return float((error.to(torch.float64) <= bound).to(torch.float64).mean())


def _pool_adjacent_violators(values: list[float], weights: list[float]) -> list[float]:
    blocks: list[list[float]] = []  # [value, weight, length]
    for value, weight in zip(values, weights, strict=True):
        blocks.append([value, weight, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks[-1]
            total = left[1] + right[1]
            left[0] = (left[0] * left[1] + right[0] * right[1]) / total
            left[1] = total
            left[2] += right[2]
    fitted: list[float] = []
    for value, _, length in blocks:
        fitted.extend([value] * int(length))
    return fitted


def fit_quantile_curve(
    indicator: torch.Tensor,
    error: torch.Tensor,
    *,
    quantile: float = 0.9,
    bins: int = 20,
) -> QuantileCurve:
    """Equal-count bins, per-bin error quantile, then isotonic regression."""

    if indicator.ndim != 1 or indicator.shape != error.shape:
        raise ValueError("indicator and error must be equal-length vectors")
    if not 0.0 < quantile < 1.0 or bins < 1 or indicator.shape[0] < bins:
        raise ValueError("need 0 < quantile < 1 and at least one point per bin")
    order = torch.argsort(indicator.to(torch.float64), stable=True)
    sorted_indicator = indicator.to(torch.float64)[order]
    sorted_error = error.to(torch.float64)[order]
    splits = torch.tensor_split(torch.arange(indicator.shape[0]), bins)
    edges: list[float] = []
    raw: list[float] = []
    weights: list[float] = []
    for split in splits:
        edges.append(float(sorted_indicator[split].max()))
        raw.append(float(torch.quantile(sorted_error[split], quantile)))
        weights.append(float(split.numel()))
    return QuantileCurve(
        quantile, tuple(edges), tuple(_pool_adjacent_violators(raw, weights))
    )


__all__ = (
    "INDICATOR_VARIANTS",
    "QuantileCurve",
    "boundary_traction_residual",
    "cell_quadrature_points",
    "equilibrium_residual",
    "fit_quantile_curve",
    "indicator_variants",
    "spearman",
    "vertex_mesh_size",
    "vertex_patch_mean",
)
