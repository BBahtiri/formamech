"""Lifting evidence for a ProblemSpec over its parameter box (ADR 0106 d. 4).

For every feasible parameter vector, on the morphed mesh:

- ``node_residual``: at the nodes of every Dirichlet region, the lifted
  displacement for a random correction meets the FEM data (nominal strain
  times the region's largest coordinate, or the expression), relative to the
  displacement scale;
- for ``adf`` also ``conformity`` (largest distance of a region's nodes to
  the condition's pieces, relative to the frame's larger extent),
  ``orientation`` (largest depth of a mesh node on the wrong side of a piece,
  same unit) and ``boundary_residual`` along every piece at each refinement;
- ``mutant_miss``: the smallest residual of the mutants, each lifting with one
  piece dropped and the identity lifting (``offset = 0``, ``scale = d``),
  which must clearly miss.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from piginot.contracts import CanonicalMesh, hash_spec
from piginot.contracts.problem_spec import (
    COMPONENTS,
    ProblemSpec,
    region_nodes,
)
from piginot.lifting import (
    AdfLifting,
    AffineDirichletLifting,
    DirichletLifting,
    DirichletPiece,
    boundary_residual,
)
from piginot.operators.architecture import LiftingVerification
from piginot.problem.lifting import build_lifting, dirichlet_piece

EXACT_TOLERANCE = 1e-12
CONFORMITY_TOLERANCE = 1e-9
MUTANT_MINIMUM_MISS = 1e-3
REFINEMENTS = (320, 4800)


def box_points(
    spec: ProblemSpec, *, interior: int = 64, seed: int = 0
) -> list[dict[str, float]]:
    """Every box corner and the centre that meet the spec's constraints, then
    ``interior`` seeded uniform draws that meet them."""

    def feasible(values: Mapping[str, float]) -> bool:
        return all(
            c.lo <= c.expression.evaluate(spec.scope(values)) <= c.hi
            for c in spec.constraints
        )

    ranges = [(p.name, p.lo, p.hi) for p in spec.parameters]
    varying = [(name, lo, hi) for name, lo, hi in ranges if lo < hi]
    fixed = {name: lo for name, lo, hi in ranges if lo == hi}
    points: list[dict[str, float]] = []
    for mask in range(2 ** len(varying)):
        corner = dict(fixed)
        for bit, (name, lo, hi) in enumerate(varying):
            corner[name] = hi if mask >> bit & 1 else lo
        if feasible(corner):
            points.append(corner)
    centre = dict(fixed)
    centre.update({name: 0.5 * (lo + hi) for name, lo, hi in varying})
    if feasible(centre):
        points.append(centre)
    generator = random.Random(seed)
    draws = 0
    while draws < interior:
        draw = dict(fixed)
        draw.update({name: generator.uniform(lo, hi) for name, lo, hi in varying})
        if feasible(draw):
            points.append(draw)
            draws += 1
    return points


@dataclass(frozen=True, slots=True)
class LiftingEvidence:
    """The measured lifting evidence of one spec; see the module docstring."""

    problem_hash: str
    kind: str
    parameter_vectors: int
    refused_geometries: int
    refinements: tuple[int, ...]
    node_residual: float
    conformity: float | None
    orientation: float | None
    boundary_residual: Mapping[int, float]
    mutant_miss: float

    @property
    def passed(self) -> bool:
        exact = [self.node_residual, *self.boundary_residual.values()]
        return (
            self.parameter_vectors > 0
            and all(value <= EXACT_TOLERANCE for value in exact)
            and (self.conformity or 0.0) <= CONFORMITY_TOLERANCE
            and (self.orientation or 0.0) <= CONFORMITY_TOLERANCE
            and self.mutant_miss > MUTANT_MINIMUM_MISS
        )

    def document(self) -> dict[str, Any]:
        return {
            "problem_hash": self.problem_hash,
            "kind": self.kind,
            "parameter_vectors": self.parameter_vectors,
            "refused_geometries": self.refused_geometries,
            "refinements": list(self.refinements),
            "node_residual": self.node_residual,
            "conformity": self.conformity,
            "orientation": self.orientation,
            "boundary_residual": {
                str(level): value for level, value in self.boundary_residual.items()
            },
            "mutant_miss": self.mutant_miss,
            "thresholds": {
                "exact": EXACT_TOLERANCE,
                "conformity": CONFORMITY_TOLERANCE,
                "mutant_minimum_miss": MUTANT_MINIMUM_MISS,
            },
            "passed": self.passed,
        }

    def verification(self, spec: ProblemSpec) -> LiftingVerification:
        """The ADR 0027 record; refused unless the evidence passed."""

        if not self.passed:
            raise ValueError("the lifting evidence did not pass")
        ranges = {
            "parameters": [p.document() for p in spec.parameters],
            "constraints": [c.document(canonical=True) for c in spec.constraints],
        }
        return LiftingVerification(
            verification_id=f"{spec.name}:{self.kind}-lifting",
            parameter_range_hash=hash_spec(ranges, "content"),
            verification_hash=hash_spec(self.document(), "content"),
            boundary_refinement_levels=self.refinements,
            full_parameter_range_verified=True,
            max_boundary_residual=max(
                [self.node_residual, *self.boundary_residual.values()]
            ),
        )


def _segment_distance(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    fraction = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (
        dx * dx + dy * dy
    )
    fraction = min(1.0, max(0.0, fraction))
    return math.hypot(
        point[0] - start[0] - fraction * dx, point[1] - start[1] - fraction * dy
    )


def _arc_distance(
    point: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
    center: Sequence[float],
) -> float:
    def cross(a: Sequence[float], b: Sequence[float]) -> float:
        return a[0] * b[1] - a[1] * b[0]

    first = (start[0] - center[0], start[1] - center[1])
    last = (end[0] - center[0], end[1] - center[1])
    direction = (point[0] - center[0], point[1] - center[1])
    turn = math.copysign(1.0, cross(first, last))
    if turn * cross(first, direction) >= 0 and turn * cross(direction, last) >= 0:
        return abs(math.hypot(*direction) - math.hypot(*first))
    return min(
        math.hypot(point[0] - start[0], point[1] - start[1]),
        math.hypot(point[0] - end[0], point[1] - end[1]),
    )


def _piece_distance(point: Sequence[float], piece: DirichletPiece) -> float:
    start: list[float] = piece.start[0].tolist()
    end: list[float] = piece.end[0].tolist()
    if piece.center is not None:
        return _arc_distance(point, start, end, piece.center[0].tolist())
    if piece.trimmed:
        return _segment_distance(point, start, end)
    dx, dy = end[0] - start[0], end[1] - start[1]
    return abs(dx * (point[1] - start[1]) - dy * (point[0] - start[0])) / math.hypot(
        dx, dy
    )


def _data(
    spec: ProblemSpec,
    mesh: CanonicalMesh,
    values: Mapping[str, float],
) -> list[tuple[int, list[int], float]]:
    """``(component, nodes, prescribed value)`` per Dirichlet condition
    component, by the FEM rule."""

    result = []
    for condition in spec.boundary_conditions:
        if not condition.is_dirichlet:
            continue
        nodes = sorted(region_nodes(mesh, condition.region))
        for component in condition.components:
            axis = COMPONENTS.index(component)
            value = condition.value(component)
            if value is None:
                amount = 0.0
            elif value.nominal_strain is not None:
                amount = values[value.nominal_strain] * max(
                    float(mesh.nodes[i][axis]) for i in nodes
                )
            else:
                assert value.expression is not None
                amount = value.expression.evaluate(values)
            result.append((axis, nodes, amount))
    return result


def _node_residual(
    lifting: DirichletLifting,
    mesh: CanonicalMesh,
    data: Sequence[tuple[int, list[int], float]],
    scale: float,
    generator: torch.Generator,
) -> float:
    worst = 0.0
    for axis, nodes, amount in data:
        points = torch.tensor([[mesh.nodes[i] for i in nodes]], dtype=torch.float64)
        lifted = lifting(points)
        phi = torch.randn(lifted.offset.shape, generator=generator, dtype=torch.float64)
        u = lifted.offset + lifted.scale * phi
        error = float((u[0, :, axis] - amount).abs().max().item()) / abs(scale)
        worst = max(worst, error if math.isfinite(error) else math.inf)
    return worst


def _identity(displacement: float) -> DirichletLifting:
    def lifting(query_xy: torch.Tensor) -> AffineDirichletLifting:
        return AffineDirichletLifting(
            torch.zeros_like(query_xy), torch.full_like(query_xy, displacement)
        )

    return lifting


def verify_lifting(
    spec: ProblemSpec,
    mesh_for: Callable[[Mapping[str, float]], CanonicalMesh | None],
    points: Sequence[Mapping[str, float]],
    *,
    refinements: tuple[int, ...] = REFINEMENTS,
    seed: int = 0,
) -> LiftingEvidence:
    """Measure the evidence on the meshes ``mesh_for`` builds (``None`` when
    the geometry is refused)."""

    generator = torch.Generator().manual_seed(seed)
    node = conformity = orientation = 0.0
    miss = math.inf
    adf = spec.lifting.kind == "adf"
    boundary = dict.fromkeys(refinements, 0.0) if adf else {}
    measured = refused = 0
    for point in points:
        values = spec.scope(point)
        mesh = mesh_for(values)
        if mesh is None:
            refused += 1
            continue
        measured += 1
        lifting = build_lifting(spec, [values])
        displacement = spec.lifting.displacement_scale.evaluate(values)
        data = _data(spec, mesh, values)
        node = max(node, _node_residual(lifting, mesh, data, displacement, generator))
        mutants: list[DirichletLifting] = [_identity(displacement)]
        if isinstance(lifting, AdfLifting):
            conformity = max(conformity, _conformity(spec, mesh, values, lifting))
            orientation = max(orientation, _orientation(mesh, lifting))
            for level in refinements:
                boundary[level] = max(
                    boundary[level],
                    boundary_residual(lifting, level, generator=generator),
                )
            for condition in spec.boundary_conditions:
                if not condition.is_dirichlet:
                    continue
                for index in range(len(spec.lifting.pieces[condition.name])):
                    try:
                        mutants.append(
                            build_lifting(spec, [values], drop=(condition.name, index))
                        )
                    except ValueError:
                        continue
        for mutant in mutants:
            miss = min(
                miss, _node_residual(mutant, mesh, data, displacement, generator)
            )
    return LiftingEvidence(
        problem_hash=spec.problem_hash,
        kind=spec.lifting.kind,
        parameter_vectors=measured,
        refused_geometries=refused,
        refinements=refinements,
        node_residual=node,
        conformity=conformity if adf else None,
        orientation=orientation if adf else None,
        boundary_residual=boundary,
        mutant_miss=miss if measured else 0.0,
    )


def _conformity(
    spec: ProblemSpec,
    mesh: CanonicalMesh,
    values: Mapping[str, float],
    lifting: AdfLifting,
) -> float:
    length = float(lifting.length[0].item())
    worst = 0.0
    for condition in spec.boundary_conditions:
        if not condition.is_dirichlet:
            continue
        pieces = [
            dirichlet_piece(
                piece, condition, condition.components[0], [values], torch.float64
            )
            for piece in spec.lifting.pieces[condition.name]
        ]
        for index in region_nodes(mesh, condition.region):
            point = mesh.nodes[index]
            distance = min(_piece_distance(point, piece) for piece in pieces)
            worst = max(worst, distance / length)
    return worst


def _orientation(mesh: CanonicalMesh, lifting: AdfLifting) -> float:
    nodes = torch.tensor([mesh.nodes], dtype=torch.float64)
    length = lifting.length
    query = (nodes - lifting.origin.unsqueeze(1)) / length[:, None, None]
    worst = 0.0
    for component in lifting.pieces:
        for piece in component:
            signed, trim = piece.signed_and_trim(query, lifting.origin, length)
            relevant = signed if trim is None else signed[trim > 0]
            if relevant.numel():
                worst = max(worst, float((-relevant).clamp(min=0.0).max().item()))
    return worst


__all__ = (
    "CONFORMITY_TOLERANCE",
    "EXACT_TOLERANCE",
    "MUTANT_MINIMUM_MISS",
    "REFINEMENTS",
    "LiftingEvidence",
    "box_points",
    "verify_lifting",
)
