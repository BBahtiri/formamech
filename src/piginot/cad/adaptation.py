"""Bounded CAD preparation driven by a declared TRAIN-pilot jump heuristic.

The mesher callback must reimport the original CAD using the supplied policy;
the evaluator owns the physical FEM request and returns cell-average in-plane
first-Piola stress. No fitting, acceptance-screen cases or error certificate is
part of this coordinator. All callback failures are terminal and retained.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from piginot.cad.mesh_policy import MeshPolicy, PointSize
from piginot.contracts import CanonicalMesh, canonical_json_bytes

_Stress = tuple[tuple[float, float], tuple[float, float]]
_Edge = tuple[int, int]
_INDICATOR = "relative-cell-average-interior-traction-jump-v1"


def _number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _sha(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _stress_values(values: object) -> tuple[_Stress, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError("cell stress must contain one 2x2 tensor per cell")
    result: list[_Stress] = []
    for cell in values:
        if not isinstance(cell, (tuple, list)) or len(cell) != 2:
            raise ValueError("cell stress must contain 2x2 tensors")
        if any(not isinstance(row, (tuple, list)) or len(row) != 2 for row in cell):
            raise ValueError("cell stress must contain 2x2 tensors")
        result.append(
            (
                (_number(cell[0][0], "stress"), _number(cell[0][1], "stress")),
                (_number(cell[1][0], "stress"), _number(cell[1][1], "stress")),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PilotCase:
    case_id: str
    partition: str
    parameters: Mapping[str, float]

    def __post_init__(self) -> None:
        _identifier(self.case_id, "pilot case_id")
        if self.partition != "TRAIN-pilot":
            raise ValueError("adaptation cases must explicitly use TRAIN-pilot")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("pilot parameters must be a data mapping")
        values = {}
        for name, value in self.parameters.items():
            _identifier(name, "pilot parameter name")
            values[name] = _number(value, "pilot parameter")
        object.__setattr__(self, "parameters", MappingProxyType(values))

    def document(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "partition": self.partition,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class PilotStress:
    case_id: str
    mesh_id: str
    cell_stress: tuple[_Stress, ...]

    def __post_init__(self) -> None:
        _identifier(self.case_id, "pilot case_id")
        _identifier(self.mesh_id, "pilot mesh_id")
        object.__setattr__(self, "cell_stress", _stress_values(self.cell_stress))

    def document(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mesh_id": self.mesh_id,
            "cell_average_inplane_first_piola": self.cell_stress,
        }

    @property
    def sha256(self) -> str:
        return _sha(self.document())


def _validated_mesh(mesh: CanonicalMesh) -> CanonicalMesh:
    from piginot.cad.refinement import refine_fixed_mesh

    validated = refine_fixed_mesh(mesh, 1)
    if canonical_json_bytes(validated) != canonical_json_bytes(mesh):
        raise ValueError("mesh metadata differs from its canonical content")
    return validated


def _edges(mesh: CanonicalMesh) -> dict[_Edge, tuple[int, ...]]:
    owners: dict[_Edge, tuple[int, ...]] = {}
    for index, cell in enumerate(mesh.cells):
        for a, b in zip(cell, (*cell[1:], cell[0]), strict=True):
            edge = (min(a, b), max(a, b))
            owners[edge] = (*owners.get(edge, ()), index)
    return owners


def _areas(mesh: CanonicalMesh) -> tuple[float, ...]:
    return tuple(
        abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
        for cell in mesh.cells
        for a, b, c in [tuple(mesh.nodes[index] for index in cell)]
    )


def stress_jump_indicator(
    mesh: CanonicalMesh, cell_stress: tuple[_Stress, ...]
) -> tuple[float, ...]:
    """Relative interior traction jumps; excludes boundary residuals and body force.

    Each cell receives half of L**2 * ||(P_left-P_right)n||**2 per
    interior edge. Divide by cell area and by the area-weighted global
    mean squared stress, then take the square root. This dimensionless
    marking heuristic is neither an equilibrium residual nor an error bound.
    """
    mesh = _validated_mesh(mesh)
    values = _stress_values(cell_stress)
    if len(values) != len(mesh.cells):
        raise ValueError("cell stress count differs from the mesh cell count")
    scale = max(abs(value) for cell in values for row in cell for value in row)
    if scale == 0:
        return (0.0,) * len(values)
    values = tuple(
        (
            (cell[0][0] / scale, cell[0][1] / scale),
            (cell[1][0] / scale, cell[1][1] / scale),
        )
        for cell in values
    )
    areas = _areas(mesh)
    mean_square = sum(
        area * sum(value * value for row in stress for value in row)
        for area, stress in zip(areas, values, strict=True)
    ) / sum(areas)
    scores = [0.0] * len(mesh.cells)
    for (a, b), owners in _edges(mesh).items():
        if len(owners) != 2:
            continue
        left, right = owners
        dx = mesh.nodes[b][0] - mesh.nodes[a][0]
        dy = mesh.nodes[b][1] - mesh.nodes[a][1]
        # L*n = (-dy, dx), so L**2 cancels the normalization of n.
        jump = sum(
            (
                -(values[left][row][0] - values[right][row][0]) * dy
                + (values[left][row][1] - values[right][row][1]) * dx
            )
            ** 2
            for row in (0, 1)
        )
        scores[left] += jump / 2
        scores[right] += jump / 2
    indicators = tuple(
        math.sqrt(score / (area * mean_square))
        for score, area in zip(scores, areas, strict=True)
    )
    if any(not math.isfinite(value) for value in indicators):
        raise ValueError("stress jump indicator is nonfinite")
    return indicators


@dataclass(frozen=True, slots=True)
class AdaptationAttempt:
    round_index: int
    policy: MeshPolicy
    mesh: CanonicalMesh | None
    pilots: tuple[PilotStress, ...]
    indicators: tuple[float, ...]
    status: str
    reason: str

    def document(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "policy": self.policy.document(),
            "policy_sha256": self.policy.sha256,
            "mesh_id": self.mesh.mesh_id if self.mesh is not None else None,
            "mesh_sha256": _sha(self.mesh) if self.mesh is not None else None,
            "source_sha256": self.mesh.source_provenance.get("source_sha256")
            if self.mesh is not None
            else None,
            "pilots": [
                {"sha256": pilot.sha256, "result": pilot.document()}
                for pilot in self.pilots
            ],
            "cell_indicators": self.indicators,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class AdaptationReport:
    status: str
    pilot_cases: tuple[PilotCase, ...]
    screen_case_ids: tuple[str, ...]
    attempts: tuple[AdaptationAttempt, ...]
    solves_used: int
    settings: Mapping[str, float | int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))

    def document(self) -> dict[str, Any]:
        return {
            "schema": "piginot.cad-adaptation-v1",
            "status": self.status,
            "indicator": _INDICATOR,
            "error_bound": "NOT_ESTABLISHED",
            "pilot_cases": [case.document() for case in self.pilot_cases],
            "screen_case_ids": self.screen_case_ids,
            "settings": dict(self.settings),
            "solves_used": self.solves_used,
            "attempts": [attempt.document() for attempt in self.attempts],
        }

    @property
    def sha256(self) -> str:
        return _sha(self.document())


def _populations(pilots: tuple[PilotCase, ...], screens: tuple[str, ...]) -> None:
    if (
        not isinstance(pilots, tuple)
        or not pilots
        or any(not isinstance(case, PilotCase) for case in pilots)
        or len({case.case_id for case in pilots}) != len(pilots)
    ):
        raise ValueError("declare a nonempty tuple of unique TRAIN-pilot cases")
    if not isinstance(screens, tuple) or len(set(screens)) != len(screens):
        raise ValueError("screen case IDs must be a tuple without duplicates")
    for case_id in screens:
        _identifier(case_id, "screen case_id")
    if {case.case_id for case in pilots}.intersection(screens):
        raise ValueError("pilot and final-screen case IDs overlap")


def _limits(mesh: CanonicalMesh, policy: MeshPolicy) -> None:
    vertices, edges, cells = len(mesh.nodes), len(_edges(mesh)), len(mesh.cells)
    for _ in range(2):
        vertices, edges, cells = vertices + edges, 2 * edges + 3 * cells, 4 * cells
    if (
        cells > policy.maximum_triangles
        or 2 * (vertices + edges) > policy.maximum_p2_dofs
    ):
        raise ValueError("largest reference mesh exceeds triangle or P2 DOF limit")


def _remeshed(
    mesh: CanonicalMesh, policy: MeshPolicy, previous: tuple[AdaptationAttempt, ...]
) -> CanonicalMesh:
    mesh = _validated_mesh(mesh)
    source = mesh.source_provenance.get("source_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", source):
        raise ValueError("remesh source_sha256 is not bound")
    if mesh.source_provenance.get("mesh_policy_sha256") != policy.sha256:
        raise ValueError("remesh policy binding differs")
    if previous:
        base = previous[0].mesh
        assert base is not None
        if (
            source != base.source_provenance["source_sha256"]
            or mesh.topology_signature != base.topology_signature
            or set(mesh.semantic_tags) != set(base.semantic_tags)
        ):
            raise ValueError("remesh source, topology or boundary vocabulary differs")
    return mesh


def _marked_policy(
    policy: MeshPolicy,
    mesh: CanonicalMesh,
    indicators: tuple[float, ...],
    settings: Mapping[str, float | int],
) -> tuple[MeshPolicy | None, str]:
    remaining = int(settings["max_markers"]) - len(policy.point_sizes)
    count = max(1, math.ceil(float(settings["mark_fraction"]) * len(mesh.cells)))
    chosen = sorted(range(len(mesh.cells)), key=lambda i: (-indicators[i], i))[:count]
    fields = policy.point_sizes
    marker_limit = False
    for index in chosen:
        if indicators[index] <= float(settings["indicator_target"]):
            continue
        nodes = tuple(mesh.nodes[node] for node in mesh.cells[index])
        size = max(
            math.dist(a, b) for a, b in zip(nodes, (*nodes[1:], nodes[0]), strict=True)
        )
        center = (sum(p[0] for p in nodes) / 3, sum(p[1] for p in nodes) / 3)
        old = next(
            (field for field in policy.point_sizes if field.center_mm == center), None
        )
        current_size = min(policy.global_size_mm, size, old.size_mm if old else size)
        target = max(
            policy.minimum_size_mm, current_size / float(settings["refinement_factor"])
        )
        if target >= current_size:
            continue
        if old is not None:
            fields = tuple(
                replace(field, size_mm=target) if field is old else field
                for field in fields
            )
            continue
        if remaining <= 0:
            marker_limit = True
            continue
        fields += (PointSize(center, target, size / 2, 2 * size),)
        remaining -= 1
    if fields == policy.point_sizes:
        return None, "MARKER_LIMIT" if marker_limit else "MINIMUM_SIZE_REACHED"
    return replace(policy, point_sizes=fields), "REFINED"


def adapt_mesh(
    policy: MeshPolicy,
    pilot_cases: tuple[PilotCase, ...],
    screen_case_ids: tuple[str, ...],
    *,
    remesh: Callable[[MeshPolicy], CanonicalMesh],
    evaluate: Callable[[CanonicalMesh, PilotCase], PilotStress],
    max_rounds: int,
    max_solves: int,
    indicator_target: float,
    mark_fraction: float,
    refinement_factor: float,
    max_markers: int,
) -> AdaptationReport:
    """Prepare meshes with finite pilots; preserve failures and never auto-retry.

    max_rounds counts all mesher invocations. max_solves counts attempted
    evaluator calls, including failures. Mesh limits apply to the prepared
    reference ladder's largest level (4), before any evaluator is called.
    A round starts only if every pilot fits the remaining solve allowance.
    """
    if (
        not isinstance(policy, MeshPolicy)
        or not callable(remesh)
        or not callable(evaluate)
    ):
        raise ValueError("declare a MeshPolicy and callable mesher/evaluator")
    _populations(pilot_cases, screen_case_ids)
    for name, value in (
        ("max_rounds", max_rounds),
        ("max_solves", max_solves),
        ("max_markers", max_markers),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if len(policy.point_sizes) > max_markers:
        raise ValueError("initial point fields exceed max_markers")
    target = _number(indicator_target, "indicator_target")
    fraction = _number(mark_fraction, "mark_fraction")
    factor = _number(refinement_factor, "refinement_factor")
    if target < 0 or not 0 < fraction <= 1 or factor <= 1:
        raise ValueError(
            "require target >= 0, 0 < mark_fraction <= 1 and refinement_factor > 1"
        )
    settings = {
        "max_rounds": max_rounds,
        "max_solves": max_solves,
        "max_markers": max_markers,
        "indicator_target": target,
        "mark_fraction": fraction,
        "refinement_factor": factor,
    }
    attempts: tuple[AdaptationAttempt, ...] = ()
    solves = 0

    def report(status: str) -> AdaptationReport:
        return AdaptationReport(
            status, pilot_cases, screen_case_ids, attempts, solves, settings
        )

    for round_index in range(1, max_rounds + 1):
        if max_solves - solves < len(pilot_cases):
            return report("SOLVE_LIMIT")
        attempt = AdaptationAttempt(round_index, policy, None, (), (), "STARTED", "")
        try:
            mesh = remesh(policy)
            if not isinstance(mesh, CanonicalMesh):
                raise ValueError("mesher must return a CanonicalMesh")
            attempt = replace(attempt, mesh=mesh)
            mesh = _remeshed(mesh, policy, attempts)
        except Exception as error:
            attempts += (replace(attempt, status="REMESH_FAILURE", reason=str(error)),)
            return report("REMESH_FAILURE")
        try:
            _limits(mesh, policy)
        except ValueError as error:
            attempts += (replace(attempt, status="MESH_LIMIT", reason=str(error)),)
            return report("MESH_LIMIT")
        if any(
            old.mesh is not None and old.mesh.mesh_id == mesh.mesh_id
            for old in attempts
        ):
            attempts += (
                replace(
                    attempt, status="STALLED", reason="mesher repeated an existing mesh"
                ),
            )
            return report("STALLED")
        indicators = (0.0,) * len(mesh.cells)
        for case in pilot_cases:
            solves += 1
            try:
                pilot = evaluate(mesh, case)
                if (
                    not isinstance(pilot, PilotStress)
                    or pilot.case_id != case.case_id
                    or pilot.mesh_id != mesh.mesh_id
                ):
                    raise ValueError(
                        "pilot result case or mesh differs from requested pilot"
                    )
                values = stress_jump_indicator(mesh, pilot.cell_stress)
                indicators = tuple(
                    max(a, b) for a, b in zip(indicators, values, strict=True)
                )
                attempt = replace(
                    attempt, pilots=(*attempt.pilots, pilot), indicators=indicators
                )
            except Exception as error:
                attempts += (
                    replace(
                        attempt,
                        status="PILOT_FAILURE",
                        reason=f"{case.case_id}: {error}",
                    ),
                )
                return report("PILOT_FAILURE")
        attempts += (replace(attempt, status="EVALUATED"),)
        if max(indicators) <= target:
            return report("HEURISTIC_TARGET_MET")
        if round_index == max_rounds:
            return report("ROUND_LIMIT")
        following, status = _marked_policy(policy, mesh, indicators, settings)
        if following is None:
            return report(status)
        policy = following
    raise AssertionError("validated positive round budget did not execute")
