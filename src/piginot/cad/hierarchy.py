"""Nested CAD mesh hierarchy, cross-mesh transfer, and geometry-vs-FEM convergence.

This module separates two distinct numerical sources of error:

1. **CAD-to-polygon approximation** (``GeometryConvergenceReport``): changing the
   polygonal boundary that approximates curved CAD edges via fresh remeshes.
2. **Fixed-polygon FEM discretization** (``ReferenceConvergenceReport``): refining
   a single frozen polygonal domain through exact midpoint levels ``M0 -> M1 -> M2``.

Refining a single fixed polygon never measures convergence to curved source CAD.
All reports carry canonical SHA-256 provenance digests and never claim continuum
error bounds without separate mathematical proof.
"""

from __future__ import annotations

import enum
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from piginot.cad.refinement import refine_fixed_mesh
from piginot.contracts import CanonicalMesh, canonical_json_bytes

HIERARCHY_SCHEMA = "piginot.cad-mesh-hierarchy-v1"
GEOMETRY_CONVERGENCE_SCHEMA = "piginot.cad-geometry-convergence-v1"
REFERENCE_CONVERGENCE_SCHEMA = "piginot.cad-reference-convergence-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _check_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 64-hex SHA-256 digest")
    return value


def _check_finite(
    value: object, label: str, *, nonnegative: bool = False, positive: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{label} must be strictly positive")
    if nonnegative and number < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return number


def _mesh_cell_areas(mesh: CanonicalMesh) -> tuple[float, ...]:
    areas: list[float] = []
    for a, b, c in mesh.cells:
        ax, ay = mesh.nodes[a]
        bx, by = mesh.nodes[b]
        cx, cy = mesh.nodes[c]
        area = 0.5 * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        if area <= 0.0:
            raise ValueError("mesh contains a degenerate or inverted triangle cell")
        areas.append(area)
    return tuple(areas)


def _mesh_perimeter(mesh: CanonicalMesh) -> float:
    total = 0.0
    for a, b in mesh.boundary_facets:
        ax, ay = mesh.nodes[a]
        bx, by = mesh.nodes[b]
        total += math.hypot(bx - ax, by - ay)
    return total


def _require_nested_refinement(
    coarse: CanonicalMesh, fine: CanonicalMesh, expected_level: int, label: str
) -> None:
    expected = refine_fixed_mesh(coarse, expected_level)
    attributes = (
        "nodes",
        "cells",
        "boundary_facets",
        "semantic_tags",
        "region_ids",
        "cell_type",
    )
    if any(getattr(expected, attr) != getattr(fine, attr) for attr in attributes):
        raise ValueError(
            f"{label} must be an exact level-{expected_level} midpoint refinement"
        )


@dataclass(frozen=True, slots=True)
class MeshHierarchy:
    """Frozen three-level nested mesh ladder ``(M0, M1, M2)`` on one fixed polygon.

    * ``m0_coarse``: coarse training / base polygonal mesh (level 1).
    * ``m1_training``: adapted or refined training mesh (level 2 midpoint of M0).
    * ``m2_reference``: fine reference FEM mesh (level 4 midpoint of M0, level 2 of M1).
    """

    m0_coarse: CanonicalMesh
    m1_training: CanonicalMesh
    m2_reference: CanonicalMesh
    cad_source_hash: str
    geometry_tolerance_mm: float
    quadrature_rule: str
    refinement_history_hash: str | None = None

    def __post_init__(self) -> None:
        for label, mesh in (
            ("m0_coarse", self.m0_coarse),
            ("m1_training", self.m1_training),
            ("m2_reference", self.m2_reference),
        ):
            if not isinstance(mesh, CanonicalMesh):
                raise TypeError(f"{label} must be a CanonicalMesh")
        _check_digest(self.cad_source_hash, "cad_source_hash")
        _check_finite(
            self.geometry_tolerance_mm, "geometry_tolerance_mm", positive=True
        )
        if (
            not isinstance(self.quadrature_rule, str)
            or not self.quadrature_rule.strip()
        ):
            raise ValueError("quadrature_rule must be non-empty text")
        if self.refinement_history_hash is not None:
            _check_digest(self.refinement_history_hash, "refinement_history_hash")
        _require_nested_refinement(self.m0_coarse, self.m1_training, 2, "m1_training")
        _require_nested_refinement(self.m0_coarse, self.m2_reference, 4, "m2_reference")
        _require_nested_refinement(
            self.m1_training, self.m2_reference, 2, "m2_reference vs m1_training"
        )
        m0_tags = set(self.m0_coarse.semantic_tags)
        if (
            set(self.m1_training.semantic_tags) != m0_tags
            or set(self.m2_reference.semantic_tags) != m0_tags
        ):
            raise ValueError(
                "semantic boundary tags must remain invariant across M0-M2"
            )
        area0 = math.fsum(_mesh_cell_areas(self.m0_coarse))
        area1 = math.fsum(_mesh_cell_areas(self.m1_training))
        area2 = math.fsum(_mesh_cell_areas(self.m2_reference))
        tol = max(1e-12 * area0, 1e-15)
        if abs(area1 - area0) > tol or abs(area2 - area0) > tol:
            raise ValueError("total domain area must be conserved across M0-M2")

    @property
    def area_mm2(self) -> float:
        return math.fsum(_mesh_cell_areas(self.m0_coarse))

    def document(self) -> dict[str, Any]:
        return {
            "schema": HIERARCHY_SCHEMA,
            "cad_source_hash": self.cad_source_hash,
            "geometry_tolerance_mm": self.geometry_tolerance_mm,
            "quadrature_rule": self.quadrature_rule,
            "refinement_history_hash": self.refinement_history_hash,
            "area_mm2": self.area_mm2,
            "semantic_tags": sorted(self.m0_coarse.semantic_tags),
            "levels": {
                "M0": {
                    "mesh_id": self.m0_coarse.mesh_id,
                    "node_count": len(self.m0_coarse.nodes),
                    "cell_count": len(self.m0_coarse.cells),
                },
                "M1": {
                    "mesh_id": self.m1_training.mesh_id,
                    "node_count": len(self.m1_training.nodes),
                    "cell_count": len(self.m1_training.cells),
                },
                "M2": {
                    "mesh_id": self.m2_reference.mesh_id,
                    "node_count": len(self.m2_reference.nodes),
                    "cell_count": len(self.m2_reference.cells),
                },
            },
        }

    @property
    def hierarchy_hash(self) -> str:
        return _sha256_doc(self.document())


def build_mesh_hierarchy(
    base_mesh: CanonicalMesh,
    *,
    cad_source_hash: str,
    geometry_tolerance_mm: float,
    quadrature_rule: str = "potential-energy-stratified-triangles-v1",
    refinement_history_hash: str | None = None,
) -> MeshHierarchy:
    """Build a verified ``(M0, M1, M2)`` midpoint hierarchy from a frozen polygon."""
    m0 = refine_fixed_mesh(base_mesh, 1)
    m1 = refine_fixed_mesh(m0, 2)
    m2 = refine_fixed_mesh(m0, 4)
    return MeshHierarchy(
        m0_coarse=m0,
        m1_training=m1,
        m2_reference=m2,
        cad_source_hash=cad_source_hash,
        geometry_tolerance_mm=geometry_tolerance_mm,
        quadrature_rule=quadrature_rule,
        refinement_history_hash=refinement_history_hash,
    )


@dataclass(frozen=True, slots=True)
class CrossMeshTransfer:
    """Exact nested midpoint transfer between a coarse mesh and its level-2 child.

    Handles canonicalized (lexicographically reordered) node and cell indices:

    * Nodal prolongation uses exact P1 midpoint interpolation (exact for affine fields).
    * Nodal injection maps coinciding fine vertices back onto coarse vertices.
    * Cell restriction uses exact area-weighted averaging over the four nested children
      (conservative for any integrable density: ``sum(A_c * f_c) == sum(A_f * f_f)``).
    """

    coarse_mesh_id: str
    fine_mesh_id: str
    coarse_node_count: int
    fine_node_count: int
    coarse_cell_count: int
    fine_cell_count: int
    fine_node_parents: tuple[tuple[int, int], ...]
    coarse_to_fine_nodes: tuple[int, ...]
    coarse_to_fine_cells: tuple[tuple[int, int, int, int], ...]
    coarse_cell_areas: tuple[float, ...]
    fine_cell_areas: tuple[float, ...]

    @classmethod
    def between(cls, coarse: CanonicalMesh, fine: CanonicalMesh) -> CrossMeshTransfer:
        _require_nested_refinement(coarse, fine, 2, "fine")
        edge_set: set[tuple[int, int]] = set()
        for cell in coarse.cells:
            for a, b in zip(cell, (*cell[1:], cell[0]), strict=True):
                edge_set.add((a, b) if a < b else (b, a))
        edges: tuple[tuple[int, int], ...] = tuple(sorted(edge_set))
        n_coarse = len(coarse.nodes)
        if len(fine.nodes) != n_coarse + len(edges):
            raise ValueError("midpoint node layout differs from nested refinement")

        # Build exact coordinate lookup -> parent pair (i, i) for vertex, (a, b) for edge.
        coord_to_parent: dict[tuple[float, float], tuple[int, int]] = {}
        for idx, (x, y) in enumerate(coarse.nodes):
            coord_to_parent[(round(x, 12), round(y, 12))] = (idx, idx)
        for a, b in edges:
            mx = 0.5 * (coarse.nodes[a][0] + coarse.nodes[b][0])
            my = 0.5 * (coarse.nodes[a][1] + coarse.nodes[b][1])
            coord_to_parent[(round(mx, 12), round(my, 12))] = (a, b)

        fine_node_parents: list[tuple[int, int]] = []
        coarse_to_fine_map: dict[int, int] = {}
        for fine_idx, (fx, fy) in enumerate(fine.nodes):
            parent = coord_to_parent.get((round(fx, 12), round(fy, 12)))
            if parent is None:
                raise ValueError(
                    "fine vertex does not match any coarse vertex or edge midpoint"
                )
            fine_node_parents.append(parent)
            if parent[0] == parent[1]:
                coarse_to_fine_map[parent[0]] = fine_idx
        if len(coarse_to_fine_map) != n_coarse:
            raise ValueError("missing coinciding coarse vertices in fine mesh")
        coarse_to_fine_nodes = tuple(coarse_to_fine_map[i] for i in range(n_coarse))

        # Match each fine cell to its unique coarse parent cell by exact vertex/midpoint sets.
        fine_cell_by_verts: dict[frozenset[int], int] = {
            frozenset(cell): idx for idx, cell in enumerate(fine.cells)
        }
        parent_pair_to_fine: dict[tuple[int, int], int] = {
            pair: idx for idx, pair in enumerate(fine_node_parents)
        }

        def fine_vertex(u: int, v: int) -> int:
            key = (u, v) if u <= v else (v, u)
            return parent_pair_to_fine[key]

        coarse_to_fine_cells: list[tuple[int, int, int, int]] = []
        for a, b, c in coarse.cells:
            va, vb, vc = fine_vertex(a, a), fine_vertex(b, b), fine_vertex(c, c)
            vab, vbc, vca = fine_vertex(a, b), fine_vertex(b, c), fine_vertex(c, a)
            child_sets = (
                frozenset((va, vab, vca)),
                frozenset((vab, vb, vbc)),
                frozenset((vca, vbc, vc)),
                frozenset((vab, vbc, vca)),
            )
            children = tuple(fine_cell_by_verts[s] for s in child_sets)
            assert len(children) == 4
            coarse_to_fine_cells.append(
                (children[0], children[1], children[2], children[3])
            )

        return cls(
            coarse_mesh_id=coarse.mesh_id,
            fine_mesh_id=fine.mesh_id,
            coarse_node_count=n_coarse,
            fine_node_count=len(fine.nodes),
            coarse_cell_count=len(coarse.cells),
            fine_cell_count=len(fine.cells),
            fine_node_parents=tuple(fine_node_parents),
            coarse_to_fine_nodes=coarse_to_fine_nodes,
            coarse_to_fine_cells=tuple(coarse_to_fine_cells),
            coarse_cell_areas=_mesh_cell_areas(coarse),
            fine_cell_areas=_mesh_cell_areas(fine),
        )

    def prolong_nodal(
        self, coarse_values: Sequence[Sequence[float]]
    ) -> tuple[tuple[float, ...], ...]:
        """Exact P1 nodal prolongation from coarse vertices to nested fine vertices."""
        if len(coarse_values) != self.coarse_node_count:
            raise ValueError("coarse_values length must match coarse_node_count")
        if not coarse_values:
            raise ValueError("coarse_values must be non-empty")
        width = len(coarse_values[0])
        if width == 0 or any(len(row) != width for row in coarse_values):
            raise ValueError("all coarse nodal rows must share a positive width")
        clean: list[tuple[float, ...]] = [
            tuple(_check_finite(v, "nodal value") for v in row) for row in coarse_values
        ]
        out: list[tuple[float, ...]] = []
        for a, b in self.fine_node_parents:
            if a == b:
                out.append(clean[a])
            else:
                row_a, row_b = clean[a], clean[b]
                out.append(tuple(0.5 * (row_a[k] + row_b[k]) for k in range(width)))
        return tuple(out)

    def restrict_nodal_injection(
        self, fine_values: Sequence[Sequence[float]]
    ) -> tuple[tuple[float, ...], ...]:
        """Exact nodal injection from fine vertices onto coinciding coarse vertices."""
        if len(fine_values) != self.fine_node_count:
            raise ValueError("fine_values length must match fine_node_count")
        width = len(fine_values[0])
        if width == 0 or any(len(row) != width for row in fine_values):
            raise ValueError("all fine nodal rows must share a positive width")
        return tuple(
            tuple(_check_finite(v, "nodal value") for v in fine_values[fine_idx])
            for fine_idx in self.coarse_to_fine_nodes
        )

    def restrict_cell_conservative(
        self, fine_cell_densities: Sequence[float]
    ) -> tuple[float, ...]:
        """Conservative area-weighted restriction of cell densities (fine -> coarse)."""
        if len(fine_cell_densities) != self.fine_cell_count:
            raise ValueError("fine_cell_densities length must match fine_cell_count")
        clean = [_check_finite(v, "cell density") for v in fine_cell_densities]
        restricted: list[float] = []
        for children in self.coarse_to_fine_cells:
            weighted_sum = 0.0
            area_sum = 0.0
            for child_idx in children:
                area = self.fine_cell_areas[child_idx]
                weighted_sum += clean[child_idx] * area
                area_sum += area
            restricted.append(weighted_sum / area_sum)
        return tuple(restricted)

    def prolong_cell_piecewise_constant(
        self, coarse_cell_densities: Sequence[float]
    ) -> tuple[float, ...]:
        """Conservative piecewise-constant prolongation of cell densities."""
        if len(coarse_cell_densities) != self.coarse_cell_count:
            raise ValueError(
                "coarse_cell_densities length must match coarse_cell_count"
            )
        values = [_check_finite(v, "cell density") for v in coarse_cell_densities]
        out = [0.0] * self.fine_cell_count
        for parent_idx, children in enumerate(self.coarse_to_fine_cells):
            for child_idx in children:
                out[child_idx] = values[parent_idx]
        return tuple(out)


class GeometryConvergenceStatus(enum.StrEnum):
    CONVERGED = "GEOMETRY_CONVERGED"
    NOT_CONVERGED = "GEOMETRY_NOT_CONVERGED"
    MEASURED_CHANGES_ONLY = "MEASURED_CHANGES_ONLY"


@dataclass(frozen=True, slots=True)
class GeometryConvergenceSample:
    """One distinct CAD-to-polygon discretization in a geometry-refinement ladder."""

    label: str
    mesh_id: str
    cad_source_hash: str
    declared_curve_tolerance_mm: float
    achieved_max_chord_error_mm: float
    area_mm2: float
    perimeter_mm: float
    node_count: int
    cell_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be non-empty text")
        if not isinstance(self.mesh_id, str) or not self.mesh_id.strip():
            raise ValueError("mesh_id must be non-empty text")
        _check_digest(self.cad_source_hash, "cad_source_hash")
        _check_finite(
            self.declared_curve_tolerance_mm,
            "declared_curve_tolerance_mm",
            positive=True,
        )
        _check_finite(
            self.achieved_max_chord_error_mm,
            "achieved_max_chord_error_mm",
            nonnegative=True,
        )
        _check_finite(self.area_mm2, "area_mm2", positive=True)
        _check_finite(self.perimeter_mm, "perimeter_mm", positive=True)
        if type(self.node_count) is not int or self.node_count < 3:
            raise ValueError("node_count must be an integer >= 3")
        if type(self.cell_count) is not int or self.cell_count < 1:
            raise ValueError("cell_count must be a positive integer")

    @classmethod
    def from_mesh(
        cls,
        label: str,
        mesh: CanonicalMesh,
        *,
        cad_source_hash: str,
        declared_curve_tolerance_mm: float,
        achieved_max_chord_error_mm: float,
    ) -> GeometryConvergenceSample:
        areas = _mesh_cell_areas(mesh)
        return cls(
            label=label,
            mesh_id=mesh.mesh_id,
            cad_source_hash=cad_source_hash,
            declared_curve_tolerance_mm=declared_curve_tolerance_mm,
            achieved_max_chord_error_mm=achieved_max_chord_error_mm,
            area_mm2=math.fsum(areas),
            perimeter_mm=_mesh_perimeter(mesh),
            node_count=len(mesh.nodes),
            cell_count=len(mesh.cells),
        )

    def document(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mesh_id": self.mesh_id,
            "cad_source_hash": self.cad_source_hash,
            "declared_curve_tolerance_mm": self.declared_curve_tolerance_mm,
            "achieved_max_chord_error_mm": self.achieved_max_chord_error_mm,
            "area_mm2": self.area_mm2,
            "perimeter_mm": self.perimeter_mm,
            "node_count": self.node_count,
            "cell_count": self.cell_count,
        }


@dataclass(frozen=True, slots=True)
class GeometryConvergenceReport:
    """CAD-to-polygon convergence report across distinct remeshes of the same CAD.

    Separately measures chord deviation, domain area/perimeter changes, and optional
    common-interior FEM field sensitivity across fresh CAD remeshes. Passing these
    checks never constitutes a continuum stress error bound.
    """

    cad_source_hash: str
    samples: tuple[GeometryConvergenceSample, ...]
    status: GeometryConvergenceStatus
    final_relative_area_change: float
    final_relative_perimeter_change: float
    final_achieved_chord_error_mm: float
    tolerances: Mapping[str, float] | None
    fem_remesh_comparisons: tuple[Mapping[str, Any], ...] = ()

    def document(self) -> dict[str, Any]:
        return {
            "schema": GEOMETRY_CONVERGENCE_SCHEMA,
            "cad_source_hash": self.cad_source_hash,
            "status": self.status.value,
            "is_continuum_solution_bound": False,
            "samples": [sample.document() for sample in self.samples],
            "final_relative_area_change": self.final_relative_area_change,
            "final_relative_perimeter_change": self.final_relative_perimeter_change,
            "final_achieved_chord_error_mm": self.final_achieved_chord_error_mm,
            "tolerances": (
                None
                if self.tolerances is None
                else dict(sorted(self.tolerances.items()))
            ),
            "fem_remesh_comparisons": [
                dict(item) for item in self.fem_remesh_comparisons
            ],
        }

    @property
    def report_hash(self) -> str:
        return _sha256_doc(self.document())


def evaluate_geometry_convergence(
    samples: Sequence[GeometryConvergenceSample],
    *,
    max_relative_area_change: float | None = None,
    max_relative_perimeter_change: float | None = None,
    max_chord_error_mm: float | None = None,
    fem_remesh_comparisons: Sequence[Mapping[str, Any]] = (),
) -> GeometryConvergenceReport:
    """Evaluate CAD-to-polygon convergence across >=2 distinct remeshes."""
    if len(samples) < 2:
        raise ValueError("geometry convergence requires at least two distinct remeshes")
    source_hash = samples[0].cad_source_hash
    if any(sample.cad_source_hash != source_hash for sample in samples):
        raise ValueError(
            "all geometry convergence samples must bind the same CAD source"
        )
    mesh_ids = [sample.mesh_id for sample in samples]
    if len(set(mesh_ids)) != len(mesh_ids):
        raise ValueError("geometry convergence requires distinct polygonal remeshes")

    prev, final = samples[-2], samples[-1]
    rel_area = abs(final.area_mm2 - prev.area_mm2) / final.area_mm2
    rel_perim = abs(final.perimeter_mm - prev.perimeter_mm) / final.perimeter_mm
    chord = final.achieved_max_chord_error_mm

    tols_input = (
        max_relative_area_change,
        max_relative_perimeter_change,
        max_chord_error_mm,
    )
    if all(v is None for v in tols_input):
        tolerances: dict[str, float] | None = None
        status = GeometryConvergenceStatus.MEASURED_CHANGES_ONLY
    elif any(v is None for v in tols_input):
        raise ValueError(
            "provide all three geometry tolerances or leave all three as None"
        )
    else:
        assert max_relative_area_change is not None
        assert max_relative_perimeter_change is not None
        assert max_chord_error_mm is not None
        tolerances = {
            "max_relative_area_change": _check_finite(
                max_relative_area_change, "max_relative_area_change", nonnegative=True
            ),
            "max_relative_perimeter_change": _check_finite(
                max_relative_perimeter_change,
                "max_relative_perimeter_change",
                nonnegative=True,
            ),
            "max_chord_error_mm": _check_finite(
                max_chord_error_mm, "max_chord_error_mm", nonnegative=True
            ),
        }
        fem_ok = all(
            item.get("status") in ("PASS", "MEASURED_CHANGES_ONLY", None)
            for item in fem_remesh_comparisons
        )
        passed = (
            rel_area <= tolerances["max_relative_area_change"]
            and rel_perim <= tolerances["max_relative_perimeter_change"]
            and chord <= tolerances["max_chord_error_mm"]
            and chord <= final.declared_curve_tolerance_mm
            and fem_ok
        )
        status = (
            GeometryConvergenceStatus.CONVERGED
            if passed
            else GeometryConvergenceStatus.NOT_CONVERGED
        )

    return GeometryConvergenceReport(
        cad_source_hash=source_hash,
        samples=tuple(samples),
        status=status,
        final_relative_area_change=rel_area,
        final_relative_perimeter_change=rel_perim,
        final_achieved_chord_error_mm=chord,
        tolerances=tolerances,
        fem_remesh_comparisons=tuple(dict(item) for item in fem_remesh_comparisons),
    )


class ReferenceConvergenceStatus(enum.StrEnum):
    CONVERGED = "CONVERGED"
    REFERENCE_NOT_CONVERGED = "REFERENCE_NOT_CONVERGED"
    MEASURED_CHANGES_ONLY = "MEASURED_CHANGES_ONLY"


@dataclass(frozen=True, slots=True)
class ReferenceConvergenceReport:
    """Fixed-polygon FEM reference convergence report across ``M0 -> M1 -> M2``.

    If the finest transition ``M1 -> M2`` fails its prospectively defined
    reference-accuracy floor, status is explicitly ``REFERENCE_NOT_CONVERGED``
    and accuracy conclusions must be withheld.
    """

    hierarchy_hash: str
    status: ReferenceConvergenceStatus
    m0_to_m1: Mapping[str, Any]
    m1_to_m2: Mapping[str, Any]
    reference_floor: Mapping[str, float] | None
    withhold_accuracy_conclusions: bool

    def document(self) -> dict[str, Any]:
        return {
            "schema": REFERENCE_CONVERGENCE_SCHEMA,
            "hierarchy_hash": self.hierarchy_hash,
            "status": self.status.value,
            "withhold_accuracy_conclusions": self.withhold_accuracy_conclusions,
            "geometry_status": "SAME_POLYGON",
            "total_error_bound": None,
            "reference_floor": (
                None
                if self.reference_floor is None
                else dict(sorted(self.reference_floor.items()))
            ),
            "m0_to_m1": dict(self.m0_to_m1),
            "m1_to_m2": dict(self.m1_to_m2),
        }

    @property
    def report_hash(self) -> str:
        return _sha256_doc(self.document())


_REQUIRED_REF_KEYS = frozenset(
    ("u_relative_l2", "p_relative_l2", "absolute_peak_error")
)


def build_reference_convergence_report(
    hierarchy: MeshHierarchy,
    m0_to_m1: Mapping[str, Any],
    m1_to_m2: Mapping[str, Any],
    *,
    reference_floor: Mapping[str, float] | None,
) -> ReferenceConvergenceReport:
    """Reduce ``compare_fem_pair`` outputs across M0->M1->M2 against a floor."""
    if not isinstance(hierarchy, MeshHierarchy):
        raise TypeError("hierarchy must be a MeshHierarchy")
    if (
        m0_to_m1.get("coarse_mesh_id") != hierarchy.m0_coarse.mesh_id
        or m0_to_m1.get("fine_mesh_id") != hierarchy.m1_training.mesh_id
    ):
        raise ValueError("m0_to_m1 pair does not match hierarchy M0 -> M1")
    if (
        m1_to_m2.get("coarse_mesh_id") != hierarchy.m1_training.mesh_id
        or m1_to_m2.get("fine_mesh_id") != hierarchy.m2_reference.mesh_id
    ):
        raise ValueError("m1_to_m2 pair does not match hierarchy M1 -> M2")

    finest_quantities = m1_to_m2.get("quantities")
    if not isinstance(finest_quantities, Mapping) or not (
        _REQUIRED_REF_KEYS <= set(finest_quantities)
    ):
        raise ValueError("m1_to_m2 comparison is missing required quantities")

    if reference_floor is None:
        return ReferenceConvergenceReport(
            hierarchy_hash=hierarchy.hierarchy_hash,
            status=ReferenceConvergenceStatus.MEASURED_CHANGES_ONLY,
            m0_to_m1=dict(m0_to_m1),
            m1_to_m2=dict(m1_to_m2),
            reference_floor=None,
            withhold_accuracy_conclusions=True,
        )

    if set(reference_floor) != _REQUIRED_REF_KEYS:
        raise ValueError(
            "reference_floor must specify u_relative_l2, p_relative_l2, "
            "and absolute_peak_error"
        )
    validated_floor = {
        k: _check_finite(v, f"reference_floor.{k}", nonnegative=True)
        for k, v in reference_floor.items()
    }
    passed = all(
        _check_finite(finest_quantities[k], f"m1_to_m2.{k}", nonnegative=True)
        <= validated_floor[k]
        for k in _REQUIRED_REF_KEYS
    )
    status = (
        ReferenceConvergenceStatus.CONVERGED
        if passed
        else ReferenceConvergenceStatus.REFERENCE_NOT_CONVERGED
    )
    return ReferenceConvergenceReport(
        hierarchy_hash=hierarchy.hierarchy_hash,
        status=status,
        m0_to_m1=dict(m0_to_m1),
        m1_to_m2=dict(m1_to_m2),
        reference_floor=validated_floor,
        withhold_accuracy_conclusions=not passed,
    )


__all__ = (
    "GEOMETRY_CONVERGENCE_SCHEMA",
    "HIERARCHY_SCHEMA",
    "REFERENCE_CONVERGENCE_SCHEMA",
    "CrossMeshTransfer",
    "GeometryConvergenceReport",
    "GeometryConvergenceSample",
    "GeometryConvergenceStatus",
    "MeshHierarchy",
    "ReferenceConvergenceReport",
    "ReferenceConvergenceStatus",
    "build_mesh_hierarchy",
    "build_reference_convergence_report",
    "evaluate_geometry_convergence",
)
