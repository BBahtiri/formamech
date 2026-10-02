"""Per-case FEM reference error terms of the S14 error budget (ADR 0019, S14-D1).

The surrogate is compared with the stored S10 P2 reference at the mesh
vertices: the vertex displacement degrees of freedom and the recovered
first-Piola stress. Two S8 calibrated estimators (ADR 0002) bound the error of
that reference; both are applied to every case:

- the recovery estimator ``eta = |P_recovered - P_raw| / |P_raw|`` (quadrature
  L2, in-plane components) with safety factor 2.61, over all cells for the
  stress field and over the cells incident to the root vertices for the peak;
- the Richardson estimator with safety factor 1.97 on one coarser solve of the
  same geometry: the structured reference mesh with half the columns and rows,
  mapped by the same morph, so every coarse vertex is a vertex of the S10 mesh.
  The differences are taken at the coarse vertices. The order is fixed at 1,
  below every order observed on the TRAIN refinement check (displacement about
  2, stress field 1.0-1.7, root peak non-monotone): the DogBone meshes are not
  in the asymptotic range of the theoretical P2 orders 3 and 2, and a lower
  order gives a larger, more conservative estimate.

Budget components per case (fractions of the reference norm):

| quantity | discretisation | projection |
| --- | --- | --- |
| stress field | ``max(2.61 eta_all, 1.97 d_P)`` | ``eta_all`` |
| root peak | ``max(2.61 eta_root, 1.97 d_peak)`` | ``eta_root`` |
| displacement | ``1.97 d_u`` | ``0`` |

The projection component is the transfer from the solved field to the
comparison points. Displacement vertex values are degrees of freedom of the P2
solution, so they carry no transfer error (their coordinates are checked to be
the mesh vertices). The stress reference is the recovered field, whose
distance to the raw quadrature stress is ``eta``; together with the
discretisation bound of the raw stress this is the triangle inequality
``|P_rec - P| <= |P_rec - P_raw| + |P_raw - P|``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np

from campaigns.dogbone_v1.family import (
    _MESH_SPEC,
    _MORPH_MAP,
    _REFERENCE,
    _VALIDITY,
    parameter_point,
)
from campaigns.dogbone_v1.s11_contingency_run import qoi_vertex_indices
from campaigns.dogbone_v1.s13_fem_recovery_reference import (
    summarize_fem_recovery_reference,
)
from piginot.contracts import CanonicalMesh
from piginot.fem import FemRequest, FemResult
from piginot.fem.error_estimation import richardson_fine_grid_error
from piginot.morphing.circular_transition import (
    build_circular_transition_reference_mesh,
)
from piginot.morphing.validation import validate_deformation


class CaseGeometry(Protocol):
    """Any case record with an identifier and primitive coordinates."""

    @property
    def case_id(self) -> str: ...

    @property
    def primitive_coordinates(self) -> Mapping[str, float]: ...


RECOVERY_SAFETY_FACTOR = 2.61
RICHARDSON_SAFETY_FACTOR = 1.97
REFINEMENT_RATIO = 2.0
RICHARDSON_ORDER = 1.0
QUANTITIES = ("p_relative_l2", "absolute_peak_error", "u_relative_l2")
_GEOMETRY = ("L_total", "W_grip", "W_gauge", "R_fillet")
_ROOT_TAG = "right_arc"
_EXCLUDED_TAGS = ("right_grip",)
_COORDINATE_TOLERANCE_MM = 1.0e-9


def root_vertices(mesh: CanonicalMesh) -> tuple[int, ...]:
    """Registered root QoI vertices: ``right_arc`` without ``right_grip``."""

    return qoi_vertex_indices(
        mesh, semantic_tag=_ROOT_TAG, excluded_tags=_EXCLUDED_TAGS
    )


def level_mesh(case: CaseGeometry, scale: float) -> CanonicalMesh:
    """The S10 mesh recipe with ``scale`` times the columns and rows."""

    counts = {
        name: getattr(_MESH_SPEC, name) * scale
        for name in ("inner_columns", "transition_columns", "transverse_rows")
    }
    if any(value != int(value) or value < 1 for value in counts.values()):
        raise ValueError("scale must give whole positive column and row counts")
    spec = dataclasses.replace(
        _MESH_SPEC, **{name: int(value) for name, value in counts.items()}
    )
    reference = build_circular_transition_reference_mesh(
        _REFERENCE,
        spec,
        source_provenance={
            "generator": "circular-transition-structured-reference-v1",
            "purpose": "s14-reference-error",
            "scale": repr(float(scale)),
        },
    )
    theta = parameter_point(
        *(float(case.primitive_coordinates[name]) for name in _GEOMETRY)
    )
    mesh, report = validate_deformation(reference, _MORPH_MAP.bind(theta), _VALIDITY)
    if mesh is None:
        raise ValueError(f"{case.case_id} level mesh is invalid: {report.reason_codes}")
    return mesh


def level_request(case: CaseGeometry, request: FemRequest, scale: float) -> FemRequest:
    """The S10 request of ``case`` on the level mesh with ``scale`` resolution."""

    if request.request_id != case.case_id:
        raise ValueError("request does not belong to the case")
    return dataclasses.replace(request, mesh=level_mesh(case, scale))


def nested_vertex_map(coarse: CanonicalMesh, fine: CanonicalMesh) -> np.ndarray:
    """Index in ``fine`` of every vertex of the half-resolution mesh ``coarse``.

    Both meshes come from :func:`level_mesh`; the coarse one has half the
    columns and rows, so its vertex ``(column, row)`` is the fine vertex
    ``(2 column, 2 row)``. Coordinates are checked after the index mapping.
    """

    coarse_xy = np.asarray(coarse.nodes, dtype=np.float64)
    fine_xy = np.asarray(fine.nodes, dtype=np.float64)
    coarse_rows = _row_count(coarse_xy)
    fine_rows = _row_count(fine_xy)
    if fine_rows != 2 * coarse_rows - 1:
        raise ValueError("meshes are not two consecutive structured levels")
    column, row = np.divmod(np.arange(len(coarse_xy)), coarse_rows)
    mapped = 2 * column * fine_rows + 2 * row
    if mapped.max() >= len(fine_xy) or not np.allclose(
        coarse_xy, fine_xy[mapped], rtol=0.0, atol=_COORDINATE_TOLERANCE_MM
    ):
        raise ValueError("coarse vertices are not vertices of the finer mesh")
    return mapped


def _row_count(nodes: np.ndarray) -> int:
    """Vertices per column of a structured level mesh (equal x in column one)."""

    first = nodes[:, 0] == nodes[0, 0]
    count = int(np.argmin(first)) if not first.all() else len(nodes)
    if count < 2 or len(nodes) % count != 0 or not first[:count].all():
        raise ValueError("mesh is not a structured level mesh")
    return count


def _vertex_fields(
    request: FemRequest, result: FemResult
) -> tuple[np.ndarray, np.ndarray]:
    count = len(request.mesh.nodes)
    coordinates = np.asarray(result.displacement_dof_coordinates[:count])
    if not np.array_equal(coordinates, np.asarray(request.mesh.nodes)):
        raise ValueError("P2 vertex degrees of freedom do not sit on the vertices")
    displacement = np.asarray(result.displacement[:count], dtype=np.float64)
    stress = np.asarray(result.recovered_first_piola_stress[:count], dtype=np.float64)[
        :, :2, :2
    ]
    return displacement, stress


@dataclass(frozen=True, slots=True)
class FemReferenceTerms:
    """Measured raw ingredients of the FEM reference error of one case."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    case_id: str
    recovery_all_cells: float
    recovery_root_cells: float
    coarse_u_difference: float
    coarse_p_difference: float
    coarse_peak_difference: float
    coarse_vertex_count: int

    def __post_init__(self) -> None:
        for name in (
            "recovery_all_cells",
            "recovery_root_cells",
            "coarse_u_difference",
            "coarse_p_difference",
            "coarse_peak_difference",
        ):
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative float")

    def components(self) -> dict[str, dict[str, float]]:
        """Discretisation and projection components per budget quantity."""

        return {
            "p_relative_l2": {
                "discretisation": max(
                    RECOVERY_SAFETY_FACTOR * self.recovery_all_cells,
                    richardson_bound(self.coarse_p_difference),
                ),
                "projection": self.recovery_all_cells,
            },
            "absolute_peak_error": {
                "discretisation": max(
                    RECOVERY_SAFETY_FACTOR * self.recovery_root_cells,
                    richardson_bound(self.coarse_peak_difference),
                ),
                "projection": self.recovery_root_cells,
            },
            "u_relative_l2": {
                "discretisation": richardson_bound(self.coarse_u_difference),
                "projection": 0.0,
            },
        }


def richardson_bound(difference: float, order: float = RICHARDSON_ORDER) -> float:
    """S8 Richardson fine-grid estimate times the S8 safety factor."""

    if difference == 0.0:
        return 0.0
    estimate = richardson_fine_grid_error(
        difference, REFINEMENT_RATIO, theoretical_order=order
    )
    return RICHARDSON_SAFETY_FACTOR * estimate.fine_grid_error


def _relative(difference: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(difference) / np.linalg.norm(reference))


def level_differences(
    coarse: FemRequest,
    coarse_result: FemResult,
    fine: FemRequest,
    fine_result: FemResult,
) -> tuple[float, float, float, int]:
    """Relative ``u``, stress-field and root-peak differences at coarse vertices.

    The peak is the largest in-plane stress norm over the coarse root vertices,
    all of which are vertices of the finer mesh.
    """

    if coarse_result.mesh_id != coarse.mesh.mesh_id:
        raise ValueError("coarse result does not belong to the coarse mesh")
    if fine_result.mesh_id != fine.mesh.mesh_id:
        raise ValueError("fine result does not belong to the fine mesh")
    u_fine, p_fine = _vertex_fields(fine, fine_result)
    u_coarse, p_coarse = _vertex_fields(coarse, coarse_result)
    mapped = nested_vertex_map(coarse.mesh, fine.mesh)
    u_ref = u_fine[mapped]
    p_ref = p_fine[mapped]
    root = np.asarray(root_vertices(coarse.mesh))
    peak_ref = float(np.linalg.norm(p_ref[root], axis=(1, 2)).max())
    peak_coarse = float(np.linalg.norm(p_coarse[root], axis=(1, 2)).max())
    return (
        _relative(u_coarse - u_ref, u_ref),
        _relative(p_coarse - p_ref, p_ref),
        abs(peak_coarse - peak_ref) / peak_ref,
        len(mapped),
    )


def reference_terms(
    case_id: str,
    request: FemRequest,
    result: FemResult,
    coarse: FemRequest,
    coarse_result: FemResult,
) -> FemReferenceTerms:
    """Measure the recovery gaps and the coarse-to-S10 differences of one case."""

    if result.request_id != case_id or coarse_result.request_id != case_id:
        raise ValueError("FEM results do not belong to the case")
    recovery = {
        population.population: population.p_relative_l2
        for population in summarize_fem_recovery_reference(
            request, result, root_vertices(request.mesh)
        ).populations
    }
    u_difference, p_difference, peak_difference, count = level_differences(
        coarse, coarse_result, request, result
    )
    return FemReferenceTerms(
        case_id=case_id,
        recovery_all_cells=float(recovery["all_cells"]),
        recovery_root_cells=float(recovery["root_incident_cells"]),
        coarse_u_difference=u_difference,
        coarse_p_difference=p_difference,
        coarse_peak_difference=peak_difference,
        coarse_vertex_count=count,
    )


def refined_error_proxy(
    request: FemRequest,
    result: FemResult,
    fine: FemRequest,
    fine_result: FemResult,
) -> dict[str, float]:
    """Richardson-extrapolated error of the S10 reference from one refined solve.

    ``e = d * r**p / (r**p - 1)`` with the S10-to-refined difference ``d`` at
    the S10 vertices and the conservative order ``p = 1`` (so ``e = 2 d``): an
    independent check of the budget components, not a budget component.
    """

    u_difference, p_difference, peak_difference, _ = level_differences(
        request, result, fine, fine_result
    )

    def extrapolated(difference: float, order: float) -> float:
        growth = REFINEMENT_RATIO**order
        return difference * growth / (growth - 1.0)

    return {
        "p_relative_l2": extrapolated(p_difference, RICHARDSON_ORDER),
        "absolute_peak_error": extrapolated(peak_difference, RICHARDSON_ORDER),
        "u_relative_l2": extrapolated(u_difference, RICHARDSON_ORDER),
    }


def observed_orders(
    case: CaseGeometry,
    request: FemRequest,
    result: FemResult,
    fine: FemRequest,
    fine_result: FemResult,
    terms: FemReferenceTerms,
) -> dict[str, float | None]:
    """Apparent convergence orders from the coarse, S10 and refined levels.

    Both differences are taken at the coarse vertices, so the order is
    ``log2(d_coarse / d_refined)``; ``None`` when a difference vanishes.
    """

    coarse_mesh = level_mesh(case, 0.5)
    to_s10 = nested_vertex_map(coarse_mesh, request.mesh)
    to_fine = nested_vertex_map(request.mesh, fine.mesh)[to_s10]
    u_s10, p_s10 = _vertex_fields(request, result)
    u_fine, p_fine = _vertex_fields(fine, fine_result)
    u_s10, p_s10 = u_s10[to_s10], p_s10[to_s10]
    u_fine, p_fine = u_fine[to_fine], p_fine[to_fine]
    root = np.asarray(root_vertices(coarse_mesh))
    peak_s10 = float(np.linalg.norm(p_s10[root], axis=(1, 2)).max())
    peak_fine = float(np.linalg.norm(p_fine[root], axis=(1, 2)).max())
    refined = {
        "u_relative_l2": _relative(u_s10 - u_fine, u_fine),
        "p_relative_l2": _relative(p_s10 - p_fine, p_fine),
        "absolute_peak_error": abs(peak_s10 - peak_fine) / peak_fine,
    }
    coarse = {
        "u_relative_l2": terms.coarse_u_difference,
        "p_relative_l2": terms.coarse_p_difference,
        "absolute_peak_error": terms.coarse_peak_difference,
    }
    return {
        quantity: (
            math.log2(coarse[quantity] / refined[quantity])
            if coarse[quantity] > 0.0 and refined[quantity] > 0.0
            else None
        )
        for quantity in QUANTITIES
    }


__all__ = (
    "QUANTITIES",
    "RECOVERY_SAFETY_FACTOR",
    "REFINEMENT_RATIO",
    "RICHARDSON_ORDER",
    "RICHARDSON_SAFETY_FACTOR",
    "CaseGeometry",
    "FemReferenceTerms",
    "nested_vertex_map",
    "observed_orders",
    "level_differences",
    "level_mesh",
    "level_request",
    "reference_terms",
    "refined_error_proxy",
    "richardson_bound",
    "root_vertices",
)
