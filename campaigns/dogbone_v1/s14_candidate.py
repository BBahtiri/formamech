"""The frozen S13c candidate, served on S14 evidence (``CP-FREEZE``).

The candidate is loaded only through the registered S13c checkpoint loader and
is accepted only when its manifest, weights, model state and run hashes equal
the values frozen in ``CP-FREEZE.json``. Scoring reuses the unchanged S13c
TRAIN-screen metric code, so a TRAIN case reproduces its ``S13C-TRAIN-SCREEN``
diagnostics exactly. Each scored case also yields the pointwise in-plane
stress error at the vertices and the three equilibrium-residual indicator
variants of ADR 0019 (S14-D4).

Cases are role-neutral :class:`S14Case` records, so a HOLDOUT case, which can
never be a public ``PublishedCase``, is scored by the same code. The vertex
targets are built with the same arithmetic as the S11 projection
(``prepare_operator_case`` followed by ``normalize_case_values``).
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import torch

from campaigns.dogbone_v1.s11_contingency import STRESS_REFERENCE_MPA
from campaigns.dogbone_v1.s11_contingency_data import CaseNormalizationStatistics
from campaigns.dogbone_v1.s13_objective import _physics_material
from campaigns.dogbone_v1.s13c_data import (
    DogboneS13cGeometryCase,
    _loaded_end_displacement,
    conditioning_vector,
)
from campaigns.dogbone_v1.s13c_evaluate import (
    S13cCaseDiagnostics,
    _case_diagnostics,
    _served_vertex_output,
    load_registered_checkpoint,
)
from campaigns.dogbone_v1.s13c_trial import load_committed_s13c_protocol
from campaigns.dogbone_v1.s14_reference import root_vertices
from campaigns.dogbone_v1.strain_pilot import prescribed_displacement_mm
from piginot.doe import EvidenceRole, PublishedCase, PublishedCaseStatus
from piginot.fem import FemRequest, FemResult
from piginot.objectives import TriangleDomain
from piginot.operators import EnergyFormOperator
from piginot.verification import (
    boundary_traction_residual,
    equilibrium_residual,
    indicator_variants,
)

_ROOT = Path(__file__).resolve().parents[2]
CP_FREEZE_PATH = _ROOT / "artifacts" / "campaigns" / "dogbone-v1" / "CP-FREEZE.json"
INDICATOR_LENGTH_SCALE_MM = 1.0
_GEOMETRY_TOLERANCE_MM = 1.0e-9
_FROZEN_FIELDS = {
    "checkpoint_manifest_hash": "manifest_hash",
    "weights_hash": "model_export_hash",
    "model_state_hash": "model_state_hash",
    "run_hash": "run_hash",
}


def frozen_candidate_record(path: Path = CP_FREEZE_PATH) -> Mapping[str, object]:
    """The ``frozen_candidate`` block of ``CP-FREEZE.json``."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("checkpoint_id") != "CP-FREEZE":
        raise ValueError("not a CP-FREEZE record")
    record = document["frozen_candidate"]
    if not isinstance(record, dict):
        raise ValueError("CP-FREEZE lacks the frozen candidate")
    return record


def load_frozen_candidate(checkpoint_path: Path) -> EnergyFormOperator:
    """Load the registered checkpoint and require the ``CP-FREEZE`` identity."""

    protocol = load_committed_s13c_protocol()
    model, _, manifest = load_registered_checkpoint(protocol, checkpoint_path)
    record = frozen_candidate_record()
    for frozen, attribute in _FROZEN_FIELDS.items():
        if getattr(manifest, attribute) != record[frozen]:
            raise ValueError(f"checkpoint {attribute} differs from CP-FREEZE")
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != record["parameter_count"]
    ):
        raise ValueError("checkpoint parameter count differs from CP-FREEZE")
    served: EnergyFormOperator = copy.deepcopy(model).to(torch.float64).eval()
    return served


@dataclass(frozen=True, slots=True)
class S14Case:
    """One scored case of any evidence role (TRAIN, DEVELOPMENT or HOLDOUT)."""

    case_id: str
    role: EvidenceRole
    leakage_group_id: str
    primitive_coordinates: Mapping[str, float]
    derived_coordinates: Mapping[str, float]
    strata: Mapping[str, int]

    def __post_init__(self) -> None:
        for name in ("case_id", "leakage_group_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty text")
        object.__setattr__(self, "role", EvidenceRole(self.role))
        for name in ("primitive_coordinates", "derived_coordinates", "strata"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @classmethod
    def from_published(cls, case: PublishedCase) -> S14Case:
        """A public case; only ACTIVE cases are scored."""

        if case.status is not PublishedCaseStatus.ACTIVE:
            raise ValueError("only ACTIVE cases are scored")
        return cls(
            case.case_id,
            case.role,
            case.leakage_group_id,
            case.primitive_coordinates,
            case.derived_coordinates,
            case.strata,
        )


def served_geometry(
    case: S14Case,
    request: FemRequest,
    normalization: CaseNormalizationStatistics,
    *,
    allowed_roles: frozenset[EvidenceRole],
) -> DogboneS13cGeometryCase:
    """The S13c geometry case of one S14 case, for an explicitly allowed role."""

    if case.role not in allowed_roles:
        raise PermissionError(f"{case.role.value} evidence is not open in this step")
    if request.request_id != case.case_id:
        raise ValueError("request does not belong to the case")
    coordinates = case.primitive_coordinates
    length = float(coordinates["L_total"])
    grip_width = float(coordinates["W_grip"])
    strain = float(coordinates["nominal_axial_strain"])
    nodes = torch.tensor(request.mesh.nodes, dtype=torch.float64)
    cells = torch.tensor(request.mesh.cells, dtype=torch.int64)
    half_length = 0.5 * length
    half_grip_width = 0.5 * grip_width
    extent_error = max(
        abs(float(nodes[:, 0].min())),
        abs(float(nodes[:, 1].min())),
        abs(float(nodes[:, 0].max()) - half_length),
        abs(float(nodes[:, 1].max()) - half_grip_width),
    )
    if extent_error > _GEOMETRY_TOLERANCE_MM:
        raise ValueError(f"{case.case_id} mesh extent differs from the quarter model")
    displacement = _loaded_end_displacement(request)
    if not math.isclose(
        displacement,
        prescribed_displacement_mm(length, strain),
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise ValueError(f"{case.case_id} loaded-end displacement differs from strain")
    return DogboneS13cGeometryCase(
        case_id=case.case_id,
        leakage_group_id=case.leakage_group_id,
        request=request,
        material=_physics_material(request.material),
        domain=TriangleDomain(nodes, cells),
        conditioning=conditioning_vector(coordinates, normalization),
        half_length_mm=half_length,
        half_grip_width_mm=half_grip_width,
        prescribed_displacement_mm=displacement,
    )


@dataclass(frozen=True, slots=True)
class ScoredCase:
    """Surrogate metrics and pointwise confidence data of one case."""

    diagnostics: S13cCaseDiagnostics
    pointwise_stress_error: torch.Tensor
    indicators: Mapping[str, torch.Tensor]
    traction: torch.Tensor | None = None

    @property
    def case_id(self) -> str:
        return self.diagnostics.case_id


def vertex_targets(
    case: S14Case,
    request: FemRequest,
    result: FemResult,
    normalization: CaseNormalizationStatistics,
) -> SimpleNamespace:
    """Scaled vertex targets, as ``prepare_operator_case`` + ``normalize_case_values``.

    ``target_u`` is divided by the prescribed displacement and ``target_P`` (the
    in-plane block of the recovered first-Piola stress) by 760 MPa.
    """

    if result.request_id != case.case_id or request.request_id != case.case_id:
        raise ValueError("case, request and result identities do not match")
    if result.mesh_id != request.mesh.mesh_id:
        raise ValueError("result mesh identity does not match the request")
    if request.config.element_order != 2 or result.element_order != 2:
        raise ValueError("S14 scoring requires the verified P2 FEM recipe")
    count = len(request.mesh.nodes)
    query_xy = torch.as_tensor(request.mesh.nodes, dtype=torch.float64)
    coordinates = torch.as_tensor(
        result.displacement_dof_coordinates[:count], dtype=torch.float64
    )
    if not torch.equal(coordinates, query_xy):
        raise ValueError("P2 vertex coordinates do not align with mesh vertices")
    target_u = torch.as_tensor(result.displacement[:count], dtype=torch.float64)
    recovered = torch.as_tensor(
        result.recovered_first_piola_stress[:count], dtype=torch.float64
    )
    if target_u.shape != (count, 2) or recovered.shape != (count, 3, 3):
        raise ValueError("P2 vertex fields have unexpected shapes")
    if not bool(torch.isfinite(target_u).all()) or not bool(
        torch.isfinite(recovered).all()
    ):
        raise ValueError("P2 vertex fields must be finite")
    scale = prescribed_displacement_mm(
        float(case.primitive_coordinates["L_total"]),
        float(case.primitive_coordinates["nominal_axial_strain"]),
    )
    return SimpleNamespace(
        case_id=case.case_id,
        query_xy=query_xy,
        target_u=target_u / scale,
        target_P=recovered[:, :2, :2] / normalization.stress_reference_mpa,
        qoi_indices=torch.tensor(root_vertices(request.mesh), dtype=torch.int64),
    )


_AXES = {"x": 0, "y": 1}


def free_traction_components(
    request: FemRequest,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, ...], ...]]:
    """Boundary facets and the traction components no condition prescribes.

    A tag without a boundary condition is traction-free (both components); a
    symmetry or Dirichlet condition on some components leaves the others free.
    """

    mesh = request.mesh
    constrained: dict[str, set[int]] = {tag: set() for tag in mesh.semantic_tags}
    for condition in request.boundary_conditions:
        if condition.boundary_tag not in constrained:
            raise ValueError(f"unknown boundary tag {condition.boundary_tag!r}")
        if condition.kind not in ("symmetry", "dirichlet"):
            raise ValueError(f"unsupported boundary condition {condition.kind!r}")
        constrained[condition.boundary_tag].update(
            _AXES[name] for name in condition.components
        )
    facets = tuple(tuple(facet) for facet in mesh.boundary_facets)
    free: list[tuple[int, ...] | None] = [None] * len(facets)
    for tag, indices in mesh.semantic_tags.items():
        components = tuple(c for c in (0, 1) if c not in constrained[tag])
        for index in indices:
            if free[index] is not None:
                raise ValueError("a boundary facet carries two semantic tags")
            free[index] = components
    if any(entry is None for entry in free):
        raise ValueError("every boundary facet needs a semantic tag")
    return facets, tuple(entry for entry in free if entry is not None)  # type: ignore[return-value]


def score_case(
    model: EnergyFormOperator,
    case: S14Case,
    request: FemRequest,
    result: FemResult,
    normalization: CaseNormalizationStatistics,
    *,
    allowed_roles: frozenset[EvidenceRole],
    with_indicators: bool = True,
) -> ScoredCase:
    """Score one case with the S13c metric code and the S14-D4 indicators."""

    geometry = served_geometry(
        case, request, normalization, allowed_roles=allowed_roles
    )
    tensors = vertex_targets(case, request, result, normalization)
    u, stress, determinant = _served_vertex_output(model, geometry, tensors.query_xy)
    if not bool(torch.isfinite(stress).all().item()):
        raise ValueError(f"{case.case_id} has inadmissible vertex deformation")
    predicted_u = u / geometry.prescribed_displacement_mm
    predicted_p = stress / STRESS_REFERENCE_MPA
    diagnostics = _case_diagnostics(
        tensors,  # type: ignore[arg-type]
        predicted_u,
        predicted_p,
        determinant,
    )
    pointwise = torch.linalg.vector_norm(
        predicted_p - tensors.target_P, dim=(-2, -1)
    ).detach()
    indicators: Mapping[str, torch.Tensor] = {}
    traction: torch.Tensor | None = None
    if with_indicators:
        facets, free = free_traction_components(request)
        traction = boundary_traction_residual(
            stress.detach(),
            geometry.domain.vertices,
            geometry.domain.cells,
            facets,
            free,
            elastic_modulus=float(
                geometry.material.properties["elastic_modulus"].value
            ),
        )

        def residual(points: torch.Tensor) -> torch.Tensor:
            return equilibrium_residual(
                model,
                points,
                parameters=geometry.parameters(),
                frame=geometry.frame(),
                lifting=geometry.lifting(),
                material=geometry.material,
                length_scale=INDICATOR_LENGTH_SCALE_MM,
            )

        indicators = indicator_variants(
            residual,
            geometry.domain.vertices,
            geometry.domain.cells,
            length_scale=INDICATOR_LENGTH_SCALE_MM,
            traction=traction,
        )
    return ScoredCase(diagnostics, pointwise, indicators, traction)


def slice_ids(case: S14Case, axes: tuple[str, ...]) -> tuple[str, ...]:
    """``all`` plus one ``axis:bin`` slice per axis from the S9 strata."""

    slices = ["all"]
    for axis in axes:
        stratum = case.strata[axis]
        if isinstance(stratum, bool) or not isinstance(stratum, int):
            raise ValueError(f"{case.case_id} stratum {axis} is not an integer")
        slices.append(f"{axis}:{stratum}")
    return tuple(slices)


__all__ = (
    "CP_FREEZE_PATH",
    "INDICATOR_LENGTH_SCALE_MM",
    "S14Case",
    "ScoredCase",
    "free_traction_components",
    "frozen_candidate_record",
    "load_frozen_candidate",
    "score_case",
    "served_geometry",
    "slice_ids",
    "vertex_targets",
)
