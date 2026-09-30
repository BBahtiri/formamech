"""Read-only TRAIN reference for recovered FEM stress at stored quadrature."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from skfem import (  # type: ignore[import-untyped]
    Basis,
    ElementTriP2,
    ElementVector,
    MeshTri,
)

from campaigns.dogbone_v1.s11_contingency_run import qoi_vertex_indices
from campaigns.dogbone_v1.s11_contingency_source import (
    S10SourceBindings,
    open_selective_s10_source,
)
from campaigns.dogbone_v1.s13_overfit_diagnostic import (
    _absolute_directory,
    _exclusive_write,
    _new_external_path,
)
from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.doe import EvidenceRole
from piginot.fem.models import FemRequest, FemResult
from piginot.fem.verification_norms import (
    _finite_array,
    _interpolate_recovered_stress,
    _pack_vector_dofs,
    _reference_quadrature,
    _validate_rebuilt_layout,
    _validate_result_shapes,
    _weighted_component_norm,
)

_CASE_ID = "candidate-002485"
_EXPECTED_COUNTS = (969, 1792, 20)
_STRESS_NORMALIZATION_MPA = 760.0
_POPULATION_DEFINITION = (
    "Stored physical quadrature with dx area weights; recovered P2 stress "
    "interpolated at those points versus stored raw FEM stress, in-plane 2x2 "
    "components only. Root-incident cells contain any registered right_arc "
    "vertex excluding right_grip; other cells form the complement. This is "
    "not operator vertex error and not error against an analytic exact stress."
)


@dataclass(frozen=True, slots=True)
class FemRecoveryPopulationMetrics:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    population: str
    cell_count: int
    quadrature_point_count: int
    integrated_area_mm2: float
    p_relative_l2: float
    p_normalized_component_rms: float


@dataclass(frozen=True, slots=True)
class FemRecoverySummary:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    cell_count: int
    quadrature_point_count: int
    gradient_max_absolute_difference: float
    populations: tuple[FemRecoveryPopulationMetrics, ...]


def summarize_fem_recovery_reference(
    request: FemRequest,
    result: FemResult,
    qoi_indices: tuple[int, ...],
) -> FemRecoverySummary:
    """Rebuild the stored P2 layout without invoking a FEM solver."""

    if (
        result.request_id != request.request_id
        or result.mesh_id != request.mesh.mesh_id
    ):
        raise ValueError("FEM request/result identities differ")
    if request.config.element_order != 2 or result.element_order != 2:
        raise ValueError("FEM recovery reference requires matching P2 element_order")
    points = _finite_array(result.quadrature_points, "quadrature_points")
    weights = _finite_array(result.quadrature_weights, "quadrature_weights")
    coordinates = _finite_array(result.displacement_dof_coordinates, "DOF coordinates")
    displacement = _finite_array(result.displacement, "displacement")
    raw = _finite_array(result.first_piola_stress, "first_piola_stress")
    recovered_dofs = _finite_array(
        result.recovered_first_piola_stress, "recovered stress"
    )
    deformation = _finite_array(result.deformation_gradient, "deformation_gradient")
    _validate_result_shapes(
        request, points, weights, coordinates, displacement, raw, recovered_dofs
    )
    if deformation.shape != raw.shape:
        raise ValueError(
            "deformation_gradient must match the stored quadrature tensor layout"
        )

    nodes = np.asarray(request.mesh.nodes, dtype=np.float64)
    cells = np.asarray(request.mesh.cells, dtype=np.int32)
    if (
        type(qoi_indices) is not tuple
        or not qoi_indices
        or any(type(index) is not int for index in qoi_indices)
        or tuple(sorted(set(qoi_indices))) != qoi_indices
        or qoi_indices[0] < 0
        or qoi_indices[-1] >= len(nodes)
    ):
        raise ValueError("QoI indices must be sorted unique mesh vertex IDs")
    root_cells = np.isin(cells, qoi_indices).any(axis=1)
    if not root_cells.any() or root_cells.all():
        raise ValueError("QoI root-incident cells must have a nonempty complement")

    mesh = MeshTri(nodes.T, cells.T, sort_t=False)
    quadrature = _reference_quadrature(request, points, weights)
    scalar_basis = Basis(mesh, ElementTriP2(), quadrature=quadrature)
    vector_basis = Basis(mesh, ElementVector(ElementTriP2()), quadrature=quadrature)
    _validate_rebuilt_layout(
        scalar_basis,
        vector_basis,
        points,
        weights,
        coordinates,
        displacement,
        recovered_dofs,
    )
    gradient = np.transpose(
        np.asarray(
            vector_basis.interpolate(
                _pack_vector_dofs(vector_basis, displacement)
            ).grad,
            dtype=np.float64,
        ),
        (2, 3, 0, 1),
    )
    stored_gradient = deformation[..., :2, :2] - np.eye(2)
    recovered = _interpolate_recovered_stress(
        scalar_basis, recovered_dofs, (points.shape[0], points.shape[1])
    )
    delta = recovered[..., :2, :2] - raw[..., :2, :2]
    populations: list[FemRecoveryPopulationMetrics] = []
    for name, mask in (
        ("all_cells", np.ones(len(cells), dtype=np.bool_)),
        ("root_incident_cells", root_cells),
        ("nonroot_cells", ~root_cells),
    ):
        subset_weights = weights[mask]
        area = float(subset_weights.sum())
        error = _weighted_component_norm(
            delta[mask],
            subset_weights,
            component_axes=(2, 3),
            name="FEM recovery difference",
        )
        denominator = _weighted_component_norm(
            raw[mask, ..., :2, :2],
            subset_weights,
            component_axes=(2, 3),
            name="raw FEM stress norm",
        )
        if denominator <= 0.0:
            raise ValueError("each raw FEM stress population needs a positive norm")
        populations.append(
            FemRecoveryPopulationMetrics(
                name,
                int(mask.sum()),
                int(subset_weights.size),
                area,
                error / denominator,
                error / (_STRESS_NORMALIZATION_MPA * math.sqrt(4.0 * area)),
            )
        )
    return FemRecoverySummary(
        len(cells),
        int(weights.size),
        float(np.max(np.abs(gradient - stored_gradient))),
        tuple(populations),
    )


@dataclass(frozen=True, slots=True)
class S13FemRecoveryReferenceReport:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    diagnostic_id: str
    role: EvidenceRole
    non_gate: bool
    case_id: str
    source_bindings: S10SourceBindings
    source_bindings_hash: str
    request_content_hash: str
    result_content_hash: str
    mesh_id: str
    element_order: int
    stress_normalization_mpa: float
    root_qoi_vertex_indices: tuple[int, ...]
    population_definition: str
    summary: FemRecoverySummary


def run_s13_fem_recovery_reference(
    *,
    s10_run_root: Path,
    development_access_record: Path,
    output_path: Path,
) -> S13FemRecoveryReferenceReport:
    """Resolve only the existing registered TRAIN case and inspect its fields."""

    s10 = _absolute_directory(s10_run_root, "s10_run_root")
    roots = (s10, s10)
    access = _new_external_path(
        development_access_record, "development_access_record", source_roots=roots
    )
    output = _new_external_path(output_path, "output_path", source_roots=roots)
    if access == output:
        raise ValueError("output_path and development_access_record must differ")
    source = open_selective_s10_source(s10, development_access_record=access)
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT must remain closed before TRAIN reference")
    verified, _ = source.load_fit_with_collocation(_CASE_ID)
    if source.development_opened or verified.case.role is not EvidenceRole.TRAIN:
        raise RuntimeError("FEM recovery reference must use only TRAIN labels")
    if verified.case.case_id != _CASE_ID or verified.request.request_id != _CASE_ID:
        raise ValueError("FEM recovery reference case differs from the registered case")
    request, result = verified.request, verified.result
    qoi = qoi_vertex_indices(
        request.mesh, semantic_tag="right_arc", excluded_tags=("right_grip",)
    )
    if (len(request.mesh.nodes), len(request.mesh.cells), len(qoi)) != _EXPECTED_COUNTS:
        raise ValueError("FEM reference population differs from the registered case")
    summary = summarize_fem_recovery_reference(request, result, qoi)
    if source.development_opened or access.exists():
        raise RuntimeError("DEVELOPMENT opened during TRAIN reference")
    return S13FemRecoveryReferenceReport(
        diagnostic_id="dogbone-s13-fem-recovery-reference-v1",
        role=EvidenceRole.TRAIN,
        non_gate=True,
        case_id=_CASE_ID,
        source_bindings=source.bindings,
        source_bindings_hash=source.bindings.content_hash,
        request_content_hash=hash_spec(request, "content"),
        result_content_hash=hash_spec(result, "content"),
        mesh_id=request.mesh.mesh_id,
        element_order=result.element_order,
        stress_normalization_mpa=_STRESS_NORMALIZATION_MPA,
        root_qoi_vertex_indices=qoi,
        population_definition=_POPULATION_DEFINITION,
        summary=summary,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--development-access-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args(arguments)
    report = run_s13_fem_recovery_reference(
        s10_run_root=options.s10_run_root,
        development_access_record=options.development_access_record,
        output_path=options.output,
    )
    _exclusive_write(options.output, canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())
