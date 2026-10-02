"""Descriptive TRAIN-only comparison of exactly replayed S13 fields at FEM quadrature."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import rfc8785
import torch
from skfem import (  # type: ignore[import-untyped]
    Basis,
    ElementTriP2,
    ElementVector,
    MeshTri,
)

from campaigns.dogbone_v1.s11_contingency_models import model_state_hash
from campaigns.dogbone_v1.s11_contingency_run import qoi_vertex_indices
from campaigns.dogbone_v1.s11_contingency_source import open_selective_s10_source
from campaigns.dogbone_v1.s13_data import materialize_dogbone_s13_training_case
from campaigns.dogbone_v1.s13_initialization import load_selected_s11_initialization
from campaigns.dogbone_v1.s13_objective import (
    _COORDINATE_CONVERSION,
    DogboneS13TrainingCase,
    _physics_material,
    _scales,
    _validated_operator,
)
from campaigns.dogbone_v1.s13_overfit_diagnostic import (
    _build_vertex_sampler_plan,
    _exclusive_write,
    _preflight_paths,
    _train_one_case,
)
from campaigns.dogbone_v1.s13_quadrature_metrics import (
    QuadratureCaseMetrics,
    summarize_quadrature_case,
)
from campaigns.dogbone_v1.s13_stress_capacity import (
    StressCapacityPlan,
    _implementation_hashes,
    assert_trainable_scope,
    build_stress_arm,
    frozen_state_hash,
    trainable_names,
)
from piginot.contracts import canonical_json_bytes, hash_spec, strict_json_loads
from piginot.doe import EvidenceRole
from piginot.fem.models import FemRequest, FemResult
from piginot.fem.verification_norms import (
    _LAYOUT_ATOL,
    _finite_array,
    _pack_vector_dofs,
    _reference_quadrature,
    _validate_rebuilt_layout,
    _validate_result_shapes,
)
from piginot.objectives import displacement_gradient
from piginot.operators import PiGinotOperator, module_digest


@dataclass(frozen=True, slots=True)
class QuadratureAuditPlan:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    diagnostic_id: str = "dogbone-s13-quadrature-gradient-comparison-v1"
    role: EvidenceRole = EvidenceRole.TRAIN
    non_gate: bool = True
    case_id: str = "candidate-002485"
    v5_sha256: str = "d443f6b51ab1883fc1cc704e3094a3a408c410de0cb14979359ec5b5848b1326"
    capacity_sha256: str = (
        "a87d52ce180718d7709462855e90994e62080291d45fbcde68f5e854d607f33e"
    )
    query_chunk_size: int = 512
    cpu_num_threads: int = 8
    cpu_interop_threads: int = 8
    torch_version: str = "2.13.0+cpu"
    expected_vertices: int = 969
    expected_cells: int = 1792
    expected_points_per_cell: int = 7
    expected_qoi_vertices: int = 20
    weighting_policy: str = "stored-physical-area-grad-u-relative-H1-v1"
    material_policy: str = "admissible-only-explicit-coverage-kernel-failure-closed-v1"

    def __post_init__(self) -> None:
        if any(
            type(getattr(self, item.name)) is not type(item.default)
            or getattr(self, item.name) != item.default
            for item in fields(self)
        ):
            raise ValueError("quadrature audit plan differs from the frozen protocol")

    @property
    def content_hash(self) -> str:
        return hash_spec(self, "content")


def validate_replay_runtime(plan: QuadratureAuditPlan) -> None:
    if (
        torch.get_num_threads() != plan.cpu_num_threads
        or torch.get_num_interop_threads() != plan.cpu_interop_threads
        or str(torch.__version__) != plan.torch_version
    ):
        raise ValueError(
            "replay runtime must match the pinned PyTorch version and CPU thread counts"
        )


@dataclass(frozen=True, slots=True)
class FemQuadratureFields:
    xy_mm: torch.Tensor
    u_mm: torch.Tensor
    gradient: torch.Tensor
    raw_p_mpa: torch.Tensor
    area_weights: torch.Tensor
    root_cell_mask: torch.Tensor
    gradient_max_absolute_difference: float


def rebuild_fem_quadrature(
    request: FemRequest, result: FemResult, qoi_indices: tuple[int, ...]
) -> FemQuadratureFields:
    """Interpolate every P2 displacement DOF, retaining cell-sided derivatives."""
    if (
        result.request_id != request.request_id
        or result.mesh_id != request.mesh.mesh_id
    ):
        raise ValueError("FEM request/result identities differ")
    if request.config.element_order != 2 or result.element_order != 2:
        raise ValueError("quadrature audit requires matching P2 element_order")
    points = _finite_array(result.quadrature_points, "quadrature_points")
    weights = _finite_array(result.quadrature_weights, "quadrature_weights")
    coordinates = _finite_array(result.displacement_dof_coordinates, "DOF coordinates")
    displacement = _finite_array(result.displacement, "displacement")
    raw = _finite_array(result.first_piola_stress, "first_piola_stress")
    recovered = _finite_array(result.recovered_first_piola_stress, "recovered stress")
    deformation = _finite_array(result.deformation_gradient, "deformation_gradient")
    _validate_result_shapes(
        request, points, weights, coordinates, displacement, raw, recovered
    )
    if deformation.shape != raw.shape:
        raise ValueError("deformation_gradient must match the quadrature tensor layout")
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
    root_mask = np.isin(cells, qoi_indices).any(axis=1)
    if not root_mask.any() or root_mask.all():
        raise ValueError("root-incident cells need a nonempty complement")
    mesh = MeshTri(nodes.T, cells.T, sort_t=False)
    quadrature = _reference_quadrature(request, points, weights)
    scalar = Basis(mesh, ElementTriP2(), quadrature=quadrature)
    vector = Basis(mesh, ElementVector(ElementTriP2()), quadrature=quadrature)
    _validate_rebuilt_layout(
        scalar, vector, points, weights, coordinates, displacement, recovered
    )
    interpolated = vector.interpolate(_pack_vector_dofs(vector, displacement))
    gradient = np.transpose(np.asarray(interpolated.grad), (2, 3, 0, 1))
    discrepancy = float(
        np.max(np.abs(gradient - (deformation[..., :2, :2] - np.eye(2))))
    )
    if discrepancy > _LAYOUT_ATOL:
        raise ValueError("rebuilt FEM gradient differs from stored F-I")
    return FemQuadratureFields(
        torch.tensor(points, dtype=torch.float64),
        torch.tensor(np.moveaxis(np.asarray(interpolated), 0, -1), dtype=torch.float64),
        torch.tensor(gradient, dtype=torch.float64),
        torch.tensor(raw[..., :2, :2], dtype=torch.float64),
        torch.tensor(weights, dtype=torch.float64),
        torch.tensor(root_mask, dtype=torch.bool),
        discrepancy,
    )


@dataclass(frozen=True, slots=True)
class QuadraturePrediction:
    u_mm: torch.Tensor
    gradient: torch.Tensor
    p_mpa: torch.Tensor


def predict_quadrature(
    model: PiGinotOperator,
    case: DogboneS13TrainingCase,
    xy_mm: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> QuadraturePrediction:
    """Use the authoritative heads and existing physical coordinate conversion."""
    _validated_operator(model)
    if (
        type(case) is not DogboneS13TrainingCase
        or case.tensors.role is not EvidenceRole.TRAIN
    ):
        raise PermissionError("quadrature prediction requires the exact TRAIN case")
    if (
        type(chunk_size) is not int
        or chunk_size <= 0
        or xy_mm.ndim != 3
        or xy_mm.shape[-1] != 2
        or xy_mm.numel() == 0
        or xy_mm.dtype is not torch.float64
        or xy_mm.device.type != "cpu"
        or not bool(torch.isfinite(xy_mm).all())
    ):
        raise ValueError(
            "quadrature points must be finite CPU float64 cell/point/xy with positive chunk size"
        )
    tensors, scales = case.tensors, _scales(case.request)
    was_training = model.training
    values: list[QuadraturePrediction] = []
    model.eval()
    try:
        with (
            torch.no_grad(),
            torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
        ):
            encoding = model.encode_boundary(
                boundary_xy=tensors.boundary_xy.unsqueeze(0),
                boundary_tags=tensors.boundary_tags.unsqueeze(0),
                boundary_normals=tensors.boundary_normals.unsqueeze(0),
                boundary_weights=tensors.boundary_weights.unsqueeze(0),
                boundary_mask=tensors.boundary_mask.unsqueeze(0),
            )
        flat = xy_mm.detach().reshape(-1, 2)
        for start in range(0, len(flat), chunk_size):
            coordinates = (
                flat[start : start + chunk_size]
                .clone()
                .unsqueeze(0)
                .requires_grad_(True)
            )
            with (
                torch.enable_grad(),
                torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
            ):
                output = model.decode_queries(
                    encoding=encoding,
                    query_xy=coordinates,
                    parameters=tensors.parameters_for(
                        model.config.parameter_names
                    ).unsqueeze(0),
                )
                u = output.u * scales.displacement_mm
                gradient = displacement_gradient(u, coordinates, _COORDINATE_CONVERSION)
            values.append(
                QuadraturePrediction(
                    u[0].detach(),
                    gradient[0].detach(),
                    output.served_stress[0].detach() * scales.stress_mpa,
                )
            )
    finally:
        model.train(was_training)
    shape = xy_mm.shape[:2]
    return QuadraturePrediction(
        torch.cat([value.u_mm for value in values]).reshape(*shape, 2),
        torch.cat([value.gradient for value in values]).reshape(*shape, 2, 2),
        torch.cat([value.p_mpa for value in values]).reshape(*shape, 2, 2),
    )


def load_pinned_report(
    path: Path, expected_sha256: str, expected_type: str
) -> dict[str, object]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("pinned report must be an absolute nonsymlink file")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("pinned report SHA-256 mismatch")
    value = strict_json_loads(payload)
    if not isinstance(value, dict) or value.get("$type") != expected_type:
        raise ValueError("pinned report canonical type mismatch")
    if value.get("role") != "TRAIN" or value.get("non_gate") is not True:
        raise PermissionError("pinned report must be TRAIN-only and non-gate")
    if rfc8785.dumps(value) + b"\n" != payload:
        raise ValueError("pinned report must be canonical JSON plus newline")
    return cast(dict[str, object], value)


def require_bindings(
    report: dict[str, object], expected: dict[str, object], label: str
) -> None:
    normalized = cast(
        dict[str, object], strict_json_loads(canonical_json_bytes(expected))
    )
    if any(report.get(name) != value for name, value in normalized.items()):
        raise ValueError(f"{label} binding mismatch")


def verify_replay(
    model: PiGinotOperator,
    trace: tuple[float, ...],
    expected_trace: object,
    expected_state_hash: object,
    label: str,
) -> None:
    if list(trace) != expected_trace or model_state_hash(model) != expected_state_hash:
        raise ValueError(f"{label} replay differs from frozen trace/state evidence")


def arm_bindings(
    model: PiGinotOperator, warmstart: PiGinotOperator
) -> dict[str, object]:
    assert_trainable_scope(model)
    return {
        "config": model.config,
        "operator_module_digest": module_digest(model),
        "initial_state_hash": model_state_hash(model),
        "frozen_state_hash": frozen_state_hash(model),
        "copied_state_names": tuple(sorted(warmstart.state_dict())),
        "new_state_names": tuple(
            sorted(set(model.state_dict()) - set(warmstart.state_dict()))
        ),
        "trainable_parameter_names": trainable_names(model),
        "trainable_parameter_count": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }


def audit_implementation_hashes() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[2]
    extra = (
        "campaigns/dogbone_v1/s13_quadrature_audit.py",
        "campaigns/dogbone_v1/s13_quadrature_metrics.py",
        "src/piginot/fem/verification_norms.py",
        "src/piginot/objectives/terms.py",
        "src/piginot/physics/plane_stress.py",
        "src/piginot/physics/neo_hookean.py",
        "src/piginot/physics/_validation.py",
    )
    return _implementation_hashes() + tuple(
        (name, hashlib.sha256((root / name).read_bytes()).hexdigest()) for name in extra
    )


@dataclass(frozen=True, slots=True)
class QuadratureAuditArm:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"
    arm_id: str
    state_hash: str
    module_digest: str
    metrics: QuadratureCaseMetrics


@dataclass(frozen=True, slots=True)
class QuadratureAuditReport:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"
    role: EvidenceRole
    non_gate: bool
    case_id: str
    plan: QuadratureAuditPlan
    plan_hash: str
    source_bindings_hash: str
    s11_evidence_file_hash: str
    s11_checkpoint_file_hash: str
    request_content_hash: str
    result_content_hash: str
    mesh_id: str
    root_qoi_vertex_indices: tuple[int, ...]
    gradient_max_absolute_difference: float
    fixed_displacement_verified: bool
    implementation_sha256: tuple[tuple[str, str], ...]
    arms: tuple[QuadratureAuditArm, ...]


def run_quadrature_audit(
    *,
    s10_run_root: Path,
    s11_checkpoint_root: Path,
    development_access_record: Path,
    v5_report_path: Path,
    capacity_report_path: Path,
    output_path: Path,
) -> QuadratureAuditReport:
    plan, capacity_plan = QuadratureAuditPlan(), StressCapacityPlan()
    validate_replay_runtime(plan)
    implementation = audit_implementation_hashes()
    s10, s11, access, _ = _preflight_paths(
        s10_run_root, s11_checkpoint_root, development_access_record, output_path
    )
    v5 = load_pinned_report(
        v5_report_path, plan.v5_sha256, "piginot.campaigns.S13OverfitDiagnosticReport"
    )
    capacity = load_pinned_report(
        capacity_report_path,
        plan.capacity_sha256,
        "piginot.campaigns.StressCapacityReport",
    )
    require_bindings(
        v5,
        {
            "case_id": plan.case_id,
            "plan_hash": capacity_plan.fine_tune_plan.content_hash,
        },
        "V5 plan",
    )
    require_bindings(
        capacity,
        {
            "plan": capacity_plan,
            "plan_hash": capacity_plan.content_hash,
            "implementation_sha256": _implementation_hashes(),
            "fixed_displacement_verified": True,
        },
        "capacity plan/code",
    )
    initialization = load_selected_s11_initialization(s11)
    source = open_selective_s10_source(s10, development_access_record=access)

    def closed() -> None:
        if source.development_opened or access.exists():
            raise RuntimeError("DEVELOPMENT must remain closed during quadrature audit")

    closed()
    case = materialize_dogbone_s13_training_case(source, initialization, plan.case_id)
    verified, _ = source.load_fit_with_collocation(plan.case_id)
    if (
        case.tensors.role is not EvidenceRole.TRAIN
        or verified.case.role is not EvidenceRole.TRAIN
    ):
        raise PermissionError("quadrature audit uses only TRAIN labels")
    if verified.request != case.request or verified.case.case_id != plan.case_id:
        raise ValueError("quadrature source case/request mismatch")
    qoi = qoi_vertex_indices(
        case.request.mesh, semantic_tag="right_arc", excluded_tags=("right_grip",)
    )
    if tuple(case.tensors.qoi_indices.tolist()) != qoi:
        raise ValueError("registered geometric QoI differs from TRAIN tensor QoI")
    if (len(case.request.mesh.nodes), len(case.request.mesh.cells), len(qoi)) != (
        plan.expected_vertices,
        plan.expected_cells,
        plan.expected_qoi_vertices,
    ):
        raise ValueError("quadrature population differs from registered case")
    reference = rebuild_fem_quadrature(case.request, verified.result, qoi)
    if reference.xy_mm.shape != (plan.expected_cells, plan.expected_points_per_cell, 2):
        raise ValueError("stored quadrature count differs from registered rule")
    sampler = _build_vertex_sampler_plan(
        capacity_plan.fine_tune_plan, plan.expected_vertices
    )
    common: dict[str, object] = {
        "source_bindings_hash": source.bindings.content_hash,
        "s11_evidence_file_hash": initialization.evidence_file_blake2b256,
    }
    require_bindings(
        v5,
        {
            **common,
            "initial_checkpoint_file_hash": initialization.checkpoint_file_blake2b256,
            "initial_model_state_hash": model_state_hash(initialization.model),
            "vertex_sampler_plan_hash": sampler.plan_hash,
        },
        "V5 input",
    )
    require_bindings(
        capacity,
        {
            **common,
            "s11_checkpoint_file_hash": initialization.checkpoint_file_blake2b256,
            "sampler_hash": sampler.plan_hash,
        },
        "capacity input",
    )
    closed()
    print("Reconstructing pinned V5 state (exact 4,000-step trace)", flush=True)
    trace = _train_one_case(
        initialization.model, case, capacity_plan.fine_tune_plan, sampler
    )
    verify_replay(
        initialization.model,
        trace,
        v5.get("optimizer_loss_trace"),
        v5.get("final_model_state_hash"),
        "V5",
    )
    warmstart = initialization.model
    warm_hash = model_state_hash(warmstart)
    require_bindings(
        capacity,
        {
            "warmstart_state_hash": warm_hash,
            "warmstart_frozen_state_hash": frozen_state_hash(warmstart),
        },
        "capacity warmstart",
    )
    prediction = predict_quadrature(
        warmstart, case, reference.xy_mm, chunk_size=plan.query_chunk_size
    )

    def measure(
        model: PiGinotOperator, predicted: QuadraturePrediction, name: str
    ) -> QuadratureAuditArm:
        metrics = summarize_quadrature_case(
            grad_model=predicted.gradient,
            grad_fem=reference.gradient,
            served_p=predicted.p_mpa,
            raw_fem_p=reference.raw_p_mpa,
            area_weights=reference.area_weights,
            root_cell_mask=reference.root_cell_mask,
            xy_mm=reference.xy_mm,
            material=_physics_material(case.request.material),
            scales=_scales(case.request),
            u_model=predicted.u_mm,
            u_fem=reference.u_mm,
        )
        print(f"Measured {name} at all stored physical quadrature points", flush=True)
        return QuadratureAuditArm(
            name, model_state_hash(model), module_digest(model), metrics
        )

    arms = [measure(warmstart, prediction, "v5")]
    records = capacity.get("arms")
    if (
        not isinstance(records, list)
        or len(records) != 2
        or not all(isinstance(record, dict) for record in records)
    ):
        raise ValueError("capacity arm records malformed")
    for enriched, record in zip((False, True), records, strict=True):
        name = "enriched" if enriched else "control"
        closed()
        model = build_stress_arm(warmstart, capacity_plan, enriched=enriched)
        require_bindings(
            record,
            {"arm_id": name, **arm_bindings(model, warmstart)},
            f"{name} migration",
        )
        print(
            f"Reconstructing pinned {name} state (exact 4,000-step trace)", flush=True
        )
        trace = _train_one_case(model, case, capacity_plan.fine_tune_plan, sampler)
        verify_replay(
            model,
            trace,
            record.get("optimizer_loss_trace"),
            record.get("final_state_hash"),
            name,
        )
        assert_trainable_scope(model)
        if frozen_state_hash(model) != frozen_state_hash(warmstart):
            raise ValueError("frozen shared state changed in replay")
        actual = predict_quadrature(
            model, case, reference.xy_mm, chunk_size=plan.query_chunk_size
        )
        if not torch.equal(actual.u_mm, prediction.u_mm) or not torch.equal(
            actual.gradient, prediction.gradient
        ):
            raise ValueError("frozen quadrature displacement/gradient changed")
        arms.append(measure(model, actual, name))
    closed()
    latest, _ = source.load_fit_with_collocation(plan.case_id)
    if latest.request != verified.request or latest.result != verified.result:
        raise RuntimeError("TRAIN request/result changed during quadrature audit")
    load_pinned_report(
        v5_report_path, plan.v5_sha256, "piginot.campaigns.S13OverfitDiagnosticReport"
    )
    load_pinned_report(
        capacity_report_path,
        plan.capacity_sha256,
        "piginot.campaigns.StressCapacityReport",
    )
    closed()
    if (
        model_state_hash(warmstart) != warm_hash
        or audit_implementation_hashes() != implementation
    ):
        raise RuntimeError(
            "immutable warmstart or diagnostic source changed during audit"
        )
    validate_replay_runtime(plan)
    return QuadratureAuditReport(
        EvidenceRole.TRAIN,
        True,
        plan.case_id,
        plan,
        plan.content_hash,
        source.bindings.content_hash,
        initialization.evidence_file_blake2b256,
        initialization.checkpoint_file_blake2b256,
        hash_spec(case.request, "content"),
        hash_spec(verified.result, "content"),
        case.request.mesh.mesh_id,
        qoi,
        reference.gradient_max_absolute_difference,
        True,
        implementation,
        tuple(arms),
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "s10-run-root",
        "s11-checkpoint-root",
        "development-access-record",
        "v5-report",
        "capacity-report",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    options = parser.parse_args(arguments)
    report = run_quadrature_audit(
        s10_run_root=options.s10_run_root,
        s11_checkpoint_root=options.s11_checkpoint_root,
        development_access_record=options.development_access_record,
        v5_report_path=options.v5_report,
        capacity_report_path=options.capacity_report,
        output_path=options.output,
    )
    _exclusive_write(options.output, canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
