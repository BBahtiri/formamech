"""TRAIN-only spatial audit of frozen S13 one-case overfit states."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast

import torch

from campaigns.dogbone_v1.s11_contingency_models import model_state_hash
from campaigns.dogbone_v1.s11_contingency_source import open_selective_s10_source
from campaigns.dogbone_v1.s13_data import materialize_dogbone_s13_training_case
from campaigns.dogbone_v1.s13_initialization import load_selected_s11_initialization
from campaigns.dogbone_v1.s13_objective import (
    _COORDINATE_CONVERSION,
    DogboneS13TrainingCase,
    _physics_material,
    _scales,
)
from campaigns.dogbone_v1.s13_overfit_diagnostic import (
    _build_vertex_sampler_plan,
    _exclusive_write,
    _preflight_paths,
    _train_one_case,
    build_s13_overfit_plan,
)
from piginot.contracts import canonical_json_bytes, strict_json_loads
from piginot.doe import EvidenceRole
from piginot.objectives import constitutive_residual, displacement_gradient
from piginot.operators import PiGinotOperator
from piginot.physics._validation import DETERMINANT_MARGIN

_CASE_ID = "candidate-002485"
_FROZEN_REPORTS = (
    (
        "v5",
        "S13-OVERFIT-BUDGET-V5.json",
        "d443f6b51ab1883fc1cc704e3094a3a408c410de0cb14979359ec5b5848b1326",
    ),
    (
        "v6",
        "S13-OVERFIT-PEAK-V6.json",
        "b8da0b8d7aa5319f0f1c17ad30c98b4ab6ef6e88b8e283f40c20a3d0a25fe596",
    ),
    (
        "v7",
        "S13-OVERFIT-P-WEIGHT-V7.json",
        "02f701ef87b02b13c9c5321e35eaf9433161cb2f0fbe1b9aef22bc5159dbb4d4",
    ),
)


@dataclass(frozen=True, slots=True)
class SpatialCaseMetrics:
    """Root and non-root stress error and constitutive consistency."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    root_vertex_count: int
    nonroot_vertex_count: int
    root_p_relative_l2: float
    nonroot_p_relative_l2: float
    root_squared_p_error_share: float
    root_constitutive_rms: float
    nonroot_constitutive_rms: float
    root_min_determinant: float
    nonroot_min_determinant: float
    fem_peak_vertex_index: int
    predicted_peak_vertex_index: int
    fem_peak_xy_mm: tuple[float, float]
    predicted_peak_xy_mm: tuple[float, float]
    fem_peak_p_frobenius: float
    predicted_peak_p_frobenius: float
    fem_peak_p_components: tuple[tuple[float, float], tuple[float, float]]
    predicted_peak_p_components: tuple[tuple[float, float], tuple[float, float]]


def _matrix(value: torch.Tensor) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(value[0, 0]), float(value[0, 1])),
        (float(value[1, 0]), float(value[1, 1])),
    )


def _xy(value: torch.Tensor) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))


def summarize_spatial_case(
    predicted_p: torch.Tensor,
    target_p: torch.Tensor,
    qoi_indices: torch.Tensor,
    query_xy: torch.Tensor,
    constitutive_residual: torch.Tensor,
    determinants: torch.Tensor,
) -> SpatialCaseMetrics:
    """Measure exact registered root-QoI vertices against their complement."""

    if (
        predicted_p.ndim != 3
        or predicted_p.shape[1:] != (2, 2)
        or predicted_p.shape != target_p.shape
        or predicted_p.shape != constitutive_residual.shape
        or query_xy.shape != (len(predicted_p), 2)
        or determinants.shape != (len(predicted_p),)
    ):
        raise ValueError("spatial audit tensors must be aligned vertex fields")
    for value in (
        predicted_p,
        target_p,
        query_xy,
        constitutive_residual,
        determinants,
    ):
        if value.dtype is not torch.float64 or value.device.type != "cpu":
            raise ValueError("spatial audit fields must be CPU float64")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("spatial audit fields must be finite")
    if qoi_indices.dtype is not torch.int64 or qoi_indices.device.type != "cpu":
        raise ValueError("qoi_indices must be CPU int64")
    if (
        qoi_indices.ndim != 1
        or qoi_indices.numel() == 0
        or qoi_indices.numel() >= len(predicted_p)
        or not bool((qoi_indices[1:] > qoi_indices[:-1]).all().item())
    ):
        raise ValueError("qoi_indices must be sorted and unique with a complement")
    if int(qoi_indices[0]) < 0 or int(qoi_indices[-1]) >= len(predicted_p):
        raise ValueError("qoi_indices lie outside the vertex field")
    if not bool((determinants > DETERMINANT_MARGIN).all().item()):
        raise ValueError("spatial audit requires positive admissible determinants")

    root_mask = torch.zeros(len(predicted_p), dtype=torch.bool)
    root_mask[qoi_indices] = True
    nonroot_mask = ~root_mask
    delta = predicted_p - target_p
    root_target_norm = torch.linalg.vector_norm(target_p[root_mask])
    nonroot_target_norm = torch.linalg.vector_norm(target_p[nonroot_mask])
    if float(root_target_norm) <= 0.0 or float(nonroot_target_norm) <= 0.0:
        raise ValueError("both FEM P subsets need positive norms")
    root_error_energy = float(delta[root_mask].square().sum())
    total_error_energy = float(delta.square().sum())
    target_peak_norms = torch.linalg.vector_norm(target_p[qoi_indices], dim=(-2, -1))
    predicted_peak_norms = torch.linalg.vector_norm(
        predicted_p[qoi_indices], dim=(-2, -1)
    )
    fem_peak_index = int(qoi_indices[int(target_peak_norms.argmax())])
    predicted_peak_index = int(qoi_indices[int(predicted_peak_norms.argmax())])

    return SpatialCaseMetrics(
        root_vertex_count=int(qoi_indices.numel()),
        nonroot_vertex_count=int(nonroot_mask.sum()),
        root_p_relative_l2=float(
            torch.linalg.vector_norm(delta[root_mask]) / root_target_norm
        ),
        nonroot_p_relative_l2=float(
            torch.linalg.vector_norm(delta[nonroot_mask]) / nonroot_target_norm
        ),
        root_squared_p_error_share=(
            root_error_energy / total_error_energy if total_error_energy else 0.0
        ),
        root_constitutive_rms=math.sqrt(
            float(constitutive_residual[root_mask].square().mean())
        ),
        nonroot_constitutive_rms=math.sqrt(
            float(constitutive_residual[nonroot_mask].square().mean())
        ),
        root_min_determinant=float(determinants[root_mask].min()),
        nonroot_min_determinant=float(determinants[nonroot_mask].min()),
        fem_peak_vertex_index=fem_peak_index,
        predicted_peak_vertex_index=predicted_peak_index,
        fem_peak_xy_mm=_xy(query_xy[fem_peak_index]),
        predicted_peak_xy_mm=_xy(query_xy[predicted_peak_index]),
        fem_peak_p_frobenius=float(target_peak_norms.max()),
        predicted_peak_p_frobenius=float(predicted_peak_norms.max()),
        fem_peak_p_components=_matrix(target_p[fem_peak_index]),
        predicted_peak_p_components=_matrix(predicted_p[predicted_peak_index]),
    )


@dataclass(frozen=True, slots=True)
class SpatialAuditArm:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    version: str
    source_report_sha256: str
    plan_hash: str
    sampler_hash: str
    final_model_state_hash: str
    metrics: SpatialCaseMetrics


@dataclass(frozen=True, slots=True)
class S13SpatialAuditReport:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    role: EvidenceRole
    non_gate: bool
    case_id: str
    source_bindings_hash: str
    arms: tuple[SpatialAuditArm, ...]


def _load_pinned_report(path: Path, expected_sha256: str) -> dict[str, object]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"frozen report SHA-256 mismatch: {path.name}")
    report = strict_json_loads(raw)
    if (
        type(report) is not dict
        or report.get("role") != EvidenceRole.TRAIN.value
        or report.get("non_gate") is not True
        or report.get("case_id") != _CASE_ID
    ):
        raise ValueError("frozen report is not the registered TRAIN diagnostic")
    return cast(dict[str, object], report)


def _vertex_prediction_and_physics(
    model: PiGinotOperator,
    case: DogboneS13TrainingCase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = case.tensors
    scales = _scales(case.request)
    coordinates = tensors.query_xy.detach().clone().unsqueeze(0)
    coordinates.requires_grad_(True)
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                output = model(
                    boundary_xy=tensors.boundary_xy.unsqueeze(0),
                    boundary_tags=tensors.boundary_tags.unsqueeze(0),
                    boundary_normals=tensors.boundary_normals.unsqueeze(0),
                    boundary_weights=tensors.boundary_weights.unsqueeze(0),
                    boundary_mask=tensors.boundary_mask.unsqueeze(0),
                    query_xy=coordinates,
                    parameters=tensors.parameters_for(
                        model.config.parameter_names
                    ).unsqueeze(0),
                )
                gradient = displacement_gradient(
                    output.u * scales.displacement_mm,
                    coordinates,
                    _COORDINATE_CONVERSION,
                )
                deformation = torch.eye(2, dtype=gradient.dtype) + gradient
                determinant = torch.linalg.det(deformation).reshape(-1)
                if not bool((determinant > DETERMINANT_MARGIN).all().item()):
                    raise ValueError("replayed vertex deformation is not admissible")
                residual = constitutive_residual(
                    output.P * scales.stress_mpa,
                    gradient,
                    _physics_material(case.request.material),
                    scales,
                )
    finally:
        model.train(was_training)
    return output.P[0].detach(), residual[0].detach(), determinant.detach()


def run_s13_spatial_audit(
    *,
    s10_run_root: Path,
    s11_checkpoint_root: Path,
    development_access_record: Path,
    artifact_root: Path,
    output_path: Path,
) -> S13SpatialAuditReport:
    """Replay sealed TRAIN arms; refuse spatial interpretation on any mismatch."""

    s10, s11, access, _ = _preflight_paths(
        s10_run_root,
        s11_checkpoint_root,
        development_access_record,
        output_path,
    )
    if not artifact_root.is_absolute() or not artifact_root.is_dir():
        raise ValueError("artifact_root must be an absolute existing directory")
    frozen_reports = tuple(
        (version, digest, _load_pinned_report(artifact_root / filename, digest))
        for version, filename, digest in _FROZEN_REPORTS
    )
    first_initialization = load_selected_s11_initialization(s11)
    source = open_selective_s10_source(s10, development_access_record=access)
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened before TRAIN spatial audit")
    case = materialize_dogbone_s13_training_case(source, first_initialization, _CASE_ID)
    if source.development_opened or case.tensors.role is not EvidenceRole.TRAIN:
        raise RuntimeError("spatial audit must use only TRAIN labels")
    if int(case.tensors.qoi_indices.numel()) != 20:
        raise ValueError("registered root QoI must contain exactly 20 vertices")

    arms: list[SpatialAuditArm] = []
    for index, (version, digest, prior) in enumerate(frozen_reports):
        plan = build_s13_overfit_plan(version=version)
        if len(case.tensors.query_xy) != plan.expected_vertex_count:
            raise ValueError("TRAIN case vertex count differs from frozen plans")
        initialization = (
            first_initialization
            if index == 0
            else load_selected_s11_initialization(s11)
        )
        sampler = _build_vertex_sampler_plan(plan, len(case.tensors.query_xy))
        if (
            prior.get("plan_hash") != plan.content_hash
            or prior.get("vertex_sampler_plan_hash") != sampler.plan_hash
            or prior.get("source_bindings_hash") != source.bindings.content_hash
            or prior.get("s11_evidence_file_hash")
            != initialization.evidence_file_blake2b256
            or prior.get("initial_checkpoint_file_hash")
            != initialization.checkpoint_file_blake2b256
            or prior.get("initial_model_state_hash")
            != model_state_hash(initialization.model)
        ):
            raise ValueError(f"{version} frozen input bindings differ")
        trace = _train_one_case(initialization.model, case, plan, sampler)
        expected_trace = prior.get("optimizer_loss_trace")
        if (
            type(expected_trace) is not list
            or len(expected_trace) != plan.optimizer_steps
            or any(type(value) not in (int, float) for value in expected_trace)
        ):
            raise ValueError(f"{version} frozen optimizer trace is invalid")
        if tuple(expected_trace) != trace or prior.get(
            "final_model_state_hash"
        ) != model_state_hash(initialization.model):
            raise ValueError(f"{version} deterministic training replay differs")
        predicted_p, constitutive, determinants = _vertex_prediction_and_physics(
            initialization.model, case
        )
        metrics = summarize_spatial_case(
            predicted_p,
            case.tensors.target_P,
            case.tensors.qoi_indices,
            case.tensors.query_xy,
            constitutive,
            determinants,
        )
        if source.development_opened:
            raise RuntimeError("DEVELOPMENT labels opened during TRAIN spatial audit")
        arms.append(
            SpatialAuditArm(
                version=version,
                source_report_sha256=digest,
                plan_hash=plan.content_hash,
                sampler_hash=sampler.plan_hash,
                final_model_state_hash=model_state_hash(initialization.model),
                metrics=metrics,
            )
        )
    return S13SpatialAuditReport(
        role=EvidenceRole.TRAIN,
        non_gate=True,
        case_id=_CASE_ID,
        source_bindings_hash=source.bindings.content_hash,
        arms=tuple(arms),
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--s11-checkpoint-root", required=True, type=Path)
    parser.add_argument("--development-access-record", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args(arguments)
    report = run_s13_spatial_audit(
        s10_run_root=options.s10_run_root,
        s11_checkpoint_root=options.s11_checkpoint_root,
        development_access_record=options.development_access_record,
        artifact_root=options.artifact_root,
        output_path=options.output,
    )
    _exclusive_write(options.output, canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = (
    "S13SpatialAuditReport",
    "SpatialAuditArm",
    "SpatialCaseMetrics",
    "main",
    "run_s13_spatial_audit",
    "summarize_spatial_case",
)
