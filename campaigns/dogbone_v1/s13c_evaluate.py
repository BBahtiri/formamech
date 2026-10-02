"""Read-only TRAIN-screen evaluation and frozen selection for S13c.

The registered checkpoint is scored with the unchanged S11/S13 metric code on
the 64 TRAIN-screen FEM cases: per-case relative L2 of the vertex displacement
(normalised by the prescribed displacement) and of the in-plane first Piola
stress (normalised by 760 MPa), and the signed error of the maximum
Frobenius stress over the registered root vertices (``right_arc`` minus
``right_grip``). The served stress is the constitutive stress of the predicted
displacement gradient. Orientation is checked at the 128 replayed interior
collocation points of all 256 TRAIN cases. DEVELOPMENT and HOLDOUT are never
opened; the S10 source records any DEVELOPMENT access durably.
"""

from __future__ import annotations

import argparse
import copy
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from campaigns.dogbone_v1.s10_evidence import DOGBONE_S10_DATASET_POLICY
from campaigns.dogbone_v1.s11_contingency import STRESS_REFERENCE_MPA
from campaigns.dogbone_v1.s11_contingency_data import (
    normalize_case_values,
    prepare_operator_case,
)
from campaigns.dogbone_v1.s11_contingency_source import (
    SelectiveS10Source,
    VerifiedS10Request,
    open_selective_s10_source,
)
from campaigns.dogbone_v1.s11_contingency_tensors import ArchitectureCaseTensors
from campaigns.dogbone_v1.s11_contingency_training import (
    calculate_architecture_screen_metrics,
)
from campaigns.dogbone_v1.s13c_data import (
    DogboneS13cGeometryCase,
    case_energy,
    geometry_case,
)
from campaigns.dogbone_v1.s13c_protocol import (
    DogboneS13cProtocol,
    build_dogbone_s13c_sampler_plan,
    build_dogbone_s13c_trainer_config,
    build_initial_operator,
    load_s13c_normalization,
)
from campaigns.dogbone_v1.s13c_trial import (
    DEVELOPMENT_ACCESS_NAME,
    RUN_BINDING_NAME,
    load_committed_s13c_protocol,
)
from piginot.contracts import ExecutionProfile, canonical_json_bytes, strict_json_loads
from piginot.datasets import SemanticBCClass
from piginot.doe import EvidenceRole
from piginot.operators import (
    EnergyFormOperator,
    OperatorOutput,
    constitutive_in_plane_stress,
)
from piginot.physics._validation import DETERMINANT_MARGIN
from piginot.training import (
    CheckpointLineage,
    NoEligibleTrialError,
    SelectionDecision,
    SelectionMetrics,
    TrainingCheckpointManifest,
    TrainingRunBinding,
    TrialResult,
    load_training_checkpoint,
    select_checkpoint,
)

_ENERGY_EVALUATION_SEED = 13_104
_ENERGY_GAP_POINTS = 12
_QUERY_CHUNK = 2_048


@dataclass(frozen=True, slots=True)
class S13cCaseDiagnostics:
    """Descriptive per-case TRAIN-screen measurements (no extra gate)."""

    case_id: str
    signed_peak_error: float
    u_relative_l2: float
    p_relative_l2: float
    root_p_relative_l2: float
    non_root_p_relative_l2: float
    predicted_peak_vertex: int
    reference_peak_vertex: int
    peak_vertex_distance_mm: float
    minimum_vertex_determinant: float
    model_strain_energy: float = math.nan
    fem_strain_energy: float = math.nan

    @property
    def relative_energy_gap(self) -> float:
        """``(Pi_model - Pi_FEM) / Pi_FEM``; both are upper bounds of the minimum."""

        return (self.model_strain_energy - self.fem_strain_energy) / (
            self.fem_strain_energy
        )


@dataclass(frozen=True, slots=True)
class S13cEvaluation:
    """Complete read-only S13c screening outcome."""

    protocol_hash: str
    run_binding: TrainingRunBinding
    manifest: TrainingCheckpointManifest
    result: TrialResult
    decision: SelectionDecision | None
    failed_guardrails: tuple[str, ...]
    diagnostics: tuple[S13cCaseDiagnostics, ...]
    invalid_case_count: int
    minimum_collocation_determinant: float
    development_opened: bool


def load_run_binding(path: Path) -> TrainingRunBinding:
    """Parse ``run-binding.json`` and require an exact canonical round trip."""

    raw = path.read_bytes()
    document = strict_json_loads(raw)
    if not isinstance(document, dict):
        raise ValueError("run binding must be a JSON object")
    values: dict[str, Any] = {k: v for k, v in document.items() if k != "$type"}
    profile_document = values.get("execution_profile")
    if not isinstance(profile_document, dict):
        raise ValueError("run binding lacks an execution profile")
    profile = ExecutionProfile(
        **{k: v for k, v in profile_document.items() if k != "$type"}
    )
    values["execution_profile"] = profile
    binding = TrainingRunBinding(**values)
    if canonical_json_bytes(binding) != raw:
        raise ValueError("run binding is not an exact canonical TrainingRunBinding")
    return binding


def _verify_binding(
    binding: TrainingRunBinding,
    protocol: DogboneS13cProtocol,
) -> None:
    """Check a run binding against the protocol without re-initialising weights.

    Initial weights are compared with the protocol's pinned hash rather than
    with a freshly initialised operator, so a checkpoint can be evaluated on a
    CPU whose random-initialisation kernels round differently.
    """

    config = build_dogbone_s13c_trainer_config(protocol)
    plan = build_dogbone_s13c_sampler_plan(protocol)
    expected = {
        "dataset_snapshot_hash": protocol.dataset_snapshot_hash,
        "public_manifest_hash": protocol.public_manifest_hash,
        "normalization_hash": protocol.normalization_hash,
        "architecture_id": protocol.architecture_id,
        "architecture_module_digest": protocol.architecture_module_digest,
        "objective_registry_hash": protocol.objective_hash,
        "trainer_config_hash": config.config_hash,
        "sampler_plan_hash": plan.plan_hash,
        "initial_weights_hash": protocol.initial_weights_hash,
        "seed": protocol.training_seed,
    }
    for name, value in expected.items():
        if getattr(binding, name) != value:
            raise ValueError(f"run binding {name} differs from the S13c protocol")
    profile = binding.execution_profile
    if (
        profile.dtype != protocol.execution_dtype
        or profile.blas_threads != protocol.execution_blas_threads
        or profile.device != "cpu"
    ):
        raise ValueError("run binding execution profile differs from the protocol")


def load_registered_checkpoint(
    protocol: DogboneS13cProtocol,
    checkpoint_path: Path,
    *,
    require_final: bool = True,
) -> tuple[EnergyFormOperator, TrainingRunBinding, TrainingCheckpointManifest]:
    """Validate the complete checkpoint against its run binding and protocol."""

    binding = load_run_binding(checkpoint_path.parent / RUN_BINDING_NAME)
    _verify_binding(binding, protocol)
    config = build_dogbone_s13c_trainer_config(protocol)
    model = build_initial_operator()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.schedule.learning_rate(0),
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.adam_weight_decay,
        foreach=False,
        fused=False,
    )
    generator_state = torch.random.get_rng_state()
    try:
        loaded = load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            binding=binding,
        )
    finally:
        torch.random.set_rng_state(generator_state)
    manifest = loaded.manifest
    if manifest.run_hash != binding.run_hash:
        raise ValueError("checkpoint run_hash differs from its run binding")
    if require_final and manifest.step != config.trial.optimizer_steps:
        raise ValueError("checkpoint is not the final registered checkpoint")
    return model, binding, manifest


def _served_vertex_output(
    model: EnergyFormOperator,
    case: DogboneS13cGeometryCase,
    query_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``u`` (mm), in-plane ``P`` (MPa) and ``det F`` at the queries."""

    u_parts: list[torch.Tensor] = []
    p_parts: list[torch.Tensor] = []
    det_parts: list[torch.Tensor] = []
    for start in range(0, query_xy.shape[0], _QUERY_CHUNK):
        chunk = query_xy[start : start + _QUERY_CHUNK].unsqueeze(0)
        u, gradient = model.displacement_and_gradient(
            query_xy=chunk,
            parameters=case.parameters(),
            frame=case.frame(),
            lifting=case.lifting(),
            create_graph=False,
        )
        deformation = torch.eye(2, dtype=gradient.dtype) + gradient[0]
        determinant = (
            deformation[:, 0, 0] * deformation[:, 1, 1]
            - deformation[:, 0, 1] * deformation[:, 1, 0]
        )
        det_parts.append(determinant)
        u_parts.append(u[0])
        if bool((determinant <= DETERMINANT_MARGIN).any().item()):
            p_parts.append(torch.full_like(gradient[0], math.nan))
        else:
            p_parts.append(constitutive_in_plane_stress(gradient[0], case.material))
    return torch.cat(u_parts), torch.cat(p_parts), torch.cat(det_parts)


def _case_diagnostics(
    tensors: ArchitectureCaseTensors,
    predicted_u: torch.Tensor,
    predicted_p: torch.Tensor,
    determinant: torch.Tensor,
) -> S13cCaseDiagnostics:
    target_u = tensors.target_u
    target_p = tensors.target_P
    root = tensors.qoi_indices
    mask = torch.ones(target_p.shape[0], dtype=torch.bool)
    mask[root] = False
    true_norms = torch.linalg.vector_norm(target_p, dim=(-2, -1))
    pred_norms = torch.linalg.vector_norm(predicted_p, dim=(-2, -1))
    true_peak = int(root[int(torch.argmax(true_norms[root]))])
    pred_peak = int(root[int(torch.argmax(pred_norms[root]))])
    distance = float(
        torch.linalg.vector_norm(
            tensors.query_xy[pred_peak] - tensors.query_xy[true_peak]
        )
    )

    def relative(prediction: torch.Tensor, target: torch.Tensor) -> float:
        return float(
            torch.linalg.vector_norm(prediction - target)
            / torch.linalg.vector_norm(target)
        )

    return S13cCaseDiagnostics(
        case_id=tensors.case_id,
        signed_peak_error=float(
            (pred_norms[root].amax() - true_norms[root].amax())
            / true_norms[root].amax()
        ),
        u_relative_l2=relative(predicted_u, target_u),
        p_relative_l2=relative(predicted_p, target_p),
        root_p_relative_l2=relative(predicted_p[root], target_p[root]),
        non_root_p_relative_l2=relative(predicted_p[mask], target_p[mask]),
        predicted_peak_vertex=pred_peak,
        reference_peak_vertex=true_peak,
        peak_vertex_distance_mm=distance,
        minimum_vertex_determinant=float(determinant.min()),
    )


def _interior_determinants(
    model: EnergyFormOperator,
    source: SelectiveS10Source,
    verified: VerifiedS10Request,
    case: DogboneS13cGeometryCase,
) -> torch.Tensor:
    collocation = source.replay_train_collocation_for_request(verified)
    interior = tuple(
        point.coordinates
        for point in collocation.points
        if point.semantic_class is SemanticBCClass.INTERIOR
    )
    if len(interior) != DOGBONE_S10_DATASET_POLICY.interior_count:
        raise ValueError("TRAIN case has an unexpected interior point count")
    coordinates = torch.tensor(interior, dtype=torch.float64)
    _, _, determinant = _served_vertex_output(model, case, coordinates)
    if not bool(torch.isfinite(determinant).all().item()):
        raise ValueError("TRAIN interior deformation determinant must be finite")
    return determinant


def evaluate_s13c_checkpoint(
    *,
    s10_run_root: Path,
    checkpoint_path: Path,
    require_final: bool = True,
) -> S13cEvaluation:
    """Score one S13c checkpoint on TRAIN evidence and apply frozen selection."""

    protocol = load_committed_s13c_protocol()
    config = build_dogbone_s13c_trainer_config(protocol)
    model, binding, manifest = load_registered_checkpoint(
        protocol, checkpoint_path, require_final=require_final
    )
    evaluation_model = copy.deepcopy(model).to(torch.float64).eval()
    run_root = checkpoint_path.parent
    source = open_selective_s10_source(
        s10_run_root,
        development_access_record=run_root / DEVELOPMENT_ACCESS_NAME,
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels must remain closed during S13c")
    if (
        source.bindings.content_hash != protocol.source_bindings_hash
        or source.fit_case_ids != protocol.fit_case_ids
        or source.train_screen_case_ids != protocol.screen_case_ids
        or source.train_snapshot.content_hash != protocol.dataset_snapshot_hash
    ):
        raise ValueError("S10 source or TRAIN population differs from the protocol")
    normalization = load_s13c_normalization()

    cases: list[ArchitectureCaseTensors] = []
    predictions: list[OperatorOutput] = []
    diagnostics: list[S13cCaseDiagnostics] = []
    energies: list[float] = []
    invalid = 0
    invalid_cases = 0
    minimum_determinant = math.inf
    generator = torch.Generator().manual_seed(_ENERGY_EVALUATION_SEED)
    for verified in source.iter_train_screen():
        if verified.case.role is not EvidenceRole.TRAIN:
            raise PermissionError("non-TRAIN evidence cannot enter S13c screening")
        tensors = normalize_case_values(
            prepare_operator_case(verified.case, verified.request, verified.result),
            normalization,
        )
        request = VerifiedS10Request(verified.case, verified.request)
        geometry = geometry_case(request, normalization, dtype=torch.float64)
        u, stress, determinant = _served_vertex_output(
            evaluation_model, geometry, tensors.query_xy
        )
        if not bool(torch.isfinite(stress).all().item()):
            raise ValueError(f"{tensors.case_id} has inadmissible vertex deformation")
        predicted_u = u / geometry.prescribed_displacement_mm
        predicted_p = stress / STRESS_REFERENCE_MPA
        cases.append(tensors)
        predictions.append(
            OperatorOutput(predicted_u.unsqueeze(0), predicted_p.unsqueeze(0))
        )
        fine = case_energy(
            evaluation_model,
            geometry,
            replace(protocol.objective.energy, points_per_cell=_ENERGY_GAP_POINTS),
            generator=generator,
            create_graph=False,
        )
        diagnostics.append(
            replace(
                _case_diagnostics(tensors, predicted_u, predicted_p, determinant),
                model_strain_energy=float(fine.energy)
                * float(geometry.material.properties["elastic_modulus"].value)
                * geometry.domain.area,
                fem_strain_energy=float(verified.result.total_strain_energy),
            )
        )
        interior = _interior_determinants(evaluation_model, source, request, geometry)
        count = int((interior <= DETERMINANT_MARGIN).sum().item())
        invalid += count
        invalid_cases += int(count > 0)
        minimum_determinant = min(minimum_determinant, float(interior.min()))
        energies.append(
            float(
                case_energy(
                    evaluation_model,
                    geometry,
                    protocol.objective.energy,
                    generator=generator,
                    create_graph=False,
                ).total
            )
        )
    for verified_request in source.iter_train_geometry(protocol.fit_case_ids):
        geometry = geometry_case(verified_request, normalization, dtype=torch.float64)
        interior = _interior_determinants(
            evaluation_model, source, verified_request, geometry
        )
        count = int((interior <= DETERMINANT_MARGIN).sum().item())
        invalid += count
        invalid_cases += int(count > 0)
        minimum_determinant = min(minimum_determinant, float(interior.min()))
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during S13c screening")

    accuracy = calculate_architecture_screen_metrics(
        config.trial.trial_id,
        tuple(cases),
        tuple(predictions),
        trainable_parameter_count=sum(
            parameter.numel() for parameter in model.parameters()
        ),
        elapsed_seconds=0.0,
    )
    metrics = SelectionMetrics(
        median_absolute_peak_error=accuracy.median_absolute_peak_error,
        p90_absolute_peak_error=accuracy.p90_absolute_peak_error,
        median_signed_peak_error=accuracy.median_signed_peak_error,
        p_relative_l2=accuracy.median_p_relative_l2,
        u_relative_l2=accuracy.median_u_relative_l2,
        hybrid_loss=statistics.fmean(energies),
        invalid_collocation_points=invalid,
    )
    result = TrialResult(
        lineage=CheckpointLineage(
            trial_id=config.trial.trial_id,
            run_hash=manifest.run_hash,
            dataset_snapshot_hash=manifest.dataset_snapshot_hash,
            execution_profile_hash=manifest.execution_profile_hash,
            weights_hash=manifest.model_export_hash,
            model_state_hash=manifest.model_state_hash,
            checkpoint_hash=manifest.manifest_hash,
        ),
        role=EvidenceRole.TRAIN,
        metrics=metrics,
    )
    try:
        decision: SelectionDecision | None = select_checkpoint(
            (result,), policy=protocol.selection_policy
        )
    except NoEligibleTrialError:
        decision = None
    return S13cEvaluation(
        protocol_hash=protocol.protocol_hash,
        run_binding=binding,
        manifest=manifest,
        result=result,
        decision=decision,
        failed_guardrails=failed_guardrails(protocol, metrics),
        diagnostics=tuple(diagnostics),
        invalid_case_count=invalid_cases,
        minimum_collocation_determinant=minimum_determinant,
        development_opened=source.development_opened,
    )


def failed_guardrails(
    protocol: DogboneS13cProtocol,
    metrics: SelectionMetrics,
) -> tuple[str, ...]:
    """Name every frozen guardrail that ``metrics`` violates."""

    guardrails = protocol.selection_policy.guardrails
    limit = 1.0 + guardrails.maximum_field_relative_regression
    underprediction_limit = (1.0 - guardrails.minimum_underprediction_reduction) * max(
        0.0, -guardrails.incumbent_median_signed_peak_error
    )
    checks = (
        ("invalid_collocation_points", metrics.invalid_collocation_points == 0),
        (
            "median_signed_peak_error",
            max(0.0, -metrics.median_signed_peak_error) <= underprediction_limit,
        ),
        (
            "median_absolute_peak_error",
            metrics.median_absolute_peak_error
            <= guardrails.incumbent_median_absolute_peak_error,
        ),
        (
            "p90_absolute_peak_error",
            metrics.p90_absolute_peak_error
            <= guardrails.incumbent_p90_absolute_peak_error,
        ),
        (
            "u_relative_l2",
            metrics.u_relative_l2 <= limit * guardrails.incumbent_u_relative_l2,
        ),
        (
            "p_relative_l2",
            metrics.p_relative_l2 <= limit * guardrails.incumbent_p_relative_l2,
        ),
    )
    return tuple(name for name, passed in checks if not passed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the registered S13c checkpoint on TRAIN-screen only."
    )
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    evaluation = evaluate_s13c_checkpoint(
        s10_run_root=options.s10_run_root,
        checkpoint_path=options.checkpoint_path,
    )
    print(canonical_json_bytes(evaluation.result).decode("utf-8"), flush=True)
    print(
        "selected"
        if evaluation.decision is not None
        else "NoEligibleTrialError: " + ",".join(evaluation.failed_guardrails),
        flush=True,
    )
    return 0 if evaluation.decision is not None else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = (
    "S13cCaseDiagnostics",
    "S13cEvaluation",
    "build_parser",
    "evaluate_s13c_checkpoint",
    "failed_guardrails",
    "load_registered_checkpoint",
    "load_run_binding",
    "main",
)
