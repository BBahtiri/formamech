"""Frozen protocol reproduction and fresh S1 confirmation for iteration 8."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from experiments.spike.confirm import (
    CONFIRMATION_GEOMETRY_SEED,
    CONFIRMATION_STATISTICS_SEED,
    build_fresh_campaign_manifest,
    load_final_checkpoint,
    run_confirmation,
    save_final_checkpoint,
)
from experiments.spike.geometry import (
    SpikeMesh,
    generate_geometry,
    sample_geometry_cases,
)
from experiments.spike.run import (
    CAMPAIGN_SEED,
    GEOMETRY_BATCH_SIZE,
    GRAD_CLIP_NORM,
    ITERATION3_SCREEN_SHA256,
    NO_SERVED_STRESS_PHYSICS,
    PHYSICS_QUERY_COUNT,
    SPLIT_SEED,
    STRESS_HEAD_CONDITIONING,
    STRESS_HEAD_OUTPUT_SCALE,
    SUPERVISED_QUERY_COUNT,
    UPSTREAM_APPLIED_DISPLACEMENT,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    _fem_content_hash,
    _file_sha256,
    _load_upstream_checkpoint,
    _lumped_nodal_areas,
    _new_pretrained_model,
    _solve_with_cache,
    _stress_magnitude,
    _tensor_dataset,
    _train_fixed_epochs,
    _weighted_squared_norm,
    _write_evaluation_artifacts,
    split_by_geometry,
    write_json_strict,
)

REPRODUCTION_MODEL_SEED = 314159
REPRODUCTION_SAMPLER_SEED = 314179
REPRODUCTION_LEARNING_RATE = 1e-3
REPRODUCTION_EPOCHS = 1500
REPRODUCTION_MAX_SECONDS = 7200.0

ITERATION8_ARTIFACT_DIRECTORY = "iteration8"
REPRODUCTION_CHECKPOINT_NAME = "iteration3_protocol_reproduction.pt"
REPRODUCTION_REPORT_NAME = "iteration3_protocol_reproduction.json"
PREREGISTRATION_NAME = "iteration8_preregistration.json"
CORE_CONFIRMATION_REPORT_NAME = "iteration8_confirmation_core.json"
CONFIRMATION_REPORT_NAME = "iteration8_confirmation.json"


def select_reproduction_training_cases(
    historical_cases: Sequence[SpikeMesh],
) -> tuple[SpikeMesh, ...]:
    """Return the original deterministic 20-per-family S1 training split."""
    split = split_by_geometry(historical_cases, master_seed=SPLIT_SEED)
    if len(split.train) != 40:
        raise ValueError("iteration-3 reproduction requires exactly 40 training cases")
    return split.train


def build_iteration8_preregistration(
    *,
    checkpoint_sha256: str,
    checkpoint_provenance: Mapping[str, object],
    fresh_campaign: object,
    fem_content_sha256: str,
    created_utc: str,
) -> Mapping[str, Any]:
    """Build the immutable declaration written before confirmation FEM solves."""
    if len(checkpoint_sha256) != 64 or len(fem_content_sha256) != 64:
        raise ValueError("checkpoint and FEM identities must be SHA-256 digests")
    manifest = getattr(fresh_campaign, "manifest", None)
    manifest_sha256 = getattr(fresh_campaign, "sha256", None)
    if not isinstance(manifest, Mapping) or not isinstance(manifest_sha256, str):
        raise ValueError("fresh_campaign must expose manifest and sha256")
    return {
        "schema_version": "iteration8-preregistration-v1",
        "step": "S1",
        "role": "LEGACY",
        "no_release_evidence": True,
        "created_utc": created_utc,
        "created_before_fem_solves": True,
        "no_tuning": True,
        "frozen_model": {
            "sha256": checkpoint_sha256,
            "provenance": dict(checkpoint_provenance),
        },
        "fresh_campaign": {
            "manifest_sha256": manifest_sha256,
            "manifest": dict(manifest),
        },
        "fem": {
            "content_sha256": fem_content_sha256,
            "cache_namespace": "confirm-v1",
            "applied_displacement_mm": 1.0,
        },
        "statistics": {
            "bootstrap_seed": 2026081302,
            "whole_geometry_resamples": 10_000,
        },
        "gates": {
            "minimum_converged_per_family": 24,
            "minimum_converged_total": 48,
            "family_relative_l2_P_max": 0.15,
            "family_relative_l2_P_bootstrap_upper_max": 0.15,
            "family_fraction_P_within_15_percent_min": 0.80,
            "family_wilson_lower_fraction_min": 0.60,
            "family_relative_l2_u_max": 0.075,
            "family_per_case_P_90th_percentile_max": 0.20,
            "family_worst_peak_error_max": 0.60,
            "family_mean_inference_seconds_max": 0.05,
        },
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identities_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def _update_array_digest(digest: Any, name: str, array: np.ndarray) -> None:
    values = np.ascontiguousarray(array)
    digest.update(name.encode())
    digest.update(str(values.dtype).encode())
    digest.update(json.dumps(values.shape).encode())
    digest.update(values.tobytes())


def _training_data_sha256(
    training_pairs: Sequence[tuple[SpikeMesh, object]],
) -> str:
    digest = hashlib.sha256()
    for mesh, result in sorted(training_pairs, key=lambda pair: pair[0].geometry_id):
        digest.update(mesh.geometry_id.encode())
        digest.update(mesh.mesh_id.encode())
        digest.update(str(mesh.seed).encode())
        digest.update(
            json.dumps(mesh.parameters, sort_keys=True, separators=(",", ":")).encode()
        )
        _update_array_digest(digest, "points", mesh.points)
        _update_array_digest(digest, "cells", mesh.cells)
        displacement = getattr(result, "displacement", None)
        stress = getattr(result, "first_piola", None)
        if not isinstance(displacement, np.ndarray) or not isinstance(
            stress, np.ndarray
        ):
            raise ValueError(
                "reproduction training data must contain converged FEM fields"
            )
        _update_array_digest(digest, "displacement", displacement)
        _update_array_digest(digest, "first_piola", stress)
    return digest.hexdigest()


def _tensor_state_sha256(model: object) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):  # type: ignore[union-attr]
        values = tensor.detach().cpu().contiguous().numpy()
        _update_array_digest(digest, name, values)
    return digest.hexdigest()


def _probe_output_sha256(model: object, mesh: SpikeMesh) -> str:
    import torch

    was_training = bool(model.training)  # type: ignore[union-attr]
    model.eval()  # type: ignore[union-attr]
    try:
        with torch.no_grad():
            points = torch.from_numpy(mesh.points.astype(np.float32))[None]
            boundary = torch.from_numpy(mesh.boundary_points.astype(np.float32))[None]
            x_max = torch.tensor([float(np.max(mesh.points[:, 0]))])
            y_max = torch.tensor([float(np.max(mesh.points[:, 1]))])
            u_delta = torch.full_like(x_max, UPSTREAM_APPLIED_DISPLACEMENT)
            output = model(points, boundary, u_delta, x_max, y_max)
    finally:
        model.train(was_training)  # type: ignore[union-attr]
    digest = hashlib.sha256()
    _update_array_digest(digest, "u", output.u.detach().cpu().numpy())
    _update_array_digest(
        digest,
        "P_served",
        output.P_served.detach().cpu().numpy(),
    )
    return digest.hexdigest()


def _source_hashes(repo_root: Path) -> Mapping[str, str]:
    paths = (
        "experiments/spike/model.py",
        "experiments/spike/run.py",
        "experiments/spike/confirm.py",
        "experiments/spike/iteration8.py",
        "experiments/spike/iteration8_reproduction.py",
    )
    return {relative: _file_sha256(repo_root / relative) for relative in paths}


def _environment() -> Mapping[str, object]:
    import scipy
    import skfem
    import torch

    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_fem": skfem.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "dtype": "float32",
    }


def _artifact_root(repo_root: Path) -> Path:
    return (
        repo_root
        / "experiments"
        / "spike"
        / "artifacts"
        / ITERATION8_ARTIFACT_DIRECTORY
    )


def reproduce_iteration3_protocol(
    *,
    repo_root: Path,
    upstream_checkpoint_path: Path,
    artifact_root: Path | None = None,
) -> Mapping[str, Any]:
    """Train and freeze a semantic reproduction of the selected all-40 protocol."""
    import torch

    root = artifact_root or _artifact_root(repo_root)
    checkpoint_path = root / "checkpoints" / REPRODUCTION_CHECKPOINT_NAME
    report_path = root / REPRODUCTION_REPORT_NAME
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError(
            "iteration-3 protocol reproduction artifact already exists"
        )

    fem_hash = _fem_content_hash(repo_root)
    historical = sample_geometry_cases(
        master_seed=CAMPAIGN_SEED,
        count_per_family=30,
        nx=32,
        ny=10,
    )
    training_cases = select_reproduction_training_cases(historical)
    cache_dir = Path(tempfile.gettempdir()) / "pi-ginot-s1-legacy-cache-019fe6e9"
    training_pairs = tuple(
        (mesh, _solve_with_cache(mesh, cache_dir, fem_hash)) for mesh in training_cases
    )
    failed = [
        mesh.geometry_id
        for mesh, result in training_pairs
        if getattr(result, "converged", None) is not True
    ]
    if failed:
        raise RuntimeError(f"reproduction training FEM failures: {failed}")

    checkpoint = _load_upstream_checkpoint(upstream_checkpoint_path)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.manual_seed(REPRODUCTION_MODEL_SEED)
    model = _new_pretrained_model(checkpoint.state_dict)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    stress_parameters = sum(
        parameter.numel() for parameter in model.p_head.parameters()
    )
    if (total_parameters, stress_parameters) != (614_939, 4_484):
        raise RuntimeError("iteration-3 reproduction parameter identity changed")

    training_started = perf_counter()
    final_loss = _train_fixed_epochs(
        model,
        _tensor_dataset(training_pairs),
        range(len(training_pairs)),
        learning_rate=REPRODUCTION_LEARNING_RATE,
        epochs=REPRODUCTION_EPOCHS,
        random_seed=REPRODUCTION_SAMPLER_SEED,
        label="iteration3 protocol reproduction",
        peak_supervised_p_weight=0.0,
        served_stress_physics=NO_SERVED_STRESS_PHYSICS,
        stress_supervision="nodal_patch",
        max_seconds=REPRODUCTION_MAX_SECONDS,
    )
    training_seconds = perf_counter() - training_started
    if not math.isfinite(final_loss):
        raise RuntimeError("reproduction finished with a non-finite loss")
    if any(
        not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    ):
        raise RuntimeError("reproduction produced non-finite model parameters")

    ordered_ids = [mesh.geometry_id for mesh in training_cases]
    probe_mesh = min(training_cases, key=lambda mesh: mesh.geometry_id)
    probe_sha = _probe_output_sha256(model, probe_mesh)
    created_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    provenance: dict[str, object] = {
        "schema_version": "iteration3-protocol-reproduction-v1",
        "role": "LEGACY",
        "no_release_evidence": True,
        "configuration_alias": "iteration3_joint_adam",
        "reproduction_kind": "semantic_protocol_reproduction",
        "historical_checkpoint_missing": True,
        "architecture": "conditioned_mlp",
        "stress_head_conditioning": list(STRESS_HEAD_CONDITIONING),
        "stress_head_output_scale": STRESS_HEAD_OUTPUT_SCALE,
        "served_stress": "P_served = P_predicted",
        "optimizer": "Adam",
        "learning_rate": REPRODUCTION_LEARNING_RATE,
        "epochs": REPRODUCTION_EPOCHS,
        "model_seed": REPRODUCTION_MODEL_SEED,
        "sampler_seed": REPRODUCTION_SAMPLER_SEED,
        "batch_size": GEOMETRY_BATCH_SIZE,
        "supervised_query_count": SUPERVISED_QUERY_COUNT,
        "physics_query_count": PHYSICS_QUERY_COUNT,
        "gradient_clip_norm": GRAD_CLIP_NORM,
        "loss_weights": {
            "supervised_u": 1.0,
            "supervised_P": 1.0,
            "equilibrium_on_P_served": 1.0,
            "constitutive_in_plane": 1.0,
            "peak_supervised_P": 0.0,
            "traction_free_on_P_served": 0.0,
            "partial_traction_on_P_served": 0.0,
            "section_resultant_on_P_served": 0.0,
        },
        "objective": "joint_u_P_equilibrium_constitutive",
        "training_geometry_ids": ordered_ids,
        "training_geometry_digest": _identities_sha256(ordered_ids),
        "training_mesh_digest": _identities_sha256(
            [mesh.mesh_id for mesh in training_cases]
        ),
        "training_data_sha256": _training_data_sha256(training_pairs),
        "fem_content_sha256": fem_hash,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_checkpoint_sha256": checkpoint.sha256,
        "upstream_checkpoint_epoch": checkpoint.epoch,
        "sealed_iteration3_screen_sha256": ITERATION3_SCREEN_SHA256,
        "source_sha256": dict(_source_hashes(repo_root)),
        "model_parameter_count": total_parameters,
        "stress_head_parameter_count": stress_parameters,
        "final_hybrid_loss": final_loss,
        "training_seconds": training_seconds,
        "probe_geometry_id": probe_mesh.geometry_id,
        "probe_output_sha256": probe_sha,
        "created_utc": created_utc,
        "environment": dict(_environment()),
    }
    checkpoint_sha = save_final_checkpoint(
        checkpoint_path,
        model,
        provenance=provenance,
    )
    restored = load_final_checkpoint(
        checkpoint_path,
        expected_sha256=checkpoint_sha,
        expected_provenance=provenance,
    )
    restored_probe_sha = _probe_output_sha256(restored, probe_mesh)
    if restored_probe_sha != probe_sha:
        raise RuntimeError("saved reproduction does not preserve probe outputs")
    state_sha = _tensor_state_sha256(restored)
    report: dict[str, Any] = {
        "schema_version": "iteration3-protocol-reproduction-report-v1",
        "step": "S1",
        "role": "LEGACY",
        "no_release_evidence": True,
        "decision": "frozen semantic reproduction; not a recovered historical checkpoint",
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha,
            "tensor_state_sha256": state_sha,
            "provenance": provenance,
            "reload_probe_output_sha256": restored_probe_sha,
        },
    }
    write_json_strict(report_path, report)
    print(f"ITERATION8_REPRODUCTION_WROTE {report_path}", flush=True)
    return report


def _evaluate_confirmation_pairs(
    model: object,
    pairs: Sequence[tuple[SpikeMesh, object]],
) -> tuple[Mapping[str, object], ...]:
    import torch

    records: list[Mapping[str, object]] = []
    for mesh, result in pairs:
        target_u = np.asarray(result.displacement, dtype=np.float64)
        target_p = np.asarray(result.first_piola, dtype=np.float64)
        points = torch.from_numpy(mesh.points.astype(np.float32))[None]
        boundary = torch.from_numpy(mesh.boundary_points.astype(np.float32))[None]
        x_max = torch.tensor([float(np.max(mesh.points[:, 0]))])
        y_max = torch.tensor([float(np.max(mesh.points[:, 1]))])
        u_delta = torch.full_like(x_max, UPSTREAM_APPLIED_DISPLACEMENT)
        model(points, boundary, u_delta, x_max, y_max)
        times: list[float] = []
        for _ in range(5):
            started = perf_counter()
            output = model(points, boundary, u_delta, x_max, y_max)
            times.append(perf_counter() - started)
        predicted_u = output.u[0].numpy().astype(np.float64)
        predicted_p = output.P_served[0].numpy().astype(np.float64)
        weights = _lumped_nodal_areas(mesh.points, mesh.cells)
        u_error = _weighted_squared_norm(predicted_u - target_u, weights)
        u_reference = _weighted_squared_norm(target_u, weights)
        p_error = _weighted_squared_norm(predicted_p - target_p, weights)
        p_reference = _weighted_squared_norm(target_p, weights)
        fem_peak = float(np.max(_stress_magnitude(target_p[mesh.qoi_node_ids])))
        predicted_peak = float(
            np.max(_stress_magnitude(predicted_p[mesh.qoi_node_ids]))
        )
        records.append(
            {
                "geometry_id": mesh.geometry_id,
                "mesh_id": mesh.mesh_id,
                "family": mesh.family,
                "rng_seed": mesh.seed,
                "parameters": dict(mesh.parameters),
                "P_squared_error": p_error,
                "P_squared_reference": p_reference,
                "u_squared_error": u_error,
                "u_squared_reference": u_reference,
                "relative_l2_P": math.sqrt(p_error / max(p_reference, 1e-24)),
                "relative_l2_u": math.sqrt(u_error / max(u_reference, 1e-24)),
                "fem_peak_stress_MPa": fem_peak,
                "predicted_peak_stress_MPa": predicted_peak,
                "relative_peak_stress_error": abs(predicted_peak - fem_peak)
                / max(fem_peak, 1e-12),
                "inference_seconds": float(np.mean(times)),
                "fem_solve_seconds": float(result.solve_seconds),
                "fem_newton_iterations": int(result.newton_iterations),
            }
        )
    return tuple(records)


def _write_confirmation_fields(
    model: object,
    pairs: Sequence[tuple[SpikeMesh, object]],
    artifact_dir: Path,
) -> Mapping[str, object]:
    metadata = dict(_write_evaluation_artifacts(model, pairs, artifact_dir))
    geometry_ids = sorted(mesh.geometry_id for mesh, _result in pairs)
    metadata.update(
        {
            "prediction_geometry_ids": geometry_ids,
            "plot_geometry_ids": geometry_ids,
        }
    )
    return metadata


def _confirmation_regularity(
    *,
    cases: Sequence[SpikeMesh],
    cache_dir: Path,
    fem_content_sha256: str,
) -> Mapping[str, object]:
    samples: list[Mapping[str, object]] = []
    all_converged = True
    all_last_changes_within_limit = True
    selected_ids: list[str] = []
    for family in ("dogbone", "centre_notched"):
        family_cases = tuple(case for case in cases if case.family == family)
        for family_index in (0, 12, 24):
            source = family_cases[family_index]
            selected_ids.append(source.geometry_id)
            levels: list[Mapping[str, object]] = []
            peaks: list[float] = []
            case_converged = True
            for nx, ny in ((16, 5), (32, 10), (64, 20)):
                mesh = generate_geometry(
                    source.family,
                    source.parameters,
                    source.seed,
                    nx=nx,
                    ny=ny,
                )
                result = _solve_with_cache(mesh, cache_dir, fem_content_sha256)
                converged = getattr(result, "converged", None) is True
                case_converged = case_converged and converged
                peak = None
                if converged:
                    stress = np.asarray(result.first_piola)
                    peak = float(np.max(_stress_magnitude(stress[mesh.qoi_node_ids])))
                    peaks.append(peak)
                levels.append(
                    {
                        "nx": nx,
                        "ny": ny,
                        "mesh_id": mesh.mesh_id,
                        "converged": converged,
                        "failure_reason": getattr(result, "failure_reason", None),
                        "peak_stress_MPa": peak,
                    }
                )
            last_change = None
            if case_converged:
                last_change = abs(peaks[-1] - peaks[-2]) / max(abs(peaks[-1]), 1e-12)
                all_last_changes_within_limit = (
                    all_last_changes_within_limit and last_change <= 0.10
                )
            all_converged = all_converged and case_converged
            samples.append(
                {
                    "geometry_id": source.geometry_id,
                    "family": source.family,
                    "family_index": family_index,
                    "levels": levels,
                    "relative_change_32_to_64": last_change,
                }
            )
    return {
        "selected_geometry_ids": selected_ids,
        "relative_change_threshold": 0.10,
        "all_converged": all_converged,
        "all_last_changes_within_limit": all_last_changes_within_limit,
        "well_posed": all_converged and all_last_changes_within_limit,
        "samples": samples,
    }


def _load_reproduction(report_path: Path) -> tuple[Path, str, Mapping[str, object]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("reproduction report has no checkpoint")
    path = Path(str(checkpoint.get("path", "")))
    sha256 = str(checkpoint.get("sha256", ""))
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("reproduction checkpoint has no provenance")
    return path, sha256, provenance


def _apply_peak_regularity_decision(
    report: dict[str, Any],
    regularity: Mapping[str, object],
) -> None:
    """Make the peak gate non-binding when mesh refinement is inconclusive."""
    if regularity.get("well_posed") is True:
        return
    gates = report.get("gates")
    decision = report.get("decision")
    if not isinstance(gates, dict) or not isinstance(decision, dict):
        raise ValueError("confirmation report is missing decision gates")
    family_gates = gates.get("families")
    if not isinstance(family_gates, Mapping):
        raise ValueError("confirmation report is missing family gates")
    binding_names = (
        "point_estimate_pass",
        "pass_fraction_gate",
        "wilson_lower_gate",
        "displacement_gate",
        "stress_p90_gate",
        "inference_time_gate",
    )
    all_other_pass = True
    for family, values in family_gates.items():
        if not isinstance(values, dict):
            raise ValueError(f"confirmation gates for {family} must be mutable")
        family_pass = all(values.get(name) is True for name in binding_names)
        values["peak_stress_gate"] = "NOT_RUN"
        values["all_non_bootstrap_gates_pass"] = family_pass
        all_other_pass = all_other_pass and family_pass
    gates["all_other_gates_pass"] = all_other_pass
    report["status"] = "INCONCLUSIVE" if all_other_pass else "NOT_CONFIRMED"
    decision["status"] = report["status"]
    decision["regularity_reason"] = (
        "peak-stress mesh regularity did not satisfy the preregistered 10% gate; "
        "peak_stress_gate is NOT_RUN"
    )


def run_iteration8_confirmation(
    *,
    repo_root: Path,
    reproduction_report_path: Path,
    artifact_root: Path | None = None,
) -> Mapping[str, Any]:
    """Consume a fresh 50-case FEM set once with the frozen reproduction."""
    from experiments.spike.fem import solve_uniaxial_square

    root = artifact_root or _artifact_root(repo_root)
    report_path = root / CONFIRMATION_REPORT_NAME
    core_report_path = root / CORE_CONFIRMATION_REPORT_NAME
    preregistration_path = root / PREREGISTRATION_NAME
    if (
        report_path.exists()
        or core_report_path.exists()
        or preregistration_path.exists()
    ):
        raise FileExistsError("iteration-8 confirmation has already been started")
    checkpoint_path, checkpoint_sha, provenance = _load_reproduction(
        reproduction_report_path
    )
    model = load_final_checkpoint(
        checkpoint_path,
        expected_sha256=checkpoint_sha,
        expected_provenance=provenance,
    )
    historical = sample_geometry_cases(
        master_seed=CAMPAIGN_SEED,
        count_per_family=30,
        nx=32,
        ny=10,
    )
    fresh = build_fresh_campaign_manifest(
        historical_cases=historical,
        master_seed=CONFIRMATION_GEOMETRY_SEED,
        nx=32,
        ny=10,
    )
    fem_hash = _fem_content_hash(repo_root)
    preregistration = dict(
        build_iteration8_preregistration(
            checkpoint_sha256=checkpoint_sha,
            checkpoint_provenance=provenance,
            fresh_campaign=fresh,
            fem_content_sha256=fem_hash,
            created_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
    )
    preregistration["regularity_geometry_ids"] = [
        family_cases[index].geometry_id
        for family in ("dogbone", "centre_notched")
        for family_cases in [
            tuple(case for case in fresh.cases if case.family == family)
        ]
        for index in (0, 12, 24)
    ]
    write_json_strict(preregistration_path, preregistration)
    preregistration_sha = _file_sha256(preregistration_path)

    sanity = solve_uniaxial_square(axial_stretch=1.20, nx=6, ny=6)
    if not sanity.converged or sanity.max_error > 1e-6:
        raise RuntimeError("confirmation FEM sanity gate failed")
    cache_dir = (
        Path(tempfile.gettempdir())
        / f"pi-ginot-s1-confirm-v1-{CONFIRMATION_GEOMETRY_SEED}"
    )

    def solve(mesh: SpikeMesh) -> object:
        return _solve_with_cache(mesh, cache_dir, fem_hash)

    field_artifact_dir = root / "confirmation_fields"
    report = run_confirmation(
        model=model,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_provenance=provenance,
        cases=fresh.cases,
        historical_cases=historical,
        campaign_manifest=fresh.manifest,
        fem_solver=solve,
        evaluator=_evaluate_confirmation_pairs,
        artifact_writer=_write_confirmation_fields,
        artifact_dir=field_artifact_dir,
        report_path=core_report_path,
        bootstrap_seed=CONFIRMATION_STATISTICS_SEED,
        bootstrap_resamples=10_000,
    )
    regularity = _confirmation_regularity(
        cases=fresh.cases,
        cache_dir=cache_dir,
        fem_content_sha256=fem_hash,
    )
    report["created_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    report["preregistration"] = {
        "path": str(preregistration_path.resolve()),
        "sha256": preregistration_sha,
        "created_before_fem_solves": True,
    }
    report["core_evaluation"] = {
        "path": str(core_report_path.resolve()),
        "sha256": _file_sha256(core_report_path),
    }
    report["fem_content_sha256"] = fem_hash
    report["fem_sanity"] = asdict(sanity)
    report["peak_regularity"] = regularity
    _apply_peak_regularity_decision(report, regularity)
    from experiments.spike.iteration8_plots import write_summary_plots

    summary_plots = write_summary_plots(
        report["per_case_metrics"],
        report["statistics"],
        regularity,
        root / "summary_plots",
    )
    report["summary_plots"] = summary_plots
    write_json_strict(report_path, report)
    print(f"ITERATION8_CONFIRMATION_WROTE {report_path}", flush=True)
    return report


def run_iteration8(
    *,
    repo_root: Path,
    upstream_checkpoint_path: Path,
) -> Mapping[str, Any]:
    """Run the reproduction, freeze it, then consume the fresh confirmation set."""
    root = _artifact_root(repo_root)
    reproduction_path = root / REPRODUCTION_REPORT_NAME
    reproduction = reproduce_iteration3_protocol(
        repo_root=repo_root,
        upstream_checkpoint_path=upstream_checkpoint_path,
        artifact_root=root,
    )
    confirmation = run_iteration8_confirmation(
        repo_root=repo_root,
        reproduction_report_path=reproduction_path,
        artifact_root=root,
    )
    return {"reproduction": reproduction, "confirmation": confirmation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-checkpoint",
        type=Path,
        default=(
            Path(tempfile.gettempdir())
            / "pi-ginot-upstream-9f679414-checkpoint"
            / "best.pt"
        ),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    result = run_iteration8(
        repo_root=repo_root,
        upstream_checkpoint_path=args.upstream_checkpoint,
    )
    print(
        json.dumps(
            {
                "status": result["confirmation"]["status"],
                "checkpoint_sha256": result["reproduction"]["checkpoint"]["sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
