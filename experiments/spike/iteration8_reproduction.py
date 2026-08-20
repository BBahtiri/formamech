"""Resumable reproduction runner for the frozen Iteration 8 incumbent."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from experiments.spike.confirm import load_final_checkpoint, save_final_checkpoint
from experiments.spike.geometry import SpikeMesh, sample_geometry_cases
from experiments.spike.iteration8 import (
    REPRODUCTION_CHECKPOINT_NAME,
    REPRODUCTION_EPOCHS,
    REPRODUCTION_LEARNING_RATE,
    REPRODUCTION_MODEL_SEED,
    REPRODUCTION_REPORT_NAME,
    REPRODUCTION_SAMPLER_SEED,
    _artifact_root,
    _environment,
    _fem_content_hash,
    _identities_sha256,
    _load_upstream_checkpoint,
    _new_pretrained_model,
    _probe_output_sha256,
    _solve_with_cache,
    _source_hashes,
    _tensor_dataset,
    _tensor_state_sha256,
    _training_data_sha256,
    select_reproduction_training_cases,
    write_json_strict,
)
from experiments.spike.run import (
    CAMPAIGN_SEED,
    GEOMETRY_BATCH_SIZE,
    GRAD_CLIP_NORM,
    ITERATION3_SCREEN_SHA256,
    NO_SERVED_STRESS_PHYSICS,
    PHYSICS_QUERY_COUNT,
    STRESS_HEAD_CONDITIONING,
    STRESS_HEAD_OUTPUT_SCALE,
    SUPERVISED_QUERY_COUNT,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    _hybrid_batch_loss,
)

WORK_STATE_NAME = "iteration3_protocol_reproduction_work.pt"


def _state_path(root: Path) -> Path:
    return root / "work" / WORK_STATE_NAME


def _save_state(path: Path, state: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(dict(state), temporary)
    os.replace(temporary, path)


def _load_state(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("reproduction work state must be a mapping")
    return state


def _training_pairs(
    repo_root: Path,
) -> tuple[str, tuple[SpikeMesh, ...], tuple[tuple[SpikeMesh, object], ...]]:
    fem_hash = _fem_content_hash(repo_root)
    historical = sample_geometry_cases(
        master_seed=CAMPAIGN_SEED,
        count_per_family=30,
        nx=32,
        ny=10,
    )
    training_cases = select_reproduction_training_cases(historical)
    cache_dir = Path(tempfile.gettempdir()) / "pi-ginot-s1-legacy-cache-019fe6e9"
    pairs = tuple(
        (mesh, _solve_with_cache(mesh, cache_dir, fem_hash)) for mesh in training_cases
    )
    failed = [
        mesh.geometry_id
        for mesh, result in pairs
        if getattr(result, "converged", None) is not True
    ]
    if failed:
        raise RuntimeError(f"reproduction training FEM failures: {failed}")
    return fem_hash, training_cases, pairs


def _new_training_state(
    *,
    repo_root: Path,
    upstream_checkpoint_path: Path,
    training_cases: Sequence[SpikeMesh],
) -> tuple[object, torch.optim.Optimizer, torch.Generator, dict[str, Any]]:
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
    optimizer = torch.optim.Adam(model.parameters(), lr=REPRODUCTION_LEARNING_RATE)
    generator = torch.Generator().manual_seed(REPRODUCTION_SAMPLER_SEED)
    metadata = {
        "schema_version": "iteration3-protocol-reproduction-work-v1",
        "completed_epochs": 0,
        "accumulated_training_seconds": 0.0,
        "final_hybrid_loss": math.inf,
        "training_geometry_ids": [mesh.geometry_id for mesh in training_cases],
        "upstream_checkpoint_sha256": checkpoint.sha256,
        "upstream_checkpoint_epoch": checkpoint.epoch,
        "repo_root": str(repo_root.resolve()),
    }
    return model, optimizer, generator, metadata


def _restore_training_state(
    path: Path,
    *,
    upstream_checkpoint_path: Path,
) -> tuple[object, torch.optim.Optimizer, torch.Generator, dict[str, Any]]:
    checkpoint = _load_upstream_checkpoint(upstream_checkpoint_path)
    state = _load_state(path)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    model = _new_pretrained_model(checkpoint.state_dict)
    model.load_state_dict(state["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=REPRODUCTION_LEARNING_RATE)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    generator = torch.Generator()
    generator.set_state(state["generator_state"])
    metadata = dict(state["metadata"])
    return model, optimizer, generator, metadata


def _train_chunk(
    *,
    model: object,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    data: Mapping[str, object],
    selected_geometry_indices: Sequence[int],
    completed_epochs: int,
    chunk_epochs: int,
) -> tuple[int, float, float]:
    selected = torch.tensor(selected_geometry_indices, dtype=torch.long)
    final_loss = math.inf
    started = perf_counter()
    model.train()
    target = min(REPRODUCTION_EPOCHS, completed_epochs + chunk_epochs)
    for epoch in range(completed_epochs, target):
        permutation = selected[torch.randperm(len(selected), generator=generator)]
        epoch_losses: list[float] = []
        for start in range(0, len(permutation), GEOMETRY_BATCH_SIZE):
            batch = permutation[start : start + GEOMETRY_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _hybrid_batch_loss(
                model,
                data,
                batch,
                generator,
                peak_supervised_p_weight=0.0,
                served_stress_physics=NO_SERVED_STRESS_PHYSICS,
            )
            if not bool(torch.isfinite(loss).detach()):
                raise RuntimeError("non-finite hybrid loss in iteration3 reproduction")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP_NORM,
            )
            if not bool(torch.isfinite(gradient_norm).detach()):
                raise RuntimeError("non-finite gradient in iteration3 reproduction")
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        final_loss = float(np.mean(epoch_losses))
        if (epoch + 1) % 10 == 0 or epoch + 1 == REPRODUCTION_EPOCHS:
            print(
                "iteration3 protocol reproduction: "
                f"epoch {epoch + 1}/{REPRODUCTION_EPOCHS}, "
                f"hybrid_loss={final_loss:.6g}",
                flush=True,
            )
    return target, final_loss, perf_counter() - started


def _save_final_report(
    *,
    repo_root: Path,
    root: Path,
    model: object,
    metadata: Mapping[str, Any],
    fem_hash: str,
    training_cases: Sequence[SpikeMesh],
    training_pairs: Sequence[tuple[SpikeMesh, object]],
) -> Mapping[str, Any]:
    checkpoint_path = root / "checkpoints" / REPRODUCTION_CHECKPOINT_NAME
    report_path = root / REPRODUCTION_REPORT_NAME
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError("iteration-3 protocol reproduction artifact exists")
    ordered_ids = [mesh.geometry_id for mesh in training_cases]
    probe_mesh = min(training_cases, key=lambda mesh: mesh.geometry_id)
    probe_sha = _probe_output_sha256(model, probe_mesh)
    final_loss = float(metadata["final_hybrid_loss"])
    training_seconds = float(metadata["accumulated_training_seconds"])
    created_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    provenance: dict[str, object] = {
        "schema_version": "iteration3-protocol-reproduction-v1",
        "role": "LEGACY",
        "no_release_evidence": True,
        "configuration_alias": "iteration3_joint_adam",
        "reproduction_kind": "semantic_protocol_reproduction_chunked",
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
        "upstream_checkpoint_sha256": metadata["upstream_checkpoint_sha256"],
        "upstream_checkpoint_epoch": metadata["upstream_checkpoint_epoch"],
        "sealed_iteration3_screen_sha256": ITERATION3_SCREEN_SHA256,
        "source_sha256": dict(_source_hashes(repo_root)),
        "model_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "stress_head_parameter_count": sum(
            parameter.numel() for parameter in model.p_head.parameters()
        ),
        "final_hybrid_loss": final_loss,
        "training_seconds": training_seconds,
        "chunked_resume_state_path": str(_state_path(root).resolve()),
        "probe_geometry_id": probe_mesh.geometry_id,
        "probe_output_sha256": probe_sha,
        "created_utc": created_utc,
        "environment": dict(_environment()),
    }
    checkpoint_sha = save_final_checkpoint(
        checkpoint_path, model, provenance=provenance
    )
    restored = load_final_checkpoint(
        checkpoint_path,
        expected_sha256=checkpoint_sha,
        expected_provenance=provenance,
    )
    restored_probe_sha = _probe_output_sha256(restored, probe_mesh)
    if restored_probe_sha != probe_sha:
        raise RuntimeError("saved reproduction does not preserve probe outputs")
    report = {
        "schema_version": "iteration3-protocol-reproduction-report-v1",
        "step": "S1",
        "role": "LEGACY",
        "no_release_evidence": True,
        "decision": "frozen semantic reproduction; not a recovered historical checkpoint",
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha,
            "tensor_state_sha256": _tensor_state_sha256(restored),
            "provenance": provenance,
            "reload_probe_output_sha256": restored_probe_sha,
        },
    }
    write_json_strict(report_path, report)
    print(f"ITERATION8_REPRODUCTION_WROTE {report_path}", flush=True)
    return report


def run_reproduction_chunk(
    *,
    repo_root: Path,
    upstream_checkpoint_path: Path,
    chunk_epochs: int,
    artifact_root: Path | None = None,
) -> Mapping[str, Any]:
    if chunk_epochs < 1 or chunk_epochs > 100:
        raise ValueError("chunk_epochs must be between 1 and 100")
    root = artifact_root or _artifact_root(repo_root)
    final_report = root / REPRODUCTION_REPORT_NAME
    if final_report.exists():
        return {"status": "complete", "report_path": str(final_report.resolve())}
    fem_hash, training_cases, pairs = _training_pairs(repo_root)
    state_path = _state_path(root)
    if state_path.exists():
        model, optimizer, generator, metadata = _restore_training_state(
            state_path,
            upstream_checkpoint_path=upstream_checkpoint_path,
        )
    else:
        model, optimizer, generator, metadata = _new_training_state(
            repo_root=repo_root,
            upstream_checkpoint_path=upstream_checkpoint_path,
            training_cases=training_cases,
        )
    completed = int(metadata["completed_epochs"])
    if completed >= REPRODUCTION_EPOCHS:
        report = _save_final_report(
            repo_root=repo_root,
            root=root,
            model=model,
            metadata=metadata,
            fem_hash=fem_hash,
            training_cases=training_cases,
            training_pairs=pairs,
        )
        return {"status": "complete", "report": report}
    completed, final_loss, seconds = _train_chunk(
        model=model,
        optimizer=optimizer,
        generator=generator,
        data=_tensor_dataset(pairs),
        selected_geometry_indices=range(len(pairs)),
        completed_epochs=completed,
        chunk_epochs=chunk_epochs,
    )
    if not math.isfinite(final_loss):
        raise RuntimeError("reproduction finished a chunk with non-finite loss")
    metadata.update(
        {
            "completed_epochs": completed,
            "final_hybrid_loss": final_loss,
            "accumulated_training_seconds": float(
                metadata["accumulated_training_seconds"]
            )
            + seconds,
        }
    )
    _save_state(
        state_path,
        {
            "metadata": metadata,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "generator_state": generator.get_state(),
        },
    )
    if completed >= REPRODUCTION_EPOCHS:
        report = _save_final_report(
            repo_root=repo_root,
            root=root,
            model=model,
            metadata=metadata,
            fem_hash=fem_hash,
            training_cases=training_cases,
            training_pairs=pairs,
        )
        return {"status": "complete", "report": report}
    return {
        "status": "in_progress",
        "completed_epochs": completed,
        "remaining_epochs": REPRODUCTION_EPOCHS - completed,
        "final_hybrid_loss": final_loss,
        "state_path": str(state_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-epochs", type=int, default=40)
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
    result = run_reproduction_chunk(
        repo_root=Path(__file__).resolve().parents[2],
        upstream_checkpoint_path=args.upstream_checkpoint,
        chunk_epochs=args.chunk_epochs,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
