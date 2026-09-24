"""Read-only TRAIN-screen evaluation of one completed registered S13 trial."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from campaigns.dogbone_v1.s11_contingency_source import open_selective_s10_source
from campaigns.dogbone_v1.s13_data import materialize_dogbone_s13_training_data
from campaigns.dogbone_v1.s13_initialization import load_selected_s11_initialization
from campaigns.dogbone_v1.s13_pilot import (
    _absolute_directory,
    _build_pilot_binding,
    _verify_training_data,
)
from campaigns.dogbone_v1.s13_protocol import (
    build_dogbone_s13_protocol,
    build_dogbone_s13_sampler_plan,
    build_dogbone_s13_trainer_configs,
)
from campaigns.dogbone_v1.s13_screen import evaluate_dogbone_s13_train_screen
from piginot.contracts import canonical_json_bytes
from piginot.doe import EvidenceRole
from piginot.training import (
    CheckpointLineage,
    TrialResult,
    load_training_checkpoint,
)

# TrainingRunBinding includes the clean implementation revision used at launch.
# Evaluation may be implemented on a later branch, but must replay that binding.
_TRAINING_IMPLEMENTATION_REVISION = "5473cdeebd06597d4ec0bdd50df7d230d00306cd"


def _preflight_paths(
    s10_run_root: Path,
    s11_checkpoint_root: Path,
    checkpoint_path: Path,
    *,
    optimizer_steps: int,
) -> tuple[Path, Path, Path, Path]:
    s10 = _absolute_directory(s10_run_root, "s10_run_root")
    s11 = _absolute_directory(s11_checkpoint_root, "s11_checkpoint_root")
    checkpoint = _absolute_directory(checkpoint_path, "checkpoint_path")
    run_root = _absolute_directory(checkpoint.parent, "registered trial root")
    if checkpoint.name != f"checkpoint-step-{optimizer_steps:04d}":
        raise ValueError("checkpoint_path must be the final registered checkpoint")
    if checkpoint.is_relative_to(s10) or checkpoint.is_relative_to(s11):
        raise ValueError("checkpoint_path must remain outside source roots")
    access_record = run_root / "development-access.json"
    if access_record.exists() or access_record.is_symlink():
        raise RuntimeError("DEVELOPMENT access was recorded in the trial run root")
    return s10, s11, checkpoint, access_record


def evaluate_registered_s13_trial(
    *,
    s10_run_root: Path,
    s11_checkpoint_root: Path,
    checkpoint_path: Path,
    trial_index: int,
) -> TrialResult:
    """Validate complete checkpoint and reduce only registered TRAIN evidence."""

    protocol = build_dogbone_s13_protocol()
    configs = build_dogbone_s13_trainer_configs(protocol)
    if type(trial_index) is not int or not 0 <= trial_index < len(configs):
        raise ValueError("trial_index must identify one registered trial")
    config = configs[trial_index]
    s10, s11, checkpoint, access_record = _preflight_paths(
        s10_run_root,
        s11_checkpoint_root,
        checkpoint_path,
        optimizer_steps=config.trial.optimizer_steps,
    )
    initialization = load_selected_s11_initialization(s11)
    binding = _build_pilot_binding(
        protocol,
        model=initialization.model,
        config=config,
        plan=build_dogbone_s13_sampler_plan(protocol),
        implementation_revision=_TRAINING_IMPLEMENTATION_REVISION,
    )
    optimizer = torch.optim.Adam(
        initialization.model.parameters(),
        lr=config.trial.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.adam_weight_decay,
        foreach=False,
        fused=False,
    )
    loaded = load_training_checkpoint(
        checkpoint,
        model=initialization.model,
        optimizer=optimizer,
        binding=binding,
    )
    manifest = loaded.manifest
    if manifest.step != config.trial.optimizer_steps:
        raise ValueError("checkpoint step differs from registered optimizer_steps")
    if manifest.run_hash != binding.run_hash:
        raise ValueError("checkpoint run_hash differs from registered trial")

    source = open_selective_s10_source(
        s10,
        development_access_record=access_record,
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels must remain closed during screening")
    data = materialize_dogbone_s13_training_data(source, initialization)
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during TRAIN-fit materialization")
    _verify_training_data(data, protocol, source.bindings.content_hash)
    metrics = evaluate_dogbone_s13_train_screen(
        protocol=protocol,
        source=source,
        training_data=data,
        model=initialization.model,
        trial=config.trial,
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during TRAIN screening")
    return TrialResult(
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one completed S13 trial on TRAIN-screen only."
    )
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--s11-checkpoint-root", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--trial-index", required=True, type=int)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    result = evaluate_registered_s13_trial(
        s10_run_root=options.s10_run_root,
        s11_checkpoint_root=options.s11_checkpoint_root,
        checkpoint_path=options.checkpoint_path,
        trial_index=options.trial_index,
    )
    print(canonical_json_bytes(result).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = ("build_parser", "evaluate_registered_s13_trial", "main")
