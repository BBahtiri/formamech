"""Matched, TRAIN-only gradient replay for the two S13 step-800 arms."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

from campaigns.dogbone_v1.s11_contingency_source import open_selective_s10_source
from campaigns.dogbone_v1.s13_data import materialize_dogbone_s13_training_data
from campaigns.dogbone_v1.s13_evaluate import (
    _TRAINING_IMPLEMENTATION_REVISION,
    _preflight_paths,
)
from campaigns.dogbone_v1.s13_initialization import load_selected_s11_initialization
from campaigns.dogbone_v1.s13_objective import DogboneS13ObjectiveBuilder
from campaigns.dogbone_v1.s13_pilot import _build_pilot_binding, _verify_training_data
from campaigns.dogbone_v1.s13_protocol import (
    build_dogbone_s13_sampler_plan,
    build_dogbone_s13_trainer_configs,
)
from campaigns.dogbone_v1.s13b_protocol import build_dogbone_s13b_pilot_plan
from campaigns.dogbone_v1.s13b_run import _build_s13b_binding
from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.doe import EvidenceRole
from piginot.training import (
    DeterministicTrainerConfig,
    LoadedTrainingCheckpoint,
    ObjectiveGradientDiagnostic,
    TrainingRunBinding,
    load_training_checkpoint,
    measure_objective_gradient_diagnostic,
)

_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_PEAK_MANIFEST_HASH = (
    "0e2cf32373767784a1ce928e1bd957d305c2901c1fc83d15ee014c2bb8cf7322"
)


@dataclass(frozen=True, slots=True)
class GradientArmReport:
    """Checkpoint identity and gradient measurement for one matched arm."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    arm_id: str
    run_hash: str
    checkpoint_manifest_hash: str
    objective_registry_hash: str
    diagnostic: ObjectiveGradientDiagnostic


@dataclass(frozen=True, slots=True)
class S13GradientDiagnosticReport:
    """A non-gate comparison that never opens DEVELOPMENT or HOLDOUT."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    role: EvidenceRole
    non_gate: bool
    pilot_id: str
    plan_hash: str
    sampler_plan_hash: str
    sample_ids: tuple[str, ...]
    control: GradientArmReport
    peak: GradientArmReport


def _load_arm_checkpoint(
    checkpoint: Path,
    *,
    model: torch.nn.Module,
    config: DeterministicTrainerConfig,
    binding: TrainingRunBinding,
    expected_manifest_hash: str | None,
) -> LoadedTrainingCheckpoint:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.trial.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.adam_weight_decay,
        foreach=False,
        fused=False,
    )
    loaded = load_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        binding=binding,
    )
    manifest = loaded.manifest
    if manifest.step != 800 or manifest.run_hash != binding.run_hash:
        raise ValueError("checkpoint step or run hash differs from its arm binding")
    if (
        expected_manifest_hash is not None
        and manifest.manifest_hash != expected_manifest_hash
    ):
        raise ValueError("checkpoint manifest differs from the pinned arm")
    return loaded


def run_s13_gradient_diagnostic(
    *,
    s10_run_root: Path,
    s11_checkpoint_root: Path,
    control_checkpoint_path: Path,
    peak_checkpoint_path: Path,
    peak_training_revision: str,
) -> S13GradientDiagnosticReport:
    """Replay the same frozen step-800 TRAIN batch for both checkpoint arms."""

    if not isinstance(peak_training_revision, str) or not _REVISION.fullmatch(
        peak_training_revision
    ):
        raise ValueError("peak_training_revision must be a 40-character commit SHA")
    if (
        control_checkpoint_path == peak_checkpoint_path
        or control_checkpoint_path.parent == peak_checkpoint_path.parent
    ):
        raise ValueError("control and peak checkpoints require separate run roots")

    pilot = build_dogbone_s13b_pilot_plan()
    s10, s11, control_cp, _ = _preflight_paths(
        s10_run_root,
        s11_checkpoint_root,
        control_checkpoint_path,
        optimizer_steps=800,
    )
    _, _, peak_cp, peak_access_record = _preflight_paths(
        s10_run_root,
        s11_checkpoint_root,
        peak_checkpoint_path,
        optimizer_steps=800,
    )
    if control_cp.parent == peak_cp.parent:
        raise ValueError("control and peak checkpoints require separate run roots")

    control_config = build_dogbone_s13_trainer_configs(pilot.base_protocol)[0]
    control_initialization = load_selected_s11_initialization(s11)
    control_binding = _build_pilot_binding(
        pilot.base_protocol,
        model=control_initialization.model,
        config=control_config,
        plan=build_dogbone_s13_sampler_plan(pilot.base_protocol),
        implementation_revision=_TRAINING_IMPLEMENTATION_REVISION,
    )
    if control_binding.run_hash != pilot.control_run_hash:
        raise ValueError("control binding differs from the pinned S13 run")
    control_loaded = _load_arm_checkpoint(
        control_cp,
        model=control_initialization.model,
        config=control_config,
        binding=control_binding,
        expected_manifest_hash=pilot.control_manifest_hash,
    )

    peak_initialization = load_selected_s11_initialization(s11)
    peak_binding = _build_s13b_binding(
        pilot,
        model=peak_initialization.model,
        config=pilot.trainer_config,
        plan=pilot.sampler_plan,
        implementation_revision=peak_training_revision,
    )
    peak_loaded = _load_arm_checkpoint(
        peak_cp,
        model=peak_initialization.model,
        config=pilot.trainer_config,
        binding=peak_binding,
        expected_manifest_hash=_PEAK_MANIFEST_HASH,
    )

    # Checkpoint identity is fully verified before any FEM label source is opened.
    source = open_selective_s10_source(
        s10,
        development_access_record=peak_access_record,
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened before gradient diagnostics")
    data = materialize_dogbone_s13_training_data(source, control_initialization)
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during TRAIN materialization")
    _verify_training_data(data, pilot.base_protocol, source.bindings.content_hash)

    batch_indices = pilot.sampler_plan.batches[799]
    sample_ids = tuple(data.population.sample_ids[index] for index in batch_indices)
    control_objective = DogboneS13ObjectiveBuilder(
        data.cases,
        include_peak_loss=False,
    )
    peak_objective = DogboneS13ObjectiveBuilder(
        data.cases,
        include_peak_loss=True,
    )
    control_diagnostic = measure_objective_gradient_diagnostic(
        control_initialization.model,
        population=data.population,
        sample_ids=sample_ids,
        step=800,
        base_registry=pilot.base_protocol.objective_registry,
        physics_weight_scale=control_config.trial.physics_weight_scale,
        adaptive_config=control_config.adaptive,
        adaptive_state=control_loaded.adaptive_state,
        objective=control_objective,
        role=EvidenceRole.TRAIN,
    )
    peak_diagnostic = measure_objective_gradient_diagnostic(
        peak_initialization.model,
        population=data.population,
        sample_ids=sample_ids,
        step=800,
        base_registry=pilot.objective_registry,
        physics_weight_scale=pilot.trainer_config.trial.physics_weight_scale,
        adaptive_config=pilot.trainer_config.adaptive,
        adaptive_state=peak_loaded.adaptive_state,
        objective=peak_objective,
        role=EvidenceRole.TRAIN,
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during gradient diagnostics")

    return S13GradientDiagnosticReport(
        role=EvidenceRole.TRAIN,
        non_gate=True,
        pilot_id=pilot.pilot_id,
        plan_hash=pilot.plan_hash,
        sampler_plan_hash=pilot.sampler_plan.plan_hash,
        sample_ids=sample_ids,
        control=GradientArmReport(
            arm_id="control",
            run_hash=control_binding.run_hash,
            checkpoint_manifest_hash=control_loaded.manifest.manifest_hash,
            objective_registry_hash=hash_spec(
                pilot.base_protocol.objective_registry, "recipe"
            ),
            diagnostic=control_diagnostic,
        ),
        peak=GradientArmReport(
            arm_id="peak",
            run_hash=peak_binding.run_hash,
            checkpoint_manifest_hash=peak_loaded.manifest.manifest_hash,
            objective_registry_hash=hash_spec(pilot.objective_registry, "recipe"),
            diagnostic=peak_diagnostic,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay matched S13/S13b step-800 TRAIN gradients."
    )
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--s11-checkpoint-root", required=True, type=Path)
    parser.add_argument("--control-checkpoint-path", required=True, type=Path)
    parser.add_argument("--peak-checkpoint-path", required=True, type=Path)
    parser.add_argument("--peak-training-revision", required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    report = run_s13_gradient_diagnostic(
        s10_run_root=options.s10_run_root,
        s11_checkpoint_root=options.s11_checkpoint_root,
        control_checkpoint_path=options.control_checkpoint_path,
        peak_checkpoint_path=options.peak_checkpoint_path,
        peak_training_revision=options.peak_training_revision,
    )
    print(canonical_json_bytes(report).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = (
    "GradientArmReport",
    "S13GradientDiagnosticReport",
    "build_parser",
    "main",
    "run_s13_gradient_diagnostic",
)
