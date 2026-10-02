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
from piginot.objectives import ObjectiveRegistry
from piginot.training import (
    AdaptiveWeightState,
    DeterministicSamplerPlan,
    DeterministicTrainerConfig,
    LoadedTrainingCheckpoint,
    ObjectiveBuilder,
    ObjectiveGradientDiagnostic,
    TrainingPopulation,
    TrainingRunBinding,
    load_training_checkpoint,
    measure_objective_gradient_diagnostic,
)

_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_PEAK_MANIFEST_HASH = (
    "0e2cf32373767784a1ce928e1bd957d305c2901c1fc83d15ee014c2bb8cf7322"
)
_PANEL_STEPS = (
    ("targeted-folded-case-step-753", 753),
    ("post-update-trace-step-800", 800),
    ("fresh-plan-step-801", 801),
    ("fresh-plan-step-802", 802),
)
_FOLDED_CASE_ID = "candidate-002485"


@dataclass(frozen=True, slots=True)
class GradientPanelBatch:
    """One predeclared sampler batch evaluated at the post-step-800 model."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    snapshot_id: str
    plan_step: int
    sample_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GradientSnapshotReport:
    """One panel batch and its post-checkpoint raw-gradient measurement."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    panel: GradientPanelBatch
    diagnostic: ObjectiveGradientDiagnostic


@dataclass(frozen=True, slots=True)
class GradientArmReport:
    """Checkpoint identity and gradient measurement for one matched arm."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    arm_id: str
    run_hash: str
    checkpoint_manifest_hash: str
    checkpoint_sampler_plan_hash: str
    objective_registry_hash: str
    scheduler_learning_rate: float
    scheduler_state_hash: str
    faithful_snapshots: tuple[GradientSnapshotReport, ...]
    common_peak_snapshots: tuple[GradientSnapshotReport, ...]


@dataclass(frozen=True, slots=True)
class S13GradientDiagnosticReport:
    """A non-gate comparison that never opens DEVELOPMENT or HOLDOUT."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    role: EvidenceRole
    non_gate: bool
    pilot_id: str
    plan_hash: str
    panel_sampler_plan_hash: str
    panel: tuple[GradientPanelBatch, ...]
    control: GradientArmReport
    peak: GradientArmReport


def _gradient_panel(
    sample_ids: tuple[str, ...],
    plan: DeterministicSamplerPlan,
) -> tuple[GradientPanelBatch, ...]:
    panel = tuple(
        GradientPanelBatch(
            snapshot_id=snapshot_id,
            plan_step=plan_step,
            sample_ids=tuple(
                sample_ids[index] for index in plan.batches[plan_step - 1]
            ),
        )
        for snapshot_id, plan_step in _PANEL_STEPS
    )
    if _FOLDED_CASE_ID not in panel[0].sample_ids:
        raise RuntimeError("targeted gradient panel no longer contains folded case")
    return panel


def _measure_panel(
    *,
    model: torch.nn.Module,
    panel: tuple[GradientPanelBatch, ...],
    population: TrainingPopulation,
    registry: ObjectiveRegistry,
    config: DeterministicTrainerConfig,
    adaptive_state: AdaptiveWeightState,
    objective: ObjectiveBuilder,
) -> tuple[GradientSnapshotReport, ...]:
    return tuple(
        GradientSnapshotReport(
            panel=batch,
            diagnostic=measure_objective_gradient_diagnostic(
                model,
                population=population,
                sample_ids=batch.sample_ids,
                step=800,
                base_registry=registry,
                physics_weight_scale=config.trial.physics_weight_scale,
                adaptive_config=config.adaptive,
                adaptive_state=adaptive_state,
                objective=objective,
                role=EvidenceRole.TRAIN,
                gradient_clip_norm=config.gradient_clip_norm,
                post_checkpoint_snapshot=True,
            ),
        )
        for batch in panel
    )


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
    base_plan = build_dogbone_s13_sampler_plan(pilot.base_protocol)
    control_initialization = load_selected_s11_initialization(s11)
    control_binding = _build_pilot_binding(
        pilot.base_protocol,
        model=control_initialization.model,
        config=control_config,
        plan=base_plan,
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

    panel = _gradient_panel(data.population.sample_ids, base_plan)
    control_objective = DogboneS13ObjectiveBuilder(
        data.cases,
        include_peak_loss=False,
    )
    peak_objective = DogboneS13ObjectiveBuilder(
        data.cases,
        include_peak_loss=True,
    )
    control_faithful = _measure_panel(
        model=control_initialization.model,
        panel=panel,
        population=data.population,
        registry=pilot.base_protocol.objective_registry,
        config=control_config,
        adaptive_state=control_loaded.adaptive_state,
        objective=control_objective,
    )
    control_common_peak = _measure_panel(
        model=control_initialization.model,
        panel=panel,
        population=data.population,
        registry=pilot.objective_registry,
        config=pilot.trainer_config,
        adaptive_state=control_loaded.adaptive_state,
        objective=peak_objective,
    )
    peak_common = _measure_panel(
        model=peak_initialization.model,
        panel=panel,
        population=data.population,
        registry=pilot.objective_registry,
        config=pilot.trainer_config,
        adaptive_state=peak_loaded.adaptive_state,
        objective=peak_objective,
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during gradient diagnostics")

    return S13GradientDiagnosticReport(
        role=EvidenceRole.TRAIN,
        non_gate=True,
        pilot_id=pilot.pilot_id,
        plan_hash=pilot.plan_hash,
        panel_sampler_plan_hash=base_plan.plan_hash,
        panel=panel,
        control=GradientArmReport(
            arm_id="control",
            run_hash=control_binding.run_hash,
            checkpoint_manifest_hash=control_loaded.manifest.manifest_hash,
            checkpoint_sampler_plan_hash=control_binding.sampler_plan_hash,
            objective_registry_hash=hash_spec(
                pilot.base_protocol.objective_registry, "recipe"
            ),
            scheduler_learning_rate=control_loaded.scheduler_state.learning_rate,
            scheduler_state_hash=hash_spec(
                control_loaded.scheduler_state, "content"
            ),
            faithful_snapshots=control_faithful,
            common_peak_snapshots=control_common_peak,
        ),
        peak=GradientArmReport(
            arm_id="peak",
            run_hash=peak_binding.run_hash,
            checkpoint_manifest_hash=peak_loaded.manifest.manifest_hash,
            checkpoint_sampler_plan_hash=peak_binding.sampler_plan_hash,
            objective_registry_hash=hash_spec(pilot.objective_registry, "recipe"),
            scheduler_learning_rate=peak_loaded.scheduler_state.learning_rate,
            scheduler_state_hash=hash_spec(peak_loaded.scheduler_state, "content"),
            faithful_snapshots=peak_common,
            common_peak_snapshots=peak_common,
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
    parser.add_argument("--output-path", type=Path)
    return parser


def _preflight_report_path(
    output_path: Path,
    *,
    forbidden_roots: tuple[Path, ...],
) -> Path:
    if not output_path.is_absolute():
        raise ValueError("output_path must be absolute")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("output_path must not already exist")
    parent = output_path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("output_path parent must be an existing non-symlink directory")
    checked = parent.resolve(strict=True) / output_path.name
    if any(checked.is_relative_to(root.absolute()) for root in forbidden_roots):
        raise ValueError("output_path must remain outside source and checkpoint roots")
    return checked


def main(arguments: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    output_path = (
        None
        if options.output_path is None
        else _preflight_report_path(
            options.output_path,
            forbidden_roots=(
                options.s10_run_root,
                options.s11_checkpoint_root,
                options.control_checkpoint_path.parent,
                options.peak_checkpoint_path.parent,
            ),
        )
    )
    report = run_s13_gradient_diagnostic(
        s10_run_root=options.s10_run_root,
        s11_checkpoint_root=options.s11_checkpoint_root,
        control_checkpoint_path=options.control_checkpoint_path,
        peak_checkpoint_path=options.peak_checkpoint_path,
        peak_training_revision=options.peak_training_revision,
    )
    payload = canonical_json_bytes(report)
    if output_path is None:
        print(payload.decode("utf-8"), flush=True)
    else:
        with output_path.open("xb") as destination:
            destination.write(payload + b"\n")
        print(f"report_path={output_path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = (
    "GradientArmReport",
    "GradientPanelBatch",
    "GradientSnapshotReport",
    "S13GradientDiagnosticReport",
    "build_parser",
    "main",
    "run_s13_gradient_diagnostic",
)
