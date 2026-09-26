"""TRAIN-only matched comparison for the ratified S13b peak pilot."""

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
    _trial_result,
)
from campaigns.dogbone_v1.s13_initialization import load_selected_s11_initialization
from campaigns.dogbone_v1.s13_pilot import _build_pilot_binding, _verify_training_data
from campaigns.dogbone_v1.s13_protocol import (
    build_dogbone_s13_sampler_plan,
    build_dogbone_s13_trainer_configs,
)
from campaigns.dogbone_v1.s13_screen import evaluate_dogbone_s13_train_screen
from campaigns.dogbone_v1.s13b_protocol import (
    DOGBONE_S13B_PILOT_ID,
    DogboneS13bPilotPlan,
    build_dogbone_s13b_pilot_plan,
)
from campaigns.dogbone_v1.s13b_run import _build_s13b_binding
from piginot.contracts import canonical_json_bytes
from piginot.doe import EvidenceRole
from piginot.training import (
    DeterministicTrainerConfig,
    TrainingCheckpointManifest,
    TrainingRunBinding,
    TrialResult,
    load_training_checkpoint,
)

_REVISION = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True, slots=True)
class S13bPilotComparison:
    """Non-gate TRAIN signal; hybrid_loss is a shared legacy diagnostic only."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    pilot_id: str
    plan_hash: str
    control: TrialResult
    peak: TrialResult
    criteria: tuple[tuple[str, bool], ...]
    pilot_signal: bool


def compare_s13b_train_results(
    pilot: DogboneS13bPilotPlan,
    *,
    control: TrialResult,
    peak: TrialResult,
) -> S13bPilotComparison:
    """Apply ADR 0015's exact matched-control criteria without gate promotion."""

    if type(pilot) is not DogboneS13bPilotPlan:
        raise TypeError("pilot must be an exact DogboneS13bPilotPlan")
    if type(control) is not TrialResult or type(peak) is not TrialResult:
        raise TypeError("control and peak must be exact TrialResult values")
    if control.role is not EvidenceRole.TRAIN or peak.role is not EvidenceRole.TRAIN:
        raise PermissionError("only TRAIN results can enter the S13b comparison")
    if (
        control.trial_id != pilot.control_trial_id
        or control.run_hash != pilot.control_run_hash
        or control.lineage.checkpoint_hash != pilot.control_manifest_hash
    ):
        raise ValueError("control checkpoint differs from the pinned S13 step 800")
    if (
        peak.trial_id != pilot.search_space.trials[0].trial_id
        or peak.run_hash == control.run_hash
        or peak.lineage.checkpoint_hash == control.lineage.checkpoint_hash
    ):
        raise ValueError("peak checkpoint is not a separate S13b pilot arm")
    if (
        control.dataset_snapshot_hash != pilot.base_protocol.dataset_snapshot_hash
        or peak.dataset_snapshot_hash != pilot.base_protocol.dataset_snapshot_hash
    ):
        raise ValueError("checkpoint datasets differ from the S13b pilot plan")

    c = control.metrics
    p = peak.metrics
    criteria = (
        (
            "median_absolute_peak_improved",
            p.median_absolute_peak_error < c.median_absolute_peak_error,
        ),
        (
            "p90_absolute_peak_improved",
            p.p90_absolute_peak_error < c.p90_absolute_peak_error,
        ),
        (
            "underprediction_reduced_10_percent",
            max(0.0, -p.median_signed_peak_error)
            <= 0.9 * max(0.0, -c.median_signed_peak_error),
        ),
        ("u_field_nonregression", p.u_relative_l2 <= 1.02 * c.u_relative_l2),
        ("p_field_nonregression", p.p_relative_l2 <= 1.02 * c.p_relative_l2),
        (
            "zero_invalid_collocation",
            c.invalid_collocation_points == p.invalid_collocation_points == 0,
        ),
    )
    return S13bPilotComparison(
        pilot_id=DOGBONE_S13B_PILOT_ID,
        plan_hash=pilot.plan_hash,
        control=control,
        peak=peak,
        criteria=criteria,
        pilot_signal=all(passed for _, passed in criteria),
    )


def _load_checked_checkpoint(
    checkpoint: Path,
    *,
    model: torch.nn.Module,
    config: DeterministicTrainerConfig,
    binding: TrainingRunBinding,
    expected_manifest_hash: str | None,
) -> TrainingCheckpointManifest:
    """Verify every checkpoint file and its exact training binding."""

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.trial.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.adam_weight_decay,
        foreach=False,
        fused=False,
    )
    manifest = load_training_checkpoint(
        checkpoint, model=model, optimizer=optimizer, binding=binding
    ).manifest
    if manifest.step != 800 or manifest.run_hash != binding.run_hash:
        raise ValueError("checkpoint step or run hash differs from the pilot")
    if (
        expected_manifest_hash is not None
        and manifest.manifest_hash != expected_manifest_hash
    ):
        raise ValueError("control manifest hash differs from the pinned checkpoint")
    return manifest


def evaluate_s13b_pilot(
    *,
    s10_run_root: Path,
    s11_checkpoint_root: Path,
    control_checkpoint_path: Path,
    peak_checkpoint_path: Path,
    peak_training_revision: str,
) -> S13bPilotComparison:
    """Load both step-800 arms before reading any TRAIN-screen FEM labels."""

    if not isinstance(peak_training_revision, str) or not _REVISION.fullmatch(
        peak_training_revision
    ):
        raise ValueError("peak_training_revision must be a 40-character commit SHA")
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
        raise ValueError("control and peak checkpoints must use separate run roots")

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
    control_manifest = _load_checked_checkpoint(
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
    peak_manifest = _load_checked_checkpoint(
        peak_cp,
        model=peak_initialization.model,
        config=pilot.trainer_config,
        binding=peak_binding,
        expected_manifest_hash=None,
    )

    source = open_selective_s10_source(
        s10, development_access_record=peak_access_record
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened before S13b TRAIN screening")
    data = materialize_dogbone_s13_training_data(source, control_initialization)
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during TRAIN materialization")
    _verify_training_data(data, pilot.base_protocol, source.bindings.content_hash)

    # Both arms use the *same* frozen S13 TRAIN-screen recipe. Its hybrid_loss
    # omits DATA_PEAK for both and is not one of the pilot decision criteria.
    control_metrics = evaluate_dogbone_s13_train_screen(
        protocol=pilot.base_protocol,
        source=source,
        training_data=data,
        model=control_initialization.model,
        trial=control_config.trial,
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during control screening")
    peak_metrics = evaluate_dogbone_s13_train_screen(
        protocol=pilot.base_protocol,
        source=source,
        training_data=data,
        model=peak_initialization.model,
        trial=control_config.trial,
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during peak screening")
    return compare_s13b_train_results(
        pilot,
        control=_trial_result(control_config, control_manifest, control_metrics),
        peak=_trial_result(pilot.trainer_config, peak_manifest, peak_metrics),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare S13b to its pinned S13 control on TRAIN only."
    )
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--s11-checkpoint-root", required=True, type=Path)
    parser.add_argument("--control-checkpoint-path", required=True, type=Path)
    parser.add_argument("--peak-checkpoint-path", required=True, type=Path)
    parser.add_argument("--peak-training-revision", required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    comparison = evaluate_s13b_pilot(
        s10_run_root=options.s10_run_root,
        s11_checkpoint_root=options.s11_checkpoint_root,
        control_checkpoint_path=options.control_checkpoint_path,
        peak_checkpoint_path=options.peak_checkpoint_path,
        peak_training_revision=options.peak_training_revision,
    )
    print(canonical_json_bytes(comparison).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = (
    "S13bPilotComparison",
    "build_parser",
    "compare_s13b_train_results",
    "evaluate_s13b_pilot",
    "main",
)
