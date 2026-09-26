"""Clean-revision, resumable TRAIN-fit execution of the S13b peak pilot arm."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import torch

from campaigns.dogbone_v1.s11_contingency_source import open_selective_s10_source
from campaigns.dogbone_v1.s13_data import materialize_dogbone_s13_training_data
from campaigns.dogbone_v1.s13_initialization import load_selected_s11_initialization
from campaigns.dogbone_v1.s13_objective import DogboneS13ObjectiveBuilder
from campaigns.dogbone_v1.s13_pilot import (
    _absolute_directory,
    _build_pilot_binding,
    _git_provenance,
    _verify_training_data,
)
from campaigns.dogbone_v1.s13_trial import (
    _latest_checkpoint,
    _prepare_run_root,
    _segment_targets,
)
from campaigns.dogbone_v1.s13b_protocol import (
    DogboneS13bPilotPlan,
    build_dogbone_s13b_pilot_plan,
)
from piginot.contracts import hash_spec
from piginot.objectives import ObjectiveTerm
from piginot.training import (
    DeterministicSamplerPlan,
    DeterministicTrainerConfig,
    TrainingRunBinding,
    TrainingRunResult,
    train_to_checkpoint,
)

Progress = Callable[[str], None]


def _build_s13b_binding(
    pilot: DogboneS13bPilotPlan,
    *,
    model: torch.nn.Module,
    config: DeterministicTrainerConfig,
    plan: DeterministicSamplerPlan,
    implementation_revision: str,
) -> TrainingRunBinding:
    """Carry S13's data/initialization checks into a distinct peak-arm identity."""

    if type(pilot) is not DogboneS13bPilotPlan:
        raise TypeError("pilot must be an exact DogboneS13bPilotPlan")
    if config != pilot.trainer_config or plan != pilot.sampler_plan:
        raise ValueError("trainer or sampler differs from the S13b pilot recipe")
    base = _build_pilot_binding(
        pilot.base_protocol,
        model=model,
        config=config,
        plan=plan,
        implementation_revision=implementation_revision,
    )
    return replace(
        base,
        objective_registry_hash=hash_spec(pilot.objective_registry, "recipe"),
    )


def run_s13b_peak_pilot(
    *,
    s10_run_root: Path,
    s11_checkpoint_root: Path,
    run_root: Path,
    segment_steps: int = 100,
    progress: Progress | None = None,
) -> TrainingRunResult:
    """Train one opt-in 800-step arm; this is not G13 or CP-FREEZE evidence."""

    if type(segment_steps) is not int or segment_steps <= 0:
        raise ValueError("segment_steps must be a positive integer")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    pilot = build_dogbone_s13b_pilot_plan()
    if ObjectiveTerm.DATA_PEAK not in pilot.objective_registry.required_terms:
        raise RuntimeError("S13b pilot requires the DATA_PEAK objective term")
    s10 = _absolute_directory(s10_run_root, "s10_run_root")
    s11 = _absolute_directory(s11_checkpoint_root, "s11_checkpoint_root")
    provenance = _git_provenance()
    if provenance.dirty_worktree:
        raise RuntimeError("S13b peak pilot requires a clean revision")

    config = pilot.trainer_config
    plan = pilot.sampler_plan
    initialization = load_selected_s11_initialization(s11)
    binding = _build_s13b_binding(
        pilot,
        model=initialization.model,
        config=config,
        plan=plan,
        implementation_revision=provenance.head,
    )
    output = _prepare_run_root(s10, s11, run_root)
    resume_from, last_step = _latest_checkpoint(
        output, expected_run_hash=binding.run_hash
    )
    targets = _segment_targets(last_step, config.trial.optimizer_steps, segment_steps)
    if progress is not None:
        progress(
            "phase=PREFLIGHT non_gate=true role=TRAIN_FIT "
            f"pilot={pilot.pilot_id} revision={provenance.head} "
            f"plan={pilot.plan_hash} run={binding.run_hash} "
            f"resume_step={last_step} targets={targets}"
        )

    source = open_selective_s10_source(
        s10,
        development_access_record=output / "development-access.json",
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels must remain closed during S13b fitting")
    data = materialize_dogbone_s13_training_data(source, initialization)
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during S13b materialization")
    _verify_training_data(data, pilot.base_protocol, source.bindings.content_hash)
    objective = DogboneS13ObjectiveBuilder(data.cases, include_peak_loss=True)
    if progress is not None:
        progress(
            "phase=TRAIN_FIT non_gate=true role=TRAIN "
            f"cases={len(data.population.sample_ids)} steps={config.trial.optimizer_steps} "
            f"source_bindings={source.bindings.content_hash}"
        )

    result: TrainingRunResult | None = None
    for target in targets:
        checkpoint_path = output / f"checkpoint-step-{target:04d}"
        result = train_to_checkpoint(
            checkpoint_path,
            model=initialization.model,
            population=data.population,
            plan=plan,
            binding=binding,
            config=config,
            base_registry=pilot.objective_registry,
            objective=objective,
            target_step=target,
            resume_from=resume_from,
        )
        if source.development_opened:
            raise RuntimeError("DEVELOPMENT labels opened during S13b fitting")
        if ObjectiveTerm.DATA_PEAK not in result.final_losses:
            raise RuntimeError("S13b checkpoint is missing DATA_PEAK loss")
        if progress is not None:
            progress(
                "phase=CHECKPOINT non_gate=true role=TRAIN "
                f"step={result.step} model={result.model_state_hash} "
                f"manifest={result.checkpoint.manifest.manifest_hash} "
                f"peak_loss={result.final_losses[ObjectiveTerm.DATA_PEAK]:.9g}"
            )
        resume_from = checkpoint_path
        if target < config.trial.optimizer_steps:
            initialization = load_selected_s11_initialization(s11)
    if result is None:
        raise RuntimeError("S13b peak pilot produced no training segment")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the separate non-gate 800-step S13b TRAIN-fit peak pilot."
    )
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--s11-checkpoint-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--segment-steps", default=100, type=int)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    result = run_s13b_peak_pilot(
        s10_run_root=options.s10_run_root,
        s11_checkpoint_root=options.s11_checkpoint_root,
        run_root=options.run_root,
        segment_steps=options.segment_steps,
        progress=lambda message: print(message, flush=True),
    )
    print(f"pilot_checkpoint={result.checkpoint.path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = ("build_parser", "main", "run_s13b_peak_pilot")
