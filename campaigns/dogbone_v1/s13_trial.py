"""Clean-revision, resumable execution of one registered S13 TRAIN-fit trial."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from campaigns.dogbone_v1.s11_contingency_source import open_selective_s10_source
from campaigns.dogbone_v1.s13_data import materialize_dogbone_s13_training_data
from campaigns.dogbone_v1.s13_initialization import load_selected_s11_initialization
from campaigns.dogbone_v1.s13_pilot import (
    _absolute_directory,
    _build_pilot_binding,
    _git_provenance,
    _verify_training_data,
)
from campaigns.dogbone_v1.s13_protocol import (
    build_dogbone_s13_protocol,
    build_dogbone_s13_sampler_plan,
    build_dogbone_s13_trainer_configs,
)
from piginot.training import (
    TrainingCheckpointManifest,
    TrainingRunResult,
    train_to_checkpoint,
)

_CHECKPOINT_NAME = re.compile(r"checkpoint-step-([0-9]{4})\Z")
Progress = Callable[[str], None]


def _segment_targets(
    last_step: int, total_steps: int, segment_steps: int
) -> tuple[int, ...]:
    if type(last_step) is not int or last_step < 0:
        raise ValueError("last_step must be a non-negative integer")
    if type(total_steps) is not int or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    if type(segment_steps) is not int or segment_steps <= 0:
        raise ValueError("segment_steps must be a positive integer")
    if last_step >= total_steps:
        raise ValueError("registered trial is already complete")
    return tuple(
        min(step, total_steps)
        for step in range(
            last_step + segment_steps, total_steps + segment_steps, segment_steps
        )
    )


def _prepare_run_root(s10: Path, s11: Path, run_root: Path) -> Path:
    if not isinstance(run_root, Path) or not run_root.is_absolute():
        raise ValueError("run_root must be an absolute pathlib.Path")
    parent = _absolute_directory(run_root.parent, "run_root parent")
    output = parent / run_root.name
    if output.is_relative_to(s10) or output.is_relative_to(s11):
        raise ValueError("run_root must remain outside source roots")
    if output.is_symlink():
        raise ValueError("run_root must not be a symlink")
    if output.exists():
        return _absolute_directory(output, "run_root")
    output.mkdir()
    return output


def _latest_checkpoint(
    run_root: Path, *, expected_run_hash: str
) -> tuple[Path | None, int]:
    checkpoints: list[tuple[int, Path]] = []
    for entry in run_root.iterdir():
        match = _CHECKPOINT_NAME.fullmatch(entry.name)
        if match is None or entry.is_symlink() or not entry.is_dir():
            raise ValueError(f"unexpected entry in S13 trial run root: {entry.name}")
        checkpoints.append((int(match.group(1)), entry))
    if not checkpoints:
        return None, 0
    step, latest = max(checkpoints)
    manifest = TrainingCheckpointManifest.from_bytes(
        (latest / "checkpoint.json").read_bytes()
    )
    if manifest.step != step:
        raise ValueError("checkpoint step differs from its directory name")
    if manifest.run_hash != expected_run_hash:
        raise ValueError("checkpoint run_hash differs from this registered trial")
    return latest, step


def run_registered_s13_trial(
    *,
    s10_run_root: Path,
    s11_checkpoint_root: Path,
    run_root: Path,
    trial_index: int,
    segment_steps: int = 100,
    progress: Progress | None = None,
) -> TrainingRunResult:
    """Train one of six frozen trials; no TRAIN-screen/DEVELOPMENT/HOLDOUT access."""

    protocol = build_dogbone_s13_protocol()
    configs = build_dogbone_s13_trainer_configs(protocol)
    if type(trial_index) is not int or not 0 <= trial_index < len(configs):
        raise ValueError("trial_index must identify one registered trial")
    if type(segment_steps) is not int or segment_steps <= 0:
        raise ValueError("segment_steps must be a positive integer")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    s10 = _absolute_directory(s10_run_root, "s10_run_root")
    s11 = _absolute_directory(s11_checkpoint_root, "s11_checkpoint_root")
    provenance = _git_provenance()
    if provenance.dirty_worktree:
        raise RuntimeError("registered S13 training requires a clean revision")
    config = configs[trial_index]
    plan = build_dogbone_s13_sampler_plan(protocol)
    initialization = load_selected_s11_initialization(s11)
    binding = _build_pilot_binding(
        protocol,
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
            "phase=PREFLIGHT role=TRAIN "
            f"trial={config.trial.trial_id} revision={provenance.head} "
            f"protocol={protocol.protocol_hash} run={binding.run_hash} "
            f"resume_step={last_step} targets={targets}"
        )
    source = open_selective_s10_source(
        s10,
        development_access_record=output / "development-access.json",
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels must remain closed during S13 training")
    data = materialize_dogbone_s13_training_data(source, initialization)
    _verify_training_data(data, protocol, source.bindings.content_hash)
    if progress is not None:
        progress(
            "phase=TRAIN_FIT role=TRAIN "
            f"cases={len(data.population.sample_ids)} "
            f"source_bindings={source.bindings.content_hash}"
        )

    result: TrainingRunResult | None = None
    objective = data.objective_builder()
    for target in targets:
        checkpoint_path = output / f"checkpoint-step-{target:04d}"
        result = train_to_checkpoint(
            checkpoint_path,
            model=initialization.model,
            population=data.population,
            plan=plan,
            binding=binding,
            config=config,
            base_registry=protocol.objective_registry,
            objective=objective,
            target_step=target,
            resume_from=resume_from,
        )
        if source.development_opened:
            raise RuntimeError("DEVELOPMENT labels opened during S13 training")
        if progress is not None:
            progress(
                "phase=CHECKPOINT role=TRAIN "
                f"trial={config.trial.trial_id} step={result.step} "
                f"model={result.model_state_hash} "
                f"manifest={result.checkpoint.manifest.manifest_hash} "
                f"losses={{{', '.join(f'{term.value}: {value:.9g}' for term, value in result.final_losses.items())}}}"
            )
        resume_from = checkpoint_path
        if target < config.trial.optimizer_steps:
            initialization = load_selected_s11_initialization(s11)
    if result is None:
        raise RuntimeError("registered trial produced no training segment")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one registered S13 TRAIN-fit trial."
    )
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--s11-checkpoint-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--trial-index", required=True, type=int)
    parser.add_argument("--segment-steps", default=100, type=int)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    result = run_registered_s13_trial(
        s10_run_root=options.s10_run_root,
        s11_checkpoint_root=options.s11_checkpoint_root,
        run_root=options.run_root,
        trial_index=options.trial_index,
        segment_steps=options.segment_steps,
        progress=lambda message: print(message, flush=True),
    )
    print(f"final_checkpoint={result.checkpoint.path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = ("build_parser", "main", "run_registered_s13_trial")
