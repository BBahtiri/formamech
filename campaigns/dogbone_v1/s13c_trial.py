"""Registered S13c TRAIN-fit training with resumable checkpoints.

Usage (from the repository root, with a clean committed revision)::

    python -m campaigns.dogbone_v1.s13c_trial \
        --s10-run-root <abs path to the S10 production run root> \
        --run-root <abs path outside the repository and the S10 root>

The run root accumulates ``checkpoint-step-NNNN`` directories, the canonical
``run-binding.json`` written at launch, and an append-only
``training-log.jsonl``. Re-running the same command resumes from the latest
checkpoint with the exact optimizer, generator, sampler and schedule state.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import TextIO

import torch

from campaigns.dogbone_v1.s11_contingency_source import open_selective_s10_source
from campaigns.dogbone_v1.s13_pilot import _absolute_directory, _git_provenance
from campaigns.dogbone_v1.s13c_data import materialize_dogbone_s13c_training_data
from campaigns.dogbone_v1.s13c_protocol import (
    DogboneS13cProtocol,
    build_dogbone_s13c_protocol,
    build_dogbone_s13c_sampler_plan,
    build_dogbone_s13c_trainer_config,
    build_initial_operator,
    load_s13c_normalization,
)
from piginot.contracts import ExecutionProfile, canonical_json_bytes
from piginot.operators import EnergyFormOperator
from piginot.training import (
    DeterministicSamplerPlan,
    EnergyStepRecord,
    EnergyTrainerConfig,
    EnergyTrainingResult,
    TrainingCheckpointManifest,
    TrainingRunBinding,
    model_state_hash,
    train_energy_to_checkpoint,
)

_ROOT = Path(__file__).resolve().parents[2]
_PROTOCOL_PATH = _ROOT / "artifacts" / "campaigns" / "dogbone-v1" / "S13C-PROTOCOL.json"
_CHECKPOINT_NAME = re.compile(r"checkpoint-step-([0-9]{4})\Z")
RUN_BINDING_NAME = "run-binding.json"
TRAINING_LOG_NAME = "training-log.jsonl"
DEVELOPMENT_ACCESS_NAME = "development-access.json"
_ALLOWED_RUN_FILES = frozenset((RUN_BINDING_NAME, TRAINING_LOG_NAME))
Progress = Callable[[str], None]


def load_committed_s13c_protocol(path: Path | None = None) -> DogboneS13cProtocol:
    """Rebuild the protocol and require byte equality with the committed copy."""

    protocol = build_dogbone_s13c_protocol()
    committed = (_PROTOCOL_PATH if path is None else path).read_bytes()
    if committed != protocol.to_bytes():
        raise ValueError("S13c protocol differs from the committed preregistration")
    return protocol


def execution_profile(protocol: DogboneS13cProtocol) -> ExecutionProfile:
    """Return the CPU execution profile bound into every S13c run hash."""

    return ExecutionProfile(
        python_version=platform.python_version(),
        library_versions={
            "torch": torch.__version__,
            "safetensors": version("safetensors"),
            "rfc8785": version("rfc8785"),
        },
        blas_threads=protocol.execution_blas_threads,
        device="cpu",
        dtype=protocol.execution_dtype,
        rng_algorithm="torch-cpu-mt19937",
        seed=protocol.training_seed,
        container_digest=None,
    )


def build_s13c_binding(
    protocol: DogboneS13cProtocol,
    *,
    model: EnergyFormOperator,
    config: EnergyTrainerConfig,
    plan: DeterministicSamplerPlan,
    implementation_revision: str,
) -> TrainingRunBinding:
    """Bind data, operator, objective, policy, code and profile into one run."""

    initial = model_state_hash(model)
    if initial != protocol.initial_weights_hash:
        raise ValueError("initial operator weights differ from the protocol")
    return TrainingRunBinding(
        dataset_snapshot_hash=protocol.dataset_snapshot_hash,
        public_manifest_hash=protocol.public_manifest_hash,
        normalization_hash=protocol.normalization_hash,
        architecture_id=protocol.architecture_id,
        architecture_module_digest=protocol.architecture_module_digest,
        objective_registry_hash=protocol.objective_hash,
        trainer_config_hash=config.config_hash,
        sampler_plan_hash=plan.plan_hash,
        initial_weights_hash=initial,
        implementation_revision=implementation_revision,
        seed=protocol.training_seed,
        execution_profile=execution_profile(protocol),
    )


def _prepare_run_root(s10: Path, run_root: Path) -> Path:
    if not isinstance(run_root, Path) or not run_root.is_absolute():
        raise ValueError("run_root must be an absolute pathlib.Path")
    parent = _absolute_directory(run_root.parent, "run_root parent")
    output = parent / run_root.name
    if output.is_relative_to(s10) or output.is_relative_to(_ROOT):
        raise ValueError("run_root must remain outside the S10 root and repository")
    if output.is_symlink():
        raise ValueError("run_root must not be a symlink")
    if not output.exists():
        output.mkdir()
    return _absolute_directory(output, "run_root")


def _bind_run_root(output: Path, binding: TrainingRunBinding) -> None:
    payload = canonical_json_bytes(binding)
    path = output / RUN_BINDING_NAME
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("run root belongs to a different S13c run binding")
        return
    path.write_bytes(payload)


def latest_checkpoint(
    run_root: Path,
    *,
    expected_run_hash: str,
) -> tuple[Path | None, int]:
    """Return the newest complete checkpoint of this run and its step."""

    checkpoints: list[tuple[int, Path]] = []
    for entry in run_root.iterdir():
        if entry.name in _ALLOWED_RUN_FILES and entry.is_file():
            continue
        if entry.name.startswith(".checkpoint-step-"):
            raise ValueError(f"incomplete checkpoint staging found: {entry.name}")
        match = _CHECKPOINT_NAME.fullmatch(entry.name)
        if match is None or entry.is_symlink() or not entry.is_dir():
            raise ValueError(f"unexpected entry in S13c run root: {entry.name}")
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


def segment_targets(
    last_step: int,
    total_steps: int,
    segment_steps: int,
    *,
    stop_after: int | None = None,
) -> tuple[int, ...]:
    """Return checkpoint steps after ``last_step`` up to the stop limit."""

    limit = total_steps if stop_after is None else stop_after
    if type(limit) is not int or not 0 < limit <= total_steps:
        raise ValueError("stop_after must lie inside the optimizer budget")
    if type(segment_steps) is not int or segment_steps <= 0:
        raise ValueError("segment_steps must be a positive integer")
    if type(last_step) is not int or last_step < 0:
        raise ValueError("last_step must be a non-negative integer")
    if last_step >= limit:
        return ()
    return tuple(
        min(step, limit)
        for step in range(
            last_step + segment_steps, limit + segment_steps, segment_steps
        )
    )


def _log_writer(stream: TextIO) -> Callable[[EnergyStepRecord], None]:
    def write(record: EnergyStepRecord) -> None:
        stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        stream.flush()

    return write


def run_registered_s13c_trial(
    *,
    s10_run_root: Path,
    run_root: Path,
    segment_steps: int = 400,
    stop_after: int | None = None,
    require_clean_revision: bool = True,
    progress: Progress | None = None,
) -> EnergyTrainingResult | None:
    """Train the single registered S13c trial on TRAIN-fit geometry only."""

    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    protocol = load_committed_s13c_protocol()
    config = build_dogbone_s13c_trainer_config(protocol)
    plan = build_dogbone_s13c_sampler_plan(protocol)
    provenance = _git_provenance()
    if require_clean_revision and provenance.dirty_worktree:
        raise RuntimeError("registered S13c training requires a clean revision")
    s10 = _absolute_directory(s10_run_root, "s10_run_root")
    binding = build_s13c_binding(
        protocol,
        model=build_initial_operator(),
        config=config,
        plan=plan,
        implementation_revision=provenance.head,
    )
    output = _prepare_run_root(s10, run_root)
    _bind_run_root(output, binding)
    resume_from, last_step = latest_checkpoint(
        output, expected_run_hash=binding.run_hash
    )
    targets = segment_targets(
        last_step,
        config.trial.optimizer_steps,
        segment_steps,
        stop_after=stop_after,
    )
    if progress is not None:
        progress(
            "phase=PREFLIGHT role=TRAIN "
            f"trial={config.trial.trial_id} revision={provenance.head} "
            f"protocol={protocol.protocol_hash} run={binding.run_hash} "
            f"resume_step={last_step} targets={targets}"
        )
    if not targets:
        return None
    source = open_selective_s10_source(
        s10,
        development_access_record=output / DEVELOPMENT_ACCESS_NAME,
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels must remain closed during S13c")
    data = materialize_dogbone_s13c_training_data(
        source,
        protocol,
        load_s13c_normalization(),
        dtype=torch.float32,
    )
    objective = data.objective(protocol.objective.energy)
    if progress is not None:
        progress(
            "phase=TRAIN_FIT role=TRAIN "
            f"cases={len(data.population.sample_ids)} "
            f"source_bindings={data.source_bindings_hash}"
        )

    result: EnergyTrainingResult | None = None
    with (output / TRAINING_LOG_NAME).open("a", encoding="utf-8") as log:
        writer = _log_writer(log)
        for target in targets:
            checkpoint_path = output / f"checkpoint-step-{target:04d}"
            result = train_energy_to_checkpoint(
                checkpoint_path,
                model=build_initial_operator(),
                population=data.population,
                plan=plan,
                binding=binding,
                config=config,
                objective=objective,
                objective_hash=protocol.objective_hash,
                target_step=target,
                resume_from=resume_from,
                progress=writer,
            )
            if source.development_opened:
                raise RuntimeError("DEVELOPMENT labels opened during S13c training")
            last = result.history[-1]
            if progress is not None:
                progress(
                    "phase=CHECKPOINT role=TRAIN "
                    f"step={result.step} model={result.model_state_hash} "
                    f"manifest={result.checkpoint.manifest.manifest_hash} "
                    f"energy={last.energy:.9g} barrier={last.barrier:.3g} "
                    f"min_det={last.minimum_determinant:.4f} lr={last.learning_rate:.3g}"
                )
            resume_from = checkpoint_path
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the registered S13c energy-form trial (TRAIN-fit only)."
    )
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--segment-steps", type=int, default=400)
    parser.add_argument("--stop-after", type=int, default=None)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    run_registered_s13c_trial(
        s10_run_root=options.s10_run_root,
        run_root=options.run_root,
        segment_steps=options.segment_steps,
        stop_after=options.stop_after,
        progress=lambda message: print(message, flush=True),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = (
    "DEVELOPMENT_ACCESS_NAME",
    "RUN_BINDING_NAME",
    "TRAINING_LOG_NAME",
    "build_parser",
    "build_s13c_binding",
    "execution_profile",
    "latest_checkpoint",
    "load_committed_s13c_protocol",
    "main",
    "run_registered_s13c_trial",
    "segment_targets",
)
