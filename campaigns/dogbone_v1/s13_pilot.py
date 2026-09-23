"""Non-gate, TRAIN-fit-only smoke run for the first registered S13 trial."""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import torch

from campaigns.dogbone_v1.s11_contingency_source import open_selective_s10_source
from campaigns.dogbone_v1.s13_data import (
    DogboneS13TrainingData,
    materialize_dogbone_s13_training_data,
)
from campaigns.dogbone_v1.s13_initialization import load_selected_s11_initialization
from campaigns.dogbone_v1.s13_protocol import (
    DogboneS13Protocol,
    build_dogbone_s13_protocol,
    build_dogbone_s13_sampler_plan,
    build_dogbone_s13_trainer_configs,
)
from piginot.contracts import ExecutionProfile, hash_spec
from piginot.git_environment import isolated_git_environment
from piginot.training import (
    DeterministicSamplerPlan,
    DeterministicTrainerConfig,
    TrainingRunBinding,
    TrainingRunResult,
    model_state_hash,
    train_to_checkpoint,
)

_ROOT = Path(__file__).resolve().parents[2]
_PILOT_STEPS = 20
Progress = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class PilotGitProvenance:
    """HEAD plus a diagnostic fingerprint; not a clean-revision attestation."""

    head: str
    dirty_worktree: bool
    status_and_tracked_diff_blake2b256: str


def _absolute_directory(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{label} must be a pathlib.Path")
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be an existing non-symlink directory")
    return path.resolve(strict=True)


def _preflight_paths(
    s10_run_root: Path,
    s11_checkpoint_root: Path,
    output_root: Path,
) -> tuple[Path, Path, Path]:
    s10 = _absolute_directory(s10_run_root, "s10_run_root")
    s11 = _absolute_directory(s11_checkpoint_root, "s11_checkpoint_root")
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a pathlib.Path")
    if not output_root.is_absolute():
        raise ValueError("output_root must be absolute")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("output_root must not already exist")
    parent = _absolute_directory(output_root.parent, "output_root parent")
    output = parent / output_root.name
    if output.is_relative_to(s10) or output.is_relative_to(s11):
        raise ValueError("output_root must remain outside source roots")
    return s10, s11, output


def _git(*arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=_ROOT,
        env=isolated_git_environment(),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Git diagnostic failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return completed.stdout


def _git_provenance() -> PilotGitProvenance:
    head = _git("rev-parse", "HEAD").decode("ascii").strip()
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    tracked_diff = _git("diff", "HEAD", "--binary")
    fingerprint = hashlib.blake2b(
        status + b"\0" + tracked_diff, digest_size=32
    ).hexdigest()
    return PilotGitProvenance(head, bool(status), fingerprint)


def _verify_training_data(
    data: DogboneS13TrainingData,
    protocol: DogboneS13Protocol,
    source_bindings_hash: str,
) -> None:
    if type(data) is not DogboneS13TrainingData:
        raise TypeError("TRAIN materialization must be exact DogboneS13TrainingData")
    if (
        data.population.sample_ids != protocol.fit_case_ids
        or data.population.dataset_snapshot_hash != protocol.dataset_snapshot_hash
        or data.population.public_manifest_hash != protocol.public_manifest_hash
    ):
        raise ValueError("TRAIN-fit population differs from the frozen protocol")
    if data.normalization.content_hash != protocol.normalization_hash:
        raise ValueError("TRAIN normalization differs from the frozen protocol")
    if (
        data.source_bindings_hash != protocol.source_bindings_hash
        or source_bindings_hash != protocol.source_bindings_hash
    ):
        raise ValueError("S10 source bindings differ from the frozen protocol")
    if (
        data.selected_candidate_id != protocol.selected_candidate_id
        or data.selected_checkpoint_file_hash
        != protocol.initial_checkpoint_file_hash
    ):
        raise ValueError("selected S11 initialization differs from the protocol")


def _build_pilot_binding(
    protocol: DogboneS13Protocol,
    *,
    model: torch.nn.Module,
    config: DeterministicTrainerConfig,
    plan: DeterministicSamplerPlan,
    implementation_revision: str,
) -> TrainingRunBinding:
    initial_hash = model_state_hash(model)
    if initial_hash != protocol.initial_trainer_weights_hash:
        raise ValueError("selected S11 model differs from initial trainer weights")
    seed = protocol.training_seed
    profile = ExecutionProfile(
        python_version=platform.python_version(),
        library_versions={
            "torch": torch.__version__,
            "safetensors": version("safetensors"),
            "rfc8785": version("rfc8785"),
        },
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="torch-cpu-mt19937",
        seed=seed,
        container_digest=None,
    )
    return TrainingRunBinding(
        dataset_snapshot_hash=protocol.dataset_snapshot_hash,
        public_manifest_hash=protocol.public_manifest_hash,
        normalization_hash=protocol.normalization_hash,
        architecture_id=protocol.selected_candidate_id,
        architecture_module_digest=protocol.architecture_module_digest,
        objective_registry_hash=hash_spec(protocol.objective_registry, "recipe"),
        trainer_config_hash=config.config_hash,
        sampler_plan_hash=plan.plan_hash,
        initial_weights_hash=initial_hash,
        implementation_revision=implementation_revision,
        seed=seed,
        execution_profile=profile,
    )


def run_s13_fit_pilot(
    *,
    s10_run_root: Path,
    s11_checkpoint_root: Path,
    output_root: Path,
    progress: Progress | None = None,
) -> TrainingRunResult:
    """Run 20 steps on TRAIN-fit only; results cannot satisfy G13 or CP-FREEZE."""

    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    s10, s11, output = _preflight_paths(
        s10_run_root, s11_checkpoint_root, output_root
    )
    protocol = build_dogbone_s13_protocol()
    provenance = _git_provenance()
    initialization = load_selected_s11_initialization(s11)
    plan = build_dogbone_s13_sampler_plan(protocol)
    config = build_dogbone_s13_trainer_configs(protocol)[0]
    binding = _build_pilot_binding(
        protocol,
        model=initialization.model,
        config=config,
        plan=plan,
        implementation_revision=provenance.head,
    )
    if progress is not None:
        progress(
            "phase=PREFLIGHT non_gate=true train_fit_only=true "
            f"head={provenance.head} dirty={str(provenance.dirty_worktree).lower()} "
            "status_and_tracked_diff_blake2b256="
            f"{provenance.status_and_tracked_diff_blake2b256} "
            f"protocol={protocol.protocol_hash} trial={config.trial.trial_id}"
        )

    output.mkdir()
    source = open_selective_s10_source(
        s10,
        development_access_record=output / "development-access.json",
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels must remain closed for the S13 pilot")
    data = materialize_dogbone_s13_training_data(source, initialization)
    _verify_training_data(data, protocol, source.bindings.content_hash)
    if progress is not None:
        progress(
            "phase=TRAIN_FIT non_gate=true "
            f"cases={len(protocol.fit_case_ids)} steps={_PILOT_STEPS} "
            f"source_bindings={source.bindings.content_hash} run={binding.run_hash}"
        )
    result = train_to_checkpoint(
        output / "checkpoint",
        model=initialization.model,
        population=data.population,
        plan=plan,
        binding=binding,
        config=config,
        base_registry=protocol.objective_registry,
        objective=data.objective_builder(),
        target_step=_PILOT_STEPS,
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels were unexpectedly opened")
    if progress is not None:
        progress(
            "phase=COMPLETE non_gate=true "
            f"step={result.step} model={result.model_state_hash}"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a non-gate 20-step S13 TRAIN-fit diagnostic."
    )
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--s11-checkpoint-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    result = run_s13_fit_pilot(
        s10_run_root=options.s10_run_root,
        s11_checkpoint_root=options.s11_checkpoint_root,
        output_root=options.output_root,
        progress=lambda message: print(message, flush=True),
    )
    print(f"checkpoint={result.checkpoint.path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = ("build_parser", "main", "run_s13_fit_pilot")
