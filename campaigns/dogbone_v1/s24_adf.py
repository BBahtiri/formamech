"""S24: the registered ADF viability run on DogBone (ADR 0106 decision 5).

The S13c recipe unchanged — protocol, data, seed, initial weights, sampler
plan, trainer configuration, objective and execution profile — except the
lifting, which is the ``adf`` lifting with the trimmed pieces of
``campaigns/dogbone_v1/problem_spec_adf.json``. The run binding binds that
spec's hash. It is judged on the 64 TRAIN-screen cases by the frozen S13
guardrails; DEVELOPMENT, HOLDOUT, H1 and H2 stay closed.

Usage (repository root, clean committed revision)::

    python -m campaigns.dogbone_v1.s24_adf protocol --write
    python -m campaigns.dogbone_v1.s24_adf train \
        --s10-run-root <S10 production run root> --run-root <new run root>
    python -m campaigns.dogbone_v1.s24_adf evaluate \
        --s10-run-root <S10 production run root> \
        --checkpoint-path <run root>/checkpoint-step-2400 --write

``train`` refuses to start unless the protocol is committed and accepted and
matches the spec and the S13c protocol; it resumes like the S13c trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from functools import cache
from pathlib import Path
from typing import Any

import torch

from campaigns.dogbone_v1.s11_contingency_data import CaseNormalizationStatistics
from campaigns.dogbone_v1.s11_contingency_source import (
    SelectiveS10Source,
    VerifiedS10Request,
    open_selective_s10_source,
)
from campaigns.dogbone_v1.s13_pilot import _absolute_directory, _git_provenance
from campaigns.dogbone_v1.s13c_data import (
    DogboneS13cGeometryCase,
    DogboneS13cTrainingData,
    geometry_case,
)
from campaigns.dogbone_v1.s13c_protocol import (
    DogboneS13cProtocol,
    build_dogbone_s13c_sampler_plan,
    build_dogbone_s13c_trainer_config,
    build_initial_operator,
    load_s13c_normalization,
)
from campaigns.dogbone_v1.s13c_trial import (
    _PROTOCOL_PATH as S13C_PROTOCOL_PATH,
)
from campaigns.dogbone_v1.s13c_trial import (
    DEVELOPMENT_ACCESS_NAME,
    TRAINING_LOG_NAME,
    Progress,
    _bind_run_root,
    _log_writer,
    _prepare_run_root,
    build_s13c_binding,
    latest_checkpoint,
    load_committed_s13c_protocol,
    segment_targets,
)
from piginot.contracts import ProblemSpec, canonical_json_bytes
from piginot.doe import EvidenceRole
from piginot.lifting import DirichletLifting
from piginot.problem import build_lifting
from piginot.training import (
    EnergyTrainingResult,
    TrainingPopulation,
    TrainingRunBinding,
    train_energy_to_checkpoint,
)

REPOSITORY = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("campaigns/dogbone_v1/problem_spec_adf.json")
BASE_SPEC_PATH = Path("campaigns/dogbone_v1/problem_spec.json")
PROTOCOL_PATH = Path("artifacts/problem/S24-PROTOCOL.json")
SCREEN_PATH = Path("artifacts/problem/S24-ADF-TRAIN-SCREEN.json")
S13C_SCREEN_PATH = Path("artifacts/campaigns/dogbone-v1/S13C-TRAIN-SCREEN.json")
LIFTING_INPUTS = ("L_total", "W_grip", "W_gauge", "nominal_axial_strain")
DISPLACEMENT_TOLERANCE = 1e-12


@cache
def load_spec(path: Path = SPEC_PATH) -> ProblemSpec:
    spec = ProblemSpec.load(REPOSITORY / path)
    if spec.lifting.kind != "adf":
        raise ValueError("the viability spec must use the adf lifting")
    return spec


@dataclass(frozen=True, slots=True)
class DogboneS24AdfCase(DogboneS13cGeometryCase):
    """An S13c geometry case whose lifting comes from the ADF spec."""

    lifting_values: Mapping[str, float]

    def lifting(self) -> DirichletLifting:  # type: ignore[override]
        return build_lifting(load_spec(), [self.lifting_values], dtype=self.dtype)


def adf_geometry_case(
    verified: VerifiedS10Request,
    normalization: CaseNormalizationStatistics,
    *,
    dtype: torch.dtype,
) -> DogboneS24AdfCase:
    """The S13c case of ``verified`` with the spec's ADF lifting."""

    base = geometry_case(verified, normalization, dtype=dtype)
    coordinates = verified.case.primitive_coordinates
    values = {name: float(coordinates[name]) for name in LIFTING_INPUTS}
    scale = load_spec().lifting.displacement_scale.evaluate(values)
    expected = base.prescribed_displacement_mm
    if abs(scale - expected) > DISPLACEMENT_TOLERANCE * abs(expected):
        raise ValueError(f"{base.case_id}: spec displacement scale differs from S10")
    return DogboneS24AdfCase(
        **{item.name: getattr(base, item.name) for item in fields(base)},
        lifting_values=values,
    )


def materialize_adf_training_data(
    source: SelectiveS10Source,
    protocol: DogboneS13cProtocol,
    normalization: CaseNormalizationStatistics,
    *,
    dtype: torch.dtype = torch.float32,
) -> DogboneS13cTrainingData:
    """The S13c TRAIN-fit population with ADF cases (same checks as S13c)."""

    if type(source) is not SelectiveS10Source:
        raise TypeError("source must be an exact verified SelectiveS10Source")
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels must remain closed")
    if source.bindings.content_hash != protocol.source_bindings_hash:
        raise ValueError("S10 source bindings differ from the S13c protocol")
    if source.fit_case_ids != protocol.fit_case_ids:
        raise ValueError("S10 TRAIN-fit population differs from the protocol")
    if normalization.content_hash != protocol.normalization_hash:
        raise ValueError("normalization differs from the S13c protocol")
    if source.train_snapshot.content_hash != protocol.dataset_snapshot_hash:
        raise ValueError("S10 TRAIN snapshot differs from the S13c protocol")
    cases = tuple(
        adf_geometry_case(verified, normalization, dtype=dtype)
        for verified in source.iter_train_geometry(protocol.fit_case_ids)
    )
    if len({case.leakage_group_id for case in cases}) != len(cases):
        raise ValueError("TRAIN-fit leakage groups must be unique")
    population = TrainingPopulation(
        dataset_snapshot_hash=protocol.dataset_snapshot_hash,
        public_manifest_hash=protocol.public_manifest_hash,
        role=EvidenceRole.TRAIN,
        sample_ids=protocol.fit_case_ids,
    )
    return DogboneS13cTrainingData(
        cases=cases,
        population=population,
        source_bindings_hash=source.bindings.content_hash,
    )


# ------------------------------------------------------------------ protocol


def _sha256(path: Path) -> str:
    return hashlib.sha256((REPOSITORY / path).read_bytes()).hexdigest()


def build_protocol(root: Path = REPOSITORY) -> dict[str, Any]:
    """The S24 preregistration (ADR 0106, accepted 2026-10-13)."""

    s13c = load_committed_s13c_protocol()
    config = build_dogbone_s13c_trainer_config(s13c)
    spec = load_spec()
    base = ProblemSpec.load(root / BASE_SPEC_PATH)
    screen = json.loads((root / S13C_SCREEN_PATH).read_text("utf-8"))
    replay = json.loads(
        (root / "artifacts/campaigns/dogbone-v1/S13C-G13-REPLAY-PROBE.json").read_text(
            "utf-8"
        )
    )
    notched = json.loads(
        (root / "artifacts/campaigns/notched-v1/S13-G13-REPLAY-PROBE.json").read_text(
            "utf-8"
        )
    )
    return {
        "schema_version": "1.0",
        "step": "S24",
        "status": "accepted",
        "decision_record": "docs/adr/0106-s24-problem-spec-and-lifting.md",
        "accepted": {
            "by": "decision owner",
            "on": "2026-10-13",
            "answers": {
                "ADR 0106": "Accept as written",
                "harmonic lifting": "Defer beyond 0.2.0",
                "viability run": "Full 2,400 steps",
                "kill rule": "Keep untrimmed lines",
            },
        },
        "viability": {
            "campaign": "dogbone-v1",
            "spec": SPEC_PATH.as_posix(),
            "problem_spec_hash": spec.problem_hash,
            "base_spec": BASE_SPEC_PATH.as_posix(),
            "base_problem_spec_hash": base.problem_hash,
            "s13c_protocol_hash": s13c.protocol_hash,
            "s13c_protocol_sha256": _sha256(S13C_PROTOCOL_PATH.relative_to(root)),
            "unchanged": [
                "TRAIN-fit population and S10 source bindings",
                "normalization",
                "architecture and initial weights",
                "objective and trainer configuration",
                "sampler plan and seed",
                "execution profile",
            ],
            "changed": "lifting: adf, trimmed pieces of the spec; the run binding "
            "binds problem_spec_hash",
            "optimizer_steps": config.trial.optimizer_steps,
            "initial_weights_hash": s13c.initial_weights_hash,
            "seed": s13c.training_seed,
            "evaluation": {
                "population": "TRAIN-screen",
                "cases": len(s13c.screen_case_ids),
                "rule": "the frozen S13 guardrails of the S13c protocol "
                "(campaigns.dogbone_v1.s13c_evaluate.failed_guardrails)",
                "guardrail_limits": screen["comparison"]["guardrail_limits"],
                "reported": {
                    "s13c_p_relative_l2": screen["comparison"]["s13c"]["p_relative_l2"],
                },
            },
            "closed_evidence": ["DEVELOPMENT", "HOLDOUT", "H1", "H2"],
            "kill_rule": "if a guardrail fails, trimmed pieces and arcs are not "
            "offered in 0.2.0; untrimmed lines stay; shipped bundles unaffected",
        },
        "replays": {
            "dogbone-v1": {
                "run": "registered S13c run",
                "run_hash": replay["run_hash"],
                "target_step": replay["target_step"],
                "model_state_hash": replay["registered_model_state_hash"],
                "command": "python -m campaigns.dogbone_v1.s13c_trial "
                "--s10-run-root <S10> --run-root <new> --stop-after 400",
            },
            "notched-v1": {
                "run": "registered notched S13 run",
                "run_hash": notched["run_hash"],
                "target_step": notched["target_step"],
                "model_state_hash": notched["registered_model_state_hash"],
                "command": "python -m campaigns.notched_v1.s13 train "
                "--run-root <new> --stop-after 400",
            },
            "rule": "every tensor of model.safetensors and state.safetensors equal "
            "and every manifest field equal except run_hash, model_export_hash and "
            "tensor_payload_hash (they hash the run hash); compared by "
            "campaigns.s24_replay",
        },
    }


def _committed(path: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", path.as_posix()],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path.as_posix()],
        cwd=REPOSITORY,
        capture_output=True,
        check=False,
    )
    return (
        result.returncode == 0
        and not result.stdout.strip()
        and (tracked.returncode == 0)
    )


def accepted_protocol(*, require_committed: bool = True) -> dict[str, Any]:
    """The committed protocol, refused unless it is current and accepted."""

    path = REPOSITORY / PROTOCOL_PATH
    if not path.is_file():
        raise RuntimeError(f"{PROTOCOL_PATH} is not written")
    if require_committed and not _committed(PROTOCOL_PATH):
        raise RuntimeError(f"{PROTOCOL_PATH} must be committed before the run")
    document: dict[str, Any] = json.loads(path.read_text("utf-8"))
    if document != build_protocol():
        raise ValueError("the committed S24 protocol differs from its rebuild")
    if document["status"] != "accepted":
        raise RuntimeError("the S24 protocol is not accepted")
    return document


# --------------------------------------------------------------------- train


def build_adf_binding(
    protocol: DogboneS13cProtocol, implementation_revision: str
) -> TrainingRunBinding:
    binding = build_s13c_binding(
        protocol,
        model=build_initial_operator(),
        config=build_dogbone_s13c_trainer_config(protocol),
        plan=build_dogbone_s13c_sampler_plan(protocol),
        implementation_revision=implementation_revision,
    )
    return replace(binding, problem_spec_hash=load_spec().problem_hash)


def run_adf_trial(
    *,
    s10_run_root: Path,
    run_root: Path,
    segment_steps: int = 400,
    stop_after: int | None = None,
    require_clean_revision: bool = True,
    progress: Progress | None = None,
) -> EnergyTrainingResult | None:
    """Train the registered ADF viability run on TRAIN-fit geometry only."""

    accepted_protocol(require_committed=require_clean_revision)
    protocol = load_committed_s13c_protocol()
    config = build_dogbone_s13c_trainer_config(protocol)
    plan = build_dogbone_s13c_sampler_plan(protocol)
    provenance = _git_provenance()
    if require_clean_revision and provenance.dirty_worktree:
        raise RuntimeError("the registered ADF run requires a clean revision")
    s10 = _absolute_directory(s10_run_root, "s10_run_root")
    binding = build_adf_binding(protocol, provenance.head)
    output = _prepare_run_root(s10, run_root)
    _bind_run_root(output, binding)
    resume_from, last_step = latest_checkpoint(
        output, expected_run_hash=binding.run_hash
    )
    targets = segment_targets(
        last_step, config.trial.optimizer_steps, segment_steps, stop_after=stop_after
    )
    if progress is not None:
        progress(
            f"phase=PREFLIGHT role=TRAIN lifting=adf revision={provenance.head} "
            f"spec={binding.problem_spec_hash} run={binding.run_hash} "
            f"resume_step={last_step} targets={targets}"
        )
    if not targets:
        return None
    source = open_selective_s10_source(
        s10, development_access_record=output / DEVELOPMENT_ACCESS_NAME
    )
    data = materialize_adf_training_data(
        source, protocol, load_s13c_normalization(), dtype=torch.float32
    )
    objective = data.objective(protocol.objective.energy)
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
                raise RuntimeError("DEVELOPMENT labels opened during the ADF run")
            if progress is not None:
                last = result.history[-1]
                progress(
                    f"phase=CHECKPOINT role=TRAIN step={result.step} "
                    f"model={result.model_state_hash} energy={last.energy:.9g} "
                    f"min_det={last.minimum_determinant:.4f}"
                )
            resume_from = checkpoint_path
    return result


# ------------------------------------------------------------------ evaluate


def screen_record(evaluation: Any, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """The TRAIN-screen record of the ADF run against the frozen guardrails."""

    metrics = evaluation.result.metrics
    viability = protocol["viability"]
    gaps = [d.relative_energy_gap for d in evaluation.diagnostics]
    return {
        "schema_version": "1.0",
        "step": "S24",
        "evidence_role": "TRAIN",
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "problem_spec_hash": evaluation.run_binding.problem_spec_hash,
        "implementation_revision": evaluation.run_binding.implementation_revision,
        "run_hash": evaluation.manifest.run_hash,
        "checkpoint_step": evaluation.manifest.step,
        "checkpoint_manifest_hash": evaluation.manifest.manifest_hash,
        "model_state_hash": evaluation.manifest.model_state_hash,
        "metrics": {
            "median_absolute_peak_error": metrics.median_absolute_peak_error,
            "p90_absolute_peak_error": metrics.p90_absolute_peak_error,
            "median_signed_peak_error": metrics.median_signed_peak_error,
            "p_relative_l2": metrics.p_relative_l2,
            "u_relative_l2": metrics.u_relative_l2,
            "invalid_collocation_points": metrics.invalid_collocation_points,
        },
        "minimum_collocation_determinant": evaluation.minimum_collocation_determinant,
        "median_relative_energy_gap": sorted(gaps)[len(gaps) // 2] if gaps else None,
        "failed_guardrails": list(evaluation.failed_guardrails),
        "guardrail_limits": viability["evaluation"]["guardrail_limits"],
        "s13c_p_relative_l2": viability["evaluation"]["reported"]["s13c_p_relative_l2"],
        "development_opened": evaluation.development_opened,
        "passed": not evaluation.failed_guardrails
        and not evaluation.development_opened,
    }


def evaluate_adf_checkpoint(*, s10_run_root: Path, checkpoint_path: Path) -> Any:
    from campaigns.dogbone_v1.s13c_evaluate import evaluate_s13c_checkpoint

    return evaluate_s13c_checkpoint(
        s10_run_root=s10_run_root,
        checkpoint_path=checkpoint_path,
        case_factory=adf_geometry_case,
        problem_spec_hash=load_spec().problem_hash,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m campaigns.dogbone_v1.s24_adf")
    sub = parser.add_subparsers(dest="command", required=True)
    protocol = sub.add_parser("protocol")
    protocol.add_argument("--write", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--s10-run-root", type=Path, required=True)
    train.add_argument("--run-root", type=Path, required=True)
    train.add_argument("--segment-steps", type=int, default=400)
    train.add_argument("--stop-after", type=int)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--s10-run-root", type=Path, required=True)
    evaluate.add_argument("--checkpoint-path", type=Path, required=True)
    evaluate.add_argument("--write", action="store_true")
    options = parser.parse_args(arguments)
    if options.command == "protocol":
        payload = json.dumps(build_protocol(), indent=2, sort_keys=True) + "\n"
        target = REPOSITORY / PROTOCOL_PATH
        if options.write:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    if options.command == "train":
        run_adf_trial(
            s10_run_root=options.s10_run_root,
            run_root=options.run_root,
            segment_steps=options.segment_steps,
            stop_after=options.stop_after,
            progress=lambda message: print(message, flush=True),
        )
        return 0
    document = accepted_protocol()
    evaluation = evaluate_adf_checkpoint(
        s10_run_root=options.s10_run_root, checkpoint_path=options.checkpoint_path
    )
    record = screen_record(evaluation, document)
    if options.write:
        (REPOSITORY / SCREEN_PATH).write_bytes(canonical_json_bytes(record) + b"\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
