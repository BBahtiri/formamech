"""S25: the registered Mooney–Rivlin viability run on DogBone (ADR 0107 d. 8).

The S13c recipe unchanged — protocol, data, seed, initial weights, sampler
plan, trainer configuration, objective form, lifting and execution profile —
except the material: Mooney–Rivlin with the shipped material's small-strain
moduli (``campaigns/dogbone_v1/problem_spec_mooney_rivlin.json``). The run
binding binds that spec's hash. Training stays label-free; the 64
TRAIN-screen geometries are re-solved with the reference FEM solver and the
same material, on their S10 meshes and settings, and the run is judged on
them by the frozen S13 guardrails. DEVELOPMENT, HOLDOUT, H1 and H2 stay
closed.

Usage (repository root, clean committed revision)::

    python -m campaigns.dogbone_v1.s25_mooney_rivlin protocol --write
    python -m campaigns.dogbone_v1.s25_mooney_rivlin references \
        --s10-run-root <S10 root> --output <references dir> [--workers 4]
    python -m campaigns.dogbone_v1.s25_mooney_rivlin train \
        --s10-run-root <S10 root> --run-root <new run root>
    python -m campaigns.dogbone_v1.s25_mooney_rivlin evaluate \
        --s10-run-root <S10 root> --references <references dir> \
        --checkpoint-path <run root>/checkpoint-step-2400 --write

``references`` and ``train`` refuse to start unless the protocol is
committed, accepted and current. ``references`` resumes: cases already
solved are verified and kept. With ``--write`` it records their hashes in
``artifacts/materials/S25-MR-REFERENCES.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from campaigns.dogbone_v1.s10_production import _decode_stored_result
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
from campaigns.dogbone_v1.s24_adf import _committed
from piginot.contracts import (
    MaterialSpec,
    ProblemSpec,
    canonical_json_bytes,
    hash_spec,
)
from piginot.doe import EvidenceRole
from piginot.fem import FemRequest, FemResult
from piginot.problem import material_spec
from piginot.training import (
    EnergyTrainingResult,
    TrainingPopulation,
    TrainingRunBinding,
    train_energy_to_checkpoint,
)

if TYPE_CHECKING:
    from campaigns.dogbone_v1.s13c_evaluate import Reference

REPOSITORY = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("campaigns/dogbone_v1/problem_spec_mooney_rivlin.json")
PROTOCOL_PATH = Path("artifacts/materials/S25-PROTOCOL.json")
REFERENCES_PATH = Path("artifacts/materials/S25-MR-REFERENCES.json")
SCREEN_PATH = Path("artifacts/materials/S25-MR-TRAIN-SCREEN.json")
S13C_SCREEN_PATH = Path("artifacts/campaigns/dogbone-v1/S13C-TRAIN-SCREEN.json")
INDEX_NAME = "index.json"


@cache
def load_spec() -> ProblemSpec:
    spec = ProblemSpec.load(REPOSITORY / SPEC_PATH)
    if spec.material.model != "mooney_rivlin":
        raise ValueError("the viability spec must use the Mooney-Rivlin model")
    return spec


def _material_values(spec: ProblemSpec) -> dict[str, float]:
    return {name: spec.parameter(name).lo for name in spec.material.parameters.values()}


def physics_material() -> MaterialSpec:
    """The kernel material of the run (fixed parameters of the spec)."""

    spec = load_spec()
    return material_spec(spec, _material_values(spec), name=spec.name)


def fem_material() -> MaterialSpec:
    """The FEM material of the reference re-solves."""

    spec = load_spec()
    return replace(physics_material(), name=spec.fem.material_name)


def mr_request(request: FemRequest) -> FemRequest:
    """An S10 request with the Mooney–Rivlin material."""

    return replace(request, material=fem_material())


def mr_geometry_case(
    verified: VerifiedS10Request,
    normalization: CaseNormalizationStatistics,
    *,
    dtype: torch.dtype,
) -> DogboneS13cGeometryCase:
    """The S13c case of ``verified`` with the Mooney–Rivlin material."""

    return replace(
        geometry_case(verified, normalization, dtype=dtype),
        material=physics_material(),
    )


def materialize_mr_training_data(
    source: SelectiveS10Source,
    protocol: DogboneS13cProtocol,
    normalization: CaseNormalizationStatistics,
    *,
    dtype: torch.dtype = torch.float32,
) -> DogboneS13cTrainingData:
    """The S13c TRAIN-fit population with the Mooney–Rivlin material."""

    if type(source) is not SelectiveS10Source:
        raise TypeError("source must be an exact verified SelectiveS10Source")
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels must remain closed")
    if (
        source.bindings.content_hash != protocol.source_bindings_hash
        or source.fit_case_ids != protocol.fit_case_ids
        or normalization.content_hash != protocol.normalization_hash
        or source.train_snapshot.content_hash != protocol.dataset_snapshot_hash
    ):
        raise ValueError("S10 source or normalization differs from the protocol")
    cases = tuple(
        mr_geometry_case(verified, normalization, dtype=dtype)
        for verified in source.iter_train_geometry(protocol.fit_case_ids)
    )
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
    """The S25 preregistration (ADR 0107, accepted 2026-10-13)."""

    s13c = load_committed_s13c_protocol()
    config = build_dogbone_s13c_trainer_config(s13c)
    spec = load_spec()
    screen = json.loads((root / S13C_SCREEN_PATH).read_text("utf-8"))
    material = physics_material()
    return {
        "schema_version": "1.0",
        "step": "S25",
        "status": "accepted",
        "decision_record": "docs/adr/0107-s25-material-registry.md",
        "accepted": {
            "by": "decision owner",
            "on": "2026-10-13",
            "answers": {
                "ADR 0107": "Accept as written",
                "viability material": "Mooney-Rivlin",
                "kill rule": "Registry, Neo-Hookean training only",
            },
        },
        "viability": {
            "campaign": "dogbone-v1",
            "spec": SPEC_PATH.as_posix(),
            "problem_spec_hash": spec.problem_hash,
            "material": {
                "model": material.model,
                "properties": {
                    name: quantity.value
                    for name, quantity in sorted(material.properties.items())
                },
                "small_strain_matches": "E = 760 MPa, nu = 0.23",
            },
            "s13c_protocol_hash": s13c.protocol_hash,
            "s13c_protocol_sha256": _sha256(S13C_PROTOCOL_PATH.relative_to(root)),
            "unchanged": [
                "TRAIN-fit population and S10 source bindings",
                "normalization",
                "architecture and initial weights",
                "objective form and trainer configuration",
                "sampler plan and seed",
                "lifting (quarter) and execution profile",
            ],
            "changed": "material: Mooney-Rivlin; the run binding binds "
            "problem_spec_hash",
            "optimizer_steps": config.trial.optimizer_steps,
            "initial_weights_hash": s13c.initial_weights_hash,
            "seed": s13c.training_seed,
            "references": {
                "cases": list(s13c.screen_case_ids),
                "rule": "each TRAIN-screen S10 request with the material replaced, "
                "solved by the reference FEM solver",
                "fem_material_name": load_spec().fem.material_name,
            },
            "evaluation": {
                "population": "TRAIN-screen",
                "cases": len(s13c.screen_case_ids),
                "rule": "the frozen S13 guardrails of the S13c protocol "
                "(campaigns.dogbone_v1.s13c_evaluate.failed_guardrails) against "
                "the Mooney-Rivlin re-solves",
                "guardrail_limits": screen["comparison"]["guardrail_limits"],
                "reported": {
                    "s13c_p_relative_l2": screen["comparison"]["s13c"]["p_relative_l2"],
                },
            },
            "closed_evidence": ["DEVELOPMENT", "HOLDOUT", "H1", "H2"],
            "kill_rule": "if a guardrail fails, 0.2.0 ships the registry and the "
            "admission gate and the S28 compiler trains only the closed-form "
            "Neo-Hookean model",
        },
    }


def accepted_protocol(*, require_committed: bool = True) -> dict[str, Any]:
    """The committed protocol, refused unless it is current and accepted."""

    path = REPOSITORY / PROTOCOL_PATH
    if not path.is_file():
        raise RuntimeError(f"{PROTOCOL_PATH} is not written")
    if require_committed and not _committed(PROTOCOL_PATH):
        raise RuntimeError(f"{PROTOCOL_PATH} must be committed before the run")
    document: dict[str, Any] = json.loads(path.read_text("utf-8"))
    if document != build_protocol():
        raise ValueError("the committed S25 protocol differs from its rebuild")
    if document["status"] != "accepted":
        raise RuntimeError("the S25 protocol is not accepted")
    return document


# ---------------------------------------------------------------- references


def encode_result(result: FemResult) -> bytes:
    """The S10 store envelope of a FEM result."""

    return canonical_json_bytes({"domain": "content", "payload": result})


def solve_reference(request: FemRequest) -> bytes:
    """One reference solve, encoded; a failed solve is an error."""

    from piginot.fem.backend import ScikitFemBackend

    torch.set_num_threads(1)
    attempt = ScikitFemBackend().solve_many((request,))[0]
    if not attempt.success or attempt.result is None:
        raise RuntimeError(f"{request.request_id}: {attempt.message}")
    return encode_result(attempt.result)


def reference_entry(request: FemRequest, payload: bytes) -> dict[str, str]:
    """The index entry of a stored reference; refuses a foreign result."""

    result = _decode_stored_result(request, payload)
    return {
        "request_content_hash": hash_spec(request, "content"),
        "result_hash": hash_spec(result, "content"),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _solved(
    solve: Callable[[FemRequest], bytes],
    requests: list[FemRequest],
    workers: int,
) -> Iterator[bytes]:
    if workers <= 1:
        yield from map(solve, requests)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(solve, requests)


def solve_references(
    requests: Mapping[str, FemRequest],
    output: Path,
    *,
    workers: int = 1,
    solve: Callable[[FemRequest], bytes] = solve_reference,
    progress: Progress | None = None,
) -> dict[str, dict[str, str]]:
    """Solve the cases not yet in ``output`` and index all of them.

    Each file is written under a temporary name and renamed, so an
    interrupted run resumes; a stored file that is not the result of its
    request is refused.
    """

    output.mkdir(parents=True, exist_ok=True)
    pending = [i for i in requests if not (output / f"{i}.json").is_file()]
    payloads = _solved(solve, [requests[i] for i in pending], workers)
    for case_id, payload in zip(pending, payloads, strict=True):
        partial = output / f"{case_id}.json.partial"
        partial.write_bytes(payload)
        os.replace(partial, output / f"{case_id}.json")
        if progress is not None:
            progress(f"phase=REFERENCE case={case_id}")
    return {
        case_id: reference_entry(request, (output / f"{case_id}.json").read_bytes())
        for case_id, request in requests.items()
    }


def references_document(
    index: Mapping[str, dict[str, str]], implementation_revision: str
) -> dict[str, Any]:
    """The references index record of the protocol's material."""

    return {
        "schema_version": "1.0",
        "step": "S25",
        "evidence_role": "TRAIN",
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "problem_spec_hash": load_spec().problem_hash,
        "material_hash": hash_spec(fem_material(), "content"),
        "implementation_revision": implementation_revision,
        "cases": dict(index),
    }


def write_references(
    *,
    s10_run_root: Path,
    output: Path,
    workers: int = 1,
    require_committed: bool = True,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Re-solve the TRAIN-screen cases with Mooney–Rivlin into ``output``."""

    protocol = accepted_protocol(require_committed=require_committed)
    provenance = _git_provenance()
    if require_committed and provenance.dirty_worktree:
        raise RuntimeError("the registered re-solves require a clean revision")
    s10 = _absolute_directory(s10_run_root, "s10_run_root")
    if not output.is_absolute() or output.is_relative_to(REPOSITORY):
        raise ValueError("output must be an absolute path outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    source = open_selective_s10_source(
        s10, development_access_record=output / DEVELOPMENT_ACCESS_NAME
    )
    cases = tuple(protocol["viability"]["references"]["cases"])
    if source.train_screen_case_ids != cases:
        raise ValueError("the TRAIN-screen cases differ from the protocol")
    requests = {
        verified.case.case_id: mr_request(verified.request)
        for verified in source.iter_train_geometry(cases)
    }
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels must remain closed")
    index = solve_references(requests, output, workers=workers, progress=progress)
    document = references_document(index, provenance.head)
    (output / INDEX_NAME).write_bytes(canonical_json_bytes(document) + b"\n")
    return document


def reference_loader(directory: Path, index: Mapping[str, Any]) -> Reference:
    """The S13c evaluator's reference hook over the re-solved cases."""

    if index != references_document(index["cases"], index["implementation_revision"]):
        raise ValueError("the references index is not this protocol's")
    cases = index["cases"]

    def reference(verified: Any) -> tuple[FemRequest, FemResult]:
        case_id = verified.case.case_id
        request = mr_request(verified.request)
        payload = (directory / f"{case_id}.json").read_bytes()
        if reference_entry(request, payload) != cases[case_id]:
            raise ValueError(f"{case_id}: the reference differs from its index")
        return request, _decode_stored_result(request, payload)

    return reference


# --------------------------------------------------------------------- train


def build_mr_binding(
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


def run_mr_trial(
    *,
    s10_run_root: Path,
    run_root: Path,
    segment_steps: int = 400,
    stop_after: int | None = None,
    require_clean_revision: bool = True,
    progress: Progress | None = None,
) -> EnergyTrainingResult | None:
    """Train the registered Mooney–Rivlin run on TRAIN-fit geometry only."""

    accepted_protocol(require_committed=require_clean_revision)
    protocol = load_committed_s13c_protocol()
    config = build_dogbone_s13c_trainer_config(protocol)
    plan = build_dogbone_s13c_sampler_plan(protocol)
    provenance = _git_provenance()
    if require_clean_revision and provenance.dirty_worktree:
        raise RuntimeError("the registered Mooney-Rivlin run requires a clean revision")
    s10 = _absolute_directory(s10_run_root, "s10_run_root")
    binding = build_mr_binding(protocol, provenance.head)
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
            f"phase=PREFLIGHT role=TRAIN material=mooney_rivlin "
            f"revision={provenance.head} spec={binding.problem_spec_hash} "
            f"run={binding.run_hash} resume_step={last_step} targets={targets}"
        )
    if not targets:
        return None
    source = open_selective_s10_source(
        s10, development_access_record=output / DEVELOPMENT_ACCESS_NAME
    )
    data = materialize_mr_training_data(
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
                raise RuntimeError("DEVELOPMENT labels opened during the run")
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


def screen_record(
    evaluation: Any, protocol: Mapping[str, Any], references: Mapping[str, Any]
) -> dict[str, Any]:
    """The TRAIN-screen record of the run against the frozen guardrails."""

    metrics = evaluation.result.metrics
    viability = protocol["viability"]
    gaps = sorted(d.relative_energy_gap for d in evaluation.diagnostics)
    return {
        "schema_version": "1.0",
        "step": "S25",
        "evidence_role": "TRAIN",
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "references_sha256": hashlib.sha256(
            canonical_json_bytes(dict(references))
        ).hexdigest(),
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
        "median_relative_energy_gap": gaps[len(gaps) // 2] if gaps else None,
        "failed_guardrails": list(evaluation.failed_guardrails),
        "guardrail_limits": viability["evaluation"]["guardrail_limits"],
        "s13c_p_relative_l2": viability["evaluation"]["reported"]["s13c_p_relative_l2"],
        "development_opened": evaluation.development_opened,
        "passed": not evaluation.failed_guardrails
        and not evaluation.development_opened,
    }


def evaluate_mr_checkpoint(
    *, s10_run_root: Path, checkpoint_path: Path, references: Path
) -> tuple[Any, dict[str, Any]]:
    from campaigns.dogbone_v1.s13c_evaluate import evaluate_s13c_checkpoint

    index: dict[str, Any] = json.loads((references / INDEX_NAME).read_text("utf-8"))
    cases = accepted_protocol()["viability"]["references"]["cases"]
    if sorted(index["cases"]) != sorted(cases):
        raise ValueError("the references do not cover the TRAIN-screen cases")
    committed = REPOSITORY / REFERENCES_PATH
    if committed.is_file() and json.loads(committed.read_text("utf-8")) != index:
        raise ValueError("the references differ from their committed index")
    evaluation = evaluate_s13c_checkpoint(
        s10_run_root=s10_run_root,
        checkpoint_path=checkpoint_path,
        case_factory=mr_geometry_case,
        problem_spec_hash=load_spec().problem_hash,
        reference=reference_loader(references, index),
    )
    return evaluation, index


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m campaigns.dogbone_v1.s25_mooney_rivlin"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    protocol = sub.add_parser("protocol")
    protocol.add_argument("--write", action="store_true")
    references = sub.add_parser("references")
    references.add_argument("--s10-run-root", type=Path, required=True)
    references.add_argument("--output", type=Path, required=True)
    references.add_argument("--workers", type=int, default=1)
    references.add_argument("--write", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--s10-run-root", type=Path, required=True)
    train.add_argument("--run-root", type=Path, required=True)
    train.add_argument("--segment-steps", type=int, default=400)
    train.add_argument("--stop-after", type=int)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--s10-run-root", type=Path, required=True)
    evaluate.add_argument("--references", type=Path, required=True)
    evaluate.add_argument("--checkpoint-path", type=Path, required=True)
    evaluate.add_argument("--write", action="store_true")
    options = parser.parse_args(arguments)
    if options.command == "protocol":
        payload = json.dumps(build_protocol(), indent=2, sort_keys=True) + "\n"
        if options.write:
            target = REPOSITORY / PROTOCOL_PATH
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    if options.command == "references":
        document = write_references(
            s10_run_root=options.s10_run_root,
            output=options.output.resolve(),
            workers=options.workers,
            progress=lambda message: print(message, flush=True),
        )
        if options.write:
            (REPOSITORY / REFERENCES_PATH).write_bytes(
                canonical_json_bytes(document) + b"\n"
            )
        print(f"references: {len(document['cases'])} cases")
        return 0
    if options.command == "train":
        run_mr_trial(
            s10_run_root=options.s10_run_root,
            run_root=options.run_root,
            segment_steps=options.segment_steps,
            stop_after=options.stop_after,
            progress=lambda message: print(message, flush=True),
        )
        return 0
    document = accepted_protocol()
    evaluation, index = evaluate_mr_checkpoint(
        s10_run_root=options.s10_run_root,
        checkpoint_path=options.checkpoint_path,
        references=options.references,
    )
    record = screen_record(evaluation, document, index)
    if options.write:
        (REPOSITORY / SCREEN_PATH).write_bytes(canonical_json_bytes(record) + b"\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
