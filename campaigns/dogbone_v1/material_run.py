"""A registered DogBone viability run on another material (S26 on, ADR 0108).

The S25 Mooney–Rivlin run (``s25_mooney_rivlin``) as a reusable recipe: the
S13c run unchanged — protocol, data, seed, initial weights, sampler plan,
trainer configuration, objective form, lifting and execution profile —
except the material, taken from a DogBone ProblemSpec whose material
parameters are fixed. The run binding binds that spec's hash. Training stays
label-free; the 64 TRAIN-screen geometries are re-solved with the reference
FEM solver and the same material, on their S10 meshes and settings, and the
run is judged on them by the frozen S13 guardrails. DEVELOPMENT, HOLDOUT, H1
and H2 stay closed.

A step instantiates :class:`MaterialViability` with its spec, paths and the
owner's answers (``s26_hgo``). Commands (repository root, clean committed
revision)::

    python -m campaigns.dogbone_v1.<step module> protocol --write
    python -m campaigns.dogbone_v1.<step module> references \
        --s10-run-root <S10 root> --output <references dir> [--workers 6] --write
    python -m campaigns.dogbone_v1.<step module> train \
        --s10-run-root <S10 root> --run-root <new run root>
    python -m campaigns.dogbone_v1.<step module> evaluate \
        --s10-run-root <S10 root> --references <references dir> \
        --checkpoint-path <run root>/checkpoint-step-2400 --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
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
from campaigns.dogbone_v1.s25_mooney_rivlin import (
    INDEX_NAME,
    S13C_SCREEN_PATH,
    reference_entry,
    solve_references,
)
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
_SPECS: dict[Path, ProblemSpec] = {}


@dataclass(frozen=True)
class MaterialViability:
    """One step's registered viability run (paths, material, owner answers)."""

    step: str
    model: str
    label: str
    spec_path: Path
    protocol_path: Path
    references_path: Path
    screen_path: Path
    decision_record: str
    accepted_on: str
    answers: Mapping[str, str]
    material_note: Mapping[str, str]
    kill_rule: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ material

    def load_spec(self) -> ProblemSpec:
        path = REPOSITORY / self.spec_path
        if path not in _SPECS:
            spec = ProblemSpec.load(path)
            if spec.material.model != self.model:
                raise ValueError(f"the viability spec must use the {self.model} model")
            _SPECS[path] = spec
        return _SPECS[path]

    def physics_material(self) -> MaterialSpec:
        """The kernel material of the run (fixed parameters of the spec)."""

        spec = self.load_spec()
        values = {n: spec.parameter(n).lo for n in spec.material.parameters.values()}
        return material_spec(spec, values, name=spec.name)

    def fem_material(self) -> MaterialSpec:
        """The FEM material of the reference re-solves."""

        return replace(self.physics_material(), name=self.load_spec().fem.material_name)

    def request(self, request: FemRequest) -> FemRequest:
        """An S10 request with this material."""

        return replace(request, material=self.fem_material())

    def geometry_case(
        self,
        verified: VerifiedS10Request,
        normalization: CaseNormalizationStatistics,
        *,
        dtype: torch.dtype,
    ) -> DogboneS13cGeometryCase:
        """The S13c case of ``verified`` with this material."""

        return replace(
            geometry_case(verified, normalization, dtype=dtype),
            material=self.physics_material(),
        )

    def materialize(
        self,
        source: SelectiveS10Source,
        protocol: DogboneS13cProtocol,
        normalization: CaseNormalizationStatistics,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> DogboneS13cTrainingData:
        """The S13c TRAIN-fit population with this material."""

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
            self.geometry_case(verified, normalization, dtype=dtype)
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

    # ------------------------------------------------------------ protocol

    def build_protocol(self, root: Path | None = None) -> dict[str, Any]:
        """The step's preregistration."""

        root = REPOSITORY if root is None else root
        s13c = load_committed_s13c_protocol()
        config = build_dogbone_s13c_trainer_config(s13c)
        spec = self.load_spec()
        screen = json.loads((root / S13C_SCREEN_PATH).read_text("utf-8"))
        material = self.physics_material()
        return {
            "schema_version": "1.0",
            "step": self.step,
            "status": "accepted",
            "decision_record": self.decision_record,
            "accepted": {
                "by": "decision owner",
                "on": self.accepted_on,
                "answers": dict(self.answers),
            },
            "viability": {
                "campaign": "dogbone-v1",
                "spec": self.spec_path.as_posix(),
                "problem_spec_hash": spec.problem_hash,
                "material": {
                    "model": material.model,
                    "properties": {
                        name: quantity.value
                        for name, quantity in sorted(material.properties.items())
                    },
                    **dict(self.material_note),
                },
                "s13c_protocol_hash": s13c.protocol_hash,
                "s13c_protocol_sha256": _sha256(
                    root, S13C_PROTOCOL_PATH.relative_to(REPOSITORY)
                ),
                "unchanged": [
                    "TRAIN-fit population and S10 source bindings",
                    "normalization",
                    "architecture and initial weights",
                    "objective form and trainer configuration",
                    "sampler plan and seed",
                    "lifting (quarter) and execution profile",
                ],
                "changed": f"material: {self.label}; the run binding binds "
                "problem_spec_hash",
                "optimizer_steps": config.trial.optimizer_steps,
                "initial_weights_hash": s13c.initial_weights_hash,
                "seed": s13c.training_seed,
                "references": {
                    "cases": list(s13c.screen_case_ids),
                    "rule": "each TRAIN-screen S10 request with the material "
                    "replaced, solved by the reference FEM solver",
                    "fem_material_name": spec.fem.material_name,
                },
                "evaluation": {
                    "population": "TRAIN-screen",
                    "cases": len(s13c.screen_case_ids),
                    "rule": "the frozen S13 guardrails of the S13c protocol "
                    "(campaigns.dogbone_v1.s13c_evaluate.failed_guardrails) "
                    f"against the {self.label} re-solves",
                    "guardrail_limits": screen["comparison"]["guardrail_limits"],
                    "reported": {
                        "s13c_p_relative_l2": screen["comparison"]["s13c"][
                            "p_relative_l2"
                        ],
                    },
                },
                "closed_evidence": ["DEVELOPMENT", "HOLDOUT", "H1", "H2"],
                "kill_rule": self.kill_rule,
                **dict(self.extra),
            },
        }

    def accepted_protocol(self, *, require_committed: bool = True) -> dict[str, Any]:
        """The committed protocol, refused unless it is current and accepted."""

        path = REPOSITORY / self.protocol_path
        if not path.is_file():
            raise RuntimeError(f"{self.protocol_path} is not written")
        if require_committed and not _committed(self.protocol_path):
            raise RuntimeError(f"{self.protocol_path} must be committed before the run")
        document: dict[str, Any] = json.loads(path.read_text("utf-8"))
        if document != self.build_protocol():
            raise ValueError(
                f"the committed {self.step} protocol differs from its rebuild"
            )
        if document["status"] != "accepted":
            raise RuntimeError(f"the {self.step} protocol is not accepted")
        return document

    # ---------------------------------------------------------- references

    def solve_reference_cases(
        self,
        requests: Mapping[str, FemRequest],
        output: Path,
        *,
        workers: int,
        progress: Progress | None,
        implementation_revision: str,
        require_committed: bool,
    ) -> dict[str, dict[str, str]]:
        return solve_references(requests, output, workers=workers, progress=progress)

    def stored_reference_entry(
        self, request: FemRequest, payload: bytes, directory: Path
    ) -> dict[str, str]:
        return reference_entry(request, payload)

    def references_document(
        self, index: Mapping[str, dict[str, str]], implementation_revision: str
    ) -> dict[str, Any]:
        """The references index record of the protocol's material."""

        return {
            "schema_version": "1.0",
            "step": self.step,
            "evidence_role": "TRAIN",
            "protocol_sha256": _sha256(REPOSITORY, self.protocol_path),
            "problem_spec_hash": self.load_spec().problem_hash,
            "material_hash": hash_spec(self.fem_material(), "content"),
            "implementation_revision": implementation_revision,
            "cases": dict(index),
        }

    def write_references(
        self,
        *,
        s10_run_root: Path,
        output: Path,
        workers: int = 1,
        require_committed: bool = True,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        """Re-solve the TRAIN-screen cases with this material into ``output``."""

        protocol = self.accepted_protocol(require_committed=require_committed)
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
            verified.case.case_id: self.request(verified.request)
            for verified in source.iter_train_geometry(cases)
        }
        if source.development_opened:
            raise RuntimeError("DEVELOPMENT labels must remain closed")
        index = self.solve_reference_cases(
            requests,
            output,
            workers=workers,
            progress=progress,
            implementation_revision=provenance.head,
            require_committed=require_committed,
        )
        document = self.references_document(index, provenance.head)
        (output / INDEX_NAME).write_bytes(canonical_json_bytes(document) + b"\n")
        return document

    def reference_loader(self, directory: Path, index: Mapping[str, Any]) -> Reference:
        """The S13c evaluator's reference hook over the re-solved cases."""

        rebuilt = self.references_document(
            index["cases"], index["implementation_revision"]
        )
        if index != rebuilt:
            raise ValueError("the references index is not this protocol's")
        cases = index["cases"]

        def reference(verified: Any) -> tuple[FemRequest, FemResult]:
            case_id = verified.case.case_id
            request = self.request(verified.request)
            payload = (directory / f"{case_id}.json").read_bytes()
            if (
                self.stored_reference_entry(request, payload, directory)
                != cases[case_id]
            ):
                raise ValueError(f"{case_id}: the reference differs from its index")
            return request, _decode_stored_result(request, payload)

        return reference

    # --------------------------------------------------------------- train

    def build_binding(
        self, protocol: DogboneS13cProtocol, implementation_revision: str
    ) -> TrainingRunBinding:
        binding = build_s13c_binding(
            protocol,
            model=build_initial_operator(),
            config=build_dogbone_s13c_trainer_config(protocol),
            plan=build_dogbone_s13c_sampler_plan(protocol),
            implementation_revision=implementation_revision,
        )
        return replace(binding, problem_spec_hash=self.load_spec().problem_hash)

    def run_trial(
        self,
        *,
        s10_run_root: Path,
        run_root: Path,
        segment_steps: int = 400,
        stop_after: int | None = None,
        require_clean_revision: bool = True,
        progress: Progress | None = None,
    ) -> EnergyTrainingResult | None:
        """Train the registered run on TRAIN-fit geometry only."""

        self.accepted_protocol(require_committed=require_clean_revision)
        protocol = load_committed_s13c_protocol()
        config = build_dogbone_s13c_trainer_config(protocol)
        plan = build_dogbone_s13c_sampler_plan(protocol)
        provenance = _git_provenance()
        if require_clean_revision and provenance.dirty_worktree:
            raise RuntimeError(
                f"the registered {self.step} run requires a clean revision"
            )
        s10 = _absolute_directory(s10_run_root, "s10_run_root")
        binding = self.build_binding(protocol, provenance.head)
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
                f"phase=PREFLIGHT role=TRAIN material={self.model} "
                f"revision={provenance.head} spec={binding.problem_spec_hash} "
                f"run={binding.run_hash} resume_step={last_step} targets={targets}"
            )
        if not targets:
            return None
        source = open_selective_s10_source(
            s10, development_access_record=output / DEVELOPMENT_ACCESS_NAME
        )
        data = self.materialize(
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

    # ------------------------------------------------------------ evaluate

    def screen_record(
        self,
        evaluation: Any,
        protocol: Mapping[str, Any],
        references: Mapping[str, Any],
    ) -> dict[str, Any]:
        """The TRAIN-screen record of the run against the frozen guardrails."""

        metrics = evaluation.result.metrics
        viability = protocol["viability"]
        gaps = sorted(d.relative_energy_gap for d in evaluation.diagnostics)
        return {
            "schema_version": "1.0",
            "step": self.step,
            "evidence_role": "TRAIN",
            "protocol_sha256": _sha256(REPOSITORY, self.protocol_path),
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
            "minimum_collocation_determinant": (
                evaluation.minimum_collocation_determinant
            ),
            "median_relative_energy_gap": gaps[len(gaps) // 2] if gaps else None,
            "failed_guardrails": list(evaluation.failed_guardrails),
            "guardrail_limits": viability["evaluation"]["guardrail_limits"],
            "s13c_p_relative_l2": viability["evaluation"]["reported"][
                "s13c_p_relative_l2"
            ],
            "development_opened": evaluation.development_opened,
            "passed": not evaluation.failed_guardrails
            and not evaluation.development_opened,
        }

    def evaluate(
        self, *, s10_run_root: Path, checkpoint_path: Path, references: Path
    ) -> tuple[Any, dict[str, Any]]:
        from campaigns.dogbone_v1.s13c_evaluate import evaluate_s13c_checkpoint

        index: dict[str, Any] = json.loads((references / INDEX_NAME).read_text("utf-8"))
        cases = self.accepted_protocol()["viability"]["references"]["cases"]
        if sorted(index["cases"]) != sorted(cases):
            raise ValueError("the references do not cover the TRAIN-screen cases")
        committed = REPOSITORY / self.references_path
        if committed.is_file() and json.loads(committed.read_text("utf-8")) != index:
            raise ValueError("the references differ from their committed index")
        evaluation = evaluate_s13c_checkpoint(
            s10_run_root=s10_run_root,
            checkpoint_path=checkpoint_path,
            case_factory=self.geometry_case,
            problem_spec_hash=self.load_spec().problem_hash,
            reference=self.reference_loader(references, index),
        )
        return evaluation, index

    # ----------------------------------------------------------------- CLI

    def main(self, arguments: Sequence[str] | None = None, prog: str = "") -> int:
        parser = argparse.ArgumentParser(prog=prog or None)
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
            payload = json.dumps(self.build_protocol(), indent=2, sort_keys=True) + "\n"
            if options.write:
                target = REPOSITORY / self.protocol_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(payload, encoding="utf-8")
            print(payload, end="")
            return 0
        if options.command == "references":
            document = self.write_references(
                s10_run_root=options.s10_run_root,
                output=options.output.resolve(),
                workers=options.workers,
                progress=lambda message: print(message, flush=True),
            )
            if options.write:
                (REPOSITORY / self.references_path).write_bytes(
                    canonical_json_bytes(document) + b"\n"
                )
            print(f"references: {len(document['cases'])} cases")
            return 0
        if options.command == "train":
            self.run_trial(
                s10_run_root=options.s10_run_root,
                run_root=options.run_root,
                segment_steps=options.segment_steps,
                stop_after=options.stop_after,
                progress=lambda message: print(message, flush=True),
            )
            return 0
        document = self.accepted_protocol()
        evaluation, index = self.evaluate(
            s10_run_root=options.s10_run_root,
            checkpoint_path=options.checkpoint_path,
            references=options.references,
        )
        record = self.screen_record(evaluation, document, index)
        if options.write:
            (REPOSITORY / self.screen_path).write_bytes(
                canonical_json_bytes(record) + b"\n"
            )
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0 if record["passed"] else 1


def _sha256(root: Path, path: Path) -> str:
    return hashlib.sha256((root / path).read_bytes()).hexdigest()


__all__ = ("REPOSITORY", "MaterialViability")
