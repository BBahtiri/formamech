"""Generic foreground campaign stages, evidence guards and explicit recovery."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from functools import partial
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol

from piginot.compiler.adapters import AdapterError
from piginot.compiler.execution_records import (
    CaseJob,
    Document,
    OutcomeStore,
    digest,
    load_bound,
    save_bound,
)
from piginot.compiler.planning import STAGES, ExecutionPlan
from piginot.compiler.records import atomic_write, run_lock, write_plan
from piginot.contracts import canonical_json_bytes


class CampaignOperations(Protocol):
    def probe_runtime(self) -> Document: ...
    def admit_material(self) -> Document: ...
    def verify_geometry(self) -> Document: ...
    def prepare_training(self, material: Document, lifting: Document) -> Document: ...
    def train(self, *, resume: bool) -> Document: ...
    def evaluate(
        self, case_id: str, partition: str, training: Document
    ) -> Document: ...
    def validate_case_cache(
        self, case_id: str, partition: str, training: Document
    ) -> list[Document] | None: ...
    def freeze(self, training: Document, screen: Document) -> Document: ...
    def export(
        self, training: Document, material: Document, lifting: Document
    ) -> Document: ...


def require_clean_implementation_identity() -> Any:
    from piginot.compiler.provenance import (
        require_clean_implementation_identity as resolve,
    )

    return resolve()


def NumericalOperations(
    plan: ExecutionPlan, adapter: Any, revision: str
) -> CampaignOperations:
    from piginot.compiler.execution_numerics import NumericalOperations as Operations

    return Operations(plan, adapter, revision)


def _require_numerical_dependencies() -> None:
    required = (
        "torch",
        "numpy",
        "scipy",
        "skfem",
        "sympy",
        "safetensors",
        "threadpoolctl",
    )
    missing = tuple(name for name in required if find_spec(name) is None)
    if missing:
        raise AdapterError(
            "capability",
            "numerical execution needs "
            + ", ".join(missing)
            + "; install pip install 'piginot[verify]'",
        )


def _overall(outcomes: tuple[Document, ...]) -> str:
    statuses = {item["status"] for item in outcomes}
    if "ERROR" in statuses:
        return "ERROR"
    if "INCOMPLETE" in statuses:
        return "INCOMPLETE"
    return "PASS" if statuses == {"PASS"} else "FAIL"


class _Run:
    def __init__(
        self, plan: ExecutionPlan, operations: CampaignOperations, resume: bool
    ):
        self.plan, self.operations, self.resume = plan, operations, resume
        self.stages: dict[str, Document] = {
            name: {"status": status}
            for name, status in plan.document()["stages"].items()
        }
        self.runtime: Document = {"status": "PENDING"}
        self.active: str | None = None
        self.cached = {
            name: load_bound(self._path(name), plan.plan_sha256) for name in STAGES
        }

    def _path(self, name: str) -> Path:
        return self.plan.output_path / f"stage-{name}.json"

    def _record(self, name: str, value: Document) -> Document:
        self.stages[name] = value
        save_bound(self._path(name), self.plan.plan_sha256, value)
        return value

    def stage(
        self, name: str, action: Callable[[], Document], *, rebuild: bool = False
    ) -> Document:
        self.active = name
        cached = self.cached[name]
        if cached is not None and cached["status"] in {
            "PASS",
            "FAIL",
            "ERROR",
            "INCOMPLETE",
        }:
            if not rebuild or cached["status"] != "PASS":
                self.stages[name] = cached
                return cached
        self._record(name, {"status": "RUNNING"})
        try:
            result = action()
            if result.get("status") not in {"PASS", "FAIL", "ERROR", "INCOMPLETE"}:
                raise ValueError("stage must return a terminal outcome")
            if rebuild and cached is not None and result != cached:
                raise ValueError(
                    "rebuilt training preparation differs from persisted evidence"
                )
        except Exception as error:
            result = {
                "status": "ERROR",
                "error": {"category": type(error).__name__, "message": str(error)},
            }
        return self._record(name, result)

    def cases(
        self,
        partition: str,
        ids: tuple[str, ...],
        training: Document,
        *,
        verify_only: bool = False,
    ) -> Document:
        from piginot.verification.budget import linear_quantile

        jobs = tuple(
            CaseJob(
                case_id,
                digest(
                    {
                        "plan": self.plan.plan_sha256,
                        "partition": partition,
                        "case_id": case_id,
                        "training": training,
                    }
                ),
                partial(self.operations.evaluate, case_id, partition, training),
            )
            for case_id in ids
        )
        store = OutcomeStore(self.plan.output_path / partition, self.plan.plan_sha256)
        for job in jobs:
            cached_case = store.cached(job)
            if cached_case is not None:
                references = self.operations.validate_case_cache(
                    job.request_id, partition, training
                )
                if (
                    references is not None
                    and cached_case.get("references") != references
                ):
                    raise ValueError("cached case references differ from raw evidence")
        if verify_only:
            cached = store.validate(jobs)
            if any(item is None for item in cached):
                raise ValueError("completed stage is missing a terminal case outcome")
            outcomes = tuple(item for item in cached if item is not None)
        else:
            outcomes = store.run(jobs)
        status = _overall(outcomes)
        quantities = self.plan.problem.verification.tolerances
        complete = len(outcomes) == len(ids) and all(
            item["status"] == "PASS"
            and all(
                isinstance(item.get("discrete_fem_errors", {}).get(name), (int, float))
                and math.isfinite(item["discrete_fem_errors"][name])
                and item["discrete_fem_errors"][name] >= 0.0
                for name in quantities
            )
            for item in outcomes
        )
        comparison: Document = {"status": "INCOMPLETE", "quantities": {}}
        if complete:
            values = {
                name: linear_quantile(
                    tuple(
                        float(item["discrete_fem_errors"][name]) for item in outcomes
                    ),
                    self.plan.problem.verification.quantile,
                )
                for name in quantities
            }
            passed = all(
                values[name] <= tolerance for name, tolerance in quantities.items()
            )
            comparison = {
                "status": "PASS" if passed else "FAIL",
                "quantities": values,
                "quantile": self.plan.problem.verification.quantile,
                "tolerances": dict(quantities),
            }
            status = comparison["status"]
        elif status == "PASS":
            status = "INCOMPLETE"
        scientific = (
            "PASS"
            if status == "PASS"
            and all(item.get("scientific_status") == "PASS" for item in outcomes)
            else "NOT_VERIFIED"
        )
        return {
            "status": status,
            "expected_cases": list(ids),
            "cases": list(outcomes),
            "discrete_fem_comparison": comparison,
            "scientific_status": scientific,
        }

    def cases_stage(
        self, name: str, ids: tuple[str, ...], training: Document
    ) -> Document:
        self.active = name
        cached = self.cached[name]
        if cached is not None and cached["status"] in {
            "PASS",
            "FAIL",
            "INCOMPLETE",
            "ERROR",
        }:
            checked = self.cases(name, ids, training, verify_only=True)
            if checked != cached:
                raise ValueError("cached case-stage summary differs from its outcomes")
        return self.stage(name, lambda: self.cases(name, ids, training))

    def pipeline(self) -> None:
        self.runtime = self.operations.probe_runtime()
        if self.runtime["status"] != "PASS":
            return
        material = self.stage("material", self.operations.admit_material)
        if material["status"] != "PASS":
            return
        lifting = self.stage("geometry-lifting", self.operations.verify_geometry)
        if lifting["status"] != "PASS":
            return
        prepared = self.stage(
            "train-preparation",
            lambda: self.operations.prepare_training(material, lifting),
            rebuild=True,
        )
        if prepared["status"] != "PASS":
            return
        training = self.stage(
            "train", lambda: self.operations.train(resume=self.resume)
        )
        if training["status"] != "PASS":
            return
        source = self.plan.config.source
        screen = self.cases_stage("train-screen", source.screen_case_ids, training)
        if screen["status"] == "PASS" and screen["scientific_status"] == "PASS":
            frozen = self.stage(
                "freeze", lambda: self.operations.freeze(training, screen), rebuild=True
            )
            if frozen["status"] == "PASS" and self.plan.config.selection.development:
                self.cases_stage(
                    "public-development", source.development_case_ids, training
                )
        self.stage(
            "export-report",
            lambda: self.operations.export(training, material, lifting),
            rebuild=True,
        )

    def report(self, *, interrupted: bool = False) -> int:
        if interrupted and self.active is not None:
            self._record(
                self.active,
                {"status": "RUNNING", "reason": "interrupted; resume explicitly"},
            )
        stages = {
            name: (
                {"status": "BLOCKED", "reason": "prerequisite did not pass"}
                if value["status"] == "PENDING"
                else value
            )
            for name, value in self.stages.items()
        }
        statuses = {value["status"] for value in stages.values()} | {
            self.runtime["status"]
        }
        code = (
            130
            if interrupted
            else 1
            if "ERROR" in statuses
            else 0
            if statuses <= {"PASS", "NOT_REQUESTED"}
            else 3
        )
        scientific = (
            "PASS" if code == 0 else "NOT_EVALUATED" if interrupted else "REFUSED"
        )
        if any(
            value.get("scientific_status") == "NOT_VERIFIED"
            for value in stages.values()
        ):
            scientific = "NOT_VERIFIED"
        document = {
            "schema": "piginot.campaign-report-v1",
            "plan_sha256": self.plan.plan_sha256,
            "problem_hash": self.plan.problem.problem_hash,
            "config_sha256": self.plan.config.config_sha256,
            "implementation": self.plan.document()["implementation"],
            "tolerances": self.plan.problem.verification.document(),
            "execution_status": "INTERRUPTED"
            if interrupted
            else "ERROR"
            if code == 1
            else "COMPLETED",
            "scientific_status": scientific,
            "claim_level": "NOT_VERIFIED",
            "exit_code": code,
            "runtime": self.runtime,
            "stages": stages,
        }
        atomic_write(
            self.plan.output_path / "report.json", canonical_json_bytes(document)
        )
        lines = [
            f"# {self.plan.config.name}",
            "",
            f"Execution: {document['execution_status']}",
            f"Scientific result: {document['scientific_status']}",
            "Claim: NOT_VERIFIED",
            "",
            "| Stage | Status |",
            "| --- | --- |",
        ]
        lines.extend(
            f"| {name} | {value['status']} |" for name, value in stages.items()
        )
        lines.extend(
            [
                "",
                "## Requested tolerances",
                "",
                "```json",
                json.dumps(document["tolerances"], sort_keys=True, indent=2),
                "```",
                "",
                "## Stage and case evidence",
                "",
                "```json",
                json.dumps(stages, sort_keys=True, indent=2),
                "```",
                "",
                f"Plan SHA-256: {self.plan.plan_sha256}",
                "",
            ]
        )
        atomic_write(self.plan.output_path / "report.md", "\n".join(lines).encode())
        return code


def execute_campaign(plan: ExecutionPlan, adapter: Any, *, resume: bool = False) -> int:
    """Run a validated plan using a clean, independently resolved implementation."""
    _require_numerical_dependencies()
    identity = require_clean_implementation_identity()
    if identity.package_sha256 != plan.package_sha256:
        raise ValueError("installed implementation differs from the execution plan")
    write_plan(plan, resume=resume)
    with run_lock(plan.output_path):
        run = _Run(
            plan, NumericalOperations(plan, adapter, identity.source_revision), resume
        )
        try:
            run.pipeline()
        except KeyboardInterrupt:
            run.report(interrupted=True)
            raise
        except Exception as error:
            value = {
                "status": "ERROR",
                "error": {"category": type(error).__name__, "message": str(error)},
            }
            if run.active is None:
                run.runtime = value
            else:
                run._record(run.active, value)
        finally:
            close = getattr(run.operations, "close", None)
            if callable(close):
                close()
        return run.report()


__all__ = ("execute_campaign",)
