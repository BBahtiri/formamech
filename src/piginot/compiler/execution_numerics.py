"""Real numerical stage adapters; no campaign family dispatch lives here."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, cast

from piginot.compiler.energy_training import EnergyTraining
from piginot.compiler.execution_records import Document, digest
from piginot.compiler.execution_runtime import RuntimeGuard
from piginot.compiler.planning import ExecutionPlan
from piginot.contracts import CanonicalMesh
from piginot.operators.architecture import LiftingVerification
from piginot.physics.errors import PhysicsError
from piginot.problem import admit_material, box_points, verify_lifting
from piginot.problem.verification import LiftingEvidence, ParameterSamplingError

_MATERIAL_CHECKS = {
    "reference",
    "objectivity",
    "symmetry",
    "consistency",
    "moduli",
    "closure",
    "ellipticity",
}


def _finite_evidence(value: Any) -> Any:
    """Retain failed nonfinite measurements in canonical evidence explicitly."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {key: _finite_evidence(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_evidence(item) for item in value]
    return value


class NumericalOperations:
    def __init__(self, plan: ExecutionPlan, adapter: Any, revision: str):
        self.plan, self.adapter, self.revision = plan, adapter, revision
        self.guard = RuntimeGuard()
        self.training: EnergyTraining | None = None

    def probe_runtime(self) -> Document:
        return self.guard.probe(
            self.plan.config.execution.training, self.plan.config.execution.reference
        )

    def close(self) -> None:
        self.guard.close()

    def admit_material(self) -> Document:
        document = admit_material(self.plan.problem, fem=True).document()
        points, reports, fem = document["points"], document["reports"], document["fem"]
        complete = (
            bool(points)
            and len(points) == len(reports) == len(fem)
            and all(
                {check["name"] for check in report["checks"]} == _MATERIAL_CHECKS
                for report in reports
            )
        )
        return {
            "status": "INCOMPLETE"
            if not complete
            else "PASS"
            if document["passed"]
            else "FAIL",
            "all_eight_checks_complete": complete,
            "evidence": _finite_evidence(document),
        }

    def verify_geometry(self) -> Document:
        refused: list[Document] = []
        parameter_names = {parameter.name for parameter in self.plan.problem.parameters}

        def mesh_for(values: Mapping[str, float]) -> CanonicalMesh | None:
            try:
                return cast(
                    CanonicalMesh,
                    self.adapter.geometry_values(
                        self.plan.problem,
                        {
                            name: value
                            for name, value in values.items()
                            if name in parameter_names
                        },
                        level=1,
                    ),
                )
            except ValueError as error:
                refused.append(
                    {
                        "parameters": dict(values),
                        "error": {
                            "category": type(error).__name__,
                            "message": str(error),
                        },
                    }
                )
                return None

        seed = self.plan.config.training.seed
        try:
            points = box_points(self.plan.problem, seed=seed)
        except ParameterSamplingError as error:
            return {
                "status": "FAIL",
                "reason": "parameter-box sampling could not supply the required feasible interior points",
                "error": {"category": type(error).__name__, "message": str(error)},
                "sampling": {
                    "requested": error.requested,
                    "accepted": error.accepted,
                    "attempts": error.attempts,
                    "seed": seed,
                },
            }
        evidence = verify_lifting(self.plan.problem, mesh_for, points, seed=seed)
        status = (
            "INCOMPLETE"
            if evidence.refused_geometries
            else "PASS"
            if evidence.passed
            else "FAIL"
        )
        result = {
            "status": status,
            "evidence": _finite_evidence(evidence.document()),
            "refused_geometries": refused,
            "requested_parameter_vectors": len(points),
        }
        if status == "PASS":
            result["verification"] = asdict(evidence.verification(self.plan.problem))
        return result

    def prepare_training(self, material: Document, lifting: Document) -> Document:
        self.training = EnergyTraining(self.plan, self.adapter, self.revision)
        return self.training.prepare()

    def _training(self) -> EnergyTraining:
        if self.training is None:
            raise ValueError("TRAIN preparation has not completed")
        return self.training

    def train(self, *, resume: bool) -> Document:
        try:
            return self._training().train(resume=resume)
        except Exception as error:
            return {
                "status": "FAIL"
                if isinstance(error, (PhysicsError, FloatingPointError))
                else "ERROR",
                "error": {"category": type(error).__name__, "message": str(error)},
                "last_checkpoint": self._last_checkpoint(),
            }

    def _last_checkpoint(self) -> str | None:
        path = self.training.last_checkpoint if self.training is not None else None
        return str(path) if path is not None else None

    def validate_case_cache(
        self, case_id: str, partition: str, training: Document
    ) -> list[Document]:
        from piginot.compiler.reference_jobs import validate_case_cache

        self._training().load_final(training)
        return validate_case_cache(
            self.plan, self.adapter, case_id, partition, training
        )

    def evaluate(self, case_id: str, partition: str, training: Document) -> Document:
        from piginot.compiler.reference_jobs import evaluate_case

        context = self._training()
        context.load_final(training)
        return evaluate_case(
            self.plan,
            self.adapter,
            case_id,
            partition,
            training,
            predict=context.predict,
        )

    def freeze(self, training: Document, screen: Document) -> Document:
        context = self._training()
        context.load_final(training)
        identities = {
            "plan_sha256": self.plan.plan_sha256,
            "problem_hash": self.plan.problem.problem_hash,
            "run_hash": context.binding.run_hash,
            "normalization_sha256": context.binding.normalization_hash,
            "objective_sha256": context.binding.objective_registry_hash,
            "architecture_module_digest": context.binding.architecture_module_digest,
            "dataset_snapshot_sha256": context.binding.dataset_snapshot_hash,
            "model_state_hash": training["model_state_hash"],
            "checkpoint_sha256": training["checkpoint_sha256"],
            "screen_sha256": digest(screen),
        }
        return {
            "status": "PASS",
            "identities": identities,
            "freeze_sha256": digest(identities),
        }

    def export(
        self, training: Document, material: Document, lifting: Document
    ) -> Document:
        from piginot.compiler.exporting import export_candidate

        context = self._training()
        context.load_final(training)
        proof = {
            **lifting["verification"],
            "boundary_refinement_levels": tuple(
                lifting["verification"]["boundary_refinement_levels"]
            ),
        }
        document = lifting["evidence"]
        evidence = LiftingEvidence(
            **{
                **{
                    name: document[name]
                    for name in (
                        "problem_hash",
                        "kind",
                        "parameter_vectors",
                        "refused_geometries",
                        "node_residual",
                        "conformity",
                        "orientation",
                        "mutant_miss",
                    )
                },
                "refinements": tuple(document["refinements"]),
                "boundary_residual": {
                    int(level): value
                    for level, value in document["boundary_residual"].items()
                },
            }
        )
        return export_candidate(
            self.plan,
            checkpoint_path=context.checkpoint_path(
                self.plan.config.training.optimizer_steps
            ),
            run_binding=context.binding,
            normalization=context.normalization,
            material_admission=material,
            lifting_verification=LiftingVerification(**proof),
            lifting_evidence=evidence,
            model=context.model,
        )


__all__ = ("NumericalOperations",)
