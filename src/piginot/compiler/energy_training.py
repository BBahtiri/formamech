"""ProblemSpec energy objective and the existing exact-resume trainer adapter."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from piginot.bundle.serving import EnergyFormServing, problem_energy_module_digest
from piginot.compiler.execution_records import Document, digest
from piginot.compiler.planning import ExecutionPlan
from piginot.compiler.records import atomic_write
from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.doe import EvidenceRole
from piginot.objectives.energy import (
    PotentialEnergyConfig,
    PotentialEnergyTerms,
    TriangleDomain,
    plane_stress_potential_energy,
    stratified_triangle_quadrature,
)
from piginot.objectives.hierarchical_energy import (
    HierarchicalEnergyConfig,
    hierarchical_plane_stress_energy,
    subdivide_triangle_domain,
)
from piginot.operators.architecture import OperatorOutput
from piginot.operators.energy_operator import (
    EnergyFormOperator,
    EnergyOperatorConfig,
)
from piginot.problem import build_lifting, material_spec, query_frame
from piginot.training.checkpoint import (
    LoadedTrainingCheckpoint,
    load_training_checkpoint,
    model_state_hash,
)
from piginot.training.energy_engine import (
    CosineTailSchedule,
    EnergyTrainerConfig,
    train_energy_to_checkpoint,
)
from piginot.training.engine import TrainingPopulation
from piginot.training.identity import TrainingRunBinding
from piginot.training.search import GridSearchSpace, TrialConfig
from piginot.training.state import DeterministicSamplerPlan


@dataclass(frozen=True, slots=True)
class ReferencePrediction(OperatorOutput):
    """The same served u/P plus the gradient needed for Cauchy-stress QoIs."""

    gradient: torch.Tensor


class EnergyTraining:
    """No FEM labels are available to this TRAIN-only objective or conditioner."""

    def __init__(self, plan: ExecutionPlan, adapter: Any, revision: str):
        self.plan, self.adapter, self.revision = plan, adapter, revision
        self.cases = {case.case_id: case for case in plan.manifest.cases}
        self.dtype = {"float32": torch.float32, "float64": torch.float64}[
            plan.config.execution.training.dtype
        ]
        self.domains: dict[str, TriangleDomain] = {}
        self.fine_domains: dict[str, TriangleDomain] = {}
        self.hierarchical_config: HierarchicalEnergyConfig | None = None
        self.prepared: Document | None = None
        self.inference_model: EnergyFormOperator | None = None
        self.last_checkpoint: Path | None = None

    def _values(self, case_id: str) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in self.cases[case_id].primitive_coordinates.items()
        }

    def _normalization(self) -> Document:
        names = self.plan.config.model.parameter_names
        ids = tuple(sorted(self.plan.config.source.fit_case_ids))
        columns = tuple(
            tuple(self._values(case_id)[name] for case_id in ids) for name in names
        )
        means = tuple(math.fsum(column) / len(column) for column in columns)
        deviations = tuple(
            math.sqrt(math.fsum((value - mean) ** 2 for value in column) / len(column))
            or 1.0
            for mean, column in zip(means, columns, strict=True)
        )
        return {
            "parameter_names": list(names),
            "fit_case_ids": list(ids),
            "mean": list(means),
            "standard_deviation": list(deviations),
            "units": [self.plan.problem.parameter(name).unit.value for name in names],
            "policy": "train-fit-population-standard-deviation-v1",
        }

    def _trainer_config(self) -> EnergyTrainerConfig:
        training = self.plan.config.training
        space = GridSearchSpace(
            (training.learning_rate,),
            (1.0,),
            training.optimizer_steps,
            training.case_batch_size,
            1,
        )
        trial = TrialConfig(
            training.learning_rate,
            1.0,
            training.optimizer_steps,
            training.case_batch_size,
        )
        return EnergyTrainerConfig(
            space,
            trial,
            CosineTailSchedule(
                training.learning_rate,
                training.final_learning_rate,
                training.constant_steps,
                training.optimizer_steps,
            ),
        )

    def prepare(self) -> Document:
        ids = tuple(sorted(self.plan.config.source.fit_case_ids))
        training = self.plan.config.training
        if training.case_batch_size > len(ids):
            raise ValueError("case batch cannot exceed the TRAIN-fit population")
        self.normalization = self._normalization()
        shape = asdict(self.plan.config.model)
        shape.pop("architecture_id")
        self.operator_config = EnergyOperatorConfig(**shape)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(training.seed)
            self.model = EnergyFormOperator(self.operator_config).to(self.dtype)
        self.initial_state = {
            name: value.clone() for name, value in self.model.state_dict().items()
        }
        meshes = {}
        for case_id in ids:
            mesh = self.adapter.geometry(
                self.plan.problem, self.cases[case_id], level=1
            )
            self.domains[case_id] = TriangleDomain(
                torch.tensor(mesh.nodes, dtype=self.dtype),
                torch.tensor(mesh.cells, dtype=torch.int64),
            )
            meshes[case_id] = hash_spec(mesh, "content")
        snapshot = {
            "cases": [
                {
                    "case_id": case_id,
                    "values": self._values(case_id),
                    "mesh_sha256": meshes[case_id],
                }
                for case_id in ids
            ]
        }
        self.population = TrainingPopulation(
            digest(snapshot), self.plan.manifest.content_hash, EvidenceRole.TRAIN, ids
        )
        generator = random.Random(training.seed)
        self.sampler = DeterministicSamplerPlan(
            "compiler-uniform-without-replacement-v1",
            len(ids),
            tuple(
                tuple(generator.sample(range(len(ids)), training.case_batch_size))
                for _ in range(training.optimizer_steps)
            ),
        )
        self.trainer = self._trainer_config()
        obj = self.plan.config.objective
        self.objective_config = PotentialEnergyConfig(
            points_per_cell=obj.points_per_cell,
            barrier_weight=obj.barrier_weight,
            barrier_threshold=obj.barrier_threshold,
            admissibility_floor=obj.admissibility_floor,
        )
        if obj.is_hierarchical:
            self.hierarchical_config = HierarchicalEnergyConfig(
                base=self.objective_config,
                cross_mesh_weight=float(obj.cross_mesh_weight or 0.0),
                local_physics_weight=float(obj.local_physics_weight or 0.0),
                quadrature_degree=int(obj.quadrature_degree or 2),
                use_deterministic_quadrature=bool(
                    obj.use_deterministic_quadrature
                    if obj.use_deterministic_quadrature is not None
                    else False
                ),
            )
            for case_id, domain in self.domains.items():
                self.fine_domains[case_id] = subdivide_triangle_domain(domain)
            self.objective_hash = digest(
                {
                    "config": asdict(self.hierarchical_config),
                    "problem_hash": self.plan.problem.problem_hash,
                }
            )
        else:
            self.hierarchical_config = None
            self.objective_hash = digest(
                {
                    "config": asdict(self.objective_config),
                    "problem_hash": self.plan.problem.problem_hash,
                }
            )
        self.binding = TrainingRunBinding(
            self.population.dataset_snapshot_hash,
            self.population.public_manifest_hash,
            digest(self.normalization),
            self.model.architecture_id,
            problem_energy_module_digest(self.model, self.plan.problem.problem_hash),
            self.objective_hash,
            self.trainer.config_hash,
            self.sampler.plan_hash,
            model_state_hash(self.model),
            self.revision,
            training.seed,
            self.plan.config.execution.training,
            self.plan.problem.problem_hash,
        )
        self.prepared = {
            "status": "PASS",
            "normalization": self.normalization,
            "training_labels": "NONE",
            "objective": "potential-energy",
            "dataset_snapshot_sha256": self.population.dataset_snapshot_hash,
            "mesh_sha256": meshes,
            "run_binding_utf8": canonical_json_bytes(self.binding).decode("utf-8"),
            "run_hash": self.binding.run_hash,
            "sampler_plan_utf8": canonical_json_bytes(self.sampler).decode("utf-8"),
        }
        return self.prepared

    def parameters(
        self, values: Mapping[str, float], dtype: torch.dtype
    ) -> torch.Tensor:
        norm = self.normalization
        return torch.tensor(
            [
                [
                    (values[name] - mean) / std
                    for name, mean, std in zip(
                        norm["parameter_names"],
                        norm["mean"],
                        norm["standard_deviation"],
                        strict=True,
                    )
                ]
            ],
            dtype=dtype,
        )

    def objective(
        self, model: torch.nn.Module, sample_id: str, step: int
    ) -> PotentialEnergyTerms:
        if not isinstance(model, EnergyFormOperator):
            raise TypeError("energy objective requires the declared energy operator")
        domain = self.domains[sample_id]
        values = self._values(sample_id)
        params = self.parameters(values, self.dtype)
        frame = query_frame(self.plan.problem, [values], self.dtype)
        lifting = build_lifting(self.plan.problem, [values], self.dtype)
        material = material_spec(self.plan.problem, values)

        if self.hierarchical_config is not None:

            def _eval_grad(pts: torch.Tensor) -> torch.Tensor:
                _, grad = model.displacement_and_gradient(
                    query_xy=pts.unsqueeze(0),
                    parameters=params,
                    frame=frame,
                    lifting=lifting,
                    create_graph=True,
                )
                return grad[0]

            h_terms = hierarchical_plane_stress_energy(
                _eval_grad,
                domain,
                material,
                config=self.hierarchical_config,
                fine_domain=self.fine_domains.get(sample_id),
            )
            return PotentialEnergyTerms(
                total=h_terms.total,
                energy=h_terms.primary.energy,
                barrier=h_terms.primary.barrier,
                minimum_determinant=h_terms.primary.minimum_determinant,
                inadmissible_points=h_terms.primary.inadmissible_points,
            )

        points, weights = stratified_triangle_quadrature(
            domain, self.objective_config.points_per_cell
        )
        _, gradient = model.displacement_and_gradient(
            query_xy=points.unsqueeze(0),
            parameters=params,
            frame=frame,
            lifting=lifting,
            create_graph=True,
        )
        return plane_stress_potential_energy(
            gradient[0],
            weights,
            material,
            config=self.objective_config,
            reference_area=domain.area,
        )

    def checkpoint_path(self, step: int) -> Path:
        return self.plan.output_path / "checkpoints" / f"step-{step:09d}"

    def _load(self, path: Path, model: EnergyFormOperator) -> LoadedTrainingCheckpoint:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.trainer.schedule.learning_rate(0),
            foreach=False,
            fused=False,
        )
        return load_training_checkpoint(
            path, model=model, optimizer=optimizer, binding=self.binding
        )

    def _checkpoints(self, resume: bool) -> tuple[int, Path | None]:
        self.last_checkpoint = None
        folder = self.plan.output_path / "checkpoints"
        if folder.is_symlink():
            raise ValueError("checkpoint folder cannot be a symlink")
        folder.mkdir(parents=True, exist_ok=True)
        paths = tuple(sorted(folder.iterdir()))
        if paths and not resume:
            raise FileExistsError(
                "existing training checkpoints require explicit resume"
            )
        targets = self._targets()
        expected = tuple(self.checkpoint_path(step) for step in targets)
        if paths != expected[: len(paths)]:
            raise ValueError("checkpoint roster is mixed, noncontiguous or unexpected")
        completed = 0
        for step, path in zip(targets, paths, strict=False):
            if path.is_symlink() or not path.is_dir():
                raise ValueError("checkpoint must be a real directory")
            with torch.random.fork_rng(devices=[]):
                self.model.load_state_dict(self.initial_state)
                loaded = self._load(path, self.model)
            if loaded.manifest.step != step or loaded.sampler_state.next_step != step:
                raise ValueError("checkpoint step differs from its registered target")
            completed = step
            self.last_checkpoint = path
        self.model.load_state_dict(self.initial_state)
        return completed, paths[-1] if paths else None

    def _targets(self) -> tuple[int, ...]:
        training = self.plan.config.training
        return tuple(
            range(
                training.checkpoint_interval,
                training.optimizer_steps,
                training.checkpoint_interval,
            )
        ) + (training.optimizer_steps,)

    def train(self, *, resume: bool) -> Document:
        from piginot._job_evidence import archive_checkpoint

        if self.prepared is None:
            raise ValueError("TRAIN preparation must precede training")
        completed, previous = self._checkpoints(resume)
        for target in self._targets():
            if target <= completed:
                continue
            self.model.load_state_dict(self.initial_state)
            result = train_energy_to_checkpoint(
                self.checkpoint_path(target),
                model=self.model,
                population=self.population,
                plan=self.sampler,
                binding=self.binding,
                config=self.trainer,
                objective=self.objective,
                objective_hash=self.objective_hash,
                target_step=target,
                resume_from=previous,
            )
            previous, completed = result.checkpoint.path, result.step
            self.last_checkpoint = previous
            progress = {
                "run_hash": self.binding.run_hash,
                "step": completed,
                "checkpoint": str(previous),
                "model_state_hash": result.model_state_hash,
                "history": [asdict(record) for record in result.history],
            }
            atomic_write(
                self.plan.output_path / "training-progress.json",
                canonical_json_bytes(progress),
            )
            archive_checkpoint(self.plan.output_path, previous, completed)
        assert previous is not None
        with torch.random.fork_rng(devices=[]):
            loaded = self._load(previous, self.model)
        self.final_training = {
            "status": "PASS",
            "selection_rule": "final",
            "step": completed,
            "run_hash": self.binding.run_hash,
            "checkpoint_path": str(previous),
            "checkpoint_sha256": loaded.manifest.manifest_hash,
            "model_state_hash": loaded.manifest.model_state_hash,
        }
        return self.final_training

    def load_final(self, training: Document) -> None:
        expected = self.checkpoint_path(self.plan.config.training.optimizer_steps)
        if (
            Path(training["checkpoint_path"]) != expected
            or training["run_hash"] != self.binding.run_hash
        ):
            raise ValueError("final checkpoint does not match the fixed selection rule")
        with torch.random.fork_rng(devices=[]):
            loaded = self._load(expected, self.model)
        if (
            loaded.manifest.manifest_hash != training["checkpoint_sha256"]
            or loaded.manifest.model_state_hash != training["model_state_hash"]
        ):
            raise ValueError("final checkpoint evidence differs")
        self.inference_model = None

    def predict(self, case_id: str, points: torch.Tensor) -> OperatorOutput:
        values = self._values(case_id)
        inference = self.inference_model
        if inference is None:
            with torch.random.fork_rng(devices=[]):
                inference = EnergyFormOperator(self.operator_config).to(torch.float64)
            inference.load_state_dict(
                {
                    name: tensor.to(torch.float64)
                    for name, tensor in self.model.state_dict().items()
                }
            )
            inference.requires_grad_(False).eval()
            self.inference_model = inference
        served = EnergyFormServing(inference).serve(
            points.to(torch.float64),
            parameters=self.parameters(values, torch.float64),
            frame=query_frame(self.plan.problem, [values]),
            lifting=build_lifting(self.plan.problem, [values]),
            material=material_spec(self.plan.problem, values),
            chunk=512,
        )
        return ReferencePrediction(
            served.u.unsqueeze(0),
            served.stress.unsqueeze(0),
            served.gradient.unsqueeze(0),
        )


__all__ = ("EnergyTraining",)
