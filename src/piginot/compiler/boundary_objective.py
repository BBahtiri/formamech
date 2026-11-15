"""FIT-only, fixed-quadrature energy/free-traction objective for the CAD pilot."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch

from piginot.bundle.serving import EnergyFormServing
from piginot.compiler.energy_training import EnergyTraining
from piginot.contracts import MaterialSpec
from piginot.objectives.energy import (
    TriangleDomain,
    plane_stress_potential_energy,
    stratified_triangle_quadrature,
)
from piginot.objectives.free_traction import (
    BoundaryQuadrature,
    boundary_quadrature,
    boundary_stress_scales,
    regional_rms,
    traction_point_weights,
)
from piginot.operators.energy_operator import (
    EnergyFormOperator,
    constitutive_in_plane_stress,
)
from piginot.problem import build_lifting, material_spec, query_frame
from piginot.training.lbfgs_stage import StageDeadline

BOUNDARY_PILOT_OBJECTIVE_ID = "normalized-energy-free-traction-fixed-v1"


@dataclass(frozen=True, slots=True)
class PilotCase:
    case_id: str
    values: Mapping[str, float]
    material: MaterialSpec
    parameters: torch.Tensor
    frame: Any
    lifting: Any
    training_points: torch.Tensor
    training_weights: torch.Tensor
    validation_points: torch.Tensor
    validation_weights: torch.Tensor
    boundary: BoundaryQuadrature
    stress_scales: torch.Tensor
    area: float
    strain_squared: float


@dataclass(frozen=True, slots=True)
class PilotMetrics:
    value: float
    bulk_energy: float
    traction_squared: float
    guard: float
    minimum_determinant: float
    admissible: bool
    regions: tuple[dict[str, Any], ...]


def _tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


class BoundaryPilotObjective:
    def __init__(
        self,
        context: EnergyTraining,
        case_ids: tuple[str, ...],
        *,
        strain_scales: Mapping[str, float],
        points_per_cell: int = 12,
        training_seed: int = 2916,
        validation_seed: int = 2917,
        chunk: int = 2048,
    ):
        if context.prepared is None:
            raise ValueError("the frozen initializer must be prepared first")
        if (
            not case_ids
            or len(set(case_ids)) != len(case_ids)
            or not set(case_ids) <= set(context.plan.config.source.fit_case_ids)
        ):
            raise ValueError("only distinct TRAIN-FIT cases may enter the pilot")
        if set(strain_scales) != set(case_ids) or any(
            not math.isfinite(v) or v < 0 for v in strain_scales.values()
        ):
            raise ValueError("finite nonnegative FIT load scales are required")
        if type(chunk) is not int or chunk <= 0 or training_seed == validation_seed:
            raise ValueError(
                "positive chunks and independent quadrature seeds required"
            )
        self.context, self.case_ids, self.chunk = context, case_ids, chunk
        self.points_per_cell, self.training_seed, self.validation_seed = (
            points_per_cell,
            training_seed,
            validation_seed,
        )
        self.energy_config = replace(
            context.objective_config, points_per_cell=points_per_cell
        )
        self.normalization = {**context.normalization}
        self.regions = tuple(
            sorted(
                condition.region
                for condition in context.plan.problem.boundary_conditions
                if condition.kind == "traction_free"
            )
        )
        if not self.regions:
            raise ValueError("the boundary pilot requires declared free regions")
        baseline = self.new_model()
        self.cases = tuple(
            self._prepare_case(name, strain_scales[name], baseline) for name in case_ids
        )

    def new_model(self) -> EnergyFormOperator:
        with torch.random.fork_rng(devices=[]):
            model = EnergyFormOperator(self.context.operator_config).double()
        model.load_state_dict(self.context.model.state_dict())
        return model.eval()

    def _prepare_case(
        self, case_id: str, strain: float, baseline: EnergyFormOperator
    ) -> PilotCase:
        context = self.context
        mesh = context.adapter.geometry(
            context.plan.problem, context.cases[case_id], level=1
        )
        domain = TriangleDomain(
            torch.tensor(mesh.nodes, dtype=torch.float64), torch.tensor(mesh.cells)
        )
        points, weights = stratified_triangle_quadrature(
            domain,
            self.points_per_cell,
            generator=torch.Generator().manual_seed(self.training_seed),
        )
        other, other_weights = stratified_triangle_quadrature(
            domain,
            self.points_per_cell,
            generator=torch.Generator().manual_seed(self.validation_seed),
        )
        rule = boundary_quadrature(mesh, self.regions)
        values = context._values(case_id)
        material = material_spec(context.plan.problem, values)
        frame, lifting = (
            query_frame(context.plan.problem, [values]),
            build_lifting(context.plan.problem, [values]),
        )
        parameters = context.parameters(values, torch.float64)
        served = EnergyFormServing(baseline).serve(
            rule.points,
            parameters=parameters,
            frame=frame,
            lifting=lifting,
            material=material,
            chunk=min(512, self.chunk),
        )
        young = float(material.properties["elastic_modulus"].value)
        scales = boundary_stress_scales(
            served.stress, rule, minimum=0.01 * young * max(strain, 1e-4)
        )
        return PilotCase(
            case_id,
            values,
            material,
            parameters,
            frame,
            lifting,
            points,
            weights,
            other,
            other_weights,
            rule,
            scales,
            domain.area,
            max(strain * strain, 1e-8),
        )

    def document(self) -> dict[str, Any]:
        return {
            "policy_id": BOUNDARY_PILOT_OBJECTIVE_ID,
            "training_labels": "NONE",
            "normalization": self.normalization,
            "points_per_cell": self.points_per_cell,
            "training_seed": self.training_seed,
            "validation_seed": self.validation_seed,
            "energy_guard_config": asdict(self.energy_config),
            "cases": [
                {
                    "case_id": case.case_id,
                    "values": dict(case.values),
                    "area_mm2": case.area,
                    "strain_squared": case.strain_squared,
                    "regions": list(case.boundary.regions),
                    "stress_scales_MPa": case.stress_scales.tolist(),
                    "tensor_sha256": {
                        name: _tensor_sha(value)
                        for name, value in self.case_tensors(case).items()
                    },
                }
                for case in self.cases
            ],
        }

    @staticmethod
    def case_tensors(case: PilotCase) -> dict[str, torch.Tensor]:
        return {
            "training_points": case.training_points,
            "training_weights": case.training_weights,
            "validation_points": case.validation_points,
            "validation_weights": case.validation_weights,
            "boundary_points": case.boundary.points,
            "boundary_normals": case.boundary.normals,
            "boundary_weights": case.boundary.weights,
            "boundary_regions": case.boundary.region_indices,
            "stress_scales": case.stress_scales,
        }

    def _gradient(
        self,
        model: EnergyFormOperator,
        case: PilotCase,
        points: torch.Tensor,
        gradients: bool,
    ) -> torch.Tensor:
        _, gradient = model.displacement_and_gradient(
            query_xy=points.unsqueeze(0),
            parameters=case.parameters,
            frame=case.frame,
            lifting=case.lifting,
            create_graph=gradients,
        )
        return gradient[0]

    @staticmethod
    def _deadline(deadline: float | None, deadline_utc: float | None = None) -> None:
        if (deadline is not None and time.monotonic() >= deadline) or (
            deadline_utc is not None and time.time() >= deadline_utc
        ):
            raise StageDeadline("pilot wall-clock allowance ended")

    def _bulk(
        self,
        model: EnergyFormOperator,
        case: PilotCase,
        *,
        gradients: bool,
        validation: bool,
        deadline: float | None,
        deadline_utc: float | None,
        energy_weight: float,
    ) -> tuple[float, float, float]:
        points, weights = (
            (case.validation_points, case.validation_weights)
            if validation
            else (case.training_points, case.training_weights)
        )
        energy, guard, minimum = 0.0, 0.0, math.inf
        for start in range(0, len(points), self.chunk):
            self._deadline(deadline, deadline_utc)
            stop = min(start + self.chunk, len(points))
            gradient = self._gradient(model, case, points[start:stop], gradients)
            terms = plane_stress_potential_energy(
                gradient,
                weights[start:stop],
                case.material,
                config=self.energy_config,
                reference_area=case.area,
            )
            value = terms.energy / case.strain_squared
            barrier = (
                self.energy_config.barrier_weight
                * terms.barrier
                * ((stop - start) / len(points))
            )
            energy += float(value.detach())
            guard += float(barrier.detach())
            minimum = min(minimum, terms.minimum_determinant)
            if gradients:
                ((energy_weight * value + barrier) / len(self.cases)).backward()  # type: ignore[no-untyped-call]
        return energy, guard, minimum

    def _surface(
        self,
        model: EnergyFormOperator,
        case: PilotCase,
        *,
        gradients: bool,
        traction_weight: float,
        deadline: float | None,
        deadline_utc: float | None,
    ) -> tuple[float, float, float, list[dict[str, Any]]]:
        rule = case.boundary
        weights = traction_point_weights(rule, case.stress_scales)
        stress_parts = []
        traction, guard, minimum = 0.0, 0.0, math.inf
        for start in range(0, len(weights), self.chunk):
            self._deadline(deadline, deadline_utc)
            stop = min(start + self.chunk, len(weights))
            gradient = self._gradient(model, case, rule.points[start:stop], gradients)
            # The same determinant barrier protects both arms' trial surfaces.
            # Stress is only evaluated in its physical domain; failed trials
            # receive a finite barrier and can be rejected by the line search.
            terms = plane_stress_potential_energy(
                gradient,
                weights.new_zeros(stop - start),
                case.material,
                config=self.energy_config,
                reference_area=case.area,
            )
            deformation = torch.eye(2, dtype=torch.float64) + gradient
            valid = (
                torch.linalg.det(deformation) > self.energy_config.admissibility_floor
            )
            safe = torch.where(
                valid[:, None, None], gradient, torch.zeros_like(gradient)
            )
            stress = constitutive_in_plane_stress(safe, case.material)
            normal = rule.normals[start:stop]
            force = (stress @ normal.unsqueeze(-1)).squeeze(-1)
            residual = (force.square().sum(1) * weights[start:stop]).sum()
            barrier = (
                self.energy_config.barrier_weight
                * terms.barrier
                * ((stop - start) / len(weights))
            )
            traction += float(residual.detach())
            guard += float(barrier.detach())
            minimum = min(minimum, terms.minimum_determinant)
            stress_parts.append(stress.detach())
            if gradients:
                ((traction_weight * residual + barrier) / len(self.cases)).backward()  # type: ignore[no-untyped-call]
        stress = torch.cat(stress_parts)
        force = (stress @ rule.normals.unsqueeze(-1)).squeeze(-1)
        rms, stress_rms = regional_rms(force, rule), regional_rms(stress, rule)
        regions = [
            {
                "case_id": case.case_id,
                "region": name,
                "traction_rms_MPa": float(rms[i]),
                "stress_rms_MPa": float(stress_rms[i]),
                "normalized_traction_rms": float(rms[i] / case.stress_scales[i]),
            }
            for i, name in enumerate(rule.regions)
        ]
        return traction, guard, minimum, regions

    def evaluate(
        self,
        model: torch.nn.Module,
        *,
        traction_weight: float,
        gradients: bool,
        validation: bool = False,
        deadline: float | None = None,
        deadline_utc: float | None = None,
        energy_weight: float = 1.0,
    ) -> PilotMetrics:
        if not isinstance(model, EnergyFormOperator) or traction_weight not in (
            0.0,
            1.0,
        ):
            raise ValueError(
                "the declared energy operator and fixed arm weight are required"
            )
        if gradients and validation:
            raise ValueError(
                "independent validation points cannot supply training gradients"
            )
        if (
            type(energy_weight) not in (int, float)
            or not math.isfinite(energy_weight)
            or energy_weight < 0
        ):
            raise ValueError("energy weight must be finite and nonnegative")
        if deadline_utc is not None and (
            type(deadline_utc) not in (int, float)
            or not math.isfinite(deadline_utc)
            or deadline_utc <= 0
        ):
            raise ValueError("UTC deadline must be finite and positive")
        if gradients:
            model.zero_grad(set_to_none=True)
        bulk, traction, guard, minimum = 0.0, 0.0, 0.0, math.inf
        regions = []
        for case in self.cases:
            energy, barrier, det = self._bulk(
                model,
                case,
                gradients=gradients,
                validation=validation,
                deadline=deadline,
                deadline_utc=deadline_utc,
                energy_weight=energy_weight,
            )
            surface, other_guard, other_det, rows = self._surface(
                model,
                case,
                gradients=gradients,
                traction_weight=traction_weight,
                deadline=deadline,
                deadline_utc=deadline_utc,
            )
            bulk += energy / len(self.cases)
            traction += surface / len(self.cases)
            guard += (barrier + other_guard) / len(self.cases)
            minimum = min(minimum, det, other_det)
            regions.extend(rows)
        return PilotMetrics(
            energy_weight * bulk + traction_weight * traction + guard,
            bulk,
            traction,
            guard,
            minimum,
            minimum > self.energy_config.admissibility_floor,
            tuple(regions),
        )
