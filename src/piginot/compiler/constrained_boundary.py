"""Bridge fixed FIT quadrature to the constrained-stage term evaluator."""

from __future__ import annotations

import torch

from piginot.compiler.boundary_objective import BoundaryPilotObjective, PilotMetrics
from piginot.objectives.energy_constraint import ConstraintTerms
from piginot.training.constraint_budget import ConstraintBudget


class BoundaryConstraintEvaluator:
    def __init__(self, data: BoundaryPilotObjective, budget: ConstraintBudget) -> None:
        self.data, self.budget = data, budget

    def __call__(
        self, model: torch.nn.Module, *, energy_weight: float, gradients: bool
    ) -> ConstraintTerms:
        self.budget.check()
        out = self.data.evaluate(
            model,
            energy_weight=energy_weight,
            traction_weight=1.0,
            gradients=gradients,
            deadline=self.budget.deadline_monotonic,
            deadline_utc=self.budget.deadline_utc,
        )
        return ConstraintTerms(
            out.bulk_energy, out.traction_squared, out.guard, out.admissible
        )

    def validate(self, model: torch.nn.Module) -> PilotMetrics:
        self.budget.reserve_validation()
        return self.data.evaluate(
            model,
            traction_weight=0.0,
            gradients=False,
            validation=True,
            deadline=self.budget.deadline_monotonic,
            deadline_utc=self.budget.deadline_utc,
        )
