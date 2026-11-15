"""Differentiate a global inequality merit with bounded-memory chunks."""

from __future__ import annotations

import math
from typing import Protocol

import torch

from piginot.objectives.energy_constraint import (
    ENERGY_RATIO_LIMIT,
    ConstraintTerms,
    PenaltyState,
    constraint_merit,
)
from piginot.training.checkpoint import model_state_hash
from piginot.training.constraint_budget import ConstraintBudget
from piginot.training.lbfgs_stage import GradientEvaluation


class TermEvaluator(Protocol):
    def __call__(
        self, model: torch.nn.Module, *, energy_weight: float, gradients: bool
    ) -> ConstraintTerms: ...


class ConstrainedObjective:
    def __init__(
        self,
        evaluator: TermEvaluator,
        reference_energy: float,
        state: PenaltyState,
        budget: ConstraintBudget,
        *,
        energy_ratio_limit: float = ENERGY_RATIO_LIMIT,
    ) -> None:
        constraint_merit(
            ConstraintTerms(0.0, 0.0),
            reference_energy,
            state,
            energy_ratio_limit=energy_ratio_limit,
        )
        self.evaluator, self.reference_energy = evaluator, reference_energy
        self.state, self.budget = state, budget
        self.energy_ratio_limit = energy_ratio_limit
        self._terms: dict[str, ConstraintTerms] = {}

    def __call__(self, model: torch.nn.Module) -> GradientEvaluation:
        self.budget.reserve_training()
        key = model_state_hash(model)
        values = self.evaluator(model, energy_weight=1.0, gradients=False)
        merit = constraint_merit(
            values,
            self.reference_energy,
            self.state,
            energy_ratio_limit=self.energy_ratio_limit,
        )
        self.budget.check()
        repeated = self.evaluator(
            model, energy_weight=merit.energy_weight, gradients=True
        )
        if model_state_hash(model) != key or repeated.admissible != values.admissible:
            raise ValueError("the two merit passes must preserve identical weights")
        if not all(
            math.isclose(
                getattr(values, name),
                getattr(repeated, name),
                rel_tol=1e-11,
                abs_tol=1e-12,
            )
            for name in ("bulk_energy", "traction_squared", "guard")
        ):
            raise ValueError("the two merit passes disagree on their fixed objective")
        self.budget.check()
        self._terms = {**self._terms, key: values}
        return GradientEvaluation(merit.value, values.admissible)

    def terms_for(self, model: torch.nn.Module) -> ConstraintTerms:
        key = model_state_hash(model)
        if key not in self._terms:
            raise ValueError("accepted weights have no evaluated constraint terms")
        return self._terms[key]
