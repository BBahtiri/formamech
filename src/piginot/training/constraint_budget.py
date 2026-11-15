"""Conservative reservations and dual-clock expiry for constrained training."""

from __future__ import annotations

import math
import time
from dataclasses import astuple, dataclass, replace

from piginot.training.lbfgs_stage import EvaluationLimit, StageDeadline


@dataclass(frozen=True, slots=True)
class ConstraintLimits:
    max_iterations: int = 50
    max_gradients: int = 100
    max_forwards: int = 200
    max_validations: int = 4
    max_wall_seconds: int = 43200

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in astuple(self)):
            raise ValueError("constraint-study limits must be positive integers")


@dataclass(frozen=True, slots=True)
class ConstraintUsage:
    iterations: int = 0
    gradient_evaluations: int = 0
    forward_evaluations: int = 0
    validation_evaluations: int = 0


class ConstraintBudget:
    def __init__(
        self,
        limits: ConstraintLimits,
        *,
        deadline_utc: float,
        usage: ConstraintUsage | None = None,
    ) -> None:
        if (
            type(deadline_utc) not in (int, float)
            or not math.isfinite(deadline_utc)
            or deadline_utc <= 0
            or deadline_utc - time.time() > limits.max_wall_seconds + 1
        ):
            raise ValueError(
                "a finite deadline within the original allowance is required"
            )
        self.limits, self.deadline_utc = limits, deadline_utc
        self.deadline_monotonic = time.monotonic() + max(
            0.0, min(limits.max_wall_seconds, deadline_utc - time.time())
        )
        usage = ConstraintUsage() if usage is None else usage
        if not self._valid(usage):
            raise ValueError("invalid restored constraint-study counters")
        self.usage = usage

    def _valid(self, usage: ConstraintUsage) -> bool:
        return (
            all(type(value) is int and value >= 0 for value in astuple(usage))
            and usage.iterations
            <= min(self.limits.max_iterations, usage.gradient_evaluations)
            and usage.gradient_evaluations <= self.limits.max_gradients
            and usage.forward_evaluations <= self.limits.max_forwards
            and usage.forward_evaluations == 2 * usage.gradient_evaluations
            and usage.validation_evaluations <= self.limits.max_validations
        )

    def check(self) -> None:
        if (
            time.time() >= self.deadline_utc
            or time.monotonic() >= self.deadline_monotonic
        ):
            raise StageDeadline("the original constrained-study allowance expired")

    def _reserve(self, usage: ConstraintUsage, *, check_clock: bool = True) -> None:
        if check_clock:
            self.check()
        if not self._valid(usage):
            raise EvaluationLimit("the constrained-study budget is exhausted")
        self.usage = usage

    def reserve_training(self) -> None:
        """Debit both passes before work; failed partial evaluations stay spent."""
        self._reserve(
            replace(
                self.usage,
                gradient_evaluations=self.usage.gradient_evaluations + 1,
                forward_evaluations=self.usage.forward_evaluations + 2,
            )
        )

    def reserve_validation(self) -> None:
        self._reserve(
            replace(
                self.usage,
                validation_evaluations=self.usage.validation_evaluations + 1,
            )
        )

    def accept_iteration(self) -> None:
        # Acceptance has already happened: expiry must not erase its debit.
        self._reserve(
            replace(self.usage, iterations=self.usage.iterations + 1), check_clock=False
        )

    def restore_usage(self, usage: ConstraintUsage) -> None:
        if not self._valid(usage) or any(
            saved < spent
            for saved, spent in zip(astuple(usage), astuple(self.usage), strict=True)
        ):
            raise ValueError("resume cannot reduce counters already spent")
        self.usage = usage
