"""Inequality augmented-Lagrangian merit for the bounded FIT-only study."""

from __future__ import annotations

import math
from dataclasses import dataclass

ENERGY_RATIO_LIMIT = 1.005
MAXIMUM_PENALTY = 1e6


def _finite(value: float, *, positive: bool = False) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


@dataclass(frozen=True, slots=True)
class ConstraintTerms:
    bulk_energy: float
    traction_squared: float
    guard: float = 0.0
    admissible: bool = True

    def __post_init__(self) -> None:
        if (
            any(
                not _finite(value)
                for value in (self.bulk_energy, self.traction_squared, self.guard)
            )
            or type(self.admissible) is not bool
        ):
            raise ValueError("constraint terms must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class PenaltyState:
    multiplier: float = 0.0
    penalty: float = 1000.0
    previous_violation: float = 0.0

    def __post_init__(self) -> None:
        if (
            not _finite(self.multiplier)
            or not _finite(self.penalty, positive=True)
            or self.penalty > MAXIMUM_PENALTY
            or not _finite(self.previous_violation)
        ):
            raise ValueError("invalid inequality multiplier or penalty state")


@dataclass(frozen=True, slots=True)
class ConstraintMerit:
    value: float
    energy_weight: float
    constraint: float


def constraint_merit(
    terms: ConstraintTerms,
    reference_energy: float,
    state: PenaltyState,
    *,
    energy_ratio_limit: float = ENERGY_RATIO_LIMIT,
) -> ConstraintMerit:
    """Return the merit and its coefficient of the global energy gradient."""
    if not _finite(reference_energy, positive=True):
        raise ValueError("reference energy must be finite and positive")
    if not _finite(energy_ratio_limit, positive=True):
        raise ValueError("energy ratio limit must be finite and positive")
    violation = terms.bulk_energy / reference_energy - energy_ratio_limit
    active = max(0.0, state.multiplier + state.penalty * violation)
    value = (
        terms.traction_squared
        + (active * active - state.multiplier * state.multiplier) / (2 * state.penalty)
        + terms.guard
    )
    coefficient = active / reference_energy
    if not all(math.isfinite(v) for v in (violation, value, coefficient)):
        raise ValueError("the derived inequality merit must be finite")
    return ConstraintMerit(value, coefficient, violation)


def update_penalty(state: PenaltyState, violation: float) -> PenaltyState:
    """Advance only between blocks, when optimizer history will be discarded."""
    if type(violation) not in (int, float) or not math.isfinite(violation):
        raise ValueError("constraint violation must be finite")
    positive = max(0.0, violation)
    penalty = state.penalty
    if positive > max(1e-6, 0.5 * state.previous_violation):
        penalty = min(MAXIMUM_PENALTY, 10 * penalty)
    return PenaltyState(
        multiplier=max(0.0, state.multiplier + state.penalty * violation),
        penalty=penalty,
        previous_violation=positive,
    )
