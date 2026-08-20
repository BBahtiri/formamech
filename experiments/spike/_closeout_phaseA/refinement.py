"""Richardson-based convergence assessment for the S1 closeout.

Supersedes the iteration-8 ``all_last_changes_within_limit`` check, which tested
only the final increment of a refinement sequence. That quantity shrinks for any
convergent sequence regardless of how much error remains, so it cannot detect slow
convergence -- which is the regime the spike is actually in. This module reports
observed order, the extrapolated limit, and the estimated remaining error at every
level, and refuses to report a number when the sequence does not support one.

Conceptually this is the S8 ``G8.estimator-*`` machinery arriving early. It is
deliberately small and standalone so S8 can reimplement it under its own tests
rather than inherit spike code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

# A model peak-stress gate at 20% cannot be evaluated against ground truth that
# carries error of the same order. Two percent keeps the ground-truth contribution
# an order of magnitude below the quantity being gated.
PRODUCTION_REMAINING_ERROR_LIMIT: Final = 0.02
DEFAULT_REFINEMENT_RATIO: Final = 2.0
PRODUCTION_LEVEL_INDEX: Final = 1
# A sequence whose two finest levels agree to within this fraction has converged
# to the noise floor. Richardson cannot extract an order from it -- the final
# increment carries no signal -- but the remaining error is still boundable.
SETTLED_TOLERANCE: Final = 1e-3


@dataclass(frozen=True)
class RichardsonEstimate:
    """Convergence assessment for one monotone refinement sequence."""

    status: str
    reason: str | None
    observed_order: float | None
    extrapolated_limit: float | None
    relative_remaining_error: tuple[float, ...]

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def production_remaining_error(
        self,
        level_index: int = PRODUCTION_LEVEL_INDEX,
    ) -> float | None:
        if not self.ok or level_index >= len(self.relative_remaining_error):
            return None
        return self.relative_remaining_error[level_index]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "observed_order": self.observed_order,
            "extrapolated_limit": self.extrapolated_limit,
            "relative_remaining_error": list(self.relative_remaining_error),
        }


def _refused(reason: str) -> RichardsonEstimate:
    return RichardsonEstimate("NOT_RUN", reason, None, None, ())


def richardson(
    values: Sequence[float],
    *,
    refinement_ratio: float = DEFAULT_REFINEMENT_RATIO,
) -> RichardsonEstimate:
    """Estimate observed order, limit and remaining error from >=3 refinements.

    ``values`` are ordered coarse to fine under a constant mesh refinement ratio.
    A sequence that is not finite, not monotone, or not converging is refused with
    a recorded reason rather than reported as a number: under STEPS section 0, a
    check that could not execute is NOT_RUN, never PASS.
    """
    if refinement_ratio <= 1.0 or not math.isfinite(refinement_ratio):
        raise ValueError("refinement_ratio must be finite and greater than one")
    if len(values) < 3:
        return _refused("at least three refinement levels are required")
    numbers = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numbers):
        return _refused("refinement values must all be finite")

    coarse_step = numbers[-2] - numbers[-3]
    fine_step = numbers[-1] - numbers[-2]
    if coarse_step == 0.0:
        return _refused("refinement sequence is stationary; order is undetermined")
    if fine_step == 0.0:
        limit = numbers[-1]
        return RichardsonEstimate(
            "OK",
            "sequence converged exactly at the finest level",
            math.inf,
            limit,
            _remaining(numbers, limit),
        )
    if coarse_step * fine_step < 0.0:
        return _refused("refinement sequence is not monotone; order is undetermined")

    ratio = fine_step / coarse_step
    if ratio >= 1.0:
        return _refused("refinement sequence is diverging; order is undetermined")

    observed_order = math.log(1.0 / ratio) / math.log(refinement_ratio)
    limit = numbers[-1] + fine_step * ratio / (1.0 - ratio)
    if not math.isfinite(limit) or limit == 0.0:
        return _refused("extrapolated limit is not finite and non-zero")
    return RichardsonEstimate(
        "OK",
        None,
        observed_order,
        limit,
        _remaining(numbers, limit),
    )


def _remaining(values: Sequence[float], limit: float) -> tuple[float, ...]:
    return tuple(abs(limit - value) / abs(limit) for value in values)


def assess_refinement_study(
    sequences: Mapping[str, Sequence[float]],
    *,
    refinement_ratio: float = DEFAULT_REFINEMENT_RATIO,
    production_level_index: int = PRODUCTION_LEVEL_INDEX,
    remaining_error_limit: float = PRODUCTION_REMAINING_ERROR_LIMIT,
) -> dict[str, object]:
    """Assess every sequence and reduce to a single gate verdict.

    Any refused sequence sets the verdict to FAIL: NOT_RUN is never PASS.

    ``production_level_index`` should be 0 -- the production mesh must be the
    *coarsest* level in the study. Iteration 8 refined 16/32/64 with the
    production mesh in the middle; the nx=16 level sits outside the asymptotic
    range, which corrupts the observed order and inflates the estimated remaining
    error. On one dogbone the 16/32/64 sequence reported order 0.85 and 2.46%
    remaining at nx=32, while 32/64/128 reported order 2.28 and 1.40% -- a FAIL
    and a PASS from the same solver on the same geometry. A non-zero index is
    permitted but recorded as a warning.
    """
    per_geometry: dict[str, object] = {}
    verdict = "PASS"
    reasons: list[str] = []
    for geometry_id, values in sequences.items():
        estimate = richardson(values, refinement_ratio=refinement_ratio)
        remaining = estimate.production_remaining_error(production_level_index)
        record = dict(estimate.as_dict())
        record["production_level_index"] = production_level_index
        record["production_remaining_error"] = remaining
        record["passes"] = bool(
            estimate.ok and remaining is not None and remaining <= remaining_error_limit
        )
        per_geometry[geometry_id] = record
        if not estimate.ok:
            fallback = direct_bound(
                values, production_level_index=production_level_index
            )
            record["direct_bound"] = fallback.as_dict()
            if fallback.ok and fallback.remaining_error is not None:
                record["production_remaining_error"] = fallback.remaining_error
                record["passes"] = fallback.remaining_error <= remaining_error_limit
                if not record["passes"]:
                    verdict = "FAIL"
                    reasons.append(
                        f"{geometry_id}: settled bound "
                        f"{fallback.remaining_error:.4f} exceeds "
                        f"{remaining_error_limit:.4f}"
                    )
            else:
                verdict = "FAIL"
                reasons.append(f"{geometry_id}: {estimate.reason}")
        elif remaining is None or remaining > remaining_error_limit:
            verdict = "FAIL"
            reasons.append(
                f"{geometry_id}: remaining error "
                f"{remaining:.4f} exceeds {remaining_error_limit:.4f}"
                if remaining is not None
                else f"{geometry_id}: production level absent"
            )
    warnings: list[str] = []
    if production_level_index != 0:
        warnings.append(
            "the production mesh is not the coarsest level in this study; levels "
            "below it may sit outside the asymptotic range and bias the order"
        )
    return {
        "remaining_error_limit": remaining_error_limit,
        "production_level_index": production_level_index,
        "warnings": warnings,
        "refinement_ratio": refinement_ratio,
        "geometries": per_geometry,
        "verdict": verdict,
        "verdict_reasons": reasons,
    }


@dataclass(frozen=True)
class CrossCheck:
    """Discretization cross-check between two element orders.

    Richardson needs a sequence inside its asymptotic range. A sharp notch root can
    leave P2 drifting almost linearly, where no order can be extracted. Two
    independent discretizations bracketing the same value is weaker evidence than a
    convergence order, and is named that way rather than dressed up as one.
    """

    status: str
    reason: str | None
    linear_value: float
    quadratic_value: float
    relative_disagreement: float

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "linear_value": self.linear_value,
            "quadratic_value": self.quadratic_value,
            "relative_disagreement": self.relative_disagreement,
            "bounds_discretization_error_only": True,
        }


def cross_check(linear_value: float, quadratic_value: float) -> CrossCheck:
    """Compare the finest P1 and P2 results for the same quantity."""
    if not (math.isfinite(linear_value) and math.isfinite(quadratic_value)):
        return CrossCheck(
            "NOT_RUN", "values must be finite", linear_value, quadratic_value, math.inf
        )
    reference = max(abs(linear_value), abs(quadratic_value))
    if reference == 0.0:
        return CrossCheck(
            "NOT_RUN", "both values are zero", linear_value, quadratic_value, math.inf
        )
    return CrossCheck(
        "OK",
        None,
        float(linear_value),
        float(quadratic_value),
        abs(linear_value - quadratic_value) / reference,
    )


@dataclass(frozen=True)
class DirectBound:
    """Conservative remaining-error bound for a sequence that has settled.

    Weaker evidence than an observed convergence order, and labelled as such. It
    applies only when the two finest levels agree to within SETTLED_TOLERANCE, so
    the sequence has stopped moving rather than failed to start converging. The
    bound is the spread between the production level and the finest level, which
    over-states the remaining error whenever the sequence is monotone.
    """

    status: str
    reason: str | None
    remaining_error: float | None
    settled_change: float | None

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "remaining_error": self.remaining_error,
            "settled_change": self.settled_change,
            "evidence_strength": "bound_only_no_observed_order",
        }


def direct_bound(
    values: Sequence[float],
    *,
    production_level_index: int = 0,
    settled_tolerance: float = SETTLED_TOLERANCE,
) -> DirectBound:
    """Bound remaining error directly, for a sequence that has settled."""
    numbers = [float(value) for value in values]
    if len(numbers) < 2 or not all(math.isfinite(value) for value in numbers):
        return DirectBound("NOT_RUN", "need at least two finite levels", None, None)
    finest = numbers[-1]
    if finest == 0.0:
        return DirectBound("NOT_RUN", "finest level is zero", None, None)
    settled_change = abs(numbers[-1] - numbers[-2]) / abs(finest)
    if settled_change > settled_tolerance:
        return DirectBound(
            "NOT_RUN",
            f"sequence has not settled: final change {settled_change:.2e} exceeds "
            f"{settled_tolerance:.0e}",
            None,
            settled_change,
        )
    production = numbers[min(production_level_index, len(numbers) - 1)]
    return DirectBound(
        "OK",
        None,
        abs(finest - production) / abs(finest),
        settled_change,
    )
