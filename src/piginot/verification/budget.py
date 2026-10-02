"""Per-slice error budget (``G14.budget``).

For every case and quantity the budget adds three non-negative components:
the surrogate-to-reference discrepancy, a conservative discretisation bound
of the reference solver and a conservative projection error. On every
coverage slice the slice statistic of those per-case totals must not exceed
the product tolerance. A missing component makes the slice ``NOT_RUN``; it is
never treated as zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from piginot.verification.status import CheckStatus

COMPONENTS = ("surrogate", "discretisation", "projection")
QUANTILE = 0.9
MINIMUM_QUANTILE_CASES = 10
ALL_CASES_SLICE = "all"


def _finite_non_negative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return number


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be non-empty canonical text")
    return value


@dataclass(frozen=True, slots=True)
class BudgetTolerance:
    """Product tolerance of one budget quantity."""

    quantity: str
    tolerance: float

    def __post_init__(self) -> None:
        _text(self.quantity, "quantity")
        object.__setattr__(
            self, "tolerance", _finite_non_negative(self.tolerance, "tolerance")
        )


@dataclass(frozen=True, slots=True)
class CaseBudgetEntry:
    """Budget components of one case, keyed ``quantity -> component -> value``.

    ``None`` marks a component that could not be measured.
    """

    case_id: str
    slices: tuple[str, ...]
    components: Mapping[str, Mapping[str, float | None]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        if type(self.slices) is not tuple or len(set(self.slices)) != len(self.slices):
            raise ValueError("slices must be a tuple of distinct slice ids")
        for slice_id in self.slices:
            _text(slice_id, "slice id")
        if ALL_CASES_SLICE not in self.slices:
            raise ValueError("every case belongs to the 'all' slice")
        frozen: dict[str, Mapping[str, float | None]] = {}
        for quantity, parts in self.components.items():
            _text(quantity, "quantity")
            if set(parts) != set(COMPONENTS):
                raise ValueError(f"{quantity} must name exactly {COMPONENTS}")
            frozen[quantity] = MappingProxyType(
                {
                    name: (
                        None
                        if parts[name] is None
                        else _finite_non_negative(parts[name], f"{quantity}.{name}")
                    )
                    for name in COMPONENTS
                }
            )
        object.__setattr__(self, "components", MappingProxyType(frozen))

    def total(self, quantity: str) -> float | None:
        """Sum the three components in fixed order, or ``None`` if any is missing."""

        parts = self.components.get(quantity)
        if parts is None:
            return None
        values: list[float] = []
        for name in COMPONENTS:
            value = parts[name]
            if value is None:
                return None
            values.append(value)
        surrogate, discretisation, projection = values
        return surrogate + discretisation + projection


def linear_quantile(values: tuple[float, ...], quantile: float) -> float:
    """Quantile with linear interpolation between order statistics."""

    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def slice_statistic(values: tuple[float, ...]) -> float:
    """p90 for slices with at least ten cases, the maximum otherwise."""

    if len(values) >= MINIMUM_QUANTILE_CASES:
        return linear_quantile(values, QUANTILE)
    if not values:
        raise ValueError("values must not be empty")
    return max(values)


@dataclass(frozen=True, slots=True)
class SliceBudgetResult:
    """Budget verdict of one quantity on one coverage slice."""

    slice_id: str
    quantity: str
    case_count: int
    statistic: float | None
    tolerance: float
    status: CheckStatus
    reason: str | None = None


def evaluate_budget(
    entries: tuple[CaseBudgetEntry, ...],
    tolerances: tuple[BudgetTolerance, ...],
    slice_ids: tuple[str, ...],
) -> tuple[SliceBudgetResult, ...]:
    """Evaluate every gated slice for every quantity, fail-closed."""

    if type(entries) is not tuple or any(
        type(entry) is not CaseBudgetEntry for entry in entries
    ):
        raise TypeError("entries must be a tuple of CaseBudgetEntry")
    if type(tolerances) is not tuple or not tolerances:
        raise ValueError("at least one tolerance is required")
    if len({tolerance.quantity for tolerance in tolerances}) != len(tolerances):
        raise ValueError("tolerances must name distinct quantities")
    if type(slice_ids) is not tuple or ALL_CASES_SLICE not in slice_ids:
        raise ValueError("slice_ids must include the 'all' slice")
    if len({entry.case_id for entry in entries}) != len(entries):
        raise ValueError("case ids must be unique")
    results: list[SliceBudgetResult] = []
    for slice_id in slice_ids:
        members = tuple(entry for entry in entries if slice_id in entry.slices)
        for tolerance in tolerances:
            if not members:
                results.append(
                    SliceBudgetResult(
                        slice_id,
                        tolerance.quantity,
                        0,
                        None,
                        tolerance.tolerance,
                        CheckStatus.NOT_RUN,
                        "slice holds no case",
                    )
                )
                continue
            totals = tuple(entry.total(tolerance.quantity) for entry in members)
            measured = tuple(total for total in totals if total is not None)
            if len(measured) != len(totals):
                results.append(
                    SliceBudgetResult(
                        slice_id,
                        tolerance.quantity,
                        len(members),
                        None,
                        tolerance.tolerance,
                        CheckStatus.NOT_RUN,
                        "a budget component is missing",
                    )
                )
                continue
            statistic = slice_statistic(measured)
            results.append(
                SliceBudgetResult(
                    slice_id,
                    tolerance.quantity,
                    len(members),
                    statistic,
                    tolerance.tolerance,
                    CheckStatus.PASS
                    if statistic <= tolerance.tolerance
                    else CheckStatus.FAIL,
                )
            )
    return tuple(results)


__all__ = (
    "ALL_CASES_SLICE",
    "COMPONENTS",
    "MINIMUM_QUANTILE_CASES",
    "QUANTILE",
    "BudgetTolerance",
    "CaseBudgetEntry",
    "SliceBudgetResult",
    "evaluate_budget",
    "linear_quantile",
    "slice_statistic",
)
