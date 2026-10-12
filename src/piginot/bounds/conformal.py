"""Split-conformal upper bounds and their exact coverage test (S23, ADR 0104).

For ``n`` exchangeable calibration scores and a miscoverage level ``alpha``,
the bound is the ``k``-th smallest score with ``k = ceil((n + 1)(1 - alpha))``.
A new exchangeable score lies at or below it with probability at least
``1 - alpha``; if ``k > n`` no finite bound exists and none is issued.

Conditional on the calibration set, the coverage is ``Beta(k, n + 1 - k)``
distributed (continuous scores), so the number of covered cases among ``m``
test cases is beta-binomial. :func:`acceptance_count` is the lower quantile
of that distribution that a coverage test compares against.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


class NoFiniteBound(ValueError):
    """``ceil((n + 1)(1 - alpha))`` exceeds the number of calibration scores."""


def _alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a real number")
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return value


def _count(value: int, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def conformal_rank(n: int, alpha: float) -> int:
    """The rank ``k`` of the bound among ``n`` sorted scores (1-based)."""

    n = _count(n, "n", minimum=1)
    a = _alpha(alpha)
    # (n + 1)(1 - alpha) in exact rational arithmetic where alpha is a short
    # decimal, so that e.g. n = 9, alpha = 0.1 gives k = 9 and not 10.
    product = (n + 1) * (1.0 - a)
    k = math.ceil(round(product, 12))
    if k > n:
        raise NoFiniteBound(
            f"no finite bound: ceil((n + 1)(1 - alpha)) = {k} exceeds n = {n}"
        )
    return max(k, 1)


@dataclass(frozen=True, slots=True)
class ConformalBound:
    """One calibrated upper bound."""

    alpha: float
    n: int
    k: int
    bound: float

    def covers(self, score: float) -> bool:
        return float(score) <= self.bound


def split_conformal_bound(scores: Sequence[float], alpha: float) -> ConformalBound:
    """The split-conformal upper bound of ``scores`` at level ``alpha``."""

    values = [float(score) for score in scores]
    if not values:
        raise ValueError("scores must not be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("scores must be finite")
    k = conformal_rank(len(values), alpha)
    return ConformalBound(float(alpha), len(values), k, sorted(values)[k - 1])


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_binomial_pmf(x: int, m: int, a: float, b: float) -> float:
    """``P(X = x)`` for ``X ~ BetaBinomial(m, a, b)``."""

    m = _count(m, "m", minimum=0)
    if isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= m:
        raise ValueError("x must be an integer in [0, m]")
    if not (a > 0.0 and b > 0.0):
        raise ValueError("a and b must be positive")
    log_choose = math.lgamma(m + 1) - math.lgamma(x + 1) - math.lgamma(m - x + 1)
    return math.exp(log_choose + _log_beta(x + a, m - x + b) - _log_beta(a, b))


def acceptance_count(n: int, alpha: float, m: int, level: float) -> int:
    """Smallest covered count ``c`` with ``P(X <= c) >= level``.

    ``X`` is the number of ``m`` exchangeable test cases covered by a
    split-conformal bound calibrated on ``n`` cases at ``alpha``. A valid
    procedure falls below ``c`` with probability below ``level``.
    """

    k = conformal_rank(n, alpha)
    m = _count(m, "m", minimum=1)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    a, b = float(k), float(n + 1 - k)
    cumulative = 0.0
    for x in range(m + 1):
        cumulative += beta_binomial_pmf(x, m, a, b)
        if cumulative >= level:
            return x
    return m  # pragma: no cover - the cumulative sum reaches 1 at x = m


__all__ = (
    "ConformalBound",
    "NoFiniteBound",
    "acceptance_count",
    "beta_binomial_pmf",
    "conformal_rank",
    "split_conformal_bound",
)
