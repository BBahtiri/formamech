"""S23 split-conformal bounds and their exact coverage test (ADR 0104)."""

from __future__ import annotations

import math
import random

import pytest

from piginot.bounds import (
    NoFiniteBound,
    acceptance_count,
    beta_binomial_pmf,
    conformal_rank,
    split_conformal_bound,
)


@pytest.mark.parametrize(
    ("n", "alpha", "k"),
    [(64, 0.1, 59), (9, 0.1, 9), (19, 0.05, 19), (64, 0.2, 52), (1, 0.5, 1)],
)
def test_the_rank_is_ceil_n_plus_one_times_one_minus_alpha(
    n: int, alpha: float, k: int
) -> None:
    assert conformal_rank(n, alpha) == k


def test_too_few_scores_give_no_finite_bound() -> None:
    with pytest.raises(NoFiniteBound):
        conformal_rank(8, 0.1)
    with pytest.raises(NoFiniteBound):
        split_conformal_bound([0.1] * 8, 0.1)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, math.nan, True, "0.1"])
def test_bad_alphas_are_refused(alpha: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        conformal_rank(64, alpha)  # type: ignore[arg-type]


@pytest.mark.parametrize("n", [0, -1, 2.0, True])
def test_bad_counts_are_refused(n: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        conformal_rank(n, 0.1)  # type: ignore[arg-type]


def test_the_bound_is_the_kth_smallest_score() -> None:
    scores = [float(v) for v in range(64, 0, -1)]
    bound = split_conformal_bound(scores, 0.1)
    assert (bound.n, bound.k, bound.bound, bound.alpha) == (64, 59, 59.0, 0.1)
    assert bound.covers(59.0) and not bound.covers(59.000001)


@pytest.mark.parametrize("scores", [[], [0.1, math.inf], [math.nan] * 20])
def test_bad_scores_are_refused(scores: list[float]) -> None:
    with pytest.raises(ValueError):
        split_conformal_bound(scores, 0.1)


@pytest.mark.parametrize(
    ("n", "alpha", "m", "level", "count"),
    [
        # the implementation plan's table (S23, 5% quantile)
        (64, 0.1, 32, 0.05, 25),
        (64, 0.1, 64, 0.05, 52),
        (64, 0.1, 128, 0.05, 106),
        (64, 0.2, 64, 0.05, 43),
        # ADR 0104 decision 4: Bonferroni over eight gated counts
        (64, 0.1, 64, 0.05 / 8, 48),
    ],
)
def test_acceptance_counts(
    n: int, alpha: float, m: int, level: float, count: int
) -> None:
    assert acceptance_count(n, alpha, m, level) == count


def test_the_beta_binomial_sums_to_one() -> None:
    total = sum(beta_binomial_pmf(x, 64, 59.0, 6.0) for x in range(65))
    assert total == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    ("x", "m", "a", "b"), [(-1, 4, 1.0, 1.0), (5, 4, 1.0, 1.0), (1, 4, 0.0, 1.0)]
)
def test_bad_beta_binomial_arguments(x: int, m: int, a: float, b: float) -> None:
    with pytest.raises(ValueError):
        beta_binomial_pmf(x, m, a, b)


@pytest.mark.parametrize("level", [0.0, 1.0, math.nan])
def test_bad_levels_are_refused(level: float) -> None:
    with pytest.raises(ValueError):
        acceptance_count(64, 0.1, 64, level)


def test_a_valid_procedure_rarely_fails_the_test() -> None:
    """Exchangeable scores fall below the 5% acceptance count about 5% of runs."""

    rng = random.Random(20261012)
    runs, failures = 2000, 0
    threshold = acceptance_count(64, 0.1, 64, 0.05)
    for _ in range(runs):
        bound = split_conformal_bound([rng.random() for _ in range(64)], 0.1)
        covered = sum(bound.covers(rng.random()) for _ in range(64))
        failures += covered < threshold
    assert failures / runs < 0.065
