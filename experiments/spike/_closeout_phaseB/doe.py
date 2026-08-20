"""Design of experiments in derived-ratio space, for the S1 closeout.

Iteration 8 sampled the raw parameter box uniformly for both families. The physics
is governed by dimensionless ratios, and a uniform box draw covers their tails
poorly: the 20-geometry centre-notched training set reached a net-section factor
of 1.56 while the 25-case confirmation draw reached 2.09. The five cases beyond
training coverage carried the reported failure, and the dogbone confirmation draw
happened to fall entirely inside its training range -- so the two families were
never tested under comparable extrapolation load.

This module samples so that strata in ratio space are filled deliberately, records
every rejected attempt with a reason, and declares an envelope that evaluation
cases are labelled against rather than silently mixed into one aggregate.

Conceptually this is the S9 coupled-domain DOE and the S14 envelope arriving
early, in the smallest form that removes the confound.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from experiments.spike.geometry import (
    SpikeMesh,
    _ranges_for,
    derived_ratios,
    generate_geometry,
)

# (primary, secondary) -- primary is the ratio that governs the concentration and
# is stratified most finely. For centre-notched, the measured correlation with
# stress error was +0.60 on notch_depth, which enters through the net section.
PRIMARY_RATIOS: Final[dict[str, tuple[str, str]]] = {
    "dogbone": ("width_ratio", "fillet_ratio"),
    "centre_notched": ("net_section_factor", "notch_sharpness"),
}
DEFAULT_STRATA: Final[tuple[int, int]] = (5, 2)
SUPPORT_PROBES: Final = 20_000
MAX_ATTEMPTS_PER_ACCEPT: Final = 400


@dataclass(frozen=True)
class DesignLedger:
    """Accepted geometries plus a full account of every rejected attempt."""

    family: str
    accepted: tuple[SpikeMesh, ...]
    rejected: tuple[Mapping[str, object], ...]
    attempted: int
    stratum_counts: tuple[int, ...]
    uncovered_strata: tuple[int, ...]
    strata: tuple[int, int]
    support: Mapping[str, tuple[float, float]]

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "accepted_count": len(self.accepted),
            "accepted_geometry_ids": [case.geometry_id for case in self.accepted],
            "rejected_count": len(self.rejected),
            "rejected": [dict(entry) for entry in self.rejected],
            "attempted": self.attempted,
            "stratum_counts": list(self.stratum_counts),
            "uncovered_strata": list(self.uncovered_strata),
            "strata": list(self.strata),
            "support": {name: list(bounds) for name, bounds in self.support.items()},
            "ledger_balances": len(self.accepted) + len(self.rejected) == self.attempted,
        }


def _sample_parameters(family: str, rng: np.random.Generator) -> dict[str, float]:
    return {
        name: float(rng.uniform(lower, upper))
        for name, (lower, upper) in _ranges_for(family).items()
    }


def ratio_support(
    family: str,
    *,
    seed: int,
    probes: int = SUPPORT_PROBES,
) -> dict[str, tuple[float, float]]:
    """Estimate the achievable range of each derived ratio inside the box.

    The box is rectangular; its image in ratio space is not. Strata must be laid
    out over what is actually reachable, not over a guessed interval.
    """
    rng = np.random.default_rng(seed)
    observed: dict[str, list[float]] = {}
    for _ in range(probes):
        parameters = _sample_parameters(family, rng)
        try:
            ratios = derived_ratios(family, parameters)
        except ValueError:
            continue
        for name, value in ratios.items():
            observed.setdefault(name, []).append(value)
    if not observed:
        raise RuntimeError(f"no admissible parameters found for {family}")
    return {
        name: (float(np.min(values)), float(np.max(values)))
        for name, values in observed.items()
    }


def _stratum_index(value: float, bounds: tuple[float, float], count: int) -> int:
    lower, upper = bounds
    if upper <= lower:
        return 0
    position = (value - lower) / (upper - lower)
    return min(count - 1, max(0, int(position * count)))


def stratified_design(
    family: str,
    *,
    count: int,
    seed: int,
    strata: tuple[int, int] = DEFAULT_STRATA,
    nx: int = 32,
    ny: int = 10,
    support: Mapping[str, tuple[float, float]] | None = None,
) -> DesignLedger:
    """Fill strata in derived-ratio space, recording every rejected attempt."""
    if count < 1:
        raise ValueError("count must be positive")
    if family not in PRIMARY_RATIOS:
        raise ValueError(f"unknown family: {family}")
    primary, secondary = PRIMARY_RATIOS[family]
    resolved = dict(support) if support is not None else ratio_support(family, seed=seed)
    n_strata = strata[0] * strata[1]
    quota = math.ceil(count / n_strata)

    rng = np.random.default_rng(seed)
    counts = [0] * n_strata
    accepted: list[SpikeMesh] = []
    rejected: list[Mapping[str, object]] = []
    attempted = 0
    budget = count * MAX_ATTEMPTS_PER_ACCEPT

    while len(accepted) < count and attempted < budget:
        attempted += 1
        parameters = _sample_parameters(family, rng)
        try:
            ratios = derived_ratios(family, parameters)
        except ValueError as error:
            rejected.append(
                {"reason": f"inadmissible geometry: {error}", "parameters": parameters}
            )
            continue
        cell = (
            _stratum_index(ratios[primary], resolved[primary], strata[0]) * strata[1]
            + _stratum_index(ratios[secondary], resolved[secondary], strata[1])
        )
        if counts[cell] >= quota:
            rejected.append(
                {
                    "reason": f"stratum {cell} already at quota {quota}",
                    "parameters": parameters,
                    "stratum": cell,
                }
            )
            continue
        case_seed = int(rng.integers(0, 2**62))
        try:
            case = generate_geometry(family, parameters, case_seed, nx=nx, ny=ny)
        except ValueError as error:
            rejected.append(
                {"reason": f"mesh generation failed: {error}", "parameters": parameters}
            )
            continue
        counts[cell] += 1
        accepted.append(case)

    if len(accepted) < count:
        raise RuntimeError(
            f"{family} design filled only {len(accepted)} of {count} within budget"
        )
    return DesignLedger(
        family=family,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        attempted=attempted,
        stratum_counts=tuple(counts),
        uncovered_strata=tuple(i for i, n in enumerate(counts) if n == 0),
        strata=strata,
        support={name: tuple(bounds) for name, bounds in resolved.items()},
    )


def declare_envelope(
    cases: Sequence[SpikeMesh],
) -> dict[str, tuple[float, float]]:
    """Declare the derived-ratio envelope a training set actually covers."""
    if not cases:
        raise ValueError("cannot declare an envelope from no cases")
    families = {case.family for case in cases}
    if len(families) != 1:
        raise ValueError("declare an envelope per family, not across families")
    collected: dict[str, list[float]] = {}
    for case in cases:
        for name, value in derived_ratios(case.family, case.parameters).items():
            collected.setdefault(name, []).append(value)
    return {
        name: (float(min(values)), float(max(values)))
        for name, values in collected.items()
    }


def label_envelope(
    family: str,
    parameters: Mapping[str, float],
    envelope: Mapping[str, tuple[float, float]],
    *,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    """Label one case in- or out-of-envelope, naming every ratio that violates."""
    ratios = derived_ratios(family, parameters)
    violations: dict[str, object] = {}
    for name, value in ratios.items():
        bounds = envelope.get(name)
        if bounds is None:
            continue
        lower, upper = bounds
        if value < lower - tolerance:
            violations[name] = {"value": value, "bound": lower, "side": "below"}
        elif value > upper + tolerance:
            violations[name] = {"value": value, "bound": upper, "side": "above"}
    return {
        "status": "out_of_envelope" if violations else "in_envelope",
        "violations": violations,
        "derived_ratios": ratios,
    }


def coverage_report(
    cases: Sequence[SpikeMesh],
    *,
    strata: int = 5,
) -> dict[str, object]:
    """Report achieved range and stratum occupancy per family, per ratio."""
    report: dict[str, object] = {}
    for family in sorted({case.family for case in cases}):
        family_cases = [case for case in cases if case.family == family]
        per_ratio: dict[str, object] = {}
        collected: dict[str, list[float]] = {}
        for case in family_cases:
            for name, value in derived_ratios(family, case.parameters).items():
                collected.setdefault(name, []).append(value)
        for name, values in collected.items():
            bounds = (min(values), max(values))
            counts = [0] * strata
            for value in values:
                counts[_stratum_index(value, bounds, strata)] += 1
            per_ratio[name] = {
                "min": bounds[0],
                "max": bounds[1],
                "count": len(values),
                "stratum_counts": counts,
                "empty_strata": sum(1 for n in counts if n == 0),
            }
        report[family] = per_ratio
    return report
