"""Notched HOLDOUT post-mortem, public-data part (S20, ADR 0101).

Implements the diagnostics D1–D6 of
``artifacts/campaigns/notched-v1/POSTMORTEM-PROTOCOL.json`` on the public S13
and S14 records, and evaluates the hypothesis criteria that depend only on
them. Diagnostics D7–D9 need the decision owner's machine; criteria that need
them are ``NOT_RUN`` here and never count as supported.

Where the protocol names a statistic loosely, this module fixes it before any
result was computed: D6 compares *medians* (TRAIN-screen error against the
DEVELOPMENT surrogate term); p90 values are reported beside them.

Run from the repository root::

    python -m campaigns.notched_v1.postmortem --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from piginot.verification.budget import slice_statistic

REPOSITORY = Path(__file__).resolve().parents[2]
CAMPAIGN = Path("artifacts/campaigns/notched-v1")
PROTOCOL_PATH = CAMPAIGN / "POSTMORTEM-PROTOCOL.json"
OUTPUT_PATH = CAMPAIGN / "POSTMORTEM-PUBLIC.json"
SPLITS = {"DEVELOPMENT": "S14-DEVELOPMENT.json", "HOLDOUT": "S14-HOLDOUT.json"}
TRAIN_SCREEN = "S13-TRAIN-SCREEN.json"
STRESS = ("p_relative_l2", "absolute_peak_error")
QUANTITIES = (*STRESS, "u_relative_l2")
COMPONENTS = ("surrogate", "discretisation", "projection")
SLICE_AXES = ("notch_radius", "width", "notch_depth", "nominal_axial_strain")


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def case_terms(case: Mapping[str, Any], quantity: str) -> dict[str, float]:
    """The three budget components of one case and quantity."""

    return {
        "surrogate": float(case["surrogate"][quantity]),
        "discretisation": float(case["components"][quantity]["discretisation"]),
        "projection": float(case["components"][quantity]["projection"]),
    }


def _total(terms: Mapping[str, float]) -> float:
    """Keep the protocol's Python 3.11 addition order on newer runtimes."""

    return terms["surrogate"] + terms["discretisation"] + terms["projection"]


def _statistic(values: Sequence[float]) -> float:
    return slice_statistic(tuple(values))


def _slices(cases: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        for name in case["slices"]:
            grouped.setdefault(name, []).append(case)
    return dict(sorted(grouped.items()))


def decomposition(
    cases: Sequence[Mapping[str, Any]], tolerances: Mapping[str, float]
) -> dict[str, Any]:
    """D1: component shares and counterfactual statistics per slice."""

    result: dict[str, Any] = {}
    for quantity in QUANTITIES:
        terms = [case_terms(case, quantity) for case in cases]
        shares = {
            component: statistics.median(
                term[component] / total if (total := _total(term)) > 0 else 0.0
                for term in terms
            )
            for component in COMPONENTS
        }
        per_slice = {}
        for name, members in _slices(cases).items():
            member_terms = [case_terms(case, quantity) for case in members]
            per_slice[name] = {
                "cases": len(members),
                "total": _statistic([_total(t) for t in member_terms]),
                "without_discretisation": _statistic(
                    [t["surrogate"] + t["projection"] for t in member_terms]
                ),
                "without_projection": _statistic(
                    [t["surrogate"] + t["discretisation"] for t in member_terms]
                ),
                "surrogate_only": _statistic([t["surrogate"] for t in member_terms]),
                "discretisation_only": _statistic(
                    [t["discretisation"] for t in member_terms]
                ),
            }
        failing = {
            variant: sorted(
                name
                for name, row in per_slice.items()
                if row[variant] > tolerances[quantity]
            )
            for variant in ("total", "without_discretisation", "surrogate_only")
        }
        result[quantity] = {
            "median_share": shares,
            "slices": per_slice,
            "failing_slices": failing,
        }
    return result


def reference_dominated(
    cases: Sequence[Mapping[str, Any]], tolerances: Mapping[str, float]
) -> dict[str, Any]:
    """D2: cases whose discretisation term alone uses half or all of the budget."""

    result = {}
    for quantity in QUANTITIES:
        terms = [case_terms(case, quantity)["discretisation"] for case in cases]
        tolerance = tolerances[quantity]
        result[quantity] = {
            "cases": len(terms),
            "at_least_half_tolerance": sum(term >= 0.5 * tolerance for term in terms),
            "at_least_tolerance": sum(term >= tolerance for term in terms),
            "fraction_at_least_tolerance": sum(term >= tolerance for term in terms)
            / len(terms),
        }
    return result


def slice_concentration(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """D3: median surrogate term per slice; sharpest over bluntest notch."""

    result: dict[str, Any] = {}
    grouped = _slices(cases)
    for quantity in QUANTITIES:
        medians = {
            name: statistics.median(
                case_terms(c, quantity)["surrogate"] for c in members
            )
            for name, members in grouped.items()
            if name.split(":")[0] in SLICE_AXES
        }
        sharp, blunt = medians.get("notch_radius:0"), medians.get("notch_radius:3")
        result[quantity] = {
            "median_surrogate_by_slice": medians,
            "sharpest_over_bluntest": (sharp / blunt) if sharp and blunt else None,
        }
    return result


def development_versus_holdout(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """D4: change of the surrogate-only all-slice statistic, DEVELOPMENT to HOLDOUT."""

    result = {}
    for quantity in QUANTITIES:
        values = {
            name: _statistic([case_terms(c, quantity)["surrogate"] for c in cases])
            for name, cases in splits.items()
        }
        result[quantity] = {
            **values,
            "relative_change": values["HOLDOUT"] / values["DEVELOPMENT"] - 1.0,
        }
    return result


def metric_fragility(
    cases: Sequence[Mapping[str, Any]], tolerances: Mapping[str, float]
) -> dict[str, Any]:
    """D5: all-slice statistic of the total without the 3 largest cases."""

    result = {}
    for quantity in QUANTITIES:
        totals = sorted(_total(case_terms(c, quantity)) for c in cases)
        trimmed = _statistic(totals[:-3])
        result[quantity] = {
            "all": _statistic(totals),
            "without_three_largest": trimmed,
            "passes_without_three_largest": trimmed <= tolerances[quantity],
        }
    return result


def train_comparison(
    screen: Mapping[str, Any], development: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """D6: TRAIN-screen errors against DEVELOPMENT surrogate terms (medians)."""

    train_cases = screen["screen_diagnostics"]["cases"]
    train = {
        "p_relative_l2": [float(c["p_relative_l2"]) for c in train_cases],
        "absolute_peak_error": [
            abs(float(c["signed_peak_error"])) for c in train_cases
        ],
        "u_relative_l2": [float(c["u_relative_l2"]) for c in train_cases],
    }
    result = {}
    for quantity in QUANTITIES:
        surrogate = [case_terms(c, quantity)["surrogate"] for c in development]
        train_median = statistics.median(train[quantity])
        development_median = statistics.median(surrogate)
        result[quantity] = {
            "train_cases": len(train[quantity]),
            "train_median": train_median,
            "train_p90": _statistic(train[quantity]),
            "development_surrogate_median": development_median,
            "development_surrogate_p90": _statistic(surrogate),
            "train_over_development_median": train_median / development_median,
        }
    return result


def _verdict(primary: bool | None, secondary: bool | None) -> str:
    """Conjunction where ``None`` means the evidence needs the owner's machine."""

    if primary is False or secondary is False:
        return "NOT_SUPPORTED"
    if primary is None or secondary is None:
        return "NOT_RUN"
    return "SUPPORTED"


def hypotheses(
    diagnostics: Mapping[str, Any], tol: Mapping[str, float]
) -> dict[str, Any]:
    d1, d2 = diagnostics["D1"], diagnostics["D2"]
    d3, d4 = diagnostics["D3"], diagnostics["D4"]
    d5, d6 = diagnostics["D5"], diagnostics["D6"]

    a_primary = any(d4[q]["relative_change"] >= 0.20 for q in STRESS)
    ratios = [d3["DEVELOPMENT"][q]["sharpest_over_bluntest"] for q in STRESS]
    b_primary = any(r is not None and r >= 1.5 for r in ratios)
    b_train = any(d6[q]["train_over_development_median"] >= 0.5 for q in STRESS)
    c_supported = all(d6[q]["train_over_development_median"] >= 0.8 for q in STRESS)
    d_supported = all(
        d5[split][q]["passes_without_three_largest"] for split in SPLITS for q in STRESS
    )
    e_counterfactual = any(
        all(
            d1[split][q]["slices"]["all"]["without_discretisation"] <= tol[q]
            for q in STRESS
        )
        for split in SPLITS
    )
    e_dominated = any(
        d2[split][q]["fraction_at_least_tolerance"] >= 0.10
        for split in SPLITS
        for q in STRESS
    )
    return {
        "a": {
            "verdict": _verdict(a_primary, None),
            "evidence": {
                "D4_relative_change": {q: d4[q]["relative_change"] for q in STRESS},
                "D7": "owner machine",
            },
        },
        "b": {
            "verdict": _verdict(b_primary, True if b_train else None),
            "evidence": {
                "D3_sharpest_over_bluntest_development": dict(
                    zip(STRESS, ratios, strict=True)
                ),
                "D6_train_over_development_median": {
                    q: d6[q]["train_over_development_median"] for q in STRESS
                },
                "D9": "owner machine (needed only if D6 is below one half)",
            },
        },
        "c": {
            "verdict": "SUPPORTED" if c_supported else "NOT_SUPPORTED",
            "evidence": {q: d6[q]["train_over_development_median"] for q in STRESS},
        },
        "d": {
            "verdict": "SUPPORTED" if d_supported else "NOT_SUPPORTED",
            "evidence": {
                split: {q: d5[split][q]["without_three_largest"] for q in STRESS}
                for split in SPLITS
            },
        },
        "e": {
            "verdict": "SUPPORTED"
            if (e_counterfactual or e_dominated)
            else "NOT_SUPPORTED",
            "evidence": {
                "D1_without_discretisation_all": {
                    split: {
                        q: d1[split][q]["slices"]["all"]["without_discretisation"]
                        for q in STRESS
                    }
                    for split in SPLITS
                },
                "D2_fraction_discretisation_at_least_tolerance": {
                    split: {
                        q: d2[split][q]["fraction_at_least_tolerance"] for q in STRESS
                    }
                    for split in SPLITS
                },
                "D8": "owner machine (strengthens or weakens)",
            },
        },
    }


def analyse(root: Path) -> dict[str, Any]:
    protocol = json.loads((root / PROTOCOL_PATH).read_text("utf-8"))
    tolerances = protocol["tolerances"]
    for path, digest in protocol["inputs"]["public_sha256_lf"].items():
        if sha256_lf(root / path) != digest:
            raise ValueError(f"{path} differs from the protocol's digest")
    splits = {
        name: json.loads((root / CAMPAIGN / file).read_text("utf-8"))["cases"]
        for name, file in SPLITS.items()
    }
    screen = json.loads((root / CAMPAIGN / TRAIN_SCREEN).read_text("utf-8"))
    diagnostics = {
        "D1": {n: decomposition(c, tolerances) for n, c in splits.items()},
        "D2": {n: reference_dominated(c, tolerances) for n, c in splits.items()},
        "D3": {n: slice_concentration(c) for n, c in splits.items()},
        "D4": development_versus_holdout(splits),
        "D5": {n: metric_fragility(c, tolerances) for n, c in splits.items()},
        "D6": train_comparison(screen, splits["DEVELOPMENT"]),
    }
    return {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256_lf": sha256_lf(root / PROTOCOL_PATH),
        "evidence_use": protocol["evidence_use"],
        "statistic_clarification": "D6 compares medians; p90 reported beside them",
        "diagnostics": diagnostics,
        "hypotheses": hypotheses(diagnostics, tolerances),
        "not_run_here": ["D7", "D8", "D9"],
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Notched post-mortem (public part).")
    parser.add_argument("--write", action="store_true", help=f"write {OUTPUT_PATH}")
    options = parser.parse_args(arguments)
    document = analyse(REPOSITORY)
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if options.write:
        (REPOSITORY / OUTPUT_PATH).write_text(text, encoding="utf-8")
    for key, entry in document["hypotheses"].items():
        print(f"({key}) {entry['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
