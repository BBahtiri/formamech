"""Notched post-mortem, owner-machine part: D7 and D8 (S20, ADR 0101).

D7 needs the spent HOLDOUT coordinates (the private S9 reveal); D8 needs the
notched S10 run root. Both run on the decision owner's machine::

    python -m campaigns.notched_v1.postmortem_owner d7 --old-reveal REVEAL
    python -m campaigns.notched_v1.postmortem_owner d8 --s10-run-root RUN
    python -m campaigns.notched_v1.postmortem_owner d8 --recompute-reference
    python -m campaigns.notched_v1.postmortem_owner verdicts --write

Statistics fixed here before any result was computed:

- D7: Mahalanobis distance to the TRAIN mean under the TRAIN covariance, on the
  primitive axes that vary across TRAIN (z-scoring does not change it);
  Spearman correlation with the surrogate term, pooled over DEVELOPMENT and
  HOLDOUT. Hypothesis (a) is SUPPORTED if some stress quantity has both a
  D4 change >= 20% and a pooled Spearman >= 0.4.
- D8: the 8 DEVELOPMENT cases with the largest recorded stress-field
  discretisation term are re-solved with the S10 recipe at twice the columns
  and rows; the observed reference error is the relative difference between
  the S10 reference and the refined solution on the reference vertices. With
  ``--recompute-reference`` (no S10 run root) the reference is re-solved with
  the S10 recipe and must first reproduce the recorded S14 coarse-level terms
  of every case within a relative 1e-6. The
  recorded bound is "not loose by more than 2x" if the median of
  bound / observed is <= 2 for both stress quantities.
- D9 is not run: by the protocol it is needed only when D6 is below one half,
  and D6 is 0.97 (field) and 1.20 (peak).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from campaigns.notched_v1 import postmortem as public

CAMPAIGN = public.CAMPAIGN
D7_PATH = CAMPAIGN / "POSTMORTEM-D7.json"
D8_PATH = CAMPAIGN / "POSTMORTEM-D8.json"
VERDICTS_PATH = CAMPAIGN / "POSTMORTEM-VERDICTS.json"
D8_CASES = 8
D8_SCALE = 2.0
STRESS = public.STRESS


def _records(root: Path) -> dict[str, list[Mapping[str, Any]]]:
    return {
        split: json.loads((root / CAMPAIGN / name).read_text("utf-8"))["cases"]
        for split, name in public.SPLITS.items()
    }


def mahalanobis(
    train: np.ndarray, points: np.ndarray, axes: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Distances of ``points`` to the TRAIN mean, on the axes TRAIN varies."""

    spread = train.std(axis=0)
    # fixed axes can carry floating-point noise (std ~ 1e-16); treat as constant
    keep = spread > 1e-9 * np.maximum(1.0, np.abs(train.mean(axis=0)))
    mean = train[:, keep].mean(axis=0)
    covariance = np.cov(train[:, keep], rowvar=False)
    inverse = np.linalg.inv(covariance)
    centred = points[:, keep] - mean
    distance = np.sqrt(np.einsum("ij,jk,ik->i", centred, inverse, centred))
    return distance, [axis for axis, k in zip(axes, keep, strict=True) if k]


def envelope_distance(
    manifest: Mapping[str, Any],
    old_holdout: Sequence[Mapping[str, float]],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """D7: distance to TRAIN against the surrogate term."""

    axes = list(manifest["primitive_axes"])
    by_id = {case["case_id"]: case for case in manifest["cases"]}
    train = np.array(
        [
            [float(case["primitive_coordinates"][a]) for a in axes]
            for case in manifest["cases"]
            if case["role"] == "TRAIN" and case["status"] == "ACTIVE"
        ]
    )
    coordinates: dict[str, list[float]] = {}
    for case in records["DEVELOPMENT"]:
        source = by_id[case["case_id"]]["primitive_coordinates"]
        coordinates[case["case_id"]] = [float(source[a]) for a in axes]
    for index, values in enumerate(old_holdout):
        coordinates[f"holdout-{index:02d}"] = [float(values[a]) for a in axes]
    distances: dict[str, dict[str, float]] = {}
    for split, cases in records.items():
        points = np.array([coordinates[case["case_id"]] for case in cases])
        values, used = mahalanobis(train, points, axes)
        distances[split] = {
            case["case_id"]: float(d) for case, d in zip(cases, values, strict=True)
        }
    result: dict[str, Any] = {"axes": used, "train_cases": int(len(train))}
    for quantity in public.QUANTITIES:
        pooled_d, pooled_s = [], []
        per_split = {}
        for split, cases in records.items():
            d = [distances[split][case["case_id"]] for case in cases]
            s = [float(case["surrogate"][quantity]) for case in cases]
            per_split[split] = float(spearmanr(d, s).statistic)
            pooled_d += d
            pooled_s += s
        result[quantity] = {
            "spearman_pooled": float(spearmanr(pooled_d, pooled_s).statistic),
            "spearman": per_split,
        }
    result["median_distance"] = {
        split: statistics.median(values.values()) for split, values in distances.items()
    }
    result["distances"] = distances
    return result


def refinement_cases(development: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """The D8 cases: largest recorded stress-field discretisation term."""

    ranked = sorted(
        development,
        key=lambda c: (
            -float(c["components"]["p_relative_l2"]["discretisation"]),
            c["case_id"],
        ),
    )
    return tuple(case["case_id"] for case in ranked[:D8_CASES])


def compare_refinement(
    observed: Mapping[str, Mapping[str, float]],
    development: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """D8: recorded discretisation bound against the observed reference error."""

    by_id = {case["case_id"]: case for case in development}
    result: dict[str, Any] = {}
    for quantity in public.QUANTITIES:
        ratios = {}
        for case_id, values in observed.items():
            bound = float(by_id[case_id]["components"][quantity]["discretisation"])
            seen = float(values[quantity])
            ratios[case_id] = bound / seen if seen > 0 else float("inf")
        result[quantity] = {
            "bound_over_observed": ratios,
            "median_bound_over_observed": statistics.median(ratios.values()),
            "observed": {
                case_id: float(v[quantity]) for case_id, v in observed.items()
            },
        }
    result["bound_within_2x"] = all(
        result[q]["median_bound_over_observed"] <= 2.0 for q in STRESS
    )
    return result


def run_refinement(s10_run_root: Path, case_ids: Sequence[str]) -> dict[str, Any]:
    """Re-solve the D8 cases at twice the S10 resolution (owner machine)."""

    from campaigns.notched_v1 import s14
    from piginot.doe.roles import EvidenceRole

    observed: dict[str, dict[str, float]] = {}
    seconds: dict[str, float] = {}
    for solved in s14.public_solved(s10_run_root, case_ids, EvidenceRole.DEVELOPMENT):
        fine = s14.level_request(solved.case, solved.request, D8_SCALE)
        fine_result, elapsed = s14._solve(fine)
        u, p, peak, _ = s14.level_differences(
            solved.request, solved.result, fine, fine_result
        )
        observed[solved.case.case_id] = {
            "u_relative_l2": u,
            "p_relative_l2": p,
            "absolute_peak_error": peak,
        }
        seconds[solved.case.case_id] = elapsed
        print(
            f"D8 {len(observed)}/{len(case_ids)} {solved.case.case_id} {elapsed:.0f} s"
        )
    return {"scale": D8_SCALE, "observed": observed, "solve_seconds": seconds}


REPRODUCTION_TOLERANCE = 1e-6  # relative, recomputed vs recorded S14 coarse terms


def _recompute_case(case_id: str) -> dict[str, Any]:
    """Recompute the S10 reference, reproduce S14's coarse term, refine 2x."""

    from campaigns.dogbone_v1.s14_rehearsal import S14Case
    from campaigns.notched_v1 import pilot, s14
    from campaigns.notched_v1.s13 import published_cases

    case = S14Case.from_published(published_cases([case_id])[0])
    reference = s14.s10_request(case, pilot.execution_profile())
    reference_result, reference_seconds = s14._solve(reference)
    coarse = s14.level_request(case, reference, 0.5)
    coarse_result, _ = s14._solve(coarse)
    terms = s14.reference_terms(
        case_id, reference, reference_result, coarse, coarse_result
    )
    fine = s14.level_request(case, reference, D8_SCALE)
    fine_result, fine_seconds = s14._solve(fine)
    u, p, peak, _ = s14.level_differences(
        reference, reference_result, fine, fine_result
    )
    return {
        "case_id": case_id,
        "observed": {
            "u_relative_l2": u,
            "p_relative_l2": p,
            "absolute_peak_error": peak,
        },
        "coarse_terms": {
            "coarse_u_difference": terms.coarse_u_difference,
            "coarse_p_difference": terms.coarse_p_difference,
            "coarse_peak_difference": terms.coarse_peak_difference,
        },
        "seconds": {"reference": reference_seconds, "fine": fine_seconds},
    }


def reproduction(
    recomputed: Mapping[str, Mapping[str, float]],
    development: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recomputed coarse-level terms against the recorded S14 terms."""

    by_id = {case["case_id"]: case for case in development}
    worst = 0.0
    for case_id, terms in recomputed.items():
        for name, value in terms.items():
            recorded = float(by_id[case_id]["terms"][name])
            worst = max(worst, abs(value - recorded) / max(abs(recorded), 1e-300))
    return {
        "max_relative_difference": worst,
        "tolerance": REPRODUCTION_TOLERANCE,
        "reproduced": worst <= REPRODUCTION_TOLERANCE,
    }


def run_recomputed(case_ids: Sequence[str], workers: int) -> dict[str, Any]:
    """D8 without an S10 run root: the S10 recipe is re-solved per case."""

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        results = []
        for result in pool.map(_recompute_case, case_ids):
            results.append(result)
            print(f"D8 {len(results)}/{len(case_ids)} {result['case_id']}", flush=True)
    return {
        "scale": D8_SCALE,
        "reference": "recomputed with the S10 recipe (no S10 run root)",
        "observed": {r["case_id"]: r["observed"] for r in results},
        "coarse_terms": {r["case_id"]: r["coarse_terms"] for r in results},
        "solve_seconds": {r["case_id"]: r["seconds"] for r in results},
    }


def verdicts(root: Path) -> dict[str, Any]:
    """Final verdicts: the public ones, completed with D7 and D8 where present."""

    pub = public.analyse(root)
    hypotheses = json.loads(json.dumps(pub["hypotheses"]))
    d4 = pub["diagnostics"]["D4"]
    d7_path, d8_path = root / D7_PATH, root / D8_PATH
    if d7_path.is_file():
        d7 = json.loads(d7_path.read_text("utf-8"))
        supported = any(
            d4[q]["relative_change"] >= 0.20 and d7[q]["spearman_pooled"] >= 0.4
            for q in STRESS
        )
        hypotheses["a"]["verdict"] = "SUPPORTED" if supported else "NOT_SUPPORTED"
        hypotheses["a"]["evidence"]["D7_spearman_pooled"] = {
            q: d7[q]["spearman_pooled"] for q in STRESS
        }
    if d8_path.is_file():
        d8 = json.loads(d8_path.read_text("utf-8"))
        hypotheses["e"]["evidence"]["D8_median_bound_over_observed"] = {
            q: d8["comparison"][q]["median_bound_over_observed"] for q in STRESS
        }
        hypotheses["e"]["evidence"]["D8_bound_within_2x"] = d8["comparison"][
            "bound_within_2x"
        ]
    hypotheses["b"]["evidence"]["D9"] = "not run: needed only if D6 < 0.5"
    return {
        "schema_version": "1.0",
        "protocol_id": pub["protocol_id"],
        "protocol_sha256_lf": pub["protocol_sha256_lf"],
        "hypotheses": hypotheses,
        "inputs": {
            "D7": d7_path.is_file(),
            "D8": d8_path.is_file(),
            "D9": "not run",
        },
    }


def _write(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", "utf-8")


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Notched post-mortem (owner part).")
    commands = parser.add_subparsers(dest="command", required=True)
    d7 = commands.add_parser("d7")
    d7.add_argument("--old-reveal", type=Path, required=True)
    d8 = commands.add_parser("d8")
    source = d8.add_mutually_exclusive_group(required=True)
    source.add_argument("--s10-run-root", type=Path)
    source.add_argument("--recompute-reference", action="store_true")
    d8.add_argument("--workers", type=int, default=4)
    final = commands.add_parser("verdicts")
    final.add_argument("--write", action="store_true")
    options = parser.parse_args(arguments)
    root = public.REPOSITORY
    records = _records(root)
    if options.command == "d7":
        manifest = json.loads(
            (root / CAMPAIGN / "public-case-manifest.json").read_text("utf-8")
        )
        reveal = json.loads(options.old_reveal.read_text("utf-8"))
        document = envelope_distance(manifest, reveal["holdout_coordinates"], records)
        _write(root / D7_PATH, document)
        print({q: round(document[q]["spearman_pooled"], 3) for q in public.QUANTITIES})
    elif options.command == "d8":
        ids = refinement_cases(records["DEVELOPMENT"])
        if options.recompute_reference:
            run = run_recomputed(ids, options.workers)
            check = reproduction(run["coarse_terms"], records["DEVELOPMENT"])
            if not check["reproduced"]:
                raise RuntimeError(
                    f"recomputed reference does not reproduce S14: {check}"
                )
            run["reproduction"] = check
        else:
            run = run_refinement(options.s10_run_root, ids)
        document = {
            **run,
            "case_ids": list(ids),
            "comparison": compare_refinement(run["observed"], records["DEVELOPMENT"]),
        }
        _write(root / D8_PATH, document)
        print(document["comparison"]["bound_within_2x"])
    else:
        document = verdicts(root)
        if options.write:
            _write(root / VERDICTS_PATH, document)
        for key, entry in document["hypotheses"].items():
            print(f"({key}) {entry['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
