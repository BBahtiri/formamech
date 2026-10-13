"""S23 H1 reveal: score the fresh H1 set once and test the bounds (ADR 0104).

Runs on the decision owner's machine, which holds the private fresh-holdout
reveal of S20 (ADR 0101), after the protocol and both calibrations are
committed::

    python -m campaigns.s23_reveal --campaign dogbone-v1 \\
        --reveal <private dir>/dogbone-v1/fresh-holdouts-v1.json \\
        --run-root <outside the repository>/s23-h1 --workers 3

Order of operations, as preregistered in ``artifacts/bounds/S23-PROTOCOL.json``:

1. Preconditions: accepted protocol; calibration sidecar and evidence
   committed and unchanged; no earlier reveal or H1 burn of the family.
2. A consumption record ``S23-<campaign>-H1-CONSUMED.json`` is written before
   the private coordinates are read.
3. The H1 coordinates are verified against the public commitment in
   ``FRESH-HOLDOUTS.json``.
4. Each H1 case is scored with its family's S14 HOLDOUT procedure (ADR 0020):
   the shipped bundle's operator against the unchanged S10 recipe solve, the
   coarse level and the recovery terms. Case results are cached under the run
   root, so an interrupted reveal resumes.
5. The coverage of every calibrated bound is counted on the in-calibration H1
   cases and compared with the preregistered beta-binomial acceptance counts.
6. ``S23-<campaign>-H1-REVEAL.json`` is written, then H1 is burned
   (``artifacts/evaluation/BURNED-H1-<campaign>.json``). A second reveal is
   refused.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from campaigns import s23_calibration as s23
from piginot.bounds import (
    BOUND_KINDS,
    IN_CALIBRATION,
    CalibrationRecord,
    acceptance_count,
)
from piginot.contracts import canonical_json_bytes
from piginot.evaluation import BurnedCaseRegistry

REPOSITORY = s23.REPOSITORY
SET_NAME = "H1"


def reveal_path(campaign_id: str) -> Path:
    return s23.OUTPUT / f"S23-{campaign_id}-H1-REVEAL.json"


def consumed_path(campaign_id: str) -> Path:
    return s23.OUTPUT / f"S23-{campaign_id}-H1-CONSUMED.json"


def burned_path(campaign_id: str) -> Path:
    return Path("artifacts/evaluation") / f"BURNED-H1-{campaign_id}.json"


class RevealRefused(PermissionError):
    """A precondition of the single H1 reveal does not hold."""


@dataclass(frozen=True)
class FamilyAdapter:
    """The family's S14 HOLDOUT scoring pieces (campaign code, unchanged)."""

    campaign_id: str
    holdout_cases: Callable[[Sequence[Mapping[str, float]]], tuple[Any, ...]]
    s10_request: Callable[[Any, Any], Any]
    rehearse_case: Callable[..., tuple[Any, Any, Any]]
    solve: Callable[[Any], tuple[Any, float]]
    solved_case: Callable[..., Any]
    record_document: Callable[[Any], dict[str, object]]
    normalization: Callable[[], Any]
    profile: Callable[[], Any]
    holdout_roles: frozenset[Any]


def _notched_normalization() -> Any:
    from campaigns.notched_v1 import s13

    committed = json.loads((REPOSITORY / s13.PROTOCOL_PATH).read_text("utf-8"))
    fields = committed["normalization"]
    normalization = s13.ConditioningNormalization(
        parameter_names=tuple(fields["parameter_names"]),
        mean=tuple(float(v) for v in fields["mean"]),
        standard_deviation=tuple(float(v) for v in fields["standard_deviation"]),
        fit_case_ids=tuple(fields["fit_case_ids"]),
    )
    if normalization.content_hash != committed["normalization_hash"]:
        raise ValueError("the notched normalization differs from its S13 protocol")
    return normalization


def adapter(campaign_id: str) -> FamilyAdapter:
    """The unchanged S14 HOLDOUT procedure of one family."""

    from campaigns.dogbone_v1 import s14_rehearsal, s14_validate

    if campaign_id == "dogbone-v1":
        from campaigns.dogbone_v1 import s14_evaluation
        from campaigns.dogbone_v1.s10_production import default_s10_execution_profile
        from campaigns.dogbone_v1.s13c_protocol import load_s13c_normalization

        return FamilyAdapter(
            campaign_id,
            s14_evaluation.holdout_cases,
            s14_evaluation.s10_request,
            s14_rehearsal.rehearse_case,
            s14_rehearsal._solve,
            s14_rehearsal.SolvedCase,
            s14_validate.record_document,
            load_s13c_normalization,
            default_s10_execution_profile,
            s14_validate._HOLDOUT,
        )
    if campaign_id == "notched-v1":
        from campaigns.notched_v1 import pilot, s14

        return FamilyAdapter(
            campaign_id,
            s14.holdout_cases,
            s14.s10_request,
            s14.rehearse_case,
            s14_rehearsal._solve,
            s14_rehearsal.SolvedCase,
            s14_validate.record_document,
            _notched_normalization,
            pilot.execution_profile,
            s14._HOLDOUT,
        )
    raise ValueError(f"unknown campaign {campaign_id!r}")


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    from piginot.git_environment import isolated_git_environment

    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
        env=isolated_git_environment(),
    )


def require_committed(root: Path, paths: Sequence[Path]) -> None:
    """Every path is tracked by Git and unchanged in the working tree."""

    for path in paths:
        if _git(root, "ls-files", "--error-unmatch", "--", path.as_posix()).returncode:
            raise RevealRefused(f"{path} is not committed")
        if _git(root, "status", "--porcelain", "--", path.as_posix()).stdout.strip():
            raise RevealRefused(f"{path} has uncommitted changes")


def preconditions(
    root: Path, campaign_id: str
) -> tuple[Mapping[str, Any], str, CalibrationRecord]:
    """Accepted protocol, committed calibration, no earlier reveal or burn."""

    protocol, protocol_sha256 = s23.accepted_protocol(root)
    family = protocol["families"][campaign_id]
    sidecar = s23.sidecar_path(family["bundle"])
    evidence = s23.evidence_path(campaign_id)
    for done in (reveal_path(campaign_id), burned_path(campaign_id)):
        if (root / done).exists():
            raise RevealRefused(f"H1 of {campaign_id} is already revealed ({done})")
    require_committed(root, [s23.PROTOCOL_PATH, sidecar, evidence])
    calibration = CalibrationRecord.from_document(
        json.loads((root / sidecar).read_text("utf-8"))
    )
    if calibration.protocol_sha256 != protocol_sha256:
        raise RevealRefused("the calibration belongs to another protocol")
    return protocol, protocol_sha256, calibration


def consume(
    root: Path, campaign_id: str, protocol_sha256: str, calibration: CalibrationRecord
) -> None:
    """Record the consumption before the private coordinates are read."""

    path = root / consumed_path(campaign_id)
    if path.exists():
        return  # an interrupted reveal resumes
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": "1.0",
        "campaign_id": campaign_id,
        "set": SET_NAME,
        "protocol_sha256": protocol_sha256,
        "calibration_sha256": calibration.sha256(),
        "consumed_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "S23 coverage test of the calibrated bounds (ADR 0104)",
    }
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(document) + b"\n")


def h1_coordinates(
    root: Path, campaign_id: str, private: Mapping[str, Any], expected: Sequence[str]
) -> tuple[list[str], tuple[Mapping[str, float], ...]]:
    """H1 IDs and coordinates, verified against the public commitment."""

    from campaigns.fresh_holdouts import verify_set_reveal

    public = json.loads(
        (root / "artifacts/campaigns" / campaign_id / "FRESH-HOLDOUTS.json").read_text(
            "utf-8"
        )
    )
    ids = list(public["sets"][SET_NAME]["case_ids"])
    if ids != list(expected):
        raise RevealRefused("the protocol's H1 IDs differ from FRESH-HOLDOUTS.json")
    coordinates = verify_set_reveal(public, private, SET_NAME)
    return ids, coordinates


@functools.cache
def _bundle_model(bundle: str) -> Any:
    from piginot.bundle import load_bundle

    loaded = load_bundle(REPOSITORY / "bundles" / bundle)
    return loaded.serving._model  # the hash-closed, float64, eval-mode operator


@functools.cache
def _normalization(campaign_id: str) -> Any:
    return adapter(campaign_id).normalization()


def score_case(
    campaign_id: str,
    bundle: str,
    case_id: str,
    coordinates: Mapping[str, float],
    run_root: str,
) -> tuple[dict[str, object], float | None]:
    """One H1 case through the family's S14 HOLDOUT scoring (cached)."""

    family = adapter(campaign_id)
    (case,) = family.holdout_cases([coordinates])
    case = dataclasses.replace(case, case_id=case_id, leakage_group_id=case_id)
    request = family.s10_request(case, family.profile())
    directory = Path(run_root) / campaign_id
    cached = (directory / "cases" / f"{case_id}.json").exists()
    seconds: float | None = None
    if cached:
        solved = family.solved_case(case, request, None)
    else:
        result, seconds = family.solve(request)
        solved = family.solved_case(case, request, result)
    record, _, _ = family.rehearse_case(
        _bundle_model(bundle),
        solved,
        _normalization(campaign_id),
        directory,
        allowed_roles=family.holdout_roles,
    )
    return family.record_document(record), seconds


def case_statuses(
    calibration: CalibrationRecord,
    records: Sequence[Mapping[str, Any]],
    coordinates: Sequence[Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Each H1 case's distance to TRAIN and calibration status."""

    statuses = []
    for record, point in zip(records, coordinates, strict=True):
        report = calibration.evaluate(point, "budget_total")
        statuses.append(
            {
                "case_id": record["case_id"],
                "distance_to_train": report["distance_to_train"],
                "status": report["status"],
            }
        )
    return statuses


def count_coverage(
    protocol: Mapping[str, Any],
    calibration: CalibrationRecord,
    records: Sequence[Mapping[str, Any]],
    statuses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Covered counts per bound kind and quantity on in-calibration cases."""

    if [r["case_id"] for r in records] != [s["case_id"] for s in statuses]:
        raise ValueError("records and statuses must list the same cases in order")
    per_case = s23.scores({"cases": list(records)})
    inside = [i for i, s in enumerate(statuses) if s["status"] == IN_CALIBRATION]
    level = float(protocol["coverage_test"]["per_check_level"])
    gated = set(protocol["coverage_test"]["gated"])
    counts: dict[str, Any] = {}
    for kind in BOUND_KINDS:
        for quantity in protocol["quantities"]:
            bound = calibration.bounds[kind][quantity]
            covered = sum(per_case[kind][quantity][i] <= bound for i in inside)
            m = len(inside)
            needed = (
                acceptance_count(calibration.n, calibration.alpha, m, level)
                if m
                else None
            )
            key = f"{calibration.campaign_id}:{kind}:{quantity}"
            counts[f"{kind}:{quantity}"] = {
                "bound": bound,
                "m": m,
                "covered": covered,
                "acceptance_count": needed,
                "gated": key in gated,
                "pass": needed is not None and covered >= needed,
            }
    return {
        "in_calibration": len(inside),
        "out_of_calibration_case_ids": [
            s["case_id"] for s in statuses if s["status"] != IN_CALIBRATION
        ],
        "per_check_level": level,
        "counts": counts,
        "case_status": [dict(s) for s in statuses],
        "scores": per_case,
    }


def coverage(
    protocol: Mapping[str, Any],
    calibration: CalibrationRecord,
    records: Sequence[Mapping[str, Any]],
    coordinates: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    """Statuses from the (private) coordinates, then the covered counts."""

    statuses = case_statuses(calibration, records, coordinates)
    return count_coverage(protocol, calibration, records, statuses)


def run(
    root: Path,
    campaign_id: str,
    private_reveal: Path,
    run_root: Path,
    workers: int,
) -> dict[str, Any]:
    if not run_root.is_absolute() or run_root.resolve().is_relative_to(root.resolve()):
        raise RevealRefused(
            "the run root must be an absolute path outside the repository"
        )
    protocol, protocol_sha256, calibration = preconditions(root, campaign_id)
    family = protocol["families"][campaign_id]
    consume(root, campaign_id, protocol_sha256, calibration)
    private = json.loads(private_reveal.read_text("utf-8"))
    ids, coordinates = h1_coordinates(root, campaign_id, private, family["h1_case_ids"])
    tasks = [
        (campaign_id, family["bundle"], case_id, dict(point), str(run_root))
        for case_id, point in zip(ids, coordinates, strict=True)
    ]
    results: list[tuple[dict[str, object], float | None]] = []
    if workers <= 1:
        for task in tasks:
            results.append(score_case(*task))
            print(f"H1 {campaign_id} {len(results)}/{len(ids)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(score_case, *task) for task in tasks]
            for future in futures:
                results.append(future.result())
                print(f"H1 {campaign_id} {len(results)}/{len(ids)}", flush=True)
    records = [document for document, _ in results]
    from piginot.training.platform_fingerprint import platform_fingerprint

    document = {
        "schema_version": "1.0",
        "evidence_role": "HOLDOUT",
        "set": SET_NAME,
        "campaign_id": campaign_id,
        "protocol_sha256": protocol_sha256,
        "calibration_sha256": calibration.sha256(),
        "bundle": family["bundle"],
        "bundle_manifest_sha256": calibration.bundle_manifest_sha256,
        "platform_fingerprint": platform_fingerprint(),
        "workers": workers,
        "fem_solve_seconds": [seconds for _, seconds in results if seconds is not None],
        "cases": records,
        "coverage": coverage(protocol, calibration, records, coordinates),
    }
    path = root / reveal_path(campaign_id)
    with path.open("xb") as handle:
        handle.write(
            json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
    registry = BurnedCaseRegistry(
        campaign_id, frozenset(ids), (reveal_path(campaign_id).as_posix(),)
    )
    burned = root / burned_path(campaign_id)
    burned.parent.mkdir(parents=True, exist_ok=True)
    with burned.open("xb") as handle:
        handle.write(
            json.dumps(registry.to_document(), indent=2).encode("utf-8") + b"\n"
        )
    return document


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S23 H1 reveal (ADR 0104).")
    parser.add_argument("--campaign", choices=tuple(s23.FAMILIES), required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    options = parser.parse_args(arguments)
    try:
        document = run(
            REPOSITORY,
            options.campaign,
            options.reveal,
            options.run_root,
            options.workers,
        )
    except RevealRefused as exc:
        print(f"s23_reveal: {exc}", file=sys.stderr)
        return 1
    summary = {
        key: {k: value[k] for k in ("m", "covered", "acceptance_count", "pass")}
        for key, value in document["coverage"]["counts"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
