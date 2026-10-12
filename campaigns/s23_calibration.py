"""S23 calibration of the v0.1.0 bundles on DEVELOPMENT (ADR 0104).

Two steps, in this order:

1. ``protocol`` writes ``artifacts/bounds/S23-PROTOCOL.json`` once the
   decision owner has accepted ADR 0104's decisions and thresholds; the
   acceptance text is recorded in the protocol. Commit it before step 2.
2. ``calibrate`` builds each family's calibration from public records only —
   the S14 DEVELOPMENT per-case terms and the public case manifest — and
   writes the sidecar ``bundles/calibration/<bundle>.json`` and the evidence
   ``artifacts/bounds/S23-<campaign>-CALIBRATION.json``. It refuses to run
   without an accepted protocol, and on any burned HOLDOUT or sealed H1/H2 ID::

    python -m campaigns.s23_calibration protocol --accepted "decision owner, ..."
    python -m campaigns.s23_calibration calibrate --campaign dogbone-v1
    python -m campaigns.s23_calibration calibrate --campaign notched-v1

No FEM runs: calibration is deterministic arithmetic on recorded terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from piginot.bounds import (
    BOUND_KINDS,
    CalibrationRecord,
    MahalanobisGate,
    acceptance_count,
    case_ids_sha256,
    split_conformal_bound,
)
from piginot.contracts import canonical_json_bytes
from piginot.evaluation import BurnedCaseRegistry, budget_entries

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = Path("artifacts/bounds")
PROTOCOL_PATH = OUTPUT / "S23-PROTOCOL.json"
PROTOCOL_ID = "s23-calibrated-bounds-v1"
FAMILIES: Mapping[str, str] = {
    "dogbone-v1": "dogbone-v1-s13c-cp-freeze",
    "notched-v1": "notched-v1-cp-freeze",
}
QUANTITIES = ("p_relative_l2", "absolute_peak_error", "u_relative_l2")
GATED_QUANTITIES = ("p_relative_l2", "absolute_peak_error")
ALPHA = 0.1
FAMILY_LEVEL = 0.05
DOGBONE_WIDTH_LIMITS = {
    "p_relative_l2": 0.05,
    "absolute_peak_error": 0.05,
    "u_relative_l2": 0.02,
}


class ProtocolNotAccepted(RuntimeError):
    """Calibration was attempted before the owner accepted the protocol."""


class SealedCaseError(PermissionError):
    """A sealed H1/H2 case reached the calibration path."""


def _read(root: Path, path: Path) -> Any:
    return json.loads((root / path).read_text("utf-8"))


def _campaign(campaign_id: str) -> Path:
    return Path("artifacts/campaigns") / campaign_id


def development_record(root: Path, campaign_id: str) -> Mapping[str, Any]:
    return _read(root, _campaign(campaign_id) / "S14-DEVELOPMENT.json")


def sealed_ids(root: Path, campaign_id: str) -> frozenset[str]:
    document = _read(root, _campaign(campaign_id) / "FRESH-HOLDOUTS.json")
    return frozenset(
        case_id
        for sealed in document["sets"].values()
        for case_id in sealed["case_ids"]
    )


def guard(root: Path, campaign_id: str, case_ids: Sequence[str]) -> None:
    """Refuse burned HOLDOUT IDs (ADR 0101) and sealed H1/H2 IDs."""

    registry = BurnedCaseRegistry.from_document(
        _read(root, Path("artifacts/evaluation") / f"BURNED-HOLDOUT-{campaign_id}.json")
    )
    registry.guard(case_ids, purpose="calibration")
    sealed = sorted(sealed_ids(root, campaign_id).intersection(case_ids))
    if sealed:
        raise SealedCaseError(
            f"{len(sealed)} sealed H1/H2 case(s) of {campaign_id} reached the "
            f"calibration path: {', '.join(sealed[:5])}"
        )


def scores(record: Mapping[str, Any]) -> dict[str, dict[str, list[float]]]:
    """Per bound kind and quantity, the per-case scores in record order.

    ``budget_total`` is the S14 per-case total (surrogate + discretisation +
    projection, summed by ``CaseBudgetEntry.total``); ``reference_terms`` is
    discretisation + projection.
    """

    entries = budget_entries(record)
    result: dict[str, dict[str, list[float]]] = {
        kind: {q: [] for q in QUANTITIES} for kind in BOUND_KINDS
    }
    for entry in entries:
        for quantity in QUANTITIES:
            total = entry.total(quantity)
            parts = entry.components[quantity]
            if total is None:
                raise ValueError(f"{entry.case_id} lacks a {quantity} term")
            result["budget_total"][quantity].append(total)
            result["reference_terms"][quantity].append(
                float(parts["discretisation"]) + float(parts["projection"])
            )
    return result


def gate(root: Path, campaign_id: str, case_ids: Sequence[str]) -> MahalanobisGate:
    """TRAIN statistics from the public manifest; threshold from ``case_ids``."""

    manifest = _read(root, _campaign(campaign_id) / "public-case-manifest.json")
    axes = list(manifest["primitive_axes"])
    by_id = {case["case_id"]: case for case in manifest["cases"]}
    train = [
        [float(case["primitive_coordinates"][a]) for a in axes]
        for case in manifest["cases"]
        if case["role"] == "TRAIN" and case["status"] == "ACTIVE"
    ]
    calibration = [
        [float(by_id[case_id]["primitive_coordinates"][a]) for a in axes]
        for case_id in case_ids
    ]
    return MahalanobisGate.fit(axes, train, calibration)


def protocol_document(root: Path, accepted: str) -> dict[str, Any]:
    """The S23 preregistration; ``accepted`` records the owner's acceptance."""

    if not accepted.strip():
        raise ProtocolNotAccepted("the protocol needs the owner's acceptance text")
    families: dict[str, Any] = {}
    for campaign_id, bundle in FAMILIES.items():
        record = development_record(root, campaign_id)
        case_ids = [case["case_id"] for case in record["cases"]]
        fresh = _read(root, _campaign(campaign_id) / "FRESH-HOLDOUTS.json")
        families[campaign_id] = {
            "bundle": bundle,
            "calibration_case_ids": case_ids,
            "calibration_source": str(_campaign(campaign_id) / "S14-DEVELOPMENT.json"),
            "h1_case_ids": list(fresh["sets"]["H1"]["case_ids"]),
        }
    gated = len(FAMILIES) * len(BOUND_KINDS) * len(GATED_QUANTITIES)
    level = FAMILY_LEVEL / gated
    return {
        "schema_version": "1.0",
        "protocol_id": PROTOCOL_ID,
        "step": "S23",
        "decision_record": "docs/adr/0104-s23-calibrated-bounds.md",
        "evidence_role": (
            "DEVELOPMENT (opened in S14) calibrates; fresh H1 tests once; "
            "burned HOLDOUTs and H2 are not touched"
        ),
        "alpha": ALPHA,
        "method": "unnormalised split conformal per family, bound kind and quantity",
        "rank": "k = ceil((n + 1)(1 - alpha)); no bound if k > n",
        "quantities": list(QUANTITIES),
        "scores": {
            "budget_total": "S14 per-case surrogate + discretisation + projection",
            "reference_terms": "S14 per-case discretisation + projection",
        },
        "bound_use": {
            "budget_total": "fast",
            "reference_terms": "certified and exact, next to the certificate",
        },
        "gate": (
            "Mahalanobis distance to the TRAIN mean under the TRAIN covariance on "
            "the primitive axes that vary across TRAIN (public case manifest); "
            "IN_CALIBRATION if at most the largest calibration-case distance"
        ),
        "families": families,
        "coverage_test": {
            "gated": [
                f"{family}:{kind}:{quantity}"
                for family in FAMILIES
                for kind in BOUND_KINDS
                for quantity in GATED_QUANTITIES
            ],
            "family_level": FAMILY_LEVEL,
            "per_check_level": level,
            "rule": (
                "covered in-calibration H1 cases >= the smallest c with "
                "P(X <= c) >= per_check_level, X ~ BetaBinomial(m, k, n + 1 - k)"
            ),
            "acceptance_count_m64": acceptance_count(64, ALPHA, 64, level),
            "reported": [f"{kind}:u_relative_l2" for kind in BOUND_KINDS],
        },
        "thresholds": {
            "dogbone_budget_total_at_most": DOGBONE_WIDTH_LIMITS,
            "notched_widths": "reported",
            "calibration_recompute": "exact",
            "determinism": "byte-identical calibration documents on rerun",
        },
        "descriptive": [
            "normalised scores on H1: S14 indicator, distance to TRAIN, Mondrian",
            "v0.1.0 models' fresh-holdout budget statistics on H1",
        ],
        "thresholds_accepted": accepted,
    }


def accepted_protocol(root: Path) -> tuple[Mapping[str, Any], str]:
    path = root / PROTOCOL_PATH
    if not path.exists():
        raise ProtocolNotAccepted(f"{PROTOCOL_PATH} does not exist")
    document = json.loads(path.read_text("utf-8"))
    if document.get("protocol_id") != PROTOCOL_ID or not document.get(
        "thresholds_accepted"
    ):
        raise ProtocolNotAccepted("the S23 protocol is not accepted")
    return document, hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate(
    root: Path, campaign_id: str, manifest_sha256: str
) -> tuple[CalibrationRecord, dict[str, Any]]:
    """The calibration sidecar and its evidence for one family."""

    protocol, protocol_sha256 = accepted_protocol(root)
    family = protocol["families"][campaign_id]
    record = development_record(root, campaign_id)
    case_ids = [case["case_id"] for case in record["cases"]]
    if case_ids != family["calibration_case_ids"]:
        raise ValueError("the DEVELOPMENT record does not match the protocol's cases")
    guard(root, campaign_id, case_ids)
    alpha = float(protocol["alpha"])
    per_case = scores(record)
    bounds: dict[str, dict[str, float]] = {}
    ranks: set[tuple[int, int]] = set()
    for kind in BOUND_KINDS:
        bounds[kind] = {}
        for quantity in protocol["quantities"]:
            bound = split_conformal_bound(per_case[kind][quantity], alpha)
            bounds[kind][quantity] = bound.bound
            ranks.add((bound.n, bound.k))
    ((n, k),) = ranks
    envelope = gate(root, campaign_id, case_ids)
    calibration = CalibrationRecord(
        bundle=family["bundle"],
        bundle_manifest_sha256=manifest_sha256,
        campaign_id=campaign_id,
        protocol_sha256=protocol_sha256,
        alpha=alpha,
        n=n,
        k=k,
        calibration_case_ids_sha256=case_ids_sha256(case_ids),
        bounds=bounds,
        gate=envelope,
    )
    evidence = {
        "schema_version": "1.0",
        "campaign_id": campaign_id,
        "protocol_sha256": protocol_sha256,
        "calibration_sha256": calibration.sha256(),
        "case_ids": case_ids,
        "scores": per_case,
        "bounds": bounds,
        "n": n,
        "k": k,
        "gate": envelope.document(),
        "dogbone_width_check": None
        if campaign_id != "dogbone-v1"
        else {
            quantity: bounds["budget_total"][quantity] <= limit
            for quantity, limit in DOGBONE_WIDTH_LIMITS.items()
        },
    }
    return calibration, evidence


def sidecar_path(bundle: str) -> Path:
    return Path("bundles/calibration") / f"{bundle}.json"


def evidence_path(campaign_id: str) -> Path:
    return OUTPUT / f"S23-{campaign_id}-CALIBRATION.json"


def _write(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(document) + b"\n")


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S23 calibration (ADR 0104).")
    commands = parser.add_subparsers(dest="command", required=True)
    protocol = commands.add_parser("protocol")
    protocol.add_argument("--accepted", required=True)
    run = commands.add_parser("calibrate")
    run.add_argument("--campaign", choices=tuple(FAMILIES), required=True)
    options = parser.parse_args(arguments)
    root = REPOSITORY
    if options.command == "protocol":
        if (root / PROTOCOL_PATH).exists():
            print(f"{PROTOCOL_PATH} exists; it is written once", file=sys.stderr)
            return 1
        _write(root / PROTOCOL_PATH, protocol_document(root, options.accepted))
        print(f"wrote {PROTOCOL_PATH}; commit it before calibrating")
        return 0
    from piginot.bundle import load_bundle

    bundle = FAMILIES[options.campaign]
    manifest = load_bundle(root / "bundles" / bundle).manifest.manifest_sha256
    calibration, evidence = calibrate(root, options.campaign, manifest)
    _write(root / sidecar_path(bundle), calibration.document())
    _write(root / evidence_path(options.campaign), evidence)
    print(json.dumps(evidence["bounds"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
