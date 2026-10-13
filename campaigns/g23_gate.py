"""G23: calibrated bounds and the calibration gate (S23, ADR 0104).

Thresholds from ``artifacts/bounds/S23-PROTOCOL.json``. Checks:
protocol-first, isolation, recompute, gate, width, reveal-once, coverage,
compat, portability (native Windows run) and quality (Linux run)::

    python -m campaigns.g23_gate --record-suite COVERAGE_JSON PYTEST_LOG
    python -m campaigns.g23_gate --record-windows PYTEST_LOG WINDOWS_CLONE
    python -m campaigns.g23_gate --write
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from campaigns import g19_gate
from campaigns import s23_calibration as s23
from campaigns import s23_reveal as reveal
from campaigns.g19_gate import execution_profile, not_run, read_text_any
from campaigns.g20_gate import (
    _added_in,
    _git,
    _pytest,
    _strictly_before,
    _tree,
    parse_pytest_log,
)
from campaigns.notched_v1.gate_io import (
    check_document,
    git_head,
    utc_now,
    write_gate_document,
)
from piginot.bounds import (
    IN_CALIBRATION,
    OUT_OF_CALIBRATION,
    CalibrationRecord,
)
from piginot.contracts import canonical_json_bytes

REPOSITORY = Path(__file__).resolve().parents[1]
GATE_PATH = Path("artifacts/gates/S23.json")
SUITE_PATH = s23.OUTPUT / "S23-SUITE.json"
WINDOWS_PATH = s23.OUTPUT / "S23-WINDOWS-SUITE.json"
GATE_FILES = (GATE_PATH, SUITE_PATH, WINDOWS_PATH)
GLOBAL_COVERAGE = 90.5
BOUNDS_COVERAGE = 90.0


def _protocol(root: Path) -> tuple[dict[str, Any], str]:
    path = root / s23.PROTOCOL_PATH
    return json.loads(path.read_text("utf-8")), hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _sidecar(root: Path, protocol: Mapping[str, Any], campaign: str) -> Path:
    return root / s23.sidecar_path(protocol["families"][campaign]["bundle"])


def _calibration(
    root: Path, protocol: Mapping[str, Any], campaign: str
) -> CalibrationRecord:
    return CalibrationRecord.from_document(
        json.loads(_sidecar(root, protocol, campaign).read_text("utf-8"))
    )


def check_protocol_first(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    commits = _git(
        root, "log", "--format=%H", "--", s23.PROTOCOL_PATH.as_posix()
    ).stdout.split()
    first = commits[-1] if commits else None
    later: dict[str, str | None] = {}
    for campaign, family in protocol["families"].items():
        later[f"{campaign}:sidecar"] = _added_in(
            root, s23.sidecar_path(family["bundle"])
        )
        later[f"{campaign}:evidence"] = _added_in(root, s23.evidence_path(campaign))
        for name, path in (
            ("consumed", reveal.consumed_path(campaign)),
            ("reveal", reveal.reveal_path(campaign)),
        ):
            if (root / path).exists():
                later[f"{campaign}:{name}"] = _added_in(root, path)
    passed = (
        first is not None
        and len(commits) == 1
        and all(
            c is not None and _strictly_before(root, first, c) for c in later.values()
        )
    )
    return check_document(
        "G23.protocol-first",
        passed,
        {"protocol_commit": first, "protocol_commits": len(commits), "later": later},
        "protocol committed once, before the calibrations and the H1 reveals",
        "all",
    )


def check_isolation(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    overlaps: dict[str, list[str]] = {}
    for campaign in protocol["families"]:
        evidence = json.loads((root / s23.evidence_path(campaign)).read_text("utf-8"))
        burned = json.loads(
            (
                root / "artifacts/evaluation" / f"BURNED-HOLDOUT-{campaign}.json"
            ).read_text("utf-8")
        )["case_ids"]
        forbidden = set(burned) | s23.sealed_ids(root, campaign)
        overlaps[campaign] = sorted(forbidden.intersection(evidence["case_ids"]))
    tests = _pytest(
        root,
        "tests/campaigns/test_s23_calibration.py::test_a_burned_holdout_case_is_refused",
        "tests/campaigns/test_s23_calibration.py::test_a_sealed_case_is_refused",
    )
    return check_document(
        "G23.isolation",
        tests["returncode"] == 0 and not any(overlaps.values()),
        {"calibration_overlaps": overlaps, "mutant_tests": tests},
        "no burned or sealed ID in a calibration; the calibration path refuses them",
        "all",
    )


GATE_STATISTICS_TOLERANCE = 1e-10
STATUS_MARGIN = 1e-9
_EXACT_SIDECAR_FIELDS = (
    "schema",
    "bundle",
    "bundle_manifest_sha256",
    "campaign_id",
    "protocol_sha256",
    "alpha",
    "n",
    "k",
    "calibration_case_ids_sha256",
    "bounds",
)
_EXACT_EVIDENCE_FIELDS = (
    "schema_version",
    "campaign_id",
    "protocol_sha256",
    "case_ids",
    "scores",
    "bounds",
    "n",
    "k",
    "dogbone_width_check",
)


def _gate_numbers(gate_document: Mapping[str, Any]) -> list[float]:
    return [
        *(float(v) for v in gate_document["mean"]),
        *(float(v) for row in gate_document["precision"] for v in row),
        float(gate_document["max_distance"]),
    ]


def gate_difference(
    committed: Mapping[str, Any], fresh: Mapping[str, Any]
) -> float | None:
    """Largest relative difference of the gate statistics; None if not comparable."""

    if committed["axes"] != fresh["axes"] or len(committed["precision"]) != len(
        fresh["precision"]
    ):
        return None
    worst = 0.0
    for a, b in zip(_gate_numbers(committed), _gate_numbers(fresh), strict=True):
        scale = max(abs(a), abs(b))
        if scale:
            worst = max(worst, abs(a - b) / scale)
    return worst


def compare_calibration(
    committed_sidecar: Mapping[str, Any],
    fresh_sidecar: Mapping[str, Any],
    committed_evidence: Mapping[str, Any],
    fresh_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """ADR 0104 decision 8: exact parts exactly, gate statistics within 1e-10."""

    difference = gate_difference(committed_sidecar["gate"], fresh_sidecar["gate"])
    evidence_gate = gate_difference(committed_evidence["gate"], fresh_evidence["gate"])
    return {
        "sidecar_exact_fields": all(
            committed_sidecar[k] == fresh_sidecar[k] for k in _EXACT_SIDECAR_FIELDS
        ),
        "evidence_exact_fields": all(
            committed_evidence[k] == fresh_evidence[k] for k in _EXACT_EVIDENCE_FIELDS
        ),
        "gate_relative_difference": difference,
        "evidence_gate_relative_difference": evidence_gate,
        "gate_within_tolerance": difference is not None
        and evidence_gate is not None
        and difference <= GATE_STATISTICS_TOLERANCE
        and evidence_gate <= GATE_STATISTICS_TOLERANCE,
    }


def _statuses_kept(
    root: Path,
    campaign: str,
    family: Mapping[str, Any],
    committed: CalibrationRecord,
    fresh: CalibrationRecord,
) -> dict[str, Any]:
    """Every calibration case keeps its status; H1 cases sit clear of the edge."""

    manifest = json.loads(
        (
            root / "artifacts/campaigns" / campaign / "public-case-manifest.json"
        ).read_text("utf-8")
    )
    by_id = {c["case_id"]: c for c in manifest["cases"]}
    development = all(
        committed.evaluate(by_id[i]["primitive_coordinates"], "budget_total")["status"]
        == fresh.evaluate(by_id[i]["primitive_coordinates"], "budget_total")["status"]
        for i in family["calibration_case_ids"]
    )
    document = _reveal_document(root, campaign)
    if document is None:
        return {"development": development, "h1": "not revealed"}
    edge = committed.gate.max_distance
    margins = [
        abs(float(s["distance_to_train"]) - edge) / edge
        for s in document["coverage"]["case_status"]
    ]
    return {
        "development": development,
        "h1_smallest_relative_margin": min(margins),
        "h1": min(margins) > STATUS_MARGIN,
    }


def check_recompute(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Byte-exact on the calibration's CPU; elsewhere ADR 0104 decision 8."""

    from piginot.bundle import load_bundle

    measured = {}
    passed = True
    for campaign, family in protocol["families"].items():
        committed = _calibration(root, protocol, campaign)
        fresh, evidence = s23.calibrate(
            root, campaign, committed.bundle_manifest_sha256
        )
        manifest = load_bundle(root / "bundles" / family["bundle"]).manifest
        sidecar_bytes = _sidecar(root, protocol, campaign).read_bytes()
        evidence_bytes = (root / s23.evidence_path(campaign)).read_bytes()
        exact = (
            canonical_json_bytes(fresh.document()) + b"\n" == sidecar_bytes
            and canonical_json_bytes(evidence) + b"\n" == evidence_bytes
        )
        entry: dict[str, Any] = {
            "byte_exact": exact,
            "binds_bundle": committed.bundle_manifest_sha256
            == manifest.manifest_sha256,
        }
        if not exact:
            entry.update(
                compare_calibration(
                    json.loads(sidecar_bytes),
                    fresh.document(),
                    json.loads(evidence_bytes),
                    evidence,
                )
            )
            entry["statuses"] = _statuses_kept(root, campaign, family, committed, fresh)
        statuses = entry.get("statuses", {})
        ok = entry["binds_bundle"] and (
            exact
            or (
                entry["sidecar_exact_fields"]
                and entry["evidence_exact_fields"]
                and entry["gate_within_tolerance"]
                and statuses.get("development") is True
                and statuses.get("h1") in (True, "not revealed")
            )
        )
        entry["pass"] = ok
        passed = passed and ok
        measured[campaign] = entry
    return check_document(
        "G23.recompute",
        passed,
        measured,
        {
            "rule": "ADR 0104 decision 8",
            "exact": "bounds, case IDs, scores, n, k, alpha, hashes; bytes on the "
            "calibration's CPU profile",
            "gate_statistics_relative_tolerance": GATE_STATISTICS_TOLERANCE,
            "statuses": "every calibration case keeps its status; every H1 case is "
            f"more than {STATUS_MARGIN:g} (relative) from the gate threshold",
        },
        "all",
    )


def _envelope_bounds(root: Path, bundle: str) -> dict[str, tuple[float, float, Any]]:
    document = json.loads(
        (root / "bundles" / bundle / "spec" / "envelope.json").read_text("utf-8")
    )
    items = [
        *document["geometry_membership"]["primitive_bounds"],
        *document["material_bounds"],
        *document["load_bounds"],
    ]
    bounds = {}
    for item in items:
        lower, upper = item["lower_bound"], item["upper_bound"]
        bounds[item["name"]] = (float(lower["value"]), float(upper["value"]), lower)
    return bounds


def _inputs(values: Mapping[str, float], bounds: Mapping[str, Any]) -> dict[str, Any]:
    from piginot.contracts import CanonicalQuantity
    from piginot.contracts.units import Dimension, Unit

    return {
        name: CanonicalQuantity(
            float(values[name]),
            Unit(bounds[name][2]["unit"]),
            Dimension(bounds[name][2]["dimension"]),
        )
        for name in bounds
    }


def constructed_outliers(
    root: Path, bundle: str, calibration: CalibrationRecord, limit: int = 4
) -> list[dict[str, Any]]:
    """In-envelope requests beyond the largest calibration distance.

    Corners of the primitive box (fixed axes at their value) that the bundle
    serves and that lie further from TRAIN than every calibration case.
    """

    from piginot.bundle import load_bundle, predict

    loaded = load_bundle(root / "bundles" / bundle)
    bounds = _envelope_bounds(root, bundle)
    varying = [n for n, (lo, hi, _) in bounds.items() if hi > lo]
    found: list[dict[str, Any]] = []
    for corner in itertools.product((0, 1), repeat=len(varying)):
        values = {n: lo for n, (lo, _, _) in bounds.items()}
        for name, side in zip(varying, corner, strict=True):
            values[name] = bounds[name][side]
        distance = calibration.gate.distance(values)
        if distance <= calibration.gate.max_distance:
            continue
        inputs = _inputs(values, bounds)
        if predict(loaded, inputs, confidence=False).verdict.value == "REFUSED":
            continue
        found.append({"values": values, "distance": distance, "inputs": inputs})
        if len(found) == limit:
            break
    return found


def check_gate(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    from piginot.bundle import load_bundle
    from piginot.serving import serve

    measured = {}
    for campaign, family in protocol["families"].items():
        calibration = _calibration(root, protocol, campaign)
        manifest = json.loads(
            (
                root / "artifacts/campaigns" / campaign / "public-case-manifest.json"
            ).read_text("utf-8")
        )
        by_id = {c["case_id"]: c for c in manifest["cases"]}
        statuses = [
            calibration.evaluate(by_id[i]["primitive_coordinates"], "budget_total")[
                "status"
            ]
            for i in family["calibration_case_ids"]
        ]
        outliers = constructed_outliers(root, family["bundle"], calibration)
        loaded = load_bundle(root / "bundles" / family["bundle"])
        served = [serve(loaded, o["inputs"], calibration=calibration) for o in outliers]
        measured[campaign] = {
            "calibration_cases_in": statuses.count(IN_CALIBRATION),
            "calibration_cases": len(statuses),
            "constructed_outliers": len(outliers),
            "outliers_flagged_without_bounds": all(
                s.prediction.in_domain
                and s.bounds is not None
                and s.bounds["status"] == OUT_OF_CALIBRATION
                and s.bounds["upper_bounds"] is None
                for s in served
            ),
            "outlier_distances": [o["distance"] for o in outliers],
            "max_calibration_distance": calibration.gate.max_distance,
        }
    passed = all(
        m["calibration_cases_in"] == m["calibration_cases"]
        and m["constructed_outliers"] > 0
        and m["outliers_flagged_without_bounds"]
        for m in measured.values()
    )
    return check_document(
        "G23.gate",
        passed,
        measured,
        "every calibration case IN_CALIBRATION; constructed in-envelope requests "
        "beyond the largest calibration distance served OUT_OF_CALIBRATION, no bound",
        "all",
    )


def check_width(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    limits = protocol["thresholds"]["dogbone_budget_total_at_most"]
    measured = {
        campaign: dict(_calibration(root, protocol, campaign).bounds["budget_total"])
        for campaign in protocol["families"]
    }
    dogbone = measured.get("dogbone-v1", {})
    passed = bool(dogbone) and all(
        dogbone[quantity] <= limit for quantity, limit in limits.items()
    )
    return check_document(
        "G23.width",
        passed,
        measured,
        {"dogbone-v1 budget_total at most": limits, "notched": "reported"},
        "all",
    )


def _reveal_document(root: Path, campaign: str) -> dict[str, Any] | None:
    path = root / reveal.reveal_path(campaign)
    return json.loads(path.read_text("utf-8")) if path.is_file() else None


def check_reveal_once(
    root: Path, protocol: Mapping[str, Any], digest: str
) -> dict[str, Any]:
    threshold = (
        "one verified H1 reveal per family after the protocol and the calibration; "
        "H1 burned; a second reveal refused"
    )
    measured: dict[str, Any] = {}
    for campaign, family in protocol["families"].items():
        document = _reveal_document(root, campaign)
        if document is None:
            return not_run(
                "G23.reveal-once", f"H1 of {campaign} not revealed", threshold
            )
        calibration = _calibration(root, protocol, campaign)
        burned_path = root / reveal.burned_path(campaign)
        burned = (
            json.loads(burned_path.read_text("utf-8")) if burned_path.is_file() else {}
        )
        try:
            reveal.preconditions(root, campaign)
            refused = False
        except reveal.RevealRefused:
            refused = True
        revealed_in = _added_in(root, reveal.reveal_path(campaign))
        calibrated_in = _added_in(root, s23.evidence_path(campaign))
        measured[campaign] = {
            "protocol_matches": document["protocol_sha256"] == digest,
            "calibration_matches": document["calibration_sha256"]
            == calibration.sha256(),
            "cases_are_h1": [c["case_id"] for c in document["cases"]]
            == family["h1_case_ids"],
            "consumed": (root / reveal.consumed_path(campaign)).is_file(),
            "burned": sorted(burned.get("case_ids", []))
            == sorted(family["h1_case_ids"]),
            "after_calibration": revealed_in is not None
            and calibrated_in is not None
            and _strictly_before(root, calibrated_in, revealed_in),
            "second_reveal_refused": refused,
        }
    return check_document(
        "G23.reveal-once",
        all(all(v.values()) for v in measured.values()),
        measured,
        threshold,
        "all",
    )


def check_coverage(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    threshold = {
        "gated": protocol["coverage_test"]["gated"],
        "rule": protocol["coverage_test"]["rule"],
        "per_check_level": protocol["coverage_test"]["per_check_level"],
    }
    measured: dict[str, Any] = {}
    gated_pass = []
    for campaign in protocol["families"]:
        document = _reveal_document(root, campaign)
        if document is None:
            return not_run("G23.coverage", f"H1 of {campaign} not revealed", threshold)
        calibration = _calibration(root, protocol, campaign)
        recounted = reveal.count_coverage(
            protocol,
            calibration,
            document["cases"],
            document["coverage"]["case_status"],
        )
        counts = recounted["counts"]
        measured[campaign] = {
            "recounts": counts == document["coverage"]["counts"],
            "in_calibration": recounted["in_calibration"],
            "counts": {
                key: {k: v[k] for k in ("m", "covered", "acceptance_count", "pass")}
                for key, v in counts.items()
                if v["gated"]
            },
            "reported": {
                key: {k: v[k] for k in ("m", "covered", "acceptance_count")}
                for key, v in counts.items()
                if not v["gated"]
            },
        }
        gated_pass.append(measured[campaign]["recounts"])
        gated_pass.extend(v["pass"] for v in counts.values() if v["gated"])
    expected = len(protocol["coverage_test"]["gated"])
    counted = sum(len(m["counts"]) for m in measured.values())
    return check_document(
        "G23.coverage",
        counted == expected and all(gated_pass),
        measured,
        threshold,
        "all",
    )


def check_compat(root: Path) -> dict[str, Any]:
    from piginot.bundle import load_bundle
    from piginot.cli import parse_request
    from piginot.serving import serve

    goldens = json.loads((root / g19_gate.GOLDEN_PATH).read_text("utf-8"))["profiles"]
    key = g19_gate.profile_key(execution_profile())
    tests = _pytest(
        root,
        "tests/test_shipped_bundles.py",
        "tests/serving/test_bounds_serving.py",
    )
    if key not in goldens:
        return not_run(
            "G23.compat", f"no S19 golden for profile {key}", "S19 goldens, tests"
        )
    measured: dict[str, Any] = {}
    for name, request_file in g19_gate.EXAMPLES:
        bundle = load_bundle(root / "bundles" / name)
        document = json.loads(
            (root / "bundles" / "requests" / request_file).read_text("utf-8")
        )
        inputs, queries = parse_request(document)
        served = serve(bundle, inputs, queries, mode="fast").document()
        digest = hashlib.sha256(
            json.dumps(served, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()
        measured[f"{name}/{request_file}"] = (
            digest == goldens[key][f"{name}/{request_file}"]
        )
    return check_document(
        "G23.compat",
        all(measured.values()) and tests["returncode"] == 0,
        {"fast_goldens": measured, "tests": tests},
        "without bounds, fast documents equal the S19 goldens; sidecars bind manifests",
        "all",
    )


def _listing(root: Path, tree: str) -> dict[str, str] | None:
    listed = _git(root, "ls-tree", "-r", tree)
    if listed.returncode != 0:
        return None
    excluded = {p.as_posix() for p in GATE_FILES}
    entries = {}
    for line in listed.stdout.splitlines():
        meta, path = line.split("\t", 1)
        if path not in excluded:
            entries[path] = meta
    return entries


def _same_code(root: Path, tree: str) -> bool:
    recorded = _listing(root, tree)
    return recorded is not None and recorded == _listing(root, "HEAD")


def record_suite(root: Path, coverage_json: Path, pytest_log: Path) -> dict[str, Any]:
    report = json.loads(read_text_any(coverage_json))
    totals = [
        f["summary"]
        for name, f in report.get("files", {}).items()
        if name.replace("\\", "/").startswith("src/piginot/bounds/")
    ]
    statements = sum(t["num_statements"] + t.get("num_branches", 0) for t in totals)
    covered = sum(t["covered_lines"] + t.get("covered_branches", 0) for t in totals)
    document = {
        "schema_version": "1.0",
        "tree": _tree(root),
        "commit": git_head(root),
        "clean": _git(root, "status", "--porcelain").stdout.strip() == "",
        "branch_coverage": bool(report.get("meta", {}).get("branch_coverage")),
        "coverage_percent": round(float(report["totals"]["percent_covered"]), 2),
        "bounds_coverage_percent": round(100.0 * covered / statements, 2)
        if statements
        else None,
        **parse_pytest_log(read_text_any(pytest_log)),
        "execution_profile": execution_profile(),
    }
    (root / SUITE_PATH).write_text(json.dumps(document, indent=2) + "\n", "utf-8")
    return document


def record_windows(root: Path, pytest_log: Path, clone: Path) -> dict[str, Any]:
    document = {
        "schema_version": "1.0",
        "tree": _tree(clone),
        "clean": _git(clone, "status", "--porcelain").stdout.strip() == "",
        **parse_pytest_log(read_text_any(pytest_log)),
    }
    (root / WINDOWS_PATH).write_text(json.dumps(document, indent=2) + "\n", "utf-8")
    return document


def check_portability(root: Path) -> dict[str, Any]:
    threshold = (
        "native Windows run of this code tree: 0 failed, 0 errors, "
        "every skip has a reason"
    )
    path = root / WINDOWS_PATH
    if not path.is_file():
        return not_run("G23.portability", f"{WINDOWS_PATH} not recorded", threshold)
    run = json.loads(path.read_text("utf-8"))
    counts = run["counts"]
    passed = (
        _same_code(root, run["tree"])
        and run["clean"]
        and counts.get("failed", 0) == 0
        and counts.get("errors", 0) == 0
        and counts.get("passed", 0) > 3000
        and all(skip["reason"].strip() for skip in run["skips"])
        and sum(skip["count"] for skip in run["skips"]) == counts.get("skipped", 0)
    )
    return check_document("G23.portability", passed, run, threshold, "all")


def check_quality(root: Path) -> dict[str, Any]:
    tools = {
        "ruff_check": [sys.executable, "-m", "ruff", "check", "."],
        "ruff_format": [sys.executable, "-m", "ruff", "format", "--check", "."],
        "mypy_strict": [sys.executable, "-m", "mypy", "--strict", "src/"],
    }
    codes = {
        name: subprocess.run(
            command, cwd=root, capture_output=True, check=False
        ).returncode
        for name, command in tools.items()
    }
    threshold = {
        "tools": "exit 0",
        "suite": "recorded Linux run of this code tree, clean, 0 failed, 0 errors",
        "global_branch_coverage_at_least": GLOBAL_COVERAGE,
        "piginot.bounds_at_least": BOUNDS_COVERAGE,
    }
    path = root / SUITE_PATH
    if not path.is_file():
        return not_run("G23.quality", f"{SUITE_PATH} not recorded", threshold)
    suite = json.loads(path.read_text("utf-8"))
    counts = suite["counts"]
    passed = (
        all(code == 0 for code in codes.values())
        and _same_code(root, suite["tree"])
        and suite["clean"]
        and suite["branch_coverage"]
        and counts.get("failed", 0) == 0
        and counts.get("errors", 0) == 0
        and suite["coverage_percent"] >= GLOBAL_COVERAGE
        and (suite["bounds_coverage_percent"] or 0.0) >= BOUNDS_COVERAGE
    )
    return check_document(
        "G23.quality", passed, {"tools": codes, "suite": suite}, threshold, "all"
    )


def build_gate(root: Path) -> dict[str, Any]:
    protocol, digest = _protocol(root)
    checks = [
        check_protocol_first(root, protocol),
        check_isolation(root, protocol),
        check_recompute(root, protocol),
        check_gate(root, protocol),
        check_width(root, protocol),
        check_reveal_once(root, protocol, digest),
        check_coverage(root, protocol),
        check_compat(root),
        check_portability(root),
        check_quality(root),
    ]
    blockers = [c["id"] for c in checks if c["status"] != "PASS"]
    return {
        "gate_id": "G23",
        "step": "S23",
        "commit": git_head(root),
        "timestamp_utc": utc_now(),
        "execution_profile": execution_profile(),
        "checks": checks,
        "decision_record": "docs/adr/0104-s23-calibrated-bounds.md",
        "verdict": "FAIL" if blockers else "PASS",
        "verdict_reason": f"{len(blockers)} required check(s) not PASS: "
        f"{', '.join(blockers)}"
        if blockers
        else "every required check PASS",
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate gate G23.")
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--record-suite", nargs=2, type=Path, metavar=("COVERAGE_JSON", "LOG")
    )
    parser.add_argument(
        "--record-windows", nargs=2, type=Path, metavar=("LOG", "WINDOWS_CLONE")
    )
    options = parser.parse_args(arguments)
    if options.record_suite:
        print(json.dumps(record_suite(REPOSITORY, *options.record_suite), indent=2))
        return 0
    if options.record_windows:
        print(json.dumps(record_windows(REPOSITORY, *options.record_windows), indent=2))
        return 0
    document = build_gate(REPOSITORY)
    if options.write:
        write_gate_document(REPOSITORY / GATE_PATH, document)
    for check in document["checks"]:
        print(f"{check['id']}: {check['status']}")
    print(f"G23: {document['verdict']} ({document['verdict_reason']})")
    return 0 if document["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
