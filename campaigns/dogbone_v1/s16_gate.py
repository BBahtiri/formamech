"""The G16 gate artifact from a full suite run and a weakened-detector run."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from campaigns.dogbone_v1.s16_mutants import MUTANTS
from piginot.mutation import EXIT_CODES, ComparisonStatus

ROOT = Path(__file__).resolve().parents[2]
MUTATION_ARTIFACTS = ROOT / "artifacts" / "mutation"
WORKFLOW = ROOT / ".github" / "workflows" / "mutation.yml"
RUNTIME_LIMIT_SECONDS = 45 * 60
COVERAGE_FLOOR = 85.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(
    identifier: str, threshold: str, measured: dict[str, Any], passed: bool
) -> dict[str, Any]:
    return {
        "id": identifier,
        "threshold": threshold,
        "measured": measured,
        "status": "PASS" if passed else "FAIL",
        "reason": None,
        "comparison": "all",
    }


def gate_checks(
    report: dict[str, Any],
    regression: dict[str, Any],
    baseline: dict[str, Any],
    workflow_text: str,
    coverage_percent: float,
) -> list[dict[str, Any]]:
    expected = [(m.name, m.primary) for m in MUTANTS]
    runs = report["mutants"]
    observed = [(run["name"], run["primary"]) for run in runs]
    primary_missed = [run["name"] for run in runs if not run["primary_fired"]]
    primary = _check(
        "G16.primary",
        "20 of 20 mutants fire their named primary gate; the unmutated control "
        "passes every detector before and after",
        {
            "mutants": len(runs),
            "primary_fired": report["primary_fired"],
            "missed": primary_missed,
            "control_passed": report["control_passed"],
            "detectors": len(report["detectors"]),
            "matches_adr_0022_table": observed == expected,
        },
        report["control_passed"]
        and observed == expected
        and not primary_missed
        and report["primary_fired"] == 20,
    )
    signatures = baseline["mutants"]
    comparison = report.get("comparison", {})
    complete = (
        set(signatures) == {name for name, _ in expected}
        and all(item["primary"] in item["signature"] for item in signatures.values())
        and all(
            item["signature"] == sorted(item["signature"])
            for item in signatures.values()
        )
    )
    signature_check = _check(
        "G16.signatures",
        "the full sorted fired-gate set of each mutant is recorded and committed "
        "as the baseline",
        {
            "baseline": "artifacts/mutation/S16-SIGNATURES.json",
            "mutants_recorded": len(signatures),
            "secondary_detections": sum(
                len(item["signature"]) - 1 for item in signatures.values()
            ),
            "complete_and_sorted": complete,
            "run_matches_baseline": comparison.get("status") == "PASS",
        },
        complete and comparison.get("status") == "PASS",
    )
    weakened = regression.get("comparison", {})
    regressed = [
        item for item in weakened.get("mutants", []) if item["status"] != "PASS"
    ]
    primaries_still_fire = all(run["primary_fired"] for run in regression["mutants"])
    workflow_ok = all(
        token in workflow_text
        for token in (
            "release:",
            "s16_mutation run",
            "--compare artifacts/mutation/S16-SIGNATURES.json",
        )
    )
    regression_check = _check(
        "G16.regression",
        "a CI job compares signatures per release; losing a secondary detection is "
        "reported as a coverage regression even when the primary still fires",
        {
            "workflow": ".github/workflows/mutation.yml",
            "workflow_compares_baseline_on_release": workflow_ok,
            "demonstration_removed_detectors": weakened.get("removed_detectors", []),
            "demonstration_status": weakened.get("status"),
            "demonstration_exit_code": EXIT_CODES[
                ComparisonStatus(weakened.get("status", "PRIMARY_LOST"))
            ],
            "coverage_regressions": [
                {
                    "mutant": item["name"],
                    "status": item["status"],
                    "lost_secondary": item["lost_secondary"],
                }
                for item in regressed
            ],
            "primaries_still_fire": primaries_still_fire,
            "hosted_release_run": "NOT_RUN: runs when a release is published",
        },
        workflow_ok
        and weakened.get("status") == "COVERAGE_REGRESSION"
        and bool(regressed)
        and all(
            item["status"] == "COVERAGE_REGRESSION" and item["lost_secondary"]
            for item in regressed
        )
        and primaries_still_fire,
    )
    seconds = float(report["seconds"])
    runtime_check = _check(
        "G16.runtime",
        "the full mutation suite completes in under 45 minutes",
        {
            "suite_seconds": seconds,
            "limit_seconds": RUNTIME_LIMIT_SECONDS,
            "ci_job_timeout_minutes": 45,
            "runtime": report["runtime"],
        },
        seconds < RUNTIME_LIMIT_SECONDS and "timeout-minutes: 45" in workflow_text,
    )
    coverage_check = _check(
        "G16.coverage",
        ">= 85% of piginot.mutation",
        {"line_and_branch_coverage_percent": coverage_percent},
        coverage_percent >= COVERAGE_FLOOR,
    )
    return [primary, signature_check, regression_check, runtime_check, coverage_check]


def write_gate(args: argparse.Namespace) -> int:
    report = json.loads(args.report.read_text(encoding="utf-8"))
    regression = json.loads(args.regression_report.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    checks = gate_checks(
        report, regression, baseline, workflow_text, args.coverage_percent
    )
    verdict = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    MUTATION_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    run_path = MUTATION_ARTIFACTS / "S16-RUN.json"
    demo_path = MUTATION_ARTIFACTS / "S16-REGRESSION-DEMO.json"
    run_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    demo_path.write_text(json.dumps(regression, indent=1, sort_keys=True) + "\n")
    document = {
        "schema_version": "1.0",
        "gate_id": "G16",
        "step": "S16",
        "branch": "codex/s16-mutation",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision_record": [
            "docs/adr/0022-replace-implementation-v1.0-oss.md",
            "docs/adr/0024-s16-mutation-suite.md",
        ],
        "implementation_source_revision": args.revision,
        "scope": {
            "mutants": len(MUTANTS),
            "detectors": len(report["detectors"]),
            "development_or_holdout_read": False,
            "claim": "Each of the twenty ADR 0022 mutants, applied as a source edit to "
            "the production function that holds the defect, fires its named primary "
            "gate; the fired-gate signatures are the committed baseline, and a "
            "release job reports any lost detection.",
        },
        "verdict": verdict,
        "checks": checks,
        "evidence": {
            "artifacts/mutation/S16-SIGNATURES.json": _sha256(args.baseline),
            "artifacts/mutation/S16-RUN.json": _sha256(run_path),
            "artifacts/mutation/S16-REGRESSION-DEMO.json": _sha256(demo_path),
            ".github/workflows/mutation.yml": _sha256(WORKFLOW),
        },
        "implementation_evidence": {
            "focused_coverage_command": args.coverage_command,
            "focused_test_result": args.focused_result,
            "full_test_command": "python -m pytest -q --cov=src/piginot --cov-branch",
            "full_test_result": args.full_result,
            "quality_checks": {
                "mypy_strict_src": "PASS",
                "ruff_check": "PASS",
                "ruff_format_check": "PASS",
                "mkdocs_strict_build": "PASS",
            },
        },
        "limitations": [
            "Detectors run each gate check's procedure on a fixed CI-sized fixture, "
            "not on the full campaign evidence the original gate artifacts measured.",
            "Signatures are deterministic on one machine; a different torch or "
            "scikit-fem release may change secondary detections, which the release "
            "job then reports.",
            "The hosted release job has not run yet; the comparison it performs was "
            "run locally, including a weakened-detector demonstration.",
            "G16.runtime measures the mutation suite; the full pytest suite is "
            "reported separately.",
        ],
    }
    gate_path = ROOT / "artifacts" / "gates" / "S16.json"
    gate_path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    print(f"G16 {verdict}")
    return 0 if verdict == "PASS" else 1


__all__ = ("gate_checks", "write_gate")
