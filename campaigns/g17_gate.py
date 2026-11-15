"""The G17 gate (examples, benchmarks, docs) from recorded evidence.

Usage::

    python -m campaigns.g17_gate --example1-summary SUMMARY.json \
        --refusal-suite artifacts/benchmarks/REFUSAL-SUITE.json

Each check reads its evidence and fails (never passes by default) when the
evidence is missing:

- ``G17.example1``: the Example 1 summary of a full run passed in under ten
  minutes, and CI declares the ``example1`` job;
- ``G17.example2-3``: both benchmark reproductions passed (bitwise replay from
  the pinned seeds and TRAIN-screen re-scoring within tolerance);
- ``G17.non-dogbone``: the notched-plate bundle passed G15 at
  ``HOLDOUT_VERIFIED`` through the same bundle API, and the framework changes
  since the S16 base are exactly the declared additions of ADR 0025. A lower
  served claim level passes only through the decision owner's signed
  acceptance (``artifacts/gates/S17-ACCEPTANCE.json``, ADR 0031) naming that
  claim level, that G15 gate and that bundle; the check then records the
  computed FAIL beside the acceptance. Undeclared framework changes are never
  accepted;
- ``G17.refusal``: every fixture of both families was refused with its
  expected reason code, through the checker and through ``predict``;
- ``G17.fixture-names``: the source-name scan passes;
- ``G17.docs``: the strict mkdocs build and the link checks pass;
- ``G17.adrs``: the six required ADRs exist and are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from campaigns.history import read_document, resolve_revision
from campaigns.notched_v1.gate_io import (
    check_document,
    gate_document,
    write_gate_document,
)
from piginot.git_environment import isolated_git_environment

REPOSITORY = Path(__file__).resolve().parents[1]
GATE_PATH = Path("artifacts/gates/S17.json")
ACCEPTANCE_PATH = Path("artifacts/gates/S17-ACCEPTANCE.json")
NOTCHED_G15_PATH = Path("artifacts/campaigns/notched-v1/gates/S15.json")
SIGNATORY = "Betim Bahtiri"
S16_BASE = "8a87cd5"
DECLARED_FRAMEWORK_CHANGES = frozenset(
    {
        "src/piginot/bundle/specs.py",
        "src/piginot/morphing/__init__.py",
        "src/piginot/morphing/blended_notch.py",
        "src/piginot/training/__init__.py",
        "src/piginot/training/selection.py",
    }
)
REQUIRED_ADRS = {
    "served-stress authority": "docs/adr/0026-served-stress-authority.md",
    "soft-BC default": "docs/adr/0027-soft-dirichlet-default.md",
    "single architecture in v1": "docs/adr/0028-single-architecture-in-v1.md",
    "preregistration over sealed audit": "docs/adr/0009-s9-preregistration-over-sealed-audit.md",
    "retention tiers": "docs/adr/0003-s3-content-store-retention.md",
    "independence vectors": "docs/adr/0029-independence-vectors.md",
}


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    document: dict[str, Any] = json.loads(path.read_text("utf-8"))
    return document


def _pytest(*tests: str) -> tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return completed.returncode == 0, lines[-1] if lines else ""


def example1(summary_path: Path) -> dict[str, Any]:
    summary = _load(summary_path) or {}
    ci = (REPOSITORY / ".github/workflows/ci.yml").read_text("utf-8")
    measured = {
        "summary_passed": summary.get("passed"),
        "total_seconds": summary.get("timings_seconds", {}).get("total"),
        "held_out_errors": summary.get("held_out_errors"),
        "ci_job_declared": "examples.fixed_mesh_sweep.sweep" in ci,
    }
    passed = (
        summary.get("passed") is True
        and isinstance(measured["total_seconds"], float)
        and measured["total_seconds"] < 600.0
        and measured["ci_job_declared"]
    )
    return check_document(
        "G17.example1",
        passed,
        measured,
        "end to end on CPU in under 600 s; CI job",
        "all",
    )


def examples23() -> dict[str, Any]:
    results = {
        family: _load(REPOSITORY / f"artifacts/benchmarks/REPRODUCTION-{family}.json")
        for family in ("dogbone-v1", "notched-v1")
    }
    measured = {
        family: None
        if result is None
        else {
            "replay_bitwise": result["replay"]["bitwise"],
            "rescore_within_tolerance": result["rescore"]["within_tolerance"],
            "max_relative_difference": max(
                result["rescore"]["relative_difference"].values()
            ),
        }
        for family, result in results.items()
    }
    passed = all(result is not None and result["passed"] for result in results.values())
    return check_document(
        "G17.example2-3",
        passed,
        measured,
        "both benchmarks reproduce from pinned seeds (bitwise replay; TRAIN-screen "
        "metrics within 1e-9)",
        "all",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def acceptance_covers(
    acceptance: dict[str, Any] | None,
    measured: dict[str, Any],
    gate_sha256: str | None,
) -> bool:
    """True only if a signed acceptance names exactly this computed shortfall.

    The acceptance must be signed by the decision owner, name the check, the
    served claim level, the notched G15 gate (LF-normalised SHA-256) and the
    bundle manifest it accepts. A failing G15 and undeclared framework changes
    are never accepted.
    """

    if acceptance is None or gate_sha256 is None:
        return False
    return (
        acceptance.get("check") == "G17.non-dogbone"
        and acceptance.get("signed_off_by") == SIGNATORY
        and measured["notched_g15_verdict"] == "PASS"
        and not measured["undeclared_framework_changes"]
        and acceptance.get("accepted_claim_level")
        == measured["notched_bundle_claim_level"]
        and acceptance.get("notched_g15_sha256") == gate_sha256
        and acceptance.get("bundle_manifest_sha256")
        == measured["notched_bundle_manifest_sha256"]
    )


def non_dogbone() -> dict[str, Any]:
    gate_path = REPOSITORY / NOTCHED_G15_PATH
    gate = _load(gate_path) or {}
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY),
            "diff",
            "--name-only",
            resolve_revision(S16_BASE),
            "HEAD",
            "--",
            "src",
        ],
        env=isolated_git_environment(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    measured: dict[str, Any] = {
        "notched_g15_verdict": gate.get("verdict"),
        "notched_bundle_claim_level": gate.get("claim_level_served"),
        "notched_bundle_manifest_sha256": gate.get("bundle_manifest_sha256"),
        "framework_files_changed_since_s16": sorted(changed),
        "undeclared_framework_changes": sorted(
            set(changed) - DECLARED_FRAMEWORK_CHANGES
        ),
    }
    passed = (
        gate.get("verdict") == "PASS"
        and gate.get("claim_level_served") == "HOLDOUT_VERIFIED"
        and not measured["undeclared_framework_changes"]
    )
    threshold = (
        "the notched bundle passes G15 at HOLDOUT_VERIFIED through the same API; "
        "framework changes are only the declared additions (no family branch)"
    )
    if passed:
        return check_document("G17.non-dogbone", True, measured, threshold, "all")
    acceptance = _load(REPOSITORY / ACCEPTANCE_PATH)
    gate_sha256 = _sha256(gate_path) if gate_path.is_file() else None
    if not acceptance_covers(acceptance, measured, gate_sha256):
        return check_document("G17.non-dogbone", False, measured, threshold, "all")
    assert acceptance is not None
    document = check_document(
        "G17.non-dogbone",
        True,
        {**measured, "computed": "FAIL"},
        threshold,
        "all",
        acceptance=ACCEPTANCE_PATH.as_posix(),
        acceptance_sha256=_sha256(REPOSITORY / ACCEPTANCE_PATH),
    )
    document["reason"] = (
        "FAIL as computed; accepted by the decision owner "
        f"({acceptance['decision_record']}) at claim level "
        f"{acceptance['accepted_claim_level']}"
    )
    return document


def refusal(path: Path) -> dict[str, Any]:
    report = _load(path) or {}
    measured = {
        "cases": report.get("cases"),
        "missed": report.get("missed"),
        "bundles_run": report.get("bundles_run"),
    }
    return check_document(
        "G17.refusal",
        report.get("passed") is True,
        measured,
        "every refusal case of both families returns its expected reason code",
        "all",
    )


def fixture_names() -> dict[str, Any]:
    passed, summary = _pytest("tests/test_no_fixture_names.py")
    return check_document(
        "G17.fixture-names", passed, summary, "no family vocabulary in src/", "eq"
    )


def docs() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as site:
        build = subprocess.run(
            [
                sys.executable,
                "-m",
                "mkdocs",
                "build",
                "--strict",
                "-q",
                "-f",
                "mkdocs.yml",
                "-d",
                site,
            ],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
        )
    links, summary = _pytest("tests/test_docs_links.py")
    measured = {
        "mkdocs_strict_build": build.returncode == 0,
        "link_and_readme_checks": summary,
    }
    return check_document(
        "G17.docs",
        build.returncode == 0 and links,
        measured,
        "mkdocs builds; internal links resolve; limitations linked above the fold",
        "all",
    )


def adrs() -> dict[str, Any]:
    measured = {}
    for topic, relative in REQUIRED_ADRS.items():
        try:
            text = read_document(REPOSITORY, relative)
        except FileNotFoundError:
            text = ""
        measured[topic] = {"path": relative, "accepted": "Status: Accepted" in text}
    return check_document(
        "G17.adrs",
        all(item["accepted"] for item in measured.values()),
        measured,
        "six ADRs written and accepted",
        "all",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m campaigns.g17_gate")
    parser.add_argument("--example1-summary", type=Path, required=True)
    parser.add_argument("--refusal-suite", type=Path, required=True)
    arguments = parser.parse_args(argv)
    checks = [
        example1(arguments.example1_summary),
        examples23(),
        non_dogbone(),
        refusal(arguments.refusal_suite),
        fixture_names(),
        docs(),
        adrs(),
    ]
    document = gate_document("G17", "S17", REPOSITORY, checks)
    document.pop("campaign_id", None)
    write_gate_document(REPOSITORY / GATE_PATH, document)
    print(document["verdict"], document["verdict_reason"])
    return 0 if document["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
