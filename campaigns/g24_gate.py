"""G24: ProblemSpec, liftings and FEM requests from the spec (S24, ADR 0106).

Thresholds from ADR 0106 (accepted 2026-10-13) and
``artifacts/problem/S24-PROTOCOL.json``. Checks: protocol-first,
lifting-exact, lifting-reduces, lifting-verified, regression-dogbone,
regression-notched, spec-hash-stable, spec-validation, adf-viability,
binding-compat, anti-scope, portability (native Windows run) and quality
(Linux run)::

    python -m campaigns.g24_gate --record-suite COVERAGE_JSON PYTEST_LOG
    python -m campaigns.g24_gate --record-windows PYTEST_LOG WINDOWS_CLONE
    python -m campaigns.g24_gate --write

A record command checks that the tree is clean apart from the gate and suite
records of S24 and later (``is_gate_record``), so recording twice needs no
``git restore`` and one suite run can be recorded for G24 and G25.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from campaigns import g19_gate
from campaigns import s24_regression as regression
from campaigns import s24_replay as replay
from campaigns.dogbone_v1 import s24_adf
from campaigns.g19_gate import execution_profile, not_run, read_text_any
from campaigns.g20_gate import (
    _git,
    _pytest,
    _tree,
    parse_pytest_log,
)
from campaigns.notched_v1.gate_io import (
    check_document,
    git_head,
    utc_now,
    write_gate_document,
)

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = Path("artifacts/problem")
GATE_PATH = Path("artifacts/gates/S24.json")
SUITE_PATH = OUTPUT / "S24-SUITE.json"
WINDOWS_PATH = OUTPUT / "S24-WINDOWS-SUITE.json"
GATE_FILES = (GATE_PATH, SUITE_PATH, WINDOWS_PATH)
# The records of the S24 and later gates: a gate document or a suite record.
# They are written after the code tree is fixed, so neither the clean check
# nor the code-tree comparison counts them; one suite run then serves G24
# and G25 together.
_GATE_RECORD = re.compile(
    r"artifacts/(gates/S(2[4-9])\.json|[a-z-]+/S(2[4-9])-(WINDOWS-)?SUITE\.json)\Z"
)


def is_gate_record(path: str) -> bool:
    """A gate document or suite record of S24 or later."""

    return bool(_GATE_RECORD.match(path))


GLOBAL_COVERAGE = 90.5
MODULE_COVERAGE = 90.0
COVERED_MODULES = (
    "src/piginot/lifting/",
    "src/piginot/problem/",
    "src/piginot/contracts/problem_spec.py",
)
ADR_PATH = Path("docs/adr/0106-s24-problem-spec-and-lifting.md")


def _protocol(root: Path) -> tuple[dict[str, Any], str]:
    path = root / s24_adf.PROTOCOL_PATH
    return json.loads(path.read_text("utf-8")), hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _json(root: Path, path: Path) -> dict[str, Any] | None:
    full = root / path
    if not full.is_file():
        return None
    document: dict[str, Any] = json.loads(full.read_text("utf-8"))
    return document


def _ancestor_or_same(root: Path, earlier: str, later: str | None) -> bool:
    return later is not None and (
        _git(root, "merge-base", "--is-ancestor", earlier, later).returncode == 0
    )


def clean(root: Path) -> bool:
    """No change in the tree apart from the gate's own record files."""

    lines = _git(
        root, "status", "--porcelain", "--untracked-files=all"
    ).stdout.splitlines()
    return all(is_gate_record(line[3:].strip('"')) for line in lines)


def check_protocol_first(root: Path) -> dict[str, Any]:
    threshold = (
        "S24-PROTOCOL.json committed once, with ADR 0106 accepted, in a revision "
        "the ADF run and both replays were trained from; S24-REPLAY-AMENDMENT.json "
        "committed once, before the replay records"
    )
    commits = _git(
        root, "log", "--format=%H", "--", s24_adf.PROTOCOL_PATH.as_posix()
    ).stdout.split()
    first = commits[-1] if commits else None
    if first is None:
        return not_run("G24.protocol-first", "protocol not committed", threshold)
    adr = _git(root, "show", f"{first}:{ADR_PATH.as_posix()}").stdout
    screen = _json(root, s24_adf.SCREEN_PATH)
    revisions: dict[str, str | None] = {
        "adf-viability": None if screen is None else screen["implementation_revision"]
    }
    for family in replay.FAMILIES:
        record = _json(root, replay.replay_path(family))
        revisions[family] = (
            None if record is None else record.get("replay_implementation_revision")
        )
    if any(value is None for value in revisions.values()):
        return not_run(
            "G24.protocol-first", "a run record or its revision is missing", threshold
        )
    amendment = _amendment_order(root)
    passed = (
        len(commits) == 1
        and "- Status: Accepted" in adr
        and all(_ancestor_or_same(root, first, rev) for rev in revisions.values())
        and amendment["passed"]
    )
    return check_document(
        "G24.protocol-first",
        passed,
        {
            "protocol_commit": first,
            "protocol_commits": len(commits),
            "runs": revisions,
            "replay_amendment": amendment,
        },
        threshold,
        "all",
    )


def _amendment_order(root: Path) -> dict[str, Any]:
    """Amendment 1 committed once, before every replay record was committed."""

    commits = _git(
        root, "log", "--format=%H", "--", replay.AMENDMENT_PATH.as_posix()
    ).stdout.split()
    first = commits[-1] if commits else None
    records: dict[str, str | None] = {}
    for family in replay.FAMILIES:
        added = _git(
            root,
            "log",
            "--diff-filter=A",
            "--format=%H",
            "--",
            replay.replay_path(family).as_posix(),
        ).stdout.split()
        records[family] = added[-1] if added else None
    passed = (
        first is not None
        and len(commits) == 1
        and all(
            record is not None
            and record != first
            and _ancestor_or_same(root, first, record)
            for record in records.values()
        )
    )
    return {
        "amendment_commit": first,
        "amendment_commits": len(commits),
        "replay_record_commits": records,
        "passed": passed,
    }


def _tests(check_id: str, threshold: str, *targets: str) -> dict[str, Any]:
    result = _pytest(REPOSITORY, *targets)
    return check_document(check_id, result["returncode"] == 0, result, threshold, "all")


def check_lifting_exact() -> dict[str, Any]:
    return _tests(
        "G24.lifting-exact",
        "1,000 random parameter vectors and corrections per kind and family: data "
        "met within 1e-12 of the displacement scale; finite values and gradients",
        "tests/lifting/test_adf.py",
        "tests/campaigns/test_problem_specs.py::"
        "test_both_liftings_meet_the_data_for_a_thousand_parameter_vectors",
    )


def check_lifting_reduces() -> dict[str, Any]:
    return _tests(
        "G24.lifting-reduces",
        "adf with untrimmed lines equals quarter within 1e-12 on 1,000 vectors",
        "tests/lifting/test_adf.py::test_untrimmed_lines_reduce_to_the_quarter_lifting",
        "tests/lifting/test_quarter.py",
    )


def _fresh_regression(root: Path) -> dict[str, dict[str, Any]]:
    fresh = {}
    for campaign in regression.SPECS:
        record = regression.build(campaign)
        record["passed"] = regression.passed(record)
        fresh[campaign] = record
    return fresh


def check_lifting_verified(
    committed: Mapping[str, Any], fresh: Mapping[str, Any]
) -> dict[str, Any]:
    measured = {
        campaign: {
            kind: {
                "committed": committed[campaign]["lifting"][kind]["passed"],
                "recomputed": fresh[campaign]["lifting"][kind]["passed"],
                "parameter_vectors": fresh[campaign]["lifting"][kind][
                    "parameter_vectors"
                ],
            }
            for kind in ("quarter", "adf")
        }
        for campaign in regression.SPECS
    }
    passed = all(
        entry["committed"] and entry["recomputed"]
        for kinds in measured.values()
        for entry in kinds.values()
    )
    return check_document(
        "G24.lifting-verified",
        passed,
        measured,
        "node and piece residuals <= 1e-12; conformity and orientation <= 1e-9; "
        "every mutant misses by > 1e-3; committed and recomputed",
        "all",
    )


def check_regression(
    root: Path,
    campaign: str,
    committed: Mapping[str, Any],
    fresh: Mapping[str, Any],
) -> dict[str, Any]:
    check_id = f"G24.regression-{campaign.split('-')[0]}"
    threshold = (
        "spec equals the bundle; lifting and frame bit-identical and FEM requests "
        "byte-identical on 64 of 64 DEVELOPMENT cases; 400-step replay tensors "
        "byte-identical to a v0.2.0-base run of the same recipe on the same CPU, "
        "run bindings equal except the revision (ADR 0106 amendment 1)"
    )
    record = _json(root, replay.replay_path(campaign))
    if record is None:
        return not_run(check_id, f"{campaign} replay not recorded", threshold)
    development = fresh[campaign]["development"]
    measured = {
        "envelope": fresh[campaign]["envelope"],
        "development": development,
        "committed_development": committed[campaign]["development"],
        "replay": {
            key: record[key]
            for key in (
                "passed",
                "reference_revision",
                "reference_is_base_revision",
                "bindings_equal_except_revision",
                "reference_matches_amendment",
                "manifest_differences",
                "model_state_hash",
                "registered_profile_comparison",
            )
        },
    }
    cases = development["cases"]
    passed = (
        cases == 64
        and all(fresh[campaign]["envelope"].values())
        and development == committed[campaign]["development"]
        and all(
            development[key] == cases
            for key in (
                "lifting_bit_identical",
                "frame_bit_identical",
                "fem_requests_byte_identical",
            )
        )
        and record["passed"] is True
        and record["reference_is_base_revision"] is True
        and record["bindings_equal_except_revision"] is True
        and record["reference_matches_amendment"] is True
    )
    return check_document(check_id, passed, measured, threshold, "all")


def check_spec_hash_stable() -> dict[str, Any]:
    return _tests(
        "G24.spec-hash-stable",
        "hash unchanged under key order, whitespace, number and expression "
        "spelling and Unicode encoding; changed by every semantic leaf edit",
        "tests/contracts/test_problem_spec.py::test_hash_is_stable_under_formatting",
        "tests/contracts/test_problem_spec.py::"
        "test_every_semantic_leaf_edit_changes_the_hash",
    )


def check_spec_validation() -> dict[str, Any]:
    return _tests(
        "G24.spec-validation",
        "every rejection rule of ADR 0106 decision 2 has a failing-spec test",
        "tests/contracts/test_problem_spec.py",
    )


def check_adf_viability(root: Path, digest: str) -> dict[str, Any]:
    threshold = (
        "registered ADF run passes the frozen S13 guardrails on TRAIN-screen; "
        "P-field L2 reported against S13c"
    )
    record = _json(root, s24_adf.SCREEN_PATH)
    if record is None:
        return not_run("G24.adf-viability", "ADF run not evaluated", threshold)
    passed = (
        record["passed"] is True
        and record["protocol_sha256"] == digest
        and record["problem_spec_hash"] == s24_adf.load_spec().problem_hash
        and record["checkpoint_step"] == 2400
        and not record["failed_guardrails"]
        and record["development_opened"] is False
    )
    return check_document("G24.adf-viability", passed, record, threshold, "all")


def check_binding_compat(root: Path) -> dict[str, Any]:
    from piginot.bundle import load_bundle
    from piginot.cli import parse_request
    from piginot.serving import serve

    tests = _pytest(
        root,
        "tests/training/test_identity.py",
        "tests/training/test_checkpoint.py",
        "tests/test_shipped_bundles.py",
    )
    goldens = json.loads((root / g19_gate.GOLDEN_PATH).read_text("utf-8"))["profiles"]
    key = g19_gate.profile_key(execution_profile())
    if key not in goldens:
        return not_run(
            "G24.binding-compat", f"no S19 golden for profile {key}", "S19 goldens"
        )
    measured: dict[str, bool] = {}
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
        "G24.binding-compat",
        tests["returncode"] == 0 and all(measured.values()),
        {"tests": tests, "fast_goldens": measured},
        "bindings without a spec keep their canonical form; checkpoint and bundle "
        "tests pass; fast documents equal the S19 goldens",
        "all",
    )


def check_anti_scope(root: Path) -> dict[str, Any]:
    tests = _pytest(
        root,
        "tests/test_s24_anti_scope.py",
        "tests/test_no_fixture_names.py",
        "tests/test_s15_anti_scope.py",
    )
    suite = _json(root, SUITE_PATH)
    modules = None if suite is None else suite.get("module_coverage_percent")
    passed = (
        tests["returncode"] == 0
        and modules is not None
        and all(value >= MODULE_COVERAGE for value in modules.values())
    )
    return check_document(
        "G24.anti-scope",
        passed,
        {"tests": tests, "module_coverage_percent": modules},
        "no family token in src/; .[runtime] FEM-free; no eval of formulas; "
        "lifting, problem and ProblemSpec branch coverage >= 90% each",
        "all",
    )


def _listing(root: Path, tree: str) -> dict[str, str] | None:
    listed = _git(root, "ls-tree", "-r", tree)
    if listed.returncode != 0:
        return None
    entries = {}
    for line in listed.stdout.splitlines():
        meta, path = line.split("\t", 1)
        if not is_gate_record(path):
            entries[path] = meta
    return entries


def _same_code(root: Path, tree: str) -> bool:
    recorded = _listing(root, tree)
    return recorded is not None and recorded == _listing(root, "HEAD")


def _module_coverage(report: Mapping[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for prefix in COVERED_MODULES:
        totals = [
            f["summary"]
            for name, f in report.get("files", {}).items()
            if name.replace("\\", "/").startswith(prefix)
        ]
        statements = sum(t["num_statements"] + t.get("num_branches", 0) for t in totals)
        covered = sum(t["covered_lines"] + t.get("covered_branches", 0) for t in totals)
        result[prefix] = round(100.0 * covered / statements, 2) if statements else None
    return result


def record_suite(root: Path, coverage_json: Path, pytest_log: Path) -> dict[str, Any]:
    report = json.loads(read_text_any(coverage_json))
    document = {
        "schema_version": "1.0",
        "tree": _tree(root),
        "commit": git_head(root),
        "clean": clean(root),
        "branch_coverage": bool(report.get("meta", {}).get("branch_coverage")),
        "coverage_percent": round(float(report["totals"]["percent_covered"]), 2),
        "module_coverage_percent": _module_coverage(report),
        **parse_pytest_log(read_text_any(pytest_log)),
        "execution_profile": execution_profile(),
    }
    (root / SUITE_PATH).write_text(json.dumps(document, indent=2) + "\n", "utf-8")
    return document


def record_windows(root: Path, pytest_log: Path, clone: Path) -> dict[str, Any]:
    document = {
        "schema_version": "1.0",
        "tree": _tree(clone),
        "clean": clean(clone),
        **parse_pytest_log(read_text_any(pytest_log)),
    }
    (root / WINDOWS_PATH).write_text(json.dumps(document, indent=2) + "\n", "utf-8")
    return document


def check_portability(root: Path) -> dict[str, Any]:
    threshold = (
        "native Windows run of this code tree: 0 failed, 0 errors, "
        "every skip has a reason"
    )
    run = _json(root, WINDOWS_PATH)
    if run is None:
        return not_run("G24.portability", f"{WINDOWS_PATH} not recorded", threshold)
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
    return check_document("G24.portability", passed, run, threshold, "all")


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
    }
    suite = _json(root, SUITE_PATH)
    if suite is None:
        return not_run("G24.quality", f"{SUITE_PATH} not recorded", threshold)
    counts = suite["counts"]
    passed = (
        all(code == 0 for code in codes.values())
        and _same_code(root, suite["tree"])
        and suite["clean"]
        and suite["branch_coverage"]
        and counts.get("failed", 0) == 0
        and counts.get("errors", 0) == 0
        and suite["coverage_percent"] >= GLOBAL_COVERAGE
    )
    return check_document(
        "G24.quality", passed, {"tools": codes, "suite": suite}, threshold, "all"
    )


def build_gate(root: Path) -> dict[str, Any]:
    _, digest = _protocol(root)
    committed = json.loads((root / regression.EVIDENCE).read_text("utf-8"))["families"]
    fresh = _fresh_regression(root)
    checks = [
        check_protocol_first(root),
        check_lifting_exact(),
        check_lifting_reduces(),
        check_lifting_verified(committed, fresh),
        check_regression(root, "dogbone-v1", committed, fresh),
        check_regression(root, "notched-v1", committed, fresh),
        check_spec_hash_stable(),
        check_spec_validation(),
        check_adf_viability(root, digest),
        check_binding_compat(root),
        check_anti_scope(root),
        check_portability(root),
        check_quality(root),
    ]
    blockers = [c["id"] for c in checks if c["status"] != "PASS"]
    return {
        "gate_id": "G24",
        "step": "S24",
        "commit": git_head(root),
        "timestamp_utc": utc_now(),
        "execution_profile": execution_profile(),
        "checks": checks,
        "decision_record": ADR_PATH.as_posix(),
        "verdict": "FAIL" if blockers else "PASS",
        "verdict_reason": f"{len(blockers)} required check(s) not PASS: "
        f"{', '.join(blockers)}"
        if blockers
        else "every required check PASS",
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate gate G24.")
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
    print(f"G24: {document['verdict']} ({document['verdict_reason']})")
    return 0 if document["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ("GATE_PATH", "build_gate", "clean", "main")
