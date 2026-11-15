"""G20: evaluation harness, notched post-mortem, fresh sealed holdouts (S20).

Checks (``STEPS-v2.0.md`` §3, ADR 0101):

- ``G20.protocol-first``: the post-mortem protocol was committed, unchanged
  since, before every post-mortem artifact;
- ``G20.harness``: the recorded harness report rebuilds exactly and says every
  S14 budget reproduces; the harness tests (with mutants) pass;
- ``G20.burned``: the burned registries rebuild exactly (64 IDs per family)
  and the guard tests (with the unguarded mutant) pass;
- ``G20.holdouts``: per family, H1 and H2 hold 64 fresh case IDs each,
  disjoint from each other and from every used case, with exact-64
  commitments bound to the set and the committed envelope, drawn after the
  ADR 0101 commit;
- ``G20.rca``: the recorded verdicts recompute, every hypothesis is
  SUPPORTED, NOT_SUPPORTED or NOT_RUN, D7 and D8 are present, and ADR 0101
  records the result;
- ``G20.portability``: a recorded native Windows run of this tree has no
  failure and every skip carries a reason;
- ``G20.binding``: the run-binding v2 tests pass (v1 canonical form pinned);
- ``G20.quality``: ruff, ruff format and mypy are clean, and a recorded full
  Linux run of this tree has no failure, global branch coverage >= 90.5% and
  ``piginot.evaluation`` >= 90%.

Usage::

    python -m campaigns.g20_gate --record-suite COVERAGE_JSON PYTEST_LOG
    python -m campaigns.g20_gate --record-windows PYTEST_LOG WINDOWS_CLONE
    python -m campaigns.g20_gate --write
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from campaigns import fresh_holdouts as fh
from campaigns import s20_evaluation
from campaigns.g19_gate import execution_profile, not_run, read_text_any
from campaigns.history import read_document
from campaigns.notched_v1 import postmortem, postmortem_owner
from campaigns.notched_v1.gate_io import (
    check_document,
    git_head,
    utc_now,
    write_gate_document,
)
from piginot.contracts import hash_spec
from piginot.git_environment import isolated_git_environment

REPOSITORY = Path(__file__).resolve().parents[1]
GATE_PATH = Path("artifacts/gates/S20.json")
SUITE_PATH = Path("artifacts/evaluation/S20-SUITE.json")
WINDOWS_PATH = Path("artifacts/evaluation/S20-WINDOWS-SUITE.json")
ADR_PATH = Path("docs/adr/0101-s20-fresh-holdouts-and-post-mortem.md")
POSTMORTEM_ARTIFACTS = (
    postmortem.OUTPUT_PATH,
    postmortem_owner.D7_PATH,
    postmortem_owner.D8_PATH,
    postmortem_owner.VERDICTS_PATH,
)
VERDICTS = frozenset({"SUPPORTED", "NOT_SUPPORTED", "NOT_RUN"})
GLOBAL_COVERAGE = 90.5
EVALUATION_COVERAGE = 90.0
GATE_FILES = (GATE_PATH, SUITE_PATH, WINDOWS_PATH)


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(root), *arguments],
        env=isolated_git_environment(),
        capture_output=True,
        text=True,
        check=False,
    )


def _added_in(root: Path, path: Path) -> str | None:
    commits = _git(root, "log", "--format=%H", "--diff-filter=A", "--", path.as_posix())
    lines = commits.stdout.split()
    return lines[-1] if lines else None


def _strictly_before(root: Path, earlier: str, later: str) -> bool:
    return earlier != later and (
        _git(root, "merge-base", "--is-ancestor", earlier, later).returncode == 0
    )


def _pytest(root: Path, *targets: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = (completed.stdout.strip().splitlines() or [""])[-1]
    return {
        "targets": list(targets),
        "returncode": completed.returncode,
        "summary": tail,
    }


def _tree(root: Path) -> str:
    return _git(root, "rev-parse", "HEAD^{tree}").stdout.strip()


def check_protocol_first(root: Path) -> dict[str, Any]:
    protocol = postmortem.PROTOCOL_PATH
    commits = _git(root, "log", "--format=%H", "--", protocol.as_posix()).stdout.split()
    first = commits[-1] if commits else None
    order = {path.as_posix(): _added_in(root, path) for path in POSTMORTEM_ARTIFACTS}
    passed = (
        first is not None
        and len(commits) == 1
        and all(
            c is not None and _strictly_before(root, first, c) for c in order.values()
        )
    )
    return check_document(
        "G20.protocol-first",
        passed,
        {
            "protocol_commit": first,
            "protocol_commits": len(commits),
            "artifacts": order,
        },
        "protocol committed once, before every post-mortem artifact",
        "all",
    )


def check_harness(root: Path) -> dict[str, Any]:
    path = root / s20_evaluation.OUTPUT / "S20-HARNESS.json"
    committed = json.loads(path.read_text("utf-8"))
    rebuilt = json.loads(json.dumps(s20_evaluation.harness_report(root)))
    tests = _pytest(root, "tests/evaluation/test_budget_records.py")
    passed = (
        committed == rebuilt and committed["reproduced"] and tests["returncode"] == 0
    )
    return check_document(
        "G20.harness",
        passed,
        {
            "rebuild_equals_record": committed == rebuilt,
            "records": [
                (r["campaign_id"], r["split"], r["mismatches"])
                for r in committed["records"]
            ],
            "tests": tests,
        },
        "every S14 budget row reproduces exactly; mutant tests pass",
        "all",
    )


def check_burned(root: Path) -> dict[str, Any]:
    rebuilt = s20_evaluation.burned_registries(root)
    committed = {
        campaign: json.loads(
            (
                root / s20_evaluation.OUTPUT / f"BURNED-HOLDOUT-{campaign}.json"
            ).read_text("utf-8")
        )
        for campaign in rebuilt
    }
    tests = _pytest(root, "tests/evaluation/test_burned.py")
    sizes = {
        campaign: len(document["case_ids"]) for campaign, document in committed.items()
    }
    passed = (
        committed == rebuilt
        and set(sizes.values()) == {64}
        and tests["returncode"] == 0
    )
    return check_document(
        "G20.burned",
        passed,
        {
            "registry_sizes": sizes,
            "rebuild_equals_record": committed == rebuilt,
            "tests": tests,
        },
        "burned registries rebuild exactly; guard and mutant tests pass",
        "all",
    )


def _holdout_family(
    root: Path, campaign: str, decision_commit: str | None
) -> dict[str, Any]:
    adapter = fh._adapters()[campaign]
    path = root / adapter.artifacts / fh.PUBLIC_NAME
    if not path.is_file():
        return {"present": False, "passed": False}
    public = json.loads(path.read_text("utf-8"))
    manifest = json.loads(
        (root / adapter.artifacts / "public-case-manifest.json").read_text("utf-8")
    )
    burned = set(
        json.loads(
            (
                root / s20_evaluation.OUTPUT / f"BURNED-HOLDOUT-{campaign}.json"
            ).read_text("utf-8")
        )["case_ids"]
    )
    used = {case["case_id"] for case in manifest["cases"]} | burned
    sets = public["sets"]
    ids = {name: sets[name]["case_ids"] for name in fh.SETS}
    envelope_hash = public["executable_envelope_hash"]
    bound = all(
        sets[name]["commitment"]["campaign_hash"]
        == fh.set_campaign_hash(campaign, envelope_hash, name)
        and sets[name]["commitment"]["n_holdout_points"] == fh.SET_SIZE
        for name in fh.SETS
    )
    envelope = hash_spec(adapter.envelope(), "content") == envelope_hash
    added = _added_in(root, adapter.artifacts / fh.PUBLIC_NAME)
    ordered = (
        decision_commit is not None
        and added is not None
        and _strictly_before(root, decision_commit, added)
    )
    sizes = {name: len(set(values)) for name, values in ids.items()}
    disjoint = not set(ids["H1"]) & set(ids["H2"])
    fresh = not used & (set(ids["H1"]) | set(ids["H2"]))
    passed = (
        public["protocol_id"] == fh.PROTOCOL_ID
        and sizes == {"H1": 64, "H2": 64}
        and disjoint
        and fresh
        and bound
        and envelope
        and ordered
        and "coordinates" not in json.dumps(public)
    )
    return {
        "present": True,
        "passed": passed,
        "sizes": sizes,
        "disjoint": disjoint,
        "fresh": fresh,
        "commitments_bound": bound,
        "envelope_matches": envelope,
        "drawn_after_decision": ordered,
        "rejected_near_duplicates": public["rejected_near_duplicates"],
    }


def check_holdouts(root: Path) -> dict[str, Any]:
    decision = _added_in(root, ADR_PATH)
    families = {
        c: _holdout_family(root, c, decision) for c in ("dogbone-v1", "notched-v1")
    }
    if not all(family["present"] for family in families.values()):
        return not_run(
            "G20.holdouts",
            "FRESH-HOLDOUTS.json missing for a family (run campaigns.fresh_holdouts)",
            "64 + 64 fresh sealed cases per family",
        )
    return check_document(
        "G20.holdouts",
        all(family["passed"] for family in families.values()),
        {"decision_commit": decision, "families": families},
        "per family: H1 and H2 of 64 fresh, disjoint, bound seals, drawn after ADR 0101",
        "all",
    )


def check_rca(root: Path) -> dict[str, Any]:
    path = root / postmortem_owner.VERDICTS_PATH
    owner_inputs = all(
        (root / p).is_file()
        for p in (postmortem_owner.D7_PATH, postmortem_owner.D8_PATH)
    )
    if not path.is_file() or not owner_inputs:
        return not_run(
            "G20.rca",
            "D7, D8 or the combined verdicts are not recorded (owner machine)",
            "every hypothesis judged; verdicts recompute; ADR 0101 records the result",
        )
    committed = json.loads(path.read_text("utf-8"))
    rebuilt = json.loads(json.dumps(postmortem_owner.verdicts(root)))
    verdicts = {key: entry["verdict"] for key, entry in committed["hypotheses"].items()}
    try:
        decision = read_document(root, ADR_PATH)
    except FileNotFoundError:
        return not_run(
            "G20.rca",
            "ADR 0101 is unavailable locally and in Git history",
            "every hypothesis judged; verdicts recompute; ADR 0101 records the result",
        )
    section = decision.split("## Post-mortem result", 1)[-1]
    section = section.split("\n## ", 1)[0]
    recorded = re.search(r"\bpending\b", section, re.IGNORECASE) is None
    passed = (
        committed == rebuilt
        and set(verdicts) == {"a", "b", "c", "d", "e"}
        and set(verdicts.values()) <= VERDICTS
        and recorded
    )
    return check_document(
        "G20.rca",
        passed,
        {
            "verdicts": verdicts,
            "rebuild_equals_record": committed == rebuilt,
            "adr_recorded": recorded,
        },
        "every hypothesis judged; verdicts recompute; ADR 0101 records the result",
        "all",
    )


_SUMMARY = re.compile(
    r"(\d+) (passed|failed|skipped|errors?|deselected|xfailed|xpassed)"
)
_SKIP = re.compile(
    r"^SKIPPED \[(?P<count>\d+)\] (?P<where>[^:]+(?::\d+)?): (?P<reason>.*)$"
)


def parse_pytest_log(text: str) -> dict[str, Any]:
    """Counts of the final summary line and the ``-rs`` skip reasons."""

    summary = ""
    for line in reversed(text.splitlines()):
        if _SUMMARY.search(line) and (" in " in line or "passed" in line):
            summary = line.strip("= ").strip()
            break
    counts: dict[str, int] = {}
    for value, key in _SUMMARY.findall(summary):
        counts["errors" if key.startswith("error") else key] = int(value)
    skips = [
        {**m.groupdict(), "count": int(m["count"])}
        for line in text.splitlines()
        if (m := _SKIP.match(line.strip()))
    ]
    return {"summary": summary, "counts": counts, "skips": skips}


def record_suite(root: Path, coverage_json: Path, pytest_log: Path) -> dict[str, Any]:
    report = json.loads(read_text_any(coverage_json))
    files = report.get("files", {})

    def package(prefix: str) -> float | None:
        totals = [
            f["summary"]
            for name, f in files.items()
            if name.replace("\\", "/").startswith(prefix)
        ]
        statements = sum(t["num_statements"] + t.get("num_branches", 0) for t in totals)
        covered = sum(t["covered_lines"] + t.get("covered_branches", 0) for t in totals)
        return round(100.0 * covered / statements, 2) if statements else None

    parsed = parse_pytest_log(read_text_any(pytest_log))
    document = {
        "schema_version": "1.0",
        "tree": _tree(root),
        "commit": git_head(root),
        "clean": _git(root, "status", "--porcelain").stdout.strip() == "",
        "branch_coverage": bool(report.get("meta", {}).get("branch_coverage")),
        "coverage_percent": round(float(report["totals"]["percent_covered"]), 2),
        "evaluation_coverage_percent": package("src/piginot/evaluation/"),
        **parsed,
        "execution_profile": execution_profile(),
    }
    (root / SUITE_PATH).write_text(json.dumps(document, indent=2) + "\n", "utf-8")
    return document


def record_windows(root: Path, pytest_log: Path, clone: Path) -> dict[str, Any]:
    parsed = parse_pytest_log(read_text_any(pytest_log))
    document = {
        "schema_version": "1.0",
        "tree": _tree(clone),
        "clean": _git(clone, "status", "--porcelain").stdout.strip() == "",
        **parsed,
    }
    (root / WINDOWS_PATH).write_text(json.dumps(document, indent=2) + "\n", "utf-8")
    return document


def _listing(root: Path, tree: str) -> dict[str, str] | None:
    """``path -> object`` of a tree, without the gate's own record files."""

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
    """Whether ``tree`` is HEAD's tree apart from the gate's record files."""

    recorded = _listing(root, tree)
    return recorded is not None and recorded == _listing(root, "HEAD")


def check_portability(root: Path) -> dict[str, Any]:
    path = root / WINDOWS_PATH
    threshold = (
        "native Windows run of this tree: 0 failed, 0 errors, every skip has a reason"
    )
    if not path.is_file():
        return not_run("G20.portability", f"{WINDOWS_PATH} not recorded", threshold)
    run = json.loads(path.read_text("utf-8"))
    reasons = all(skip["reason"].strip() for skip in run["skips"])
    counts = run["counts"]
    passed = (
        _same_code(root, run["tree"])
        and run["clean"]
        and counts.get("failed", 0) == 0
        and counts.get("errors", 0) == 0
        and counts.get("passed", 0) > 3000
        and reasons
        and sum(skip["count"] for skip in run["skips"]) == counts.get("skipped", 0)
    )
    return check_document("G20.portability", passed, run, threshold, "all")


def check_binding(root: Path) -> dict[str, Any]:
    tests = _pytest(
        root, "tests/contracts/test_profile.py", "tests/training/test_identity.py"
    )
    return check_document(
        "G20.binding",
        tests["returncode"] == 0,
        tests,
        "v1 profile canonical form pinned; platform fingerprint enters the identity",
        "all",
    )


def check_quality(root: Path) -> dict[str, Any]:
    python = sys.executable
    tools = {
        "ruff_check": [python, "-m", "ruff", "check", "."],
        "ruff_format": [python, "-m", "ruff", "format", "--check", "."],
        "mypy_strict": [python, "-m", "mypy", "--strict", "src/"],
    }
    results = {}
    for name, command in tools.items():
        completed = subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=False
        )
        results[name] = completed.returncode
    suite_path = root / SUITE_PATH
    suite = json.loads(suite_path.read_text("utf-8")) if suite_path.is_file() else None
    threshold = {
        "tools": "exit 0",
        "suite": "recorded Linux run of this tree, clean, 0 failed, 0 errors",
        "global_branch_coverage_at_least": GLOBAL_COVERAGE,
        "piginot.evaluation_at_least": EVALUATION_COVERAGE,
    }
    if suite is None:
        return not_run("G20.quality", f"{SUITE_PATH} not recorded", threshold)
    counts = suite["counts"]
    passed = (
        all(code == 0 for code in results.values())
        and _same_code(root, suite["tree"])
        and suite["clean"]
        and suite["branch_coverage"]
        and counts.get("failed", 0) == 0
        and counts.get("errors", 0) == 0
        and suite["coverage_percent"] >= GLOBAL_COVERAGE
        and (suite["evaluation_coverage_percent"] or 0.0) >= EVALUATION_COVERAGE
    )
    return check_document(
        "G20.quality", passed, {"tools": results, "suite": suite}, threshold, "all"
    )


def build_gate(root: Path) -> dict[str, Any]:
    checks = [
        check_protocol_first(root),
        check_harness(root),
        check_burned(root),
        check_holdouts(root),
        check_rca(root),
        check_portability(root),
        check_binding(root),
        check_quality(root),
    ]
    blockers = [c["id"] for c in checks if c["status"] != "PASS"]
    return {
        "gate_id": "G20",
        "step": "S20",
        "commit": git_head(root),
        "timestamp_utc": utc_now(),
        "execution_profile": execution_profile(),
        "checks": checks,
        "decision_record": ADR_PATH.as_posix(),
        "verdict": "FAIL" if blockers else "PASS",
        "verdict_reason": (
            f"{len(blockers)} required check(s) not PASS: {', '.join(blockers)}"
            if blockers
            else "every required check PASS"
        ),
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate gate G20.")
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--record-suite", nargs=2, type=Path, metavar=("COVERAGE_JSON", "LOG")
    )
    parser.add_argument(
        "--record-windows", nargs=2, type=Path, metavar=("LOG", "CLONE")
    )
    options = parser.parse_args(arguments)
    if options.record_suite:
        print(
            json.dumps(record_suite(REPOSITORY, *options.record_suite), indent=2)[:2000]
        )
        return 0
    if options.record_windows:
        print(
            json.dumps(record_windows(REPOSITORY, *options.record_windows), indent=2)[
                :2000
            ]
        )
        return 0
    document = build_gate(REPOSITORY)
    if options.write:
        write_gate_document(REPOSITORY / GATE_PATH, document)
    for check in document["checks"]:
        print(f"{check['id']}: {check['status']}")
    print(f"G20: {document['verdict']} ({document['verdict_reason']})")
    return 0 if document["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
