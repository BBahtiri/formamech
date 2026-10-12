"""G22: verified serving on DEVELOPMENT (S22, ADR 0103).

Thresholds from ``artifacts/serving/S22-PROTOCOL.json``. Checks:
protocol-first, reference-machine, recompute, integrity (recipes and cold
solves), never-silent, equivalence, speedup, correction, fast-unchanged (S19
goldens of this profile), light-install and old-bundles (their tests),
determinism (re-served under the protocol's conditions) and quality::

    python -m campaigns.g22_gate --record-suite COVERAGE_JSON PYTEST_LOG
    python -m campaigns.g22_gate --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from campaigns import g19_gate, s22_recipes
from campaigns import s22_serving as s22
from campaigns.g19_gate import execution_profile, not_run, read_text_any
from campaigns.g20_gate import (
    _added_in,
    _git,
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

REPOSITORY = Path(__file__).resolve().parents[1]
GATE_PATH = Path("artifacts/gates/S22.json")
SUITE_PATH = s22.OUTPUT / "S22-SUITE.json"
EVIDENCE_FILES = (GATE_PATH, SUITE_PATH, *(s22.results_path(c) for c in s22.FAMILIES))
GLOBAL_COVERAGE = 90.5
SERVING_COVERAGE = 90.0


def _results(root: Path) -> dict[str, dict[str, Any]] | None:
    paths = {c: root / s22.results_path(c) for c in s22.FAMILIES}
    if not all(p.is_file() for p in paths.values()):
        return None
    return {c: json.loads(p.read_text("utf-8")) for c, p in paths.items()}


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


def check_protocol_first(root: Path) -> dict[str, Any]:
    commits = _git(
        root, "log", "--format=%H", "--", s22.PROTOCOL_PATH.as_posix()
    ).stdout.split()
    first = commits[-1] if commits else None
    order = {c: _added_in(root, s22.results_path(c)) for c in s22.FAMILIES}
    passed = (
        first is not None
        and len(commits) == 1
        and all(
            c is not None and _strictly_before(root, first, c) for c in order.values()
        )
    )
    return check_document(
        "G22.protocol-first",
        passed,
        {"protocol_commit": first, "protocol_commits": len(commits), "results": order},
        "protocol committed once, before both results files",
        "all",
    )


def family_checks(
    protocol: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rules = protocol["thresholds"]
    recomputed = {c: s22.summarize(d["records"]) for c, d in results.items()}
    ordered = {
        c: [r["case_id"] for r in d["records"]] == protocol["families"][c]["case_ids"]
        for c, d in results.items()
    }
    recipes = json.loads((REPOSITORY / s22_recipes.EVIDENCE).read_text("utf-8"))[
        "families"
    ]
    checks = [
        check_document(
            "G22.reference-machine",
            all(
                d["platform_fingerprint"] == protocol["reference_machine"]
                for d in results.values()
            ),
            {c: d["platform_fingerprint"] for c, d in results.items()},
            protocol["reference_machine"],
            "all",
        ),
        check_document(
            "G22.recompute",
            all(ordered.values())
            and all(results[c]["summary"] == recomputed[c] for c in results),
            {
                "cases_in_protocol_order": ordered,
                "summary_recomputes": {
                    c: results[c]["summary"] == recomputed[c] for c in results
                },
            },
            "records cover the protocol's cases in order; summaries recompute exactly",
            "all",
        ),
    ]

    def per_family(
        check_id: str, test: Any, fields: Sequence[str], threshold: object
    ) -> None:
        verdicts = {c: bool(test(c, s)) for c, s in recomputed.items()}
        checks.append(
            check_document(
                check_id,
                all(verdicts.values()),
                {
                    c: {f: s[f] for f in fields} | {"pass": verdicts[c]}
                    for c, s in recomputed.items()
                },
                threshold,
                "all",
            )
        )

    per_family(
        "G22.integrity",
        lambda c, s: (
            s["cold_solved"] == s["cases"] == 64
            and recipes[c]["reproduced_byte_for_byte"] == recipes[c]["cases"] == 64
        ),
        ("cases", "cold_solved"),
        protocol["integrity"],
    )
    per_family(
        "G22.never-silent",
        lambda c, s: (
            s["silent_results"] == 0 and sum(s["paths"].values()) == s["cold_solved"]
        ),
        ("paths", "silent_results"),
        rules["never_silent"],
    )
    per_family(
        "G22.equivalence",
        lambda c, s: (
            s["equivalence_cases"] == s["cold_solved"]
            and s["equivalence_max_relative"] <= rules["equivalence_relative_at_most"]
        ),
        ("equivalence_cases", "equivalence_max_relative"),
        {"equivalence_relative_at_most": rules["equivalence_relative_at_most"]},
    )
    per_family(
        "G22.speedup",
        lambda c, s: s["median_speedup"] >= rules["median_speedup_at_least"],
        ("median_speedup", "min_speedup"),
        {"median_speedup_at_least": rules["median_speedup_at_least"]},
    )
    per_family(
        "G22.correction",
        lambda c, s: (
            s["median_corrected_peak_error"] is not None
            and s["corrected_cases"] == s["cold_solved"]
            and s["median_corrected_peak_error"]
            <= rules["corrected_over_operator_median_peak_error_at_most"]
            * s["median_operator_peak_error"]
        ),
        (
            "median_operator_peak_error",
            "median_corrected_peak_error",
            "median_served_peak_error",
        ),
        {
            "corrected_over_operator_median_peak_error_at_most": rules[
                "corrected_over_operator_median_peak_error_at_most"
            ]
        },
    )
    return checks


def check_fast_unchanged(root: Path) -> dict[str, Any]:
    """serve(mode='fast') documents equal the S19 golden digests of this profile."""

    from piginot.bundle import load_bundle
    from piginot.cli import parse_request
    from piginot.serving import serve

    goldens = json.loads((root / g19_gate.GOLDEN_PATH).read_text("utf-8"))["profiles"]
    key = g19_gate.profile_key(execution_profile())
    if key not in goldens:
        return not_run(
            "G22.fast-unchanged", f"no S19 golden for profile {key}", "S19 goldens"
        )
    measured = {}
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
        "G22.fast-unchanged",
        all(measured.values()),
        measured,
        "equal to the S19 goldens",
        "all",
    )


def check_tests(root: Path) -> list[dict[str, Any]]:
    light = _pytest(
        root,
        "tests/serving/test_hybrid.py::test_the_light_install_serves_fast_and_explains_the_rest",
    )
    old = _pytest(root, "tests/test_shipped_bundles.py", "tests/serving/test_recipe.py")
    return [
        check_document(
            "G22.light-install", light["returncode"] == 0, light, "test passes", "all"
        ),
        check_document(
            "G22.old-bundles", old["returncode"] == 0, old, "tests pass", "all"
        ),
    ]


def reserve(campaign: str, case_id: str) -> dict[str, str | None]:
    """Re-serve one case under the protocol's conditions (one torch thread)."""

    import torch

    from piginot.serving import serve

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        bundle = s22.s21._bundle(s22.s21.runtime(campaign).bundle)
        recipe = s22.recipe_for(campaign)
        _, inputs = s22_recipes.case_inputs(campaign, case_id)
        exact = serve(bundle, inputs, mode="exact", recipe=recipe, request_id=case_id)
        certified = serve(
            bundle,
            inputs,
            mode="certified",
            recipe=recipe,
            corrected=True,
            request_id=case_id,
        )
    finally:
        torch.set_num_threads(threads)
    return {
        "exact_result_hash": exact.exact_result_hash,
        "certified_document_sha256": hashlib.sha256(
            json.dumps(certified.document(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def check_determinism(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    from piginot.training.platform_fingerprint import platform_fingerprint

    here = platform_fingerprint()
    if any(d["platform_fingerprint"] != here for d in results.values()):
        return not_run(
            "G22.determinism", f"platform {here} differs", "re-served results equal"
        )
    measured = {}
    for campaign, document in results.items():
        record = document["records"][0]
        again = reserve(campaign, record["case_id"])
        measured[campaign] = {
            "case_id": record["case_id"],
            "exact_equal": again["exact_result_hash"] == record["exact_result_hash"],
            "certified_equal": again["certified_document_sha256"]
            == record["certified_document_sha256"],
        }
    return check_document(
        "G22.determinism",
        all(m["exact_equal"] and m["certified_equal"] for m in measured.values()),
        measured,
        "exact result hash and certified document re-served under the protocol's conditions are equal",
        "all",
    )


def _listing(root: Path, tree: str) -> dict[str, str] | None:
    listed = _git(root, "ls-tree", "-r", tree)
    if listed.returncode != 0:
        return None
    excluded = {p.as_posix() for p in EVIDENCE_FILES}
    return {
        line.split("\t", 1)[1]: line.split("\t", 1)[0]
        for line in listed.stdout.splitlines()
        if line.split("\t", 1)[1] not in excluded
    }


def record_suite(root: Path, coverage_json: Path, pytest_log: Path) -> dict[str, Any]:
    report = json.loads(read_text_any(coverage_json))
    totals = [
        f["summary"]
        for name, f in report.get("files", {}).items()
        if name.replace("\\", "/").startswith("src/piginot/serving/")
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
        "serving_coverage_percent": round(100.0 * covered / statements, 2)
        if statements
        else None,
        **parse_pytest_log(read_text_any(pytest_log)),
        "execution_profile": execution_profile(),
    }
    (root / SUITE_PATH).write_text(json.dumps(document, indent=2) + "\n", "utf-8")
    return document


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
        "piginot.serving_at_least": SERVING_COVERAGE,
    }
    path = root / SUITE_PATH
    if not path.is_file():
        return not_run("G22.quality", f"{SUITE_PATH} not recorded", threshold)
    suite = json.loads(path.read_text("utf-8"))
    recorded = _listing(root, suite["tree"])
    counts = suite["counts"]
    passed = (
        all(code == 0 for code in codes.values())
        and recorded is not None
        and recorded == _listing(root, "HEAD")
        and suite["clean"]
        and suite["branch_coverage"]
        and counts.get("failed", 0) == 0
        and counts.get("errors", 0) == 0
        and suite["coverage_percent"] >= GLOBAL_COVERAGE
        and (suite["serving_coverage_percent"] or 0.0) >= SERVING_COVERAGE
    )
    return check_document(
        "G22.quality", passed, {"tools": codes, "suite": suite}, threshold, "all"
    )


def build_gate(root: Path) -> dict[str, Any]:
    protocol = json.loads((root / s22.PROTOCOL_PATH).read_text("utf-8"))
    results = _results(root)
    checks = [check_protocol_first(root)]
    if results is None:
        checks.append(
            not_run("G22.results", "S22 results missing", "both families evaluated")
        )
    else:
        checks.extend(family_checks(protocol, results))
    checks.append(check_fast_unchanged(root))
    checks.extend(check_tests(root))
    if results is not None:
        checks.append(check_determinism(results))
    checks.append(check_quality(root))
    blockers = [c["id"] for c in checks if c["status"] != "PASS"]
    return {
        "gate_id": "G22",
        "step": "S22",
        "commit": git_head(root),
        "timestamp_utc": utc_now(),
        "execution_profile": execution_profile(),
        "checks": checks,
        "decision_record": "docs/adr/0103-s22-verified-serving.md",
        "verdict": "FAIL" if blockers else "PASS",
        "verdict_reason": f"{len(blockers)} required check(s) not PASS: {', '.join(blockers)}"
        if blockers
        else "every required check PASS",
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate gate G22.")
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--record-suite", nargs=2, type=Path, metavar=("COVERAGE_JSON", "LOG")
    )
    options = parser.parse_args(arguments)
    if options.record_suite:
        print(
            json.dumps(record_suite(REPOSITORY, *options.record_suite), indent=2)[:2000]
        )
        return 0
    document = build_gate(REPOSITORY)
    if options.write:
        write_gate_document(REPOSITORY / GATE_PATH, document)
    for check in document["checks"]:
        print(f"{check['id']}: {check['status']}")
    print(f"G22: {document['verdict']} ({document['verdict_reason']})")
    return 0 if document["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
