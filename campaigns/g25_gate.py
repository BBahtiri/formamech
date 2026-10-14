"""G25: material registry, energy formulas and admission (S25, ADR 0107).

Thresholds from ADR 0107 (accepted 2026-10-13) and
``artifacts/materials/S25-PROTOCOL.json``. Checks: protocol-first,
formula-safety, neo-hookean-identity, admission, admission-mutants, fem,
viability, binding-compat, anti-scope, portability (native Windows run) and
quality (Linux run)::

    python -m campaigns.g25_gate --record-suite COVERAGE_JSON PYTEST_LOG
    python -m campaigns.g25_gate --record-windows PYTEST_LOG WINDOWS_CLONE
    python -m campaigns.g25_gate --write

A record command checks that the tree is clean apart from the gate and suite
records of S24 and later (``is_gate_record``), so recording twice needs no
``git restore`` and one suite run can be recorded for G24 and G25.
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

from campaigns import g19_gate
from campaigns.dogbone_v1 import s25_mooney_rivlin as mr
from campaigns.g19_gate import execution_profile, not_run, read_text_any
from campaigns.g20_gate import _git, _pytest, _tree, parse_pytest_log
from campaigns.g24_gate import is_gate_record
from campaigns.notched_v1.gate_io import (
    check_document,
    git_head,
    utc_now,
    write_gate_document,
)
from piginot.contracts import canonical_json_bytes

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = Path("artifacts/materials")
GATE_PATH = Path("artifacts/gates/S25.json")
SUITE_PATH = OUTPUT / "S25-SUITE.json"
WINDOWS_PATH = OUTPUT / "S25-WINDOWS-SUITE.json"
GATE_FILES = (GATE_PATH, SUITE_PATH, WINDOWS_PATH)
GLOBAL_COVERAGE = 90.5
MODULE_COVERAGE = 90.0
COVERED_MODULES = (
    "src/piginot/materials/",
    "src/piginot/contracts/energy_formula.py",
)
ADR_PATH = Path("docs/adr/0107-s25-material-registry.md")
FINAL_STEP = 2400


def _json(root: Path, path: Path) -> dict[str, Any] | None:
    full = root / path
    if not full.is_file():
        return None
    document: dict[str, Any] = json.loads(full.read_text("utf-8"))
    return document


def _digest(root: Path, path: Path) -> str:
    return hashlib.sha256((root / path).read_bytes()).hexdigest()


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
        "S25-PROTOCOL.json committed once, with ADR 0107 accepted, in a revision "
        "the reference re-solves and the viability run descend from"
    )
    commits = _git(
        root, "log", "--format=%H", "--", mr.PROTOCOL_PATH.as_posix()
    ).stdout.split()
    first = commits[-1] if commits else None
    if first is None:
        return not_run("G25.protocol-first", "protocol not committed", threshold)
    adr = _git(root, "show", f"{first}:{ADR_PATH.as_posix()}").stdout
    references = _json(root, mr.REFERENCES_PATH)
    screen = _json(root, mr.SCREEN_PATH)
    revisions: dict[str, str | None] = {
        "references": None
        if references is None
        else references.get("implementation_revision"),
        "viability": None if screen is None else screen.get("implementation_revision"),
    }
    if any(value is None for value in revisions.values()):
        return not_run(
            "G25.protocol-first", "a run record or its revision is missing", threshold
        )
    passed = (
        len(commits) == 1
        and "- Status: Accepted" in adr
        and all(_ancestor_or_same(root, first, rev) for rev in revisions.values())
    )
    return check_document(
        "G25.protocol-first",
        passed,
        {"protocol_commit": first, "protocol_commits": len(commits), "runs": revisions},
        threshold,
        "all",
    )


def _tests(check_id: str, threshold: str, *targets: str) -> dict[str, Any]:
    result = _pytest(REPOSITORY, *targets)
    return check_document(check_id, result["returncode"] == 0, result, threshold, "all")


def check_formula_safety() -> dict[str, Any]:
    return _tests(
        "G25.formula-safety",
        "the parser refuses every construct outside the grammar; no eval, exec, "
        "compile, sympify or parse_expr in the formula path (syntax-tree scan)",
        "tests/contracts/test_energy_formula.py",
        "tests/test_s25_anti_scope.py::test_formulas_are_never_evaluated_as_code",
    )


def _fast_goldens(root: Path) -> dict[str, bool] | None:
    from piginot.bundle import load_bundle
    from piginot.cli import parse_request
    from piginot.serving import serve

    goldens = json.loads((root / g19_gate.GOLDEN_PATH).read_text("utf-8"))["profiles"]
    key = g19_gate.profile_key(execution_profile())
    if key not in goldens:
        return None
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
    return measured


def check_neo_hookean_identity(root: Path) -> dict[str, Any]:
    threshold = (
        "dispatch bit-identical to the closed-form kernels; S19 fast goldens "
        "unchanged; the Neo-Hookean formula within 1e-12 of the closed form"
    )
    tests = _pytest(
        root,
        "tests/test_s25_anti_scope.py::test_the_neo_hookean_path_is_bit_identical",
        "tests/materials/test_hyperelastic.py::"
        "test_a_neo_hookean_formula_equals_the_closed_form",
    )
    goldens = _fast_goldens(root)
    if goldens is None:
        return not_run("G25.neo-hookean-identity", "no S19 golden here", threshold)
    return check_document(
        "G25.neo-hookean-identity",
        tests["returncode"] == 0 and all(goldens.values()),
        {"tests": tests, "fast_goldens": goldens},
        threshold,
        "all",
    )


def check_admission(root: Path) -> dict[str, Any]:
    from piginot.problem import admit_material

    tests = _pytest(
        root,
        "tests/materials/test_admission.py::test_every_builtin_model_is_admitted",
        "tests/materials/test_admission.py::"
        "test_the_closed_form_and_its_formula_are_admitted_alike",
        "tests/problem/test_material.py",
    )
    viability = admit_material(mr.load_spec(), fem=True)
    return check_document(
        "G25.admission",
        tests["returncode"] == 0 and viability.passed,
        {"tests": tests, "viability_material": viability.document()},
        "every built-in model and the closed-form model admitted (all eight "
        "checks) over [0.6, 1.6]; the viability material admitted",
        "all",
    )


def check_admission_mutants() -> dict[str, Any]:
    return _tests(
        "G25.admission-mutants",
        "each seeded defect fails its check (reference, moduli, ellipticity and "
        "closure, undefined energy, symmetry, objectivity, consistency)",
        "tests/materials/test_admission.py",
    )


def check_fem() -> dict[str, Any]:
    return _tests(
        "G25.fem",
        "autograd manufactured fields equal the symbolic S8 fields within 1e-10; "
        "a formula model converges at u-order 3 +- 0.15 and P-order 2 +- 0.2",
        "tests/materials/test_fem_admission.py",
    )


def check_viability(root: Path) -> dict[str, Any]:
    threshold = (
        "the registered Mooney-Rivlin run passes the frozen S13 guardrails on the "
        "64 TRAIN-screen re-solves at step 2400; DEVELOPMENT closed"
    )
    references = _json(root, mr.REFERENCES_PATH)
    record = _json(root, mr.SCREEN_PATH)
    if references is None or record is None:
        return not_run(
            "G25.viability", "references or the viability run not recorded", threshold
        )
    protocol = _json(root, mr.PROTOCOL_PATH)
    assert protocol is not None
    digest = _digest(root, mr.PROTOCOL_PATH)
    spec_hash = mr.load_spec().problem_hash
    cases = protocol["viability"]["references"]["cases"]
    passed = (
        record["passed"] is True
        and record["protocol_sha256"] == digest
        and references["protocol_sha256"] == digest
        and record["references_sha256"]
        == hashlib.sha256(canonical_json_bytes(references)).hexdigest()
        and sorted(references["cases"]) == sorted(cases)
        and record["problem_spec_hash"] == spec_hash
        and references["problem_spec_hash"] == spec_hash
        and record["checkpoint_step"] == FINAL_STEP
        and not record["failed_guardrails"]
        and record["development_opened"] is False
    )
    return check_document("G25.viability", passed, record, threshold, "all")


def check_binding_compat(root: Path) -> dict[str, Any]:
    tests = _pytest(
        root,
        "tests/training/test_identity.py",
        "tests/training/test_checkpoint.py",
        "tests/test_shipped_bundles.py",
        "tests/campaigns/test_problem_specs.py",
    )
    return check_document(
        "G25.binding-compat",
        tests["returncode"] == 0,
        {"tests": tests},
        "existing run bindings, checkpoints, bundles and family spec hashes "
        "still verify",
        "all",
    )


def check_anti_scope(root: Path) -> dict[str, Any]:
    tests = _pytest(
        root,
        "tests/test_s25_anti_scope.py",
        "tests/test_s24_anti_scope.py",
        "tests/test_no_fixture_names.py",
    )
    suite = _json(root, SUITE_PATH)
    modules = None if suite is None else suite.get("module_coverage_percent")
    passed = (
        tests["returncode"] == 0
        and modules is not None
        and all(
            value is not None and value >= MODULE_COVERAGE for value in modules.values()
        )
    )
    return check_document(
        "G25.anti-scope",
        passed,
        {"tests": tests, "module_coverage_percent": modules},
        "isotropic hyperelastic only; .[runtime] imports piginot.materials without "
        "FEM packages; branch coverage of piginot.materials and the formula "
        "contract >= 90% each",
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
    (root / SUITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / SUITE_PATH).write_text(json.dumps(document, indent=2) + "\n", "utf-8")
    return document


def record_windows(root: Path, pytest_log: Path, clone: Path) -> dict[str, Any]:
    document = {
        "schema_version": "1.0",
        "tree": _tree(clone),
        "clean": clean(clone),
        **parse_pytest_log(read_text_any(pytest_log)),
    }
    (root / WINDOWS_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / WINDOWS_PATH).write_text(json.dumps(document, indent=2) + "\n", "utf-8")
    return document


def check_portability(root: Path) -> dict[str, Any]:
    threshold = (
        "native Windows run of this code tree: 0 failed, 0 errors, "
        "every skip has a reason"
    )
    run = _json(root, WINDOWS_PATH)
    if run is None:
        return not_run("G25.portability", f"{WINDOWS_PATH} not recorded", threshold)
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
    return check_document("G25.portability", passed, run, threshold, "all")


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
        return not_run("G25.quality", f"{SUITE_PATH} not recorded", threshold)
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
        "G25.quality", passed, {"tools": codes, "suite": suite}, threshold, "all"
    )


def build_gate(root: Path) -> dict[str, Any]:
    checks = [
        check_protocol_first(root),
        check_formula_safety(),
        check_neo_hookean_identity(root),
        check_admission(root),
        check_admission_mutants(),
        check_fem(),
        check_viability(root),
        check_binding_compat(root),
        check_anti_scope(root),
        check_portability(root),
        check_quality(root),
    ]
    blockers = [c["id"] for c in checks if c["status"] != "PASS"]
    return {
        "gate_id": "G25",
        "step": "S25",
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
    parser = argparse.ArgumentParser(description="Evaluate gate G25.")
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
    print(f"G25: {document['verdict']} ({document['verdict_reason']})")
    return 0 if document["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ("GATE_PATH", "build_gate", "clean", "main")
