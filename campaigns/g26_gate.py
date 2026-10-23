"""G26: anisotropic models with in-plane fibre families (S26, ADR 0108).

Thresholds from ADR 0108 (accepted 2026-10-14) and
``artifacts/materials/S26-PROTOCOL.json``. Checks: protocol-first,
formula-safety, isotropic-identity, fibre-kernels, admission,
admission-mutants, fem, material-symmetry, viability, binding-compat,
anti-scope, portability (native Windows run) and quality (Linux run)::

    python -m campaigns.g26_gate --record-suite COVERAGE_JSON PYTEST_LOG
    python -m campaigns.g26_gate --record-windows PYTEST_LOG WINDOWS_CLONE
    python -m campaigns.g26_gate --write

The S19 fast goldens are compared at their recorded thread count
(``campaigns.fast_goldens``), whatever ``OMP_NUM_THREADS`` says. A record
command checks that the tree is clean apart from gate and suite records of
S24 and later (``g24_gate.is_gate_record``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from campaigns import fast_goldens
from campaigns.dogbone_v1.s26_hgo import RUN
from campaigns.dogbone_v1.s26_references import (
    AMENDMENT_PATH,
    accepted_amendment,
    amendment_digest,
)
from campaigns.g19_gate import execution_profile, not_run, read_text_any
from campaigns.g20_gate import _git, _pytest, _tree, parse_pytest_log
from campaigns.g25_gate import (
    _ancestor_or_same,
    _digest,
    _json,
    _module_coverage,
    _same_code,
    clean,
)
from campaigns.notched_v1.gate_io import (
    check_document,
    git_head,
    utc_now,
    write_gate_document,
)
from piginot.contracts import canonical_json_bytes

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = Path("artifacts/materials")
GATE_PATH = Path("artifacts/gates/S26.json")
SUITE_PATH = OUTPUT / "S26-SUITE.json"
WINDOWS_PATH = OUTPUT / "S26-WINDOWS-SUITE.json"
GATE_FILES = (GATE_PATH, SUITE_PATH, WINDOWS_PATH)
GLOBAL_COVERAGE = 90.5
MODULE_COVERAGE = 90.0
ADR_PATH = Path("docs/adr/0108-s26-fibre-families.md")
FINAL_STEP = 2400
_FIBRES = "tests/materials/test_fibres.py"


def check_protocol_first(root: Path) -> dict[str, Any]:
    threshold = (
        "S26-PROTOCOL.json committed once, with ADR 0108 accepted, in a revision "
        "the reference re-solves and the viability run descend from; the accepted "
        "reference amendment committed once after the protocol and no later than "
        "the reference re-solves"
    )
    commits = _git(
        root, "log", "--format=%H", "--", RUN.protocol_path.as_posix()
    ).stdout.split()
    first = commits[-1] if commits else None
    if first is None:
        return not_run("G26.protocol-first", "protocol not committed", threshold)
    adr = _git(root, "show", f"{first}:{ADR_PATH.as_posix()}").stdout
    references = _json(root, RUN.references_path)
    screen = _json(root, RUN.screen_path)
    revisions: dict[str, str | None] = {
        "references": None
        if references is None
        else references.get("implementation_revision"),
        "viability": None if screen is None else screen.get("implementation_revision"),
    }
    if any(value is None for value in revisions.values()):
        return not_run(
            "G26.protocol-first", "a run record or its revision is missing", threshold
        )
    try:
        accepted_amendment(root)
    except (OSError, ValueError, RuntimeError) as exc:
        return check_document(
            "G26.protocol-first", False, {"amendment_error": str(exc)}, threshold, "all"
        )
    amendment_commits = _git(
        root, "log", "--format=%H", "--", AMENDMENT_PATH.as_posix()
    ).stdout.split()
    amendment_first = amendment_commits[-1] if amendment_commits else None
    passed = (
        len(commits) == 1
        and "- Status: Accepted" in adr
        and all(_ancestor_or_same(root, first, rev) for rev in revisions.values())
        and len(amendment_commits) == 1
        and amendment_first is not None
        and amendment_first != first
        and _ancestor_or_same(root, first, amendment_first)
        and _ancestor_or_same(root, amendment_first, revisions["references"])
    )
    return check_document(
        "G26.protocol-first",
        passed,
        {
            "protocol_commit": first,
            "protocol_commits": len(commits),
            "reference_amendment_commit": amendment_first,
            "reference_amendment_commits": len(amendment_commits),
            "runs": revisions,
        },
        threshold,
        "all",
    )


def _tests(check_id: str, threshold: str, *targets: str) -> dict[str, Any]:
    result = _pytest(REPOSITORY, *targets)
    return check_document(check_id, result["returncode"] == 0, result, threshold, "all")


def check_formula_safety() -> dict[str, Any]:
    return _tests(
        "G26.formula-safety",
        "every malformed fibre construct is refused; no eval, exec or compile in "
        "the formula path (syntax-tree scan)",
        "tests/contracts/test_energy_formula_fibres.py",
        "tests/contracts/test_energy_formula.py",
        "tests/test_s25_anti_scope.py::test_formulas_are_never_evaluated_as_code",
    )


def check_isotropic_identity(root: Path) -> dict[str, Any]:
    threshold = (
        "S25 formula identities unchanged; Neo-Hookean dispatch bit-identical; "
        "S19 fast goldens equal at their recorded thread count"
    )
    tests = _pytest(
        root,
        "tests/test_s26_anti_scope.py::test_the_isotropic_identities_are_unchanged",
        "tests/test_s25_anti_scope.py::test_the_neo_hookean_path_is_bit_identical",
        "tests/materials/test_hyperelastic.py::"
        "test_a_neo_hookean_formula_equals_the_closed_form",
        "tests/materials/test_admission.py::test_every_builtin_model_is_admitted",
    )
    goldens = fast_goldens.compare(root)
    if goldens is None:
        return not_run("G26.isotropic-identity", "no S19 golden here", threshold)
    return check_document(
        "G26.isotropic-identity",
        tests["returncode"] == 0 and all(goldens["documents"].values()),
        {"tests": tests, "fast_goldens": goldens},
        threshold,
        "all",
    )


def check_fibre_kernels() -> dict[str, Any]:
    return _tests(
        "G26.fibre-kernels",
        "hgo stress equals its closed form within 1e-12; zero fibre stress where "
        "I4 <= 1; the hgo closure root equals the matrix's within 1e-12",
        f"{_FIBRES}::test_the_hgo_stress_equals_its_closed_form",
        f"{_FIBRES}::test_fibres_carry_nothing_in_compression",
        f"{_FIBRES}::test_in_plane_fibres_leave_the_closure_root_alone",
    )


def check_admission(root: Path) -> dict[str, Any]:
    from piginot.problem import admit_material

    tests = _pytest(
        root,
        f"{_FIBRES}::test_the_fibre_builtins_are_admitted",
        f"{_FIBRES}::test_the_declared_symmetry_group",
    )
    viability = admit_material(RUN.load_spec(), fem=True)
    return check_document(
        "G26.admission",
        tests["returncode"] == 0 and viability.passed,
        {"tests": tests, "viability_material": viability.document()},
        "hgo, hgo_single and goh admitted (all eight checks) over [0.6, 1.6]; the "
        "viability material admitted",
        "all",
    )


def check_admission_mutants() -> dict[str, Any]:
    return _tests(
        "G26.admission-mutants",
        "a fibre direction other than declared fails symmetry; a fibre term "
        "without switch fails moduli; k1 < 0 fails ellipticity; S25 mutants fail",
        f"{_FIBRES}::test_a_fibre_direction_other_than_declared_fails_symmetry",
        f"{_FIBRES}::test_a_fibre_term_without_switch_fails_moduli",
        f"{_FIBRES}::test_softening_fibres_fail_ellipticity",
        "tests/materials/test_admission.py",
    )


def check_fem() -> dict[str, Any]:
    return _tests(
        "G26.fem",
        "fibres in tension: displacement L2 order within 0.15 of 3, first-Piola "
        "L2 order within 0.2 of 2",
        f"{_FIBRES}::test_the_fem_check_keeps_fibres_in_tension",
        f"{_FIBRES}::test_a_field_with_fibres_in_compression_is_refused",
        "tests/materials/test_fem_admission.py",
    )


def check_material_symmetry() -> dict[str, Any]:
    return _tests(
        "G26.material-symmetry",
        "a quarter-model spec with one family at 30 degrees is refused "
        "(material-symmetry); +-beta and 0 degrees pass",
        "tests/problem/test_material_symmetry.py",
    )


def check_viability(root: Path) -> dict[str, Any]:
    threshold = (
        "the registered HGO run passes the frozen S13 guardrails on the 64 "
        "TRAIN-screen re-solves at step 2400; both records bind the accepted "
        "reference amendment; DEVELOPMENT closed"
    )
    references = _json(root, RUN.references_path)
    record = _json(root, RUN.screen_path)
    if references is None or record is None:
        return not_run(
            "G26.viability", "references or the viability run not recorded", threshold
        )
    try:
        accepted_amendment(root)
    except (OSError, ValueError, RuntimeError) as exc:
        return check_document(
            "G26.viability", False, {"amendment_error": str(exc)}, threshold, "all"
        )
    reference_amendment_sha256 = amendment_digest(root)
    protocol = _json(root, RUN.protocol_path)
    assert protocol is not None
    digest = _digest(root, RUN.protocol_path)
    spec_hash = RUN.load_spec().problem_hash
    cases = protocol["viability"]["references"]["cases"]
    passed = (
        record["passed"] is True
        and record["protocol_sha256"] == digest
        and references["protocol_sha256"] == digest
        and record.get("reference_amendment_sha256") == reference_amendment_sha256
        and references.get("reference_amendment_sha256") == reference_amendment_sha256
        and record["references_sha256"]
        == hashlib.sha256(canonical_json_bytes(references)).hexdigest()
        and sorted(references["cases"]) == sorted(cases)
        and record["problem_spec_hash"] == spec_hash
        and references["problem_spec_hash"] == spec_hash
        and record["checkpoint_step"] == FINAL_STEP
        and not record["failed_guardrails"]
        and record["development_opened"] is False
    )
    return check_document("G26.viability", passed, record, threshold, "all")


def check_binding_compat(root: Path) -> dict[str, Any]:
    tests = _pytest(
        root,
        "tests/training/test_identity.py",
        "tests/training/test_checkpoint.py",
        "tests/test_shipped_bundles.py",
        "tests/campaigns/test_problem_specs.py",
        "tests/campaigns/test_dogbone_s25_mooney_rivlin.py::"
        "test_the_committed_protocol_is_the_rebuild_and_accepted",
        "tests/campaigns/test_dogbone_s24_adf.py::"
        "test_the_committed_protocol_is_the_rebuild_and_accepted",
    )
    return check_document(
        "G26.binding-compat",
        tests["returncode"] == 0,
        {"tests": tests},
        "existing run bindings, checkpoints, bundles, family spec hashes and the "
        "S24/S25 protocols still verify",
        "all",
    )


def check_anti_scope(root: Path) -> dict[str, Any]:
    tests = _pytest(
        root,
        "tests/test_s26_anti_scope.py",
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
        "G26.anti-scope",
        passed,
        {"tests": tests, "module_coverage_percent": modules},
        "in-plane fibres only, at most two families, S25 identities unchanged; "
        "branch coverage of piginot.materials and the formula contract >= 90% each",
        "all",
    )


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
        return not_run("G26.portability", f"{WINDOWS_PATH} not recorded", threshold)
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
    return check_document("G26.portability", passed, run, threshold, "all")


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
        return not_run("G26.quality", f"{SUITE_PATH} not recorded", threshold)
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
        "G26.quality", passed, {"tools": codes, "suite": suite}, threshold, "all"
    )


def build_gate(root: Path) -> dict[str, Any]:
    checks = [
        check_protocol_first(root),
        check_formula_safety(),
        check_isotropic_identity(root),
        check_fibre_kernels(),
        check_admission(root),
        check_admission_mutants(),
        check_fem(),
        check_material_symmetry(),
        check_viability(root),
        check_binding_compat(root),
        check_anti_scope(root),
        check_portability(root),
        check_quality(root),
    ]
    blockers = [c["id"] for c in checks if c["status"] != "PASS"]
    return {
        "gate_id": "G26",
        "step": "S26",
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
    parser = argparse.ArgumentParser(description="Evaluate gate G26.")
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
    print(f"G26: {document['verdict']} ({document['verdict_reason']})")
    return 0 if document["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ("GATE_PATH", "build_gate", "main")
