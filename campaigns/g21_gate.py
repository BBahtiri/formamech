"""G21: the per-prediction certificate on DEVELOPMENT (S21, ADR 0102).

Checks (thresholds from ``artifacts/certify/S21-PROTOCOL.json``):

- ``G21.protocol-first``: the protocol was committed once, before both
  results files;
- ``G21.reference-machine``: both families ran on the reference machine;
- ``G21.recompute``: each family's records cover the protocol's cases in
  order and its summary recomputes exactly from them;
- ``G21.integrity``: every certificate OK, S14 coarse terms reproduce on the
  reproduction cases, served meshes identical, vertex and Dirichlet gaps tiny;
- ``G21.rank``, ``G21.effectivity``, ``G21.stress``, ``G21.versus-s14``,
  ``G21.cost``: the five accepted thresholds, per family;
- ``G21.determinism``: one certificate per family re-issued here equals the
  recorded one byte for byte (only on the recorded platform);
- ``G21.quality``: ruff, ruff format, mypy clean; a recorded full Linux run of
  this code tree with no failure, global coverage >= 90.5% and
  ``piginot.certify`` >= 90%.

Usage::

    python -m campaigns.g21_gate --record-suite COVERAGE_JSON PYTEST_LOG
    python -m campaigns.g21_gate --write
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from campaigns import s21_certificate as s21
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
GATE_PATH = Path("artifacts/gates/S21.json")
SUITE_PATH = s21.OUTPUT / "S21-SUITE.json"
GATE_FILES = (GATE_PATH, SUITE_PATH)
# evidence written by the runs, not code: excluded when comparing code trees,
# so the full suite may run in parallel with the evaluations
EVIDENCE_FILES = (*GATE_FILES, *(s21.results_path(c) for c in s21.FAMILIES))
GLOBAL_COVERAGE = 90.5
CERTIFY_COVERAGE = 90.0


def _protocol(root: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((root / s21.PROTOCOL_PATH).read_text("utf-8"))
    return document


def _results(root: Path) -> dict[str, dict[str, Any]] | None:
    paths = {c: root / s21.results_path(c) for c in s21.FAMILIES}
    if not all(path.is_file() for path in paths.values()):
        return None
    return {c: json.loads(path.read_text("utf-8")) for c, path in paths.items()}


def check_protocol_first(root: Path) -> dict[str, Any]:
    commits = _git(
        root, "log", "--format=%H", "--", s21.PROTOCOL_PATH.as_posix()
    ).stdout.split()
    first = commits[-1] if commits else None
    order = {c: _added_in(root, s21.results_path(c)) for c in s21.FAMILIES}
    passed = (
        first is not None
        and len(commits) == 1
        and all(
            c is not None and _strictly_before(root, first, c) for c in order.values()
        )
    )
    return check_document(
        "G21.protocol-first",
        passed,
        {"protocol_commit": first, "protocol_commits": len(commits), "results": order},
        "protocol committed once, before both results files",
        "all",
    )


def _family_checks(
    protocol: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rules, integrity = protocol["thresholds"], protocol["integrity"]
    recomputed: dict[str, dict[str, Any]] = {}
    coverage_ok: dict[str, bool] = {}
    for campaign, document in results.items():
        records = document["records"]
        terms = {c["case_id"]: c["terms"] for c in s21.development_records(campaign)}
        recomputed[campaign] = s21.summarize(records, terms, protocol)
        family = protocol["families"][campaign]
        flagged = [r["case_id"] for r in records if r.get("reproduction")]
        coverage_ok[campaign] = [r["case_id"] for r in records] == family[
            "case_ids"
        ] and flagged == family["reproduction_case_ids"]
    machine = protocol["reference_machine"]
    profiles = {c: d["platform_fingerprint"] for c, d in results.items()}
    checks = [
        check_document(
            "G21.reference-machine",
            all(p == machine for p in profiles.values()),
            profiles,
            machine,
            "all",
        ),
        check_document(
            "G21.recompute",
            all(coverage_ok.values())
            and all(results[c]["summary"] == recomputed[c] for c in results),
            {
                "cases_in_protocol_order": coverage_ok,
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
        verdicts = {c: bool(test(s)) for c, s in recomputed.items()}
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
        "G21.integrity",
        lambda s: (
            s["certificates_ok"] == s["cases"]
            and s["reproduction_cases"] == 8
            and s["reproduction_max_relative"]
            <= integrity["reproduction_relative_at_most"]
            and s["mesh_identity"]
            and s["max_vertex_gap"] <= integrity["vertex_gap_at_most"]
            and s["max_dirichlet_deviation"] <= integrity["dirichlet_deviation_at_most"]
        ),
        (
            "cases",
            "certificates_ok",
            "reproduction_cases",
            "reproduction_max_relative",
            "mesh_identity",
            "max_vertex_gap",
            "max_dirichlet_deviation",
        ),
        integrity,
    )
    per_family(
        "G21.rank",
        lambda s: s["spearman_eta_energy"] >= rules["spearman_eta_energy_at_least"],
        ("spearman_eta_energy",),
        {"spearman_eta_energy_at_least": rules["spearman_eta_energy_at_least"]},
    )
    per_family(
        "G21.effectivity",
        lambda s: (
            s["effectivity_within_band"] >= rules["effectivity_within_band_at_least"]
        ),
        ("effectivity_cases", "effectivity_within_band", "effectivity_median"),
        {k: rules[k] for k in ("effectivity_band", "effectivity_within_band_at_least")},
    )
    per_family(
        "G21.stress",
        lambda s: s["spearman_stress"] >= rules["spearman_stress_at_least"],
        ("spearman_stress",),
        {"spearman_stress_at_least": rules["spearman_stress_at_least"]},
    )
    per_family(
        "G21.versus-s14",
        lambda s: (
            s["spearman_s14_stress"] is not None
            and s["spearman_stress"] >= s["spearman_s14_stress"]
        ),
        ("spearman_stress", "spearman_s14_stress"),
        "the certificate ranks the stress error at least as well as the S14 indicator",
    )
    per_family(
        "G21.cost",
        lambda s: s["cost_ratio"] <= rules["cost_ratio_at_most"],
        ("cost_ratio",),
        {"cost_ratio_at_most": rules["cost_ratio_at_most"]},
    )
    return checks


def check_determinism(
    root: Path, results: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    from campaigns.g19_gate import _differences, _relative
    from piginot.training.platform_fingerprint import platform_fingerprint

    here = platform_fingerprint()
    if any(d["platform_fingerprint"] != here for d in results.values()):
        return not_run(
            "G21.determinism",
            f"this platform {here} differs from the recorded one",
            "a re-issued certificate equals the recorded one byte for byte",
        )
    measured = {}
    for campaign, document in results.items():
        record = document["records"][0]
        reissued = reissue(campaign, record["case_id"])
        differences = _differences(record["certificate"], reissued)
        relatives = [
            value
            for value in (_relative(a, b) for _, a, b in differences)
            if value is not None
        ]
        measured[campaign] = {
            "case_id": record["case_id"],
            "equal": not differences,
            "differing_fields": [path for path, _, _ in differences][:10],
            "max_relative_difference": max(relatives) if relatives else 0.0,
        }
    return check_document(
        "G21.determinism",
        all(m["equal"] for m in measured.values()),
        measured,
        "a certificate re-issued under the protocol's conditions (one torch "
        "thread, same prediction options) equals the recorded one byte for byte",
        "all",
    )


def reissue(campaign: str, case_id: str) -> dict[str, Any]:
    """The certificate of one case, without the reference solve.

    Runs under the evaluation's conditions: one torch thread (the protocol's
    worker setting; more threads change the summation order of the network's
    matrix products) and the same prediction options.
    """

    import numpy as np
    import torch

    from campaigns.dogbone_v1.s14_candidate import S14Case
    from piginot.bundle import predict
    from piginot.certify import certify
    from piginot.fem.linearization import FemLinearization

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        family = s21.runtime(campaign)
        case = S14Case.from_published(s21._published(campaign)[case_id])
        request = family.s10_request(case, family.profile())
        linearization = FemLinearization(request)
        values = {k: float(v) for k, v in case.primitive_coordinates.items()}
        prediction = predict(
            s21._bundle(family.bundle),
            family.theta(
                values,
                elastic_modulus=values["elastic_modulus"],
                poisson_ratio=values["poisson_ratio"],
            ),
            linearization.dof_coordinates.tolist(),
            confidence=True,
        )
        assert prediction.queries is not None
        surrogate = prediction.queries.u.detach().numpy().astype(np.float64)
        document: dict[str, Any] = json.loads(
            json.dumps(certify(FemLinearization(request), surrogate).document())
        )
    finally:
        torch.set_num_threads(threads)
    return document


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


def _same_code(root: Path, tree: str) -> bool:
    recorded = _listing(root, tree)
    return recorded is not None and recorded == _listing(root, "HEAD")


def record_suite(root: Path, coverage_json: Path, pytest_log: Path) -> dict[str, Any]:
    report = json.loads(read_text_any(coverage_json))
    files = report.get("files", {})
    totals = [
        f["summary"]
        for name, f in files.items()
        if name.replace("\\", "/").startswith("src/piginot/certify/")
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
        "certify_coverage_percent": round(100.0 * covered / statements, 2)
        if statements
        else None,
        **parse_pytest_log(read_text_any(pytest_log)),
        "execution_profile": execution_profile(),
    }
    (root / SUITE_PATH).write_text(json.dumps(document, indent=2) + "\n", "utf-8")
    return document


def check_quality(root: Path) -> dict[str, Any]:
    python = sys.executable
    tools = {
        "ruff_check": [python, "-m", "ruff", "check", "."],
        "ruff_format": [python, "-m", "ruff", "format", "--check", "."],
        "mypy_strict": [python, "-m", "mypy", "--strict", "src/"],
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
        "piginot.certify_at_least": CERTIFY_COVERAGE,
    }
    path = root / SUITE_PATH
    if not path.is_file():
        return not_run("G21.quality", f"{SUITE_PATH} not recorded", threshold)
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
        and (suite["certify_coverage_percent"] or 0.0) >= CERTIFY_COVERAGE
    )
    return check_document(
        "G21.quality", passed, {"tools": codes, "suite": suite}, threshold, "all"
    )


def build_gate(root: Path) -> dict[str, Any]:
    protocol = _protocol(root)
    results = _results(root)
    checks = [check_protocol_first(root)]
    if results is None:
        checks.append(
            not_run(
                "G21.results",
                "S21 results missing (run campaigns.s21_certificate for both families)",
                "both families evaluated",
            )
        )
    else:
        checks.extend(_family_checks(protocol, results))
        checks.append(check_determinism(root, results))
    checks.append(check_quality(root))
    blockers = [c["id"] for c in checks if c["status"] != "PASS"]
    return {
        "gate_id": "G21",
        "step": "S21",
        "commit": git_head(root),
        "timestamp_utc": utc_now(),
        "execution_profile": execution_profile(),
        "checks": checks,
        "decision_record": "docs/adr/0102-s21-per-prediction-certificate.md",
        "verdict": "FAIL" if blockers else "PASS",
        "verdict_reason": (
            f"{len(blockers)} required check(s) not PASS: {', '.join(blockers)}"
            if blockers
            else "every required check PASS"
        ),
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate gate G21.")
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
    print(f"G21: {document['verdict']} ({document['verdict_reason']})")
    return 0 if document["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
