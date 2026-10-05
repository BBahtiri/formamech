from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from campaigns.dogbone_v1 import s16_mutation
from campaigns.dogbone_v1.s16_detectors import CACHES, DETECTORS, cleanup
from campaigns.dogbone_v1.s16_gate import gate_checks
from campaigns.dogbone_v1.s16_mutants import MUTANTS
from piginot.mutation import (
    Detector,
    DetectorOutcome,
    applied,
    compare_signatures,
    resolve_function,
    run_suite,
    signature_document,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs" / "specs" / "implementation-v1.0-oss-replacement.md"
BASELINE = ROOT / "artifacts" / "mutation" / "S16-SIGNATURES.json"


def _spec_table() -> list[tuple[str, str]]:
    rows = re.findall(
        r"^\| +(\d+) \| `([A-Z_]+)` \| `(G\d+\.[a-z-]+)` \|",
        SPEC.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    return [(name, gate) for _, name, gate in rows]


def test_the_registry_is_the_adr_0022_table() -> None:
    table = _spec_table()
    assert len(table) == 20
    assert [(m.name, m.primary) for m in MUTANTS] == table
    detectors = {d.check_id for d in DETECTORS}
    assert {m.primary for m in MUTANTS} <= detectors
    assert len(detectors) == len(DETECTORS) == 22


def test_every_mutant_applies_and_restores_the_production_code() -> None:
    for mutant in MUTANTS:
        functions = [resolve_function(edit.target) for edit in mutant.edits]
        before = [function.__code__ for function in functions]
        with applied(mutant):
            assert all(
                function.__code__ is not code
                for function, code in zip(functions, before, strict=True)
            )
        assert [function.__code__ for function in functions] == before


def test_the_suite_reproduces_the_committed_signatures() -> None:
    try:
        report = run_suite(MUTANTS, DETECTORS, caches=CACHES)
    finally:
        cleanup()
    assert report.control_passed
    assert report.primary_count == 20
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    comparison = compare_signatures(baseline, signature_document(report))
    assert comparison.status.value == "PASS", comparison.document()


def test_command_line_runs_subsets_and_reports_control_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "report.json"
    signatures = tmp_path / "signatures.json"
    assert (
        s16_mutation.main(
            [
                "run",
                "--report",
                str(report_path),
                "--signatures",
                str(signatures),
                "--mutant",
                "NODE_SPLIT",
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text())
    assert report["mutant_count"] == 1 and report["primary_fired"] == 1
    assert json.loads(signatures.read_text())["mutants"]["NODE_SPLIT"]["signature"] == [
        "G9.no-node-split"
    ]
    compared = tmp_path / "compared.json"
    code = s16_mutation.main(
        [
            "run",
            "--report",
            str(compared),
            "--mutant",
            "NODE_SPLIT",
            "--compare",
            str(BASELINE),
        ]
    )
    assert code == 4  # the other nineteen baseline mutants did not run
    with pytest.raises(SystemExit):
        s16_mutation.run(["NO_SUCH_MUTANT"])
    with pytest.raises(SystemExit):
        s16_mutation.run(["NODE_SPLIT"], ["G0.none"])
    broken = Detector("G9.no-node-split", "broken", lambda: DetectorOutcome(False))
    monkeypatch.setattr(s16_mutation, "DETECTORS", (broken,))
    assert (
        s16_mutation.main(
            ["run", "--report", str(report_path), "--mutant", "NODE_SPLIT"]
        )
        == s16_mutation.EXIT_CONTROL_FAILED
    )


def _report(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "control_passed": True,
        "primary_fired": 20,
        "detectors": [d.check_id for d in DETECTORS],
        "seconds": 50.0,
        "runtime": {},
        "mutants": [
            {"name": m.name, "primary": m.primary, "primary_fired": True}
            for m in MUTANTS
        ],
        "comparison": {"status": "PASS", "mutants": []},
    }
    document.update(overrides)
    return document


def test_gate_checks_pass_only_on_complete_evidence() -> None:
    baseline = {
        "mutants": {
            m.name: {"primary": m.primary, "signature": [m.primary]} for m in MUTANTS
        }
    }
    regression = _report(
        comparison={
            "status": "COVERAGE_REGRESSION",
            "removed_detectors": ["G7.loads-fields-store"],
            "mutants": [
                {
                    "name": "BODY_FORCE_SIGN",
                    "status": "COVERAGE_REGRESSION",
                    "lost_secondary": ["G7.loads-fields-store"],
                }
            ],
        }
    )
    workflow = (ROOT / ".github" / "workflows" / "mutation.yml").read_text()
    checks = gate_checks(_report(), regression, baseline, workflow, 99.0)
    assert [c["status"] for c in checks] == ["PASS"] * 5
    assert checks[2]["measured"]["demonstration_exit_code"] == 3

    failing = gate_checks(
        _report(
            primary_fired=19,
            seconds=3_000.0,
            comparison={"status": "PRIMARY_LOST", "mutants": []},
        ),
        _report(),
        baseline,
        "",
        50.0,
    )
    assert [c["status"] for c in failing] == ["FAIL"] * 5
