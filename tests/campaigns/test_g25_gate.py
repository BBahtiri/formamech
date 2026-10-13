"""G25 gate logic on synthetic records; no training and no FEM re-solve."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from campaigns import g25_gate as gate
from campaigns.dogbone_v1 import s25_mooney_rivlin as mr
from piginot.contracts import canonical_json_bytes
from tests.git_helpers import run_git


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "-q")
    (repo / "a.txt").write_text("a")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-q", "-m", "a")
    return repo


def _commit(repo: Path, message: str) -> str:
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-q", "-m", message)
    return run_git(repo, "rev-parse", "HEAD").strip()


def test_clean_ignores_only_the_gate_record_files(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    assert gate.clean(repo)
    suite = repo / gate.SUITE_PATH
    suite.parent.mkdir(parents=True)
    suite.write_text("{}")
    assert gate.clean(repo)
    (repo / "a.txt").write_text("b")
    assert not gate.clean(repo)


def test_module_coverage_counts_lines_and_branches() -> None:
    report = {
        "files": {
            "src/piginot/materials/admission.py": {
                "summary": {
                    "num_statements": 8,
                    "num_branches": 2,
                    "covered_lines": 8,
                    "covered_branches": 1,
                }
            },
        }
    }
    coverage = gate._module_coverage(report)
    assert coverage["src/piginot/materials/"] == 90.0
    assert coverage["src/piginot/contracts/energy_formula.py"] is None


def test_missing_records_are_not_run(tmp_path: Path) -> None:
    assert gate.check_viability(tmp_path)["status"] == "NOT_RUN"
    assert gate.check_portability(tmp_path)["status"] == "NOT_RUN"
    assert gate.check_protocol_first(_repository(tmp_path))["status"] == "NOT_RUN"


def _references(revision: str) -> dict[str, Any]:
    protocol = mr.build_protocol()
    cases = {
        case_id: {"request_content_hash": "0" * 64}
        for case_id in protocol["viability"]["references"]["cases"]
    }
    return mr.references_document(cases, revision)


def _screen(references: dict[str, Any], revision: str, **changes: Any) -> Any:
    record = {
        "passed": True,
        "protocol_sha256": hashlib.sha256(
            (mr.REPOSITORY / mr.PROTOCOL_PATH).read_bytes()
        ).hexdigest(),
        "references_sha256": hashlib.sha256(
            canonical_json_bytes(references)
        ).hexdigest(),
        "problem_spec_hash": mr.load_spec().problem_hash,
        "implementation_revision": revision,
        "checkpoint_step": 2400,
        "failed_guardrails": [],
        "development_opened": False,
    }
    record.update(changes)
    return record


def _write(root: Path, path: Path, document: dict[str, Any]) -> None:
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_bytes(canonical_json_bytes(document) + b"\n")


@pytest.fixture
def protocol_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / mr.PROTOCOL_PATH).parent.mkdir(parents=True)
    shutil.copy(mr.REPOSITORY / mr.PROTOCOL_PATH, root / mr.PROTOCOL_PATH)
    return root


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ({}, "PASS"),
        ({"failed_guardrails": ["u_relative_l2"], "passed": False}, "FAIL"),
        ({"checkpoint_step": 2000}, "FAIL"),
        ({"development_opened": True}, "FAIL"),
        ({"references_sha256": "0" * 64}, "FAIL"),
        ({"problem_spec_hash": "0" * 64}, "FAIL"),
    ],
    ids=["pass", "guardrail", "step", "development", "references", "spec"],
)
def test_viability_checks_the_run_against_its_references(
    protocol_root: Path, change: dict[str, Any], status: str
) -> None:
    references = _references("r" * 40)
    _write(protocol_root, mr.REFERENCES_PATH, references)
    _write(protocol_root, mr.SCREEN_PATH, _screen(references, "r" * 40, **change))
    assert gate.check_viability(protocol_root)["status"] == status


def test_references_must_cover_every_screen_case(protocol_root: Path) -> None:
    references = _references("r" * 40)
    references["cases"].pop(next(iter(references["cases"])))
    _write(protocol_root, mr.REFERENCES_PATH, references)
    _write(protocol_root, mr.SCREEN_PATH, _screen(references, "r" * 40))
    assert gate.check_viability(protocol_root)["status"] == "FAIL"


def test_protocol_first_orders_the_protocol_before_both_runs(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    adr = repo / gate.ADR_PATH
    adr.parent.mkdir(parents=True)
    adr.write_text("# ADR\n\n- Status: Accepted\n")
    (repo / mr.PROTOCOL_PATH).parent.mkdir(parents=True)
    (repo / mr.PROTOCOL_PATH).write_text("{}")
    first = _commit(repo, "protocol")
    (repo / "b.txt").write_text("b")
    later = _commit(repo, "code")
    references = _references(first)
    _write(repo, mr.REFERENCES_PATH, references)
    _write(repo, mr.SCREEN_PATH, _screen(references, later))
    record = gate.check_protocol_first(repo)
    assert record["status"] == "PASS", record
    assert record["measured"]["runs"] == {"references": first, "viability": later}

    _write(repo, mr.SCREEN_PATH, _screen(references, "f" * 40))
    assert gate.check_protocol_first(repo)["status"] == "FAIL"
    _write(repo, mr.SCREEN_PATH, _screen(references, later))
    (repo / mr.PROTOCOL_PATH).write_text('{"edited": true}')
    _commit(repo, "edit protocol")
    assert gate.check_protocol_first(repo)["status"] == "FAIL"


def test_a_proposed_adr_fails_protocol_first(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    adr = repo / gate.ADR_PATH
    adr.parent.mkdir(parents=True)
    adr.write_text("# ADR\n\n- Status: Proposed\n")
    (repo / mr.PROTOCOL_PATH).parent.mkdir(parents=True)
    (repo / mr.PROTOCOL_PATH).write_text("{}")
    first = _commit(repo, "protocol")
    references = _references(first)
    _write(repo, mr.REFERENCES_PATH, references)
    _write(repo, mr.SCREEN_PATH, _screen(references, first))
    assert gate.check_protocol_first(repo)["status"] == "FAIL"


def test_suite_records_and_portability(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps(
            {
                "meta": {"branch_coverage": True},
                "totals": {"percent_covered": 91.234},
                "files": {
                    "src/piginot/materials/registry.py": {
                        "summary": {"num_statements": 10, "covered_lines": 10}
                    }
                },
            }
        )
    )
    log = tmp_path / "pytest.log"
    log.write_text("===== 3001 passed, 1 skipped in 9.00s =====\n")
    document = gate.record_suite(repo, coverage, log)
    assert document["coverage_percent"] == 91.23
    assert document["module_coverage_percent"]["src/piginot/materials/"] == 100.0
    assert document["clean"] is True
    windows = gate.record_windows(repo, log, repo)
    assert windows["clean"] is True
    # One skip without a recorded reason.
    assert gate.check_portability(repo)["status"] == "FAIL"
    log.write_text("===== 3001 passed in 9.00s =====\n")
    gate.record_windows(repo, log, repo)
    assert gate.check_portability(repo)["status"] == "PASS"
