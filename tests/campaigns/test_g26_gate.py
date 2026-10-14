"""G26 gate logic on synthetic records; no training and no FEM re-solve."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from campaigns import g26_gate as gate
from campaigns.dogbone_v1 import material_run
from campaigns.dogbone_v1.s26_hgo import RUN
from piginot.contracts import canonical_json_bytes
from tests.git_helpers import run_git


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "-q")
    run_git(repo, "config", "user.name", "t")
    run_git(repo, "config", "user.email", "t@example.invalid")
    (repo / "a.txt").write_text("a")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-q", "-m", "a")
    return repo


def _commit(repo: Path, message: str) -> str:
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-q", "-m", message)
    return run_git(repo, "rev-parse", "HEAD").strip()


def _references(revision: str) -> dict[str, Any]:
    cases = {
        case_id: {"request_content_hash": "0" * 64}
        for case_id in RUN.build_protocol()["viability"]["references"]["cases"]
    }
    return RUN.references_document(cases, revision)


def _screen(references: dict[str, Any], revision: str, **changes: Any) -> Any:
    record = {
        "passed": True,
        "protocol_sha256": hashlib.sha256(
            (material_run.REPOSITORY / RUN.protocol_path).read_bytes()
        ).hexdigest(),
        "references_sha256": hashlib.sha256(
            canonical_json_bytes(references)
        ).hexdigest(),
        "problem_spec_hash": RUN.load_spec().problem_hash,
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
    (root / RUN.protocol_path).parent.mkdir(parents=True)
    shutil.copy(material_run.REPOSITORY / RUN.protocol_path, root / RUN.protocol_path)
    return root


def test_missing_records_are_not_run(tmp_path: Path) -> None:
    assert gate.check_viability(tmp_path)["status"] == "NOT_RUN"
    assert gate.check_portability(tmp_path)["status"] == "NOT_RUN"
    assert gate.check_quality(tmp_path)["status"] == "NOT_RUN"
    assert gate.check_protocol_first(_repository(tmp_path))["status"] == "NOT_RUN"


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ({}, "PASS"),
        ({"failed_guardrails": ["u_relative_l2"], "passed": False}, "FAIL"),
        ({"checkpoint_step": 2000}, "FAIL"),
        ({"development_opened": True}, "FAIL"),
        ({"references_sha256": "0" * 64}, "FAIL"),
    ],
    ids=["pass", "guardrail", "step", "development", "references"],
)
def test_viability_checks_the_run_against_its_references(
    protocol_root: Path, change: dict[str, Any], status: str
) -> None:
    references = _references("r" * 40)
    _write(protocol_root, RUN.references_path, references)
    _write(protocol_root, RUN.screen_path, _screen(references, "r" * 40, **change))
    assert gate.check_viability(protocol_root)["status"] == status


def test_protocol_first_orders_the_protocol_before_both_runs(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    adr = repo / gate.ADR_PATH
    adr.parent.mkdir(parents=True)
    adr.write_text("# ADR\n\n- Status: Accepted\n")
    (repo / RUN.protocol_path).parent.mkdir(parents=True)
    (repo / RUN.protocol_path).write_text("{}")
    first = _commit(repo, "protocol")
    (repo / "b.txt").write_text("b")
    later = _commit(repo, "code")
    references = _references(first)
    _write(repo, RUN.references_path, references)
    _write(repo, RUN.screen_path, _screen(references, later))
    record = gate.check_protocol_first(repo)
    assert record["status"] == "PASS", record
    _write(repo, RUN.screen_path, _screen(references, "f" * 40))
    assert gate.check_protocol_first(repo)["status"] == "FAIL"


def test_suite_records_portability_and_coverage(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps(
            {
                "meta": {"branch_coverage": True},
                "totals": {"percent_covered": 92.0},
                "files": {
                    "src/piginot/materials/values.py": {
                        "summary": {"num_statements": 10, "covered_lines": 10}
                    }
                },
            }
        )
    )
    log = tmp_path / "pytest.log"
    log.write_text("===== 3001 passed in 9.00s =====\n")
    document = gate.record_suite(repo, coverage, log)
    assert document["clean"] and document["coverage_percent"] == 92.0
    gate.record_windows(repo, log, repo)
    assert gate.check_portability(repo)["status"] == "PASS"
