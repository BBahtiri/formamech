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
from campaigns.dogbone_v1.s26_references import AMENDMENT_PATH, build_amendment
from campaigns.history import read_document_bytes
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


def _references(revision: str, root: Path) -> dict[str, Any]:
    cases = {
        case_id: {"request_content_hash": "0" * 64}
        for case_id in RUN.build_protocol()["viability"]["references"]["cases"]
    }
    return {
        **RUN.references_document(cases, revision),
        "reference_amendment_sha256": hashlib.sha256(
            (root / AMENDMENT_PATH).read_bytes()
        ).hexdigest(),
    }


def _screen(references: dict[str, Any], revision: str, **changes: Any) -> Any:
    record = {
        "passed": True,
        "protocol_sha256": hashlib.sha256(
            (material_run.REPOSITORY / RUN.protocol_path).read_bytes()
        ).hexdigest(),
        "references_sha256": hashlib.sha256(
            canonical_json_bytes(references)
        ).hexdigest(),
        "reference_amendment_sha256": references["reference_amendment_sha256"],
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
    root = _repository(tmp_path)
    (root / RUN.protocol_path).parent.mkdir(parents=True)
    shutil.copy(material_run.REPOSITORY / RUN.protocol_path, root / RUN.protocol_path)
    (root / gate.ADR_PATH).parent.mkdir(parents=True)
    (root / gate.ADR_PATH).write_bytes(
        read_document_bytes(material_run.REPOSITORY, gate.ADR_PATH)
    )
    _commit(root, "protocol")
    _write(root, AMENDMENT_PATH, build_amendment(root))
    _commit(root, "reference amendment")
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
    references = _references("r" * 40, protocol_root)
    _write(protocol_root, RUN.references_path, references)
    _write(protocol_root, RUN.screen_path, _screen(references, "r" * 40, **change))
    assert gate.check_viability(protocol_root)["status"] == status


@pytest.mark.parametrize("reference_at_amendment", [False, True])
def test_protocol_first_orders_the_protocol_before_both_runs(
    protocol_root: Path, reference_at_amendment: bool
) -> None:
    repo = protocol_root
    first = run_git(repo, "rev-parse", "HEAD^")
    amendment = run_git(repo, "rev-parse", "HEAD")
    (repo / "b.txt").write_text("b")
    later = _commit(repo, "code")
    references = _references(amendment if reference_at_amendment else later, repo)
    _write(repo, RUN.references_path, references)
    _write(repo, RUN.screen_path, _screen(references, first))
    record = gate.check_protocol_first(repo)
    assert record["status"] == "PASS", record
    _write(repo, RUN.screen_path, _screen(references, "f" * 40))
    assert gate.check_protocol_first(repo)["status"] == "FAIL"


def test_protocol_first_requires_a_reference_amendment(protocol_root: Path) -> None:
    repo = protocol_root
    revision = run_git(repo, "rev-parse", "HEAD")
    references = _references(revision, repo)
    _write(repo, RUN.references_path, references)
    _write(repo, RUN.screen_path, _screen(references, revision))
    (repo / AMENDMENT_PATH).unlink()
    assert gate.check_protocol_first(repo)["status"] == "FAIL"


def test_viability_requires_a_reference_amendment(protocol_root: Path) -> None:
    references = _references("r" * 40, protocol_root)
    _write(protocol_root, RUN.references_path, references)
    _write(protocol_root, RUN.screen_path, _screen(references, "r" * 40))
    (protocol_root / AMENDMENT_PATH).unlink()
    assert gate.check_viability(protocol_root)["status"] == "FAIL"


@pytest.mark.parametrize("change", ["tampered", "uncommitted", "multiple-commits"])
def test_protocol_first_requires_the_frozen_committed_amendment(
    protocol_root: Path, change: str
) -> None:
    path = protocol_root / AMENDMENT_PATH
    if change == "tampered":
        document = json.loads(path.read_text("utf-8"))
        _write(protocol_root, AMENDMENT_PATH, {**document, "unregistered_change": True})
        run_git(protocol_root, "add", AMENDMENT_PATH.as_posix())
        run_git(protocol_root, "commit", "--amend", "--no-edit", "-q")
    else:
        path.write_bytes(path.read_bytes() + b"\n")
        if change == "multiple-commits":
            _commit(protocol_root, "second amendment commit")
    revision = run_git(protocol_root, "rev-parse", "HEAD")
    references = _references(revision, protocol_root)
    _write(protocol_root, RUN.references_path, references)
    _write(protocol_root, RUN.screen_path, _screen(references, revision))
    assert gate.check_protocol_first(protocol_root)["status"] == "FAIL"


def test_protocol_first_refuses_an_amendment_after_the_references(
    protocol_root: Path,
) -> None:
    before_amendment = run_git(protocol_root, "rev-parse", "HEAD^")
    references = _references(before_amendment, protocol_root)
    _write(protocol_root, RUN.references_path, references)
    _write(protocol_root, RUN.screen_path, _screen(references, before_amendment))
    assert gate.check_protocol_first(protocol_root)["status"] == "FAIL"


@pytest.mark.parametrize("order", ["same-commit", "amendment-first"])
def test_the_amendment_must_follow_the_original_protocol(
    tmp_path: Path, order: str
) -> None:
    repo = _repository(tmp_path)
    protocol = repo / RUN.protocol_path
    protocol.parent.mkdir(parents=True)
    protocol.write_bytes((material_run.REPOSITORY / RUN.protocol_path).read_bytes())
    (repo / gate.ADR_PATH).parent.mkdir(parents=True)
    (repo / gate.ADR_PATH).write_bytes(
        read_document_bytes(material_run.REPOSITORY, gate.ADR_PATH)
    )
    _write(repo, AMENDMENT_PATH, build_amendment(repo))
    if order == "amendment-first":
        protocol.unlink()
        _commit(repo, "amendment before protocol")
        protocol.write_bytes((material_run.REPOSITORY / RUN.protocol_path).read_bytes())
    revision = _commit(repo, "protocol")
    references = _references(revision, repo)
    _write(repo, RUN.references_path, references)
    _write(repo, RUN.screen_path, _screen(references, revision))
    assert gate.check_protocol_first(repo)["status"] == "FAIL"


@pytest.mark.parametrize("record_name", ["references", "screen"])
@pytest.mark.parametrize("value", [None, "0" * 64], ids=["missing", "mismatched"])
def test_viability_binds_both_records_to_the_reference_amendment(
    protocol_root: Path, record_name: str, value: str | None
) -> None:
    references = _references("r" * 40, protocol_root)
    screen = _screen(references, "r" * 40)
    record = references if record_name == "references" else screen
    if value is None:
        record.pop("reference_amendment_sha256")
    else:
        record["reference_amendment_sha256"] = value
    screen["references_sha256"] = hashlib.sha256(
        canonical_json_bytes(references)
    ).hexdigest()
    _write(protocol_root, RUN.references_path, references)
    _write(protocol_root, RUN.screen_path, screen)
    assert gate.check_viability(protocol_root)["status"] == "FAIL"


def test_protocol_first_still_refuses_changes_to_the_original_protocol(
    protocol_root: Path,
) -> None:
    run_git(protocol_root, "reset", "--hard", "HEAD^")
    path = protocol_root / RUN.protocol_path
    path.write_bytes(path.read_bytes() + b"\n")
    _commit(protocol_root, "second protocol commit")
    _write(protocol_root, AMENDMENT_PATH, build_amendment(protocol_root))
    revision = _commit(protocol_root, "bind changed protocol")
    references = _references(revision, protocol_root)
    _write(protocol_root, RUN.references_path, references)
    _write(protocol_root, RUN.screen_path, _screen(references, revision))
    assert gate.check_protocol_first(protocol_root)["status"] == "FAIL"


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
