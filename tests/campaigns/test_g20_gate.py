"""G20 gate logic: log parsing, record trees, holdout seals, missing evidence."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from campaigns import fresh_holdouts as fh
from campaigns import g20_gate as gate
from piginot.doe.secret_material import generate_campaign_secrets

LOG = """
.....s....
=========================== short test summary info ============================
SKIPPED [2] tests/a.py:10: the OS refuses to create symbolic links
SKIPPED [1] tests/b.py:3: the OS has no POSIX permission bits
================= 3063 passed, 3 skipped in 3354.67s (0:55:54) =================
"""


def test_the_pytest_log_is_parsed_with_skip_reasons() -> None:
    parsed = gate.parse_pytest_log(LOG)
    assert parsed["counts"] == {"passed": 3063, "skipped": 3}
    assert [s["count"] for s in parsed["skips"]] == [2, 1]
    assert parsed["skips"][0]["reason"].startswith("the OS refuses")
    failed = gate.parse_pytest_log("== 1 failed, 5 passed, 2 errors in 1.0s ==")
    assert failed["counts"] == {"failed": 1, "passed": 5, "errors": 2}


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.name", "t"],
        ["git", "config", "user.email", "t@example.invalid"],
    ):
        subprocess.run(command, cwd=root, check=True)
    (root / "code.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "code"], cwd=root, check=True)
    return root


def test_record_files_do_not_change_the_code_tree(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    tree = gate._tree(root)
    record = root / gate.SUITE_PATH
    record.parent.mkdir(parents=True)
    record.write_text("{}\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "record"], cwd=root, check=True)
    assert gate._same_code(root, tree)
    (root / "code.py").write_text("x = 2\n")
    subprocess.run(["git", "commit", "-qam", "change"], cwd=root, check=True)
    assert not gate._same_code(root, tree)
    assert not gate._same_code(root, "0" * 40)


def test_missing_evidence_is_not_run(tmp_path: Path) -> None:
    assert gate.check_portability(tmp_path)["status"] == "NOT_RUN"
    assert gate.check_rca(tmp_path)["status"] == "NOT_RUN"


@pytest.fixture(scope="module")
def drawn() -> fh.FreshDraw:
    adapter = fh._adapters()["notched-v1"]
    return fh.draw(
        adapter,
        secrets=generate_campaign_secrets(),
        manifest=json.loads(
            (
                fh.REPOSITORY / adapter.artifacts / "public-case-manifest.json"
            ).read_text()
        ),
        old_holdout=[],
        committed_envelope=(
            fh.REPOSITORY / adapter.artifacts / "executable-envelope.json"
        ).read_bytes(),
        created_utc="2026-10-11T00:00:00Z",
    )


def _seal_repo(tmp_path: Path, public: dict) -> Path:
    root = _repo(tmp_path)
    for relative in (
        "artifacts/campaigns/notched-v1",
        "artifacts/evaluation",
        "docs/adr",
    ):
        shutil.copytree(fh.REPOSITORY / relative, root / relative, dirs_exist_ok=True)
    # the scratch "decision" commit must predate any draw, also once the real
    # FRESH-HOLDOUTS.json is committed in the repository
    (root / "artifacts/campaigns/notched-v1" / fh.PUBLIC_NAME).unlink(missing_ok=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "decision"], cwd=root, check=True)
    path = root / "artifacts/campaigns/notched-v1" / fh.PUBLIC_NAME
    path.write_text(json.dumps(public))
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "draw"], cwd=root, check=True)
    return root


def test_a_correct_seal_passes(tmp_path: Path, drawn: fh.FreshDraw) -> None:
    root = _seal_repo(tmp_path, fh.public_document(drawn))
    decision = gate._added_in(root, gate.ADR_PATH)
    family = gate._holdout_family(root, "notched-v1", decision)
    assert family["passed"], family


@pytest.mark.parametrize("defect", ["reused_id", "overlap", "short", "unbound"])
def test_defective_seals_fail(tmp_path: Path, drawn: fh.FreshDraw, defect: str) -> None:
    public = json.loads(json.dumps(fh.public_document(drawn)))
    if defect == "reused_id":
        manifest = json.loads(
            (
                fh.REPOSITORY
                / "artifacts/campaigns/notched-v1/public-case-manifest.json"
            ).read_text()
        )
        public["sets"]["H1"]["case_ids"][0] = manifest["cases"][0]["case_id"]
    elif defect == "overlap":
        public["sets"]["H2"]["case_ids"][0] = public["sets"]["H1"]["case_ids"][0]
    elif defect == "short":
        public["sets"]["H1"]["case_ids"].pop()
    else:
        public["sets"]["H1"]["commitment"]["campaign_hash"] = "f" * 64
    root = _seal_repo(tmp_path, public)
    decision = gate._added_in(root, gate.ADR_PATH)
    assert not gate._holdout_family(root, "notched-v1", decision)["passed"]


def test_the_adr_result_must_not_be_pending(tmp_path: Path) -> None:
    import re

    pattern = re.compile(r"\bpending\b", re.IGNORECASE)
    assert pattern.search("Owner-machine part: pending.")
    assert not pattern.search("the peak changed without depending on distance")
