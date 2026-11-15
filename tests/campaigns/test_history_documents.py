"""Historical documentation remains verifiable after a working-tree cleanup."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from campaigns import g17_gate, g18_gate, g20_gate
from campaigns.history import read_document, read_document_bytes
from piginot.git_environment import isolated_git_environment

DOCUMENT = Path("docs/adr/decision.md")
ACCEPTED = b"# Decision\r\n\r\n- Status: Accepted\r\n"


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", *args], cwd=root, env=isolated_git_environment()
    )


def _repo(root: Path) -> Path:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Documentation test")
    _git(root, "config", "user.email", "documentation@example.invalid")
    _git(root, "config", "core.autocrlf", "false")
    return root


def _commit_document(root: Path, relative: Path, body: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    _git(root, "add", "--", relative.as_posix())
    _git(root, "commit", "-qm", "document")
    return path


def test_a_local_document_does_not_require_git(tmp_path: Path) -> None:
    path = tmp_path / DOCUMENT
    path.parent.mkdir(parents=True)
    path.write_bytes(ACCEPTED)
    assert read_document_bytes(tmp_path, DOCUMENT) == ACCEPTED
    assert read_document(tmp_path, DOCUMENT) == path.read_text(encoding="utf-8")


def test_untracked_removal_reads_the_last_committed_bytes(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    path = _commit_document(root, DOCUMENT, ACCEPTED)
    path.unlink()
    assert read_document_bytes(root, DOCUMENT) == ACCEPTED
    assert not path.exists()


def test_a_committed_removal_keeps_the_latest_document_version(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    _commit_document(root, DOCUMENT, ACCEPTED)
    updated = "# Decision\n\nStatus: Accepted\n\nRésumé recorded.\n".encode()
    path = _commit_document(root, DOCUMENT, updated)
    _git(root, "rm", "-q", "--", DOCUMENT.as_posix())
    _git(root, "commit", "-qm", "remove local-only document")
    before = _git(root, "status", "--porcelain")
    assert read_document(root, DOCUMENT) == updated.decode()
    assert _git(root, "status", "--porcelain") == before
    assert not path.exists()


def test_present_local_changes_keep_the_existing_read_semantics(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    path = _commit_document(root, DOCUMENT, ACCEPTED)
    path.write_text("Status: Proposed\n", encoding="utf-8")
    assert read_document(root, DOCUMENT) == "Status: Proposed\n"


def test_committed_reads_do_not_trust_an_ignored_local_edit(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    path = _commit_document(root, DOCUMENT, ACCEPTED)
    path.write_bytes(b"Status: Proposed\n")
    assert read_document_bytes(root, DOCUMENT, committed=True) == ACCEPTED


@pytest.mark.parametrize("git_repository", [False, True])
def test_an_unknown_document_fails_closed(tmp_path: Path, git_repository: bool) -> None:
    root = _repo(tmp_path / "repo") if git_repository else tmp_path
    with pytest.raises(FileNotFoundError, match="document|history"):
        read_document(root, DOCUMENT)


@pytest.mark.parametrize("relative", ["", ".", "../outside.md", "/outside.md"])
def test_document_paths_must_stay_inside_the_repository(
    tmp_path: Path, relative: str
) -> None:
    with pytest.raises(ValueError, match="relative|repository"):
        read_document(tmp_path, relative)


def test_a_symlink_cannot_read_outside_the_repository(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_bytes(ACCEPTED)
    path = root / "linked.md"
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("the OS refuses to create symbolic links")
    with pytest.raises(ValueError, match="repository"):
        read_document(root, "linked.md")


def test_unavailable_shallow_history_is_not_success(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    _commit_document(root, DOCUMENT, ACCEPTED)
    _git(root, "rm", "-q", "--", DOCUMENT.as_posix())
    _git(root, "commit", "-qm", "remove document")
    clone = tmp_path / "shallow"
    _git(tmp_path, "clone", "-q", "--depth=1", "--no-local", str(root), str(clone))
    with pytest.raises(FileNotFoundError, match="history"):
        read_document(clone, DOCUMENT)


def test_repository_local_git_environment_does_not_redirect_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path / "repo")
    path = _commit_document(root, DOCUMENT, ACCEPTED)
    path.unlink()
    other = _repo(tmp_path / "other")
    _commit_document(other, DOCUMENT, b"Status: Proposed\n")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    assert read_document_bytes(root, DOCUMENT) == ACCEPTED


def test_a_missing_historical_blob_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = iter(
        [
            subprocess.CompletedProcess([], 0, b"a" * 40 + b"\n", b""),
            subprocess.CompletedProcess([], 128, b"", b"missing blob"),
        ]
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(results))
    with pytest.raises(FileNotFoundError, match="history"):
        read_document_bytes(tmp_path, DOCUMENT)


def test_history_reads_disable_network_and_credential_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    results = iter(
        [
            subprocess.CompletedProcess([], 0, b"a" * 40 + b"\n", b""),
            subprocess.CompletedProcess([], 0, ACCEPTED, b""),
        ]
    )

    def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(kwargs)
        return next(results)

    monkeypatch.setattr(subprocess, "run", run)
    assert read_document_bytes(tmp_path, DOCUMENT) == ACCEPTED
    assert len(calls) == 2
    for call in calls:
        assert call["env"]["GIT_ALLOW_PROTOCOL"] == ""
        assert call["env"]["GIT_NO_LAZY_FETCH"] == "1"
        assert call["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_g17_reads_accepted_adrs_from_history_without_restoring_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path / "repo")
    path = _commit_document(root, DOCUMENT, ACCEPTED)
    _git(root, "rm", "-q", "--", DOCUMENT.as_posix())
    _git(root, "commit", "-qm", "remove document")
    monkeypatch.setattr(g17_gate, "REPOSITORY", root)
    monkeypatch.setattr(g17_gate, "REQUIRED_ADRS", {"test decision": str(DOCUMENT)})
    assert g17_gate.adrs()["status"] == "PASS"
    assert not path.exists()


def test_g17_does_not_accept_missing_or_unaccepted_documentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(g17_gate, "REPOSITORY", tmp_path)
    monkeypatch.setattr(g17_gate, "REQUIRED_ADRS", {"test decision": str(DOCUMENT)})
    assert g17_gate.adrs()["status"] == "FAIL"
    path = tmp_path / DOCUMENT
    path.parent.mkdir(parents=True)
    path.write_text("Status: Proposed\n")
    assert g17_gate.adrs()["status"] == "FAIL"


def test_g17_does_not_fall_back_past_the_latest_unaccepted_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path / "repo")
    _commit_document(root, DOCUMENT, ACCEPTED)
    path = _commit_document(root, DOCUMENT, b"Status: Proposed\n")
    path.unlink()
    monkeypatch.setattr(g17_gate, "REPOSITORY", root)
    monkeypatch.setattr(g17_gate, "REQUIRED_ADRS", {"test decision": str(DOCUMENT)})
    assert g17_gate.adrs()["status"] == "FAIL"


def test_g18_reads_recorded_explanations_from_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path / "repo")
    path = _commit_document(root, DOCUMENT, b"G0: explicitly not run.\n")
    path.unlink()
    (root / "gate.json").write_text(json.dumps({"verdict": "NOT_RUN"}))
    monkeypatch.setattr(g18_gate, "REPOSITORY", root)
    monkeypatch.setattr(g18_gate, "RELEASE_ADR", DOCUMENT)
    monkeypatch.setattr(g18_gate, "GATE_FILES", {"G0": (Path("gate.json"),)})
    assert g18_gate.all_gates()["status"] == "PASS"
    assert not path.exists()


def test_g18_does_not_accept_an_unexplained_missing_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "gate.json").write_text(json.dumps({"verdict": "NOT_RUN"}))
    monkeypatch.setattr(g18_gate, "REPOSITORY", tmp_path)
    monkeypatch.setattr(g18_gate, "RELEASE_ADR", DOCUMENT)
    monkeypatch.setattr(g18_gate, "GATE_FILES", {"G0": (Path("gate.json"),)})
    assert g18_gate.all_gates()["status"] == "FAIL"


def _rca_inputs(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    result = {"hypotheses": {name: {"verdict": "SUPPORTED"} for name in "abcde"}}
    for relative in (
        g20_gate.postmortem_owner.D7_PATH,
        g20_gate.postmortem_owner.D8_PATH,
        g20_gate.postmortem_owner.VERDICTS_PATH,
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
    monkeypatch.setattr(g20_gate.postmortem_owner, "verdicts", lambda root: result)
    return result


def test_g20_reads_the_recorded_result_from_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path / "repo")
    path = _commit_document(
        root, g20_gate.ADR_PATH, b"# ADR\n\n## Post-mortem result\n\nRecorded.\n"
    )
    path.unlink()
    _rca_inputs(root, monkeypatch)
    assert g20_gate.check_rca(root)["status"] == "PASS"
    assert not path.exists()


def test_g20_does_not_accept_missing_historical_documentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _rca_inputs(tmp_path, monkeypatch)
    assert g20_gate.check_rca(tmp_path)["status"] == "NOT_RUN"
