from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import campaigns.dogbone_v1.run_s11_contingency as runner
from campaigns.dogbone_v1.s11_contingency_artifacts import RUN_RESERVATION_NAME
from campaigns.dogbone_v1.s11_contingency_run import load_committed_protocol

_REVISION = "1" * 40


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "runner@example.invalid")
    _git(repo, "config", "user.name", "Runner Test")
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "test fixture")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_verified_clean_head_accepts_only_the_exact_clean_revision(
    tmp_path: Path,
) -> None:
    repo, revision = _repository(tmp_path)

    assert runner.verified_clean_head(repo, revision) == revision

    with pytest.raises(ValueError, match="HEAD"):
        runner.verified_clean_head(repo, "0" * 40)
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean"):
        runner.verified_clean_head(repo, revision)


def test_preflight_failure_occurs_before_source_access_or_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_opened = False

    def fail_preflight(repo_root: Path, revision: str) -> str:
        raise RuntimeError("worktree must be clean")

    def forbidden_source(*args: object, **kwargs: object) -> object:
        nonlocal source_opened
        source_opened = True
        raise AssertionError("source must remain closed")

    monkeypatch.setattr(runner, "verified_clean_head", fail_preflight)
    monkeypatch.setattr(runner, "open_selective_s10_source", forbidden_source)
    output_root = tmp_path / "output"

    with pytest.raises(RuntimeError, match="clean"):
        runner.run_architecture_contingency(
            repo_root=tmp_path,
            s10_run_root=tmp_path / "s10",
            output_root=output_root,
            implementation_revision=_REVISION,
        )

    assert not source_opened
    assert not output_root.exists()


def test_empty_output_is_reserved_before_source_and_full_run_is_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_committed_protocol()
    s10_root = tmp_path / "s10"
    s10_root.mkdir()
    output_root = tmp_path / "output"
    events: list[str] = []
    fake_source = SimpleNamespace(bindings=object())
    fake_result = object()
    fake_evidence = SimpleNamespace(content_hash="f" * 64)

    monkeypatch.setattr(
        runner,
        "verified_clean_head",
        lambda repo_root, revision: events.append("git") or revision,
    )
    monkeypatch.setattr(
        runner,
        "load_committed_protocol",
        lambda: events.append("protocol") or protocol,
    )

    def open_source(
        run_root: Path,
        *,
        development_access_record: Path,
    ) -> object:
        assert (output_root / RUN_RESERVATION_NAME).is_file()
        assert run_root == s10_root
        assert development_access_record == (
            output_root / "DEVELOPMENT-ACCESS.json"
        )
        events.append("source")
        return fake_source

    def execute(
        protocol_arg: object,
        source_arg: object,
        *,
        implementation_revision: str,
        progress: object,
    ) -> object:
        assert protocol_arg is protocol
        assert source_arg is fake_source
        assert implementation_revision == _REVISION
        assert callable(progress)
        progress("candidate:start:A-s11-baseline")
        events.append("execute")
        return fake_result

    def persist(
        reservation: object,
        *,
        protocol: object,
        source_bindings: object,
        result: object,
    ) -> object:
        assert reservation.output_root == output_root
        assert source_bindings is fake_source.bindings
        assert result is fake_result
        events.append("persist")
        return fake_evidence

    monkeypatch.setattr(runner, "open_selective_s10_source", open_source)
    monkeypatch.setattr(runner, "execute_architecture_contingency", execute)
    monkeypatch.setattr(runner, "persist_contingency_result", persist)

    evidence = runner.run_architecture_contingency(
        repo_root=tmp_path,
        s10_run_root=s10_root,
        output_root=output_root,
        implementation_revision=_REVISION,
        progress=events.append,
    )

    assert evidence is fake_evidence
    assert events.index("git") < events.index("source")
    assert events.index("source") < events.index("execute")
    assert events.index("execute") < events.index("persist")
    assert "candidate:start:A-s11-baseline" in events


def test_nonempty_output_fails_before_source_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_committed_protocol()
    s10_root = tmp_path / "s10"
    s10_root.mkdir()
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / "previous.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "verified_clean_head", lambda *_args: _REVISION)
    monkeypatch.setattr(runner, "load_committed_protocol", lambda: protocol)
    monkeypatch.setattr(
        runner,
        "open_selective_s10_source",
        lambda *_args, **_kwargs: pytest.fail("source must remain closed"),
    )

    with pytest.raises(FileExistsError, match="empty"):
        runner.run_architecture_contingency(
            repo_root=tmp_path,
            s10_run_root=s10_root,
            output_root=output_root,
            implementation_revision=_REVISION,
        )


def test_output_inside_s10_is_refused_before_reservation_or_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s10_root = tmp_path / "s10"
    s10_root.mkdir()
    output_root = s10_root / "s11-output"
    monkeypatch.setattr(runner, "verified_clean_head", lambda *_args: _REVISION)
    monkeypatch.setattr(
        runner,
        "open_selective_s10_source",
        lambda *_args, **_kwargs: pytest.fail("source must remain closed"),
    )

    with pytest.raises(ValueError, match="outside the S10"):
        runner.run_architecture_contingency(
            repo_root=tmp_path,
            s10_run_root=s10_root,
            output_root=output_root,
            implementation_revision=_REVISION,
        )

    assert not output_root.exists()


def test_cli_requires_explicit_paths_and_streams_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    s10_root = tmp_path / "s10"
    output_root = tmp_path / "output"
    for path in (repo_root, s10_root):
        path.mkdir()

    def fake_run(**kwargs: object) -> object:
        progress = kwargs["progress"]
        assert callable(progress)
        progress("phase:TRAIN_FIT")
        progress("candidate:start:A-s11-baseline")
        progress("candidate:complete:A-s11-baseline")
        return SimpleNamespace(content_hash="e" * 64)

    monkeypatch.setattr(runner, "run_architecture_contingency", fake_run)

    assert runner.main(
        [
            "--repo-root",
            str(repo_root),
            "--s10-run-root",
            str(s10_root),
            "--output-root",
            str(output_root),
            "--implementation-revision",
            _REVISION,
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "phase:TRAIN_FIT" in output
    assert "candidate:start:A-s11-baseline" in output
    assert "candidate:complete:A-s11-baseline" in output
    assert "e" * 64 in output


def test_cli_exposes_no_budget_or_role_override() -> None:
    parser = runner.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--optimizer-steps" not in option_strings
    assert "--case-ids" not in option_strings
    assert "--role" not in option_strings
    assert "--resume" not in option_strings
    assert "--overwrite" not in option_strings
