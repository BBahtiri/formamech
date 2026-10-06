"""G18 install and sign-off checks pass only on complete, matching evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from campaigns import g18_gate


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n')
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/limitations.md").write_text("Limits.\r\n")
    monkeypatch.setattr(g18_gate, "REPOSITORY", tmp_path)
    return tmp_path


def _result(os_name: str, **changes: Any) -> dict[str, Any]:
    result = {
        "os": os_name,
        "installed_version": "0.1.0",
        "passed": True,
        "checks": {"version": True, "claim_ceiling": True},
    }
    result.update(changes)
    return result


def _write_install(root: Path, results: list[dict[str, Any]], **changes: Any) -> None:
    evidence = {
        "run_url": "https://github.com/o/r/actions/runs/1",
        "index": "https://test.pypi.org/simple/",
        "version": "0.1.0",
        "results": results,
    }
    evidence.update(changes)
    path = root / g18_gate.INSTALL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence))


def test_install_is_not_run_without_evidence(repository: Path) -> None:
    assert g18_gate.install()["status"] == "NOT_RUN"


def test_install_passes_on_all_three_systems(repository: Path) -> None:
    _write_install(
        repository, [_result(name) for name in ("Linux", "Darwin", "Windows")]
    )
    check = g18_gate.install()
    assert check["status"] == "PASS"
    assert check["measured"]["passed_on"] == ["Darwin", "Linux", "Windows"]


@pytest.mark.parametrize(
    ("results", "changes"),
    [
        ([_result("Linux"), _result("Darwin")], {}),
        ([_result("Linux"), _result("Darwin"), _result("Windows", passed=False)], {}),
        (
            [
                _result("Linux"),
                _result("Darwin"),
                _result("Windows", checks={"version": False}),
            ],
            {},
        ),
        (
            [
                _result(name, installed_version="0.0.9")
                for name in ("Linux", "Darwin", "Windows")
            ],
            {},
        ),
        (
            [_result(name) for name in ("Linux", "Darwin", "Windows")],
            {"index": "https://pypi.org/simple/"},
        ),
        ([_result(name) for name in ("Linux", "Darwin", "Windows")], {"run_url": ""}),
    ],
)
def test_install_fails_on_incomplete_evidence(
    repository: Path, results: list[dict[str, Any]], changes: dict[str, Any]
) -> None:
    _write_install(repository, results, **changes)
    assert g18_gate.install()["status"] == "FAIL"


def _write_signoff(root: Path, **changes: Any) -> None:
    content = (root / "docs/limitations.md").read_bytes().replace(b"\r\n", b"\n")
    record = {
        "signed_off_by": "Betim Bahtiri",
        "signed_off_utc": "2026-10-06T00:00:00Z",
        "limitations_sha256": hashlib.sha256(content).hexdigest(),
        "confirmed": True,
    }
    record.update(changes)
    path = root / g18_gate.SIGNOFF_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def test_signoff_needs_the_owner_and_the_current_digest(repository: Path) -> None:
    assert g18_gate.signoff()["status"] == "NOT_RUN"
    _write_signoff(repository)
    assert g18_gate.signoff()["status"] == "PASS"
    _write_signoff(repository, signed_off_by="someone")
    assert g18_gate.signoff()["status"] == "FAIL"
    _write_signoff(repository, confirmed=False)
    assert g18_gate.signoff()["status"] == "FAIL"
    _write_signoff(repository)
    (repository / "docs/limitations.md").write_text("Changed after sign-off.\n")
    assert g18_gate.signoff()["status"] == "FAIL"
