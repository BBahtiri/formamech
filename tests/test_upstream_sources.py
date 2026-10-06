"""G18.licenses: third-party GINOT modules are pinned, fetched, never committed."""

from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path

import pytest

from experiments.spike import fetch_upstream
from piginot.git_environment import isolated_git_environment

ROOT = Path(__file__).resolve().parents[1]
MODULES = "experiments/spike/upstream_pi_ginot/models/modules"


def _spike_pins() -> dict[str, str]:
    tree = ast.parse((ROOT / "experiments/spike/model.py").read_text("utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "UPSTREAM_SOURCE_SHA256"
        ):
            pins: dict[str, str] = ast.literal_eval(node.value)  # type: ignore[arg-type]
            return pins
    raise AssertionError("UPSTREAM_SOURCE_SHA256 not found")


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        env=isolated_git_environment(),
        capture_output=True,
        text=True,
        check=False,
    ).stdout


def test_manifest_pins_every_ginot_module_with_the_spike_digest() -> None:
    manifest = fetch_upstream.load_manifest()
    files = {item["path"]: item["sha256"] for item in manifest["files"]}  # type: ignore[attr-defined]
    pins = _spike_pins()
    assert set(files) == {path for path in pins if path.startswith("models/modules/")}
    assert all(files[path] == pins[path] for path in files)
    assert manifest["upstream_commit"] == "9f6794141beca9d3e66433465c3b07039a7e41d2"
    assert manifest["record"] == "docs/adr/upstream-sources.md"


def test_ginot_modules_are_not_tracked_and_are_ignored() -> None:
    assert _git("ls-files", "--", MODULES).strip() == ""
    assert _git("check-ignore", f"{MODULES}/point_encoding.py").strip()


def test_fetch_refuses_a_mismatching_file_and_writes_nothing(tmp_path: Path) -> None:
    manifest = {
        "files": [
            {"path": "a.py", "upstream_path": "a.py", "sha256": "0" * 64},
        ]
    }
    with pytest.raises(ValueError, match="differs from the pin"):
        fetch_upstream.fetch(lambda _: b"x = 1\n", manifest=manifest, root=tmp_path)
    assert not (tmp_path / "a.py").exists()


def test_fetch_writes_lf_content_that_matches_its_pin(tmp_path: Path) -> None:
    content = b"x = 1\r\ny = 2\r\n"
    digest = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
    manifest = {
        "files": [
            {"path": "pkg/a.py", "upstream_path": "src/a.py", "sha256": digest},
        ]
    }
    written = fetch_upstream.fetch(
        {"src/a.py": content}.__getitem__, manifest=manifest, root=tmp_path
    )
    assert written == ["pkg/a.py"]
    assert (tmp_path / "pkg/a.py").read_bytes() == b"x = 1\ny = 2\n"
