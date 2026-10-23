"""S22 CLI: --mode certified/exact; fast keeps the v0.1.0 output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from piginot.cli import main

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "bundles" / "notched-v1-cp-freeze"
REQUEST = ROOT / "bundles" / "requests" / "notched-in-domain.json"


def _run(tmp_path: Path, *extra: str) -> tuple[int, dict]:
    out = tmp_path / "out.json"
    code = main(
        [
            "predict",
            str(BUNDLE),
            "--request",
            str(REQUEST),
            "--output",
            str(out),
            *extra,
        ]
    )
    return code, json.loads(out.read_text()) if out.exists() else {}


def test_fast_is_the_default_and_unchanged(tmp_path: Path) -> None:
    code, default = _run(tmp_path)
    code_fast, fast = _run(tmp_path, "--mode", "fast")
    assert code == code_fast == 0 and default == fast
    assert "mode" not in fast


def test_certified_with_correction(tmp_path: Path) -> None:
    code, document = _run(tmp_path, "--mode", "certified", "--corrected")
    assert code == 0
    assert document["path"] == "operator+certificate"
    assert document["certificate"]["verdict"] == "OK"
    assert document["corrected_displacement"]


@pytest.mark.parametrize(
    "extra",
    [
        ("--mode", "fast", "--escalate"),
        ("--mode", "exact", "--tolerance", "0.01"),
        ("--mode", "certified", "--tolerance", "0.01"),
        ("--mode", "certified", "--escalate", "--tolerance", "0"),
        ("--mode", "certified", "--escalate", "--tolerance", "nan"),
    ],
)
def test_ignored_escalation_options_are_errors(
    tmp_path: Path, extra: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    code, document = _run(tmp_path, *extra)
    assert code == 1 and document == {}
    assert "piginot: --" in capsys.readouterr().err


def test_a_missing_recipe_is_an_error(tmp_path: Path) -> None:
    code, document = _run(
        tmp_path, "--mode", "exact", "--recipe", str(tmp_path / "none.json")
    )
    assert code == 1 and document == {}
