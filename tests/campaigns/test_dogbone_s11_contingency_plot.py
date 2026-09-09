from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from campaigns.dogbone_v1.s11_contingency_plot import (
    load_screen_metrics,
    render_screen_svg,
    write_screen_svg,
)

_EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "campaigns"
    / "dogbone-v1"
    / "S11-CONTINGENCY-EVIDENCE.json"
)
_SCREEN = _EVIDENCE.with_name("S11-CONTINGENCY-SCREEN.svg")
_SCREEN_SHA256 = "d92255515dbfaaa1d646eaf74253364211d028c5590727d712e36f90a5eaefae"


def test_screen_plot_is_deterministic_and_records_the_fallback_reason() -> None:
    metrics = load_screen_metrics(_EVIDENCE)

    first = render_screen_svg(metrics)
    second = render_screen_svg(metrics)

    assert first == second
    assert first.startswith(b'<svg xmlns="http://www.w3.org/2000/svg"')
    assert b"S11 contingency: frozen TRAIN-screen errors" in first
    assert b"A: baseline" in first
    assert b"B: derived ratios" in first
    assert b"C: local stress" in first
    assert b"D: combined" in first
    assert b"Formal outcome: fallback to A" in first
    assert b"C failed only the +2% displacement guardrail" in first
    tracked = _SCREEN.read_bytes().replace(b"\r\n", b"\n")
    assert tracked == first
    assert hashlib.sha256(first).hexdigest() == _SCREEN_SHA256


def test_screen_plot_writer_is_create_once_and_refuses_divergence(
    tmp_path: Path,
) -> None:
    payload = render_screen_svg(load_screen_metrics(_EVIDENCE))
    target = tmp_path / "screen.svg"

    write_screen_svg(payload, target)
    write_screen_svg(payload, target)
    assert target.read_bytes() == payload

    with pytest.raises(ValueError, match="different plot"):
        write_screen_svg(payload + b"changed", target)
