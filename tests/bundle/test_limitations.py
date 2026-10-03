from __future__ import annotations

from pathlib import Path

from piginot.bundle import REQUIRED_LIMITATION_PARAGRAPH

ROOT = Path(__file__).resolve().parents[2]


def test_the_required_paragraph_is_the_published_limitation() -> None:
    published = " ".join((ROOT / "docs" / "limitations.md").read_text().split())
    assert REQUIRED_LIMITATION_PARAGRAPH in published
    for phrase in (
        "not experimental validation",
        "outside that bundle's declared input envelope",
        "not a mathematical proof of a uniform error bound",
    ):
        assert phrase in REQUIRED_LIMITATION_PARAGRAPH
