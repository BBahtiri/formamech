"""G17.docs: every relative link of the README and the docs resolves."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_FENCE = re.compile(r"^(```|~~~)")
_PAGES = sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]


def _links(path: Path) -> list[str]:
    targets: list[str] = []
    fenced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if _FENCE.match(line.strip()):
            fenced = not fenced
            continue
        if fenced:
            continue
        targets.extend(_LINK.findall(re.sub(r"`[^`]*`", "", line)))
    return targets


@pytest.mark.parametrize("page", _PAGES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_relative_link_resolves(page: Path) -> None:
    broken = []
    for target in _links(page):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            continue
        relative = target.split("#", 1)[0]
        if not (page.parent / relative).exists():
            broken.append(target)
    assert not broken, f"{page.relative_to(ROOT)} has broken links: {broken}"


def test_limitations_are_linked_above_the_fold_in_the_readme() -> None:
    lines = [
        line for line in (ROOT / "README.md").read_text().splitlines() if line.strip()
    ]
    assert any("docs/limitations.md" in line for line in lines[:8])
