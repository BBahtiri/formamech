from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "piginot"
MUTATION = SOURCE / "mutation"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def test_no_production_module_imports_the_mutation_engine() -> None:
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in SOURCE.rglob("*.py")
        if MUTATION not in path.parents
        and any(
            name == "piginot.mutation" or name.startswith("piginot.mutation.")
            for name in _imports(path)
        )
    ]
    assert offenders == []


def test_the_mutation_engine_has_no_campaign_network_or_process_imports() -> None:
    forbidden = (
        "campaigns",
        "experiments",
        "http",
        "requests",
        "urllib",
        "subprocess",
        "socket",
        "pickle",
    )
    imported = set().union(*(_imports(path) for path in MUTATION.glob("*.py")))
    assert not any(
        name == root or name.startswith(root + ".")
        for name in imported
        for root in forbidden
    )
