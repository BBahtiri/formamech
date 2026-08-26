from __future__ import annotations

import ast
import subprocess
from pathlib import Path


def _doe_sources() -> tuple[Path, ...]:
    root = Path(__file__).parents[1] / "src" / "piginot" / "doe"
    return tuple(sorted(root.glob("*.py")))


def test_s9_has_no_fem_ml_dataset_or_spike_dependency() -> None:
    forbidden_roots = {
        "experiments",
        "numpy",
        "scipy",
        "torch",
        "piginot.fem",
    }
    violations: list[str] = []

    for source in _doe_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = (node.module,)
            for module in imported:
                if any(
                    module == root or module.startswith(root + ".")
                    for root in forbidden_roots
                ):
                    violations.append(f"{source.name}:{node.lineno}:{module}")

    assert violations == []


def test_s9_never_executes_text_as_code() -> None:
    violations: list[str] = []

    for source in _doe_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                violations.append(f"{source.name}:{node.lineno}:{node.func.id}")

    assert violations == []


def test_s9_does_not_create_downstream_modules() -> None:
    forbidden_stems = {
        "collocation",
        "dataset",
        "fem_runner",
        "model",
        "trainer",
        "training",
    }

    assert not ({path.stem for path in _doe_sources()} & forbidden_stems)


def test_private_preregistration_material_is_ignored_and_untracked() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    ignore_entries = {
        line.strip()
        for line in (repository_root / ".gitignore")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    tracked = subprocess.run(
        ["git", "ls-files", "--", ".piginot-private"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert ".piginot-private/" in ignore_entries
    assert tracked.stdout.strip() == ""
