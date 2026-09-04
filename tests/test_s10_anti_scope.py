from __future__ import annotations

import ast
from pathlib import Path


def _dataset_sources() -> tuple[Path, ...]:
    root = Path(__file__).parents[1] / "src" / "piginot" / "datasets"
    return tuple(sorted(root.glob("*.py")))


def test_s10_never_imports_legacy_experiments_or_training_modules() -> None:
    forbidden_roots = {
        "experiments",
        "piginot.training",
        "piginot.models",
    }
    violations: list[str] = []

    for source in _dataset_sources():
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
                    violations.append(
                        f"{source.name}:{getattr(node, 'lineno', 0)}:{module}"
                    )

    assert violations == []


def test_s10_source_and_tests_never_reference_private_reveal_material() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    files = (
        *_dataset_sources(),
        *sorted((repository_root / "tests" / "datasets").glob("*.py")),
    )
    forbidden_literals = (
        ".piginot-private",
        "private_reveal",
        "reveal-v1.json",
        "holdout_salt",
        "CampaignSecrets",
        "generate_campaign_secrets",
    )
    violations: list[str] = []

    for source in files:
        text = source.read_text(encoding="utf-8")
        for literal in forbidden_literals:
            if literal in text:
                violations.append(f"{source.relative_to(repository_root)}:{literal}")

    assert violations == []


def test_s10_never_executes_text_as_code() -> None:
    violations: list[str] = []

    for source in _dataset_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                violations.append(f"{source.name}:{node.lineno}:{node.func.id}")

    assert violations == []
