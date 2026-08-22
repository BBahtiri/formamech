from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MORPHING_ROOT = REPOSITORY_ROOT / "src" / "piginot" / "morphing"

FORBIDDEN_IMPORTS = {
    "experiments",
    "piginot.fem",
    "piginot.physics",
    "skfem",
    "torch",
}
FORBIDDEN_OPERATION_FRAGMENTS = {
    "assign_role",
    "defeatur",
    "generate_mesh",
    "heal",
    "remesh",
    "repair_mesh",
    "run_fem",
    "train",
}


def test_morphing_has_no_experiments_fem_physics_or_ml_import_path() -> None:
    imports: set[str] = set()
    for path in MORPHING_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)

    assert all(
        not any(
            imported == forbidden or imported.startswith(f"{forbidden}.")
            for forbidden in FORBIDDEN_IMPORTS
        )
        for imported in imports
    )


def test_morphing_contains_no_healing_remeshing_doe_fem_or_training_operation() -> None:
    definitions: set[str] = set()
    for path in MORPHING_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions.update(
            node.name.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        )

    assert all(
        fragment not in definition
        for definition in definitions
        for fragment in FORBIDDEN_OPERATION_FRAGMENTS
    )
