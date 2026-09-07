from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPERATOR_ROOT = ROOT / "src" / "piginot" / "operators"


def test_operator_package_contains_no_training_serving_or_pickle_runtime() -> None:
    forbidden_roots = {
        "fastapi",
        "flask",
        "http",
        "pickle",
        "requests",
        "streamlit",
        "torch.optim",
    }
    imported: set[str] = set()
    class_names: set[str] = set()
    for path in OPERATOR_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
            elif isinstance(node, ast.ClassDef):
                class_names.add(node.name)

    assert not any(
        name == root or name.startswith(root + ".")
        for name in imported
        for root in forbidden_roots
    )
    assert "Trainer" not in class_names


def test_operator_package_does_not_load_pickle_or_legacy_experiments() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in OPERATOR_ROOT.glob("*.py")
    )

    assert "torch.load" not in source
    assert "pickle.load" not in source
    assert "experiments." not in source
