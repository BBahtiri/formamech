from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "src" / "piginot" / "training"


def test_training_package_has_no_legacy_private_reveal_or_network_imports() -> None:
    forbidden_roots = {
        "campaigns",
        "experiments",
        "fastapi",
        "flask",
        "http",
        "piginot.doe.secret_material",
        "requests",
        "streamlit",
        "subprocess",
        "urllib",
    }
    imported: set[str] = set()
    for path in TRAINING_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)

    assert not any(
        name == root or name.startswith(root + ".")
        for name in imported
        for root in forbidden_roots
    )


def test_training_package_has_no_reveal_pickle_or_active_learning_surface() -> None:
    assert (TRAINING_ROOT / "__init__.py").is_file()
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in TRAINING_ROOT.glob("*.py")
    )

    assert "private_reveal" not in source
    assert "holdout_salt" not in source
    assert "holdout_coordinates" not in source
    assert "CampaignSecrets" not in source
    assert "torch.load" not in source
    assert "pickle" not in source
    assert "active_learning" not in source
    assert "acquisition_function" not in source
    assert "release_claim" not in source
