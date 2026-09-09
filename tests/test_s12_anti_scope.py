from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE_ROOT = ROOT / "src" / "piginot" / "objectives"


def test_objective_package_has_no_legacy_campaign_training_or_network_imports() -> None:
    forbidden_roots = {
        "campaigns",
        "experiments",
        "http",
        "piginot.training",
        "requests",
        "subprocess",
        "torch.optim",
        "urllib",
    }
    imported: set[str] = set()
    for path in OBJECTIVE_ROOT.glob("*.py"):
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


def test_objective_package_has_no_private_reveal_or_serialization_surface() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in OBJECTIVE_ROOT.glob("*.py")
    )

    assert "private_reveal" not in source
    assert "holdout_salt" not in source
    assert "CampaignSecrets" not in source
    assert "torch.load" not in source
    assert "pickle" not in source
    assert "eval(" not in source
    assert "exec(" not in source
