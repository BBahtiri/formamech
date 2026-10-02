from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFICATION_ROOT = ROOT / "src" / "piginot" / "verification"


def _imports() -> set[str]:
    imported: set[str] = set()
    for path in VERIFICATION_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
    return imported


def test_verification_has_no_campaign_secret_or_network_imports() -> None:
    forbidden = {
        "campaigns",
        "experiments",
        "piginot.doe.secret_material",
        "piginot.training",
        "fastapi",
        "flask",
        "http",
        "requests",
        "subprocess",
        "urllib",
        "pickle",
    }
    imported = _imports()

    assert not any(
        name == root or name.startswith(root + ".")
        for name in imported
        for root in forbidden
    )


def test_no_claim_above_holdout_verified_is_emittable() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in VERIFICATION_ROOT.glob("*.py")
    )

    assert "SEALED_RELEASE_VERIFIED" not in source
    assert "torch.load" not in source
    assert "holdout_salt" not in source
