from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT / "src" / "piginot" / "bundle"
FEM_ONLY = ("scipy", "skfem", "meshio", "sympy")


def _imports() -> set[str]:
    imported: set[str] = set()
    for path in BUNDLE_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
    return imported


def test_bundle_has_no_campaign_secret_network_or_pickle_imports() -> None:
    forbidden = {
        "campaigns",
        "experiments",
        "piginot.doe.secret_material",
        "piginot.training",
        "piginot.fem",
        "http",
        "requests",
        "subprocess",
        "urllib",
        "pickle",
        *FEM_ONLY,
    }
    imported = _imports()
    assert not any(
        name == root or name.startswith(root + ".")
        for name in imported
        for root in forbidden
    )
    for path in BUNDLE_ROOT.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "torch.load(" not in text and "weights_only" not in text


def test_the_inference_closure_runs_without_fem_dependencies() -> None:
    script = textwrap.dedent(
        f"""
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in {FEM_ONLY!r}:
                    raise ImportError("blocked " + name)
        sys.meta_path.insert(0, Block())
        import piginot.bundle, piginot.cli
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
