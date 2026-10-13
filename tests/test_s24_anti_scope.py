"""S24 anti-scope (ADR 0106 decision 8, G24.anti-scope)."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

from piginot.contracts.problem_spec import LIFTING_KINDS, MATERIAL_MODELS, PHYSICS

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src" / "piginot"
S24_SOURCES = (
    SOURCE / "contracts" / "problem_spec.py",
    *sorted((SOURCE / "problem").glob("*.py")),
    *sorted((SOURCE / "lifting").glob("*.py")),
)
FORBIDDEN_NAMES = {"eval", "exec", "compile", "__import__", "sympify", "parse_expr"}
FORBIDDEN_ATTRIBUTES = {"sympify", "parse_expr", "eval", "exec"}


def _run(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPOSITORY,
    )
    assert result.returncode == 0, result.stderr


def test_formulas_are_never_evaluated_as_code() -> None:
    for path in S24_SOURCES:
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                assert node.func.id not in FORBIDDEN_NAMES, (
                    f"{path.name}: {node.func.id}"
                )
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in FORBIDDEN_ATTRIBUTES, path.name


def test_the_contract_needs_no_numerical_package() -> None:
    _run(
        """
        import sys
        import piginot.contracts.problem_spec
        assert not {"torch", "numpy", "scipy"} & set(sys.modules)
        """
    )


def test_problem_and_lifting_import_without_fem_packages() -> None:
    _run(
        """
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in ("scipy", "skfem", "meshio", "sympy"):
                    raise ImportError("blocked " + name)
        sys.meta_path.insert(0, Block())
        import piginot.lifting, piginot.problem
        """
    )


def test_the_step_adds_no_physics_material_or_harmonic_lifting() -> None:
    assert PHYSICS == ("2d-plane-stress-finite-strain",)
    assert set(MATERIAL_MODELS) == {"compressible_neo_hookean"}
    assert LIFTING_KINDS == ("quarter", "adf")
