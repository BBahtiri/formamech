"""S25 anti-scope (ADR 0107): formulas as data, Neo-Hookean unchanged."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import torch

from piginot import physics
from piginot.contracts.energy_formula import SYMMETRIES
from piginot.contracts.problem_spec import MATERIAL_MODELS
from piginot.materials import embed
from piginot.physics import neo_hookean, plane_stress
from tests.materials.helpers import neo_hookean as neo_hookean_material

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src" / "piginot"
S25_SOURCES = (
    SOURCE / "contracts" / "energy_formula.py",
    *sorted((SOURCE / "materials").glob("*.py")),
)
FORBIDDEN = {"eval", "exec", "compile", "__import__", "sympify", "parse_expr"}


def test_formulas_are_never_evaluated_as_code() -> None:
    for path in S25_SOURCES:
        for node in ast.walk(ast.parse(path.read_text("utf-8"))):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                assert name not in FORBIDDEN, f"{path.name} calls {name}"
            if isinstance(node, ast.Import | ast.ImportFrom):
                module = getattr(node, "module", None) or node.names[0].name
                assert not module.startswith("sympy"), path.name


def test_only_isotropic_hyperelastic_models() -> None:
    assert SYMMETRIES == ("isotropic",)
    assert set(MATERIAL_MODELS) == {
        "compressible_neo_hookean",
        "mooney_rivlin",
        "yeoh",
        "gent",
    }


def test_the_neo_hookean_path_is_bit_identical() -> None:
    material = neo_hookean_material()
    generator = torch.Generator().manual_seed(0)
    planar = torch.eye(2, dtype=torch.float64) + 0.2 * torch.randn(
        (300, 2, 2), generator=generator, dtype=torch.float64
    )
    f33 = physics.plane_stress_closure(planar, material)
    assert torch.equal(f33, plane_stress.plane_stress_closure(planar, material))
    deformation = embed(planar, f33)
    for name in ("energy_density", "first_piola", "consistent_tangent"):
        assert torch.equal(
            getattr(physics, name)(deformation, material),
            getattr(neo_hookean, name)(deformation, material),
        )
    leaf = planar.clone().requires_grad_(True)
    other = planar.clone().requires_grad_(True)
    (dispatched,) = torch.autograd.grad(
        physics.plane_stress_closure(leaf, material).sum(), leaf
    )
    (direct,) = torch.autograd.grad(
        plane_stress.plane_stress_closure(other, material).sum(), other
    )
    assert torch.equal(dispatched, direct)


def test_the_materials_package_needs_no_fem_package() -> None:
    code = textwrap.dedent(
        """
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in ("scipy", "skfem", "meshio", "sympy"):
                    raise ImportError("blocked " + name)
        sys.meta_path.insert(0, Block())
        import piginot.materials, piginot.materials.admission
        import piginot.materials.fem_admission, piginot.problem
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPOSITORY,
    )
    assert result.returncode == 0, result.stderr
