"""The S25 admission gate: built-ins pass, every seeded defect is caught."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from piginot.contracts.energy_formula import BUILTIN_FORMULAS, EnergyFormula
from piginot.materials import register
from piginot.materials.admission import (
    MaterialKernel,
    admit,
    condensed_tangent,
    declared_moduli,
    deformations_2d,
    deformations_3d,
    isotropic_tangent,
)
from tests.materials.helpers import (
    BUILTIN_VALUES,
    SHEAR,
    formula_neo_hookean,
    material,
    neo_hookean,
)

F64 = torch.float64


@pytest.mark.parametrize("model", sorted(BUILTIN_FORMULAS))
def test_every_builtin_model_is_admitted(model: str) -> None:
    report = admit(material(model, BUILTIN_VALUES[model]), samples=200)
    assert report.passed, report.document()
    assert [check.name for check in report.checks] == [
        "reference",
        "objectivity",
        "symmetry",
        "consistency",
        "moduli",
        "closure",
        "ellipticity",
    ]
    assert report.document()["passed"] is True


def test_the_closed_form_and_its_formula_are_admitted_alike() -> None:
    closed = admit(neo_hookean(), samples=200)
    formula = admit(formula_neo_hookean(), samples=200)
    assert closed.passed and formula.passed
    assert closed.shear_modulus == pytest.approx(formula.shear_modulus, rel=1e-14)
    assert closed.bulk_modulus == pytest.approx(formula.bulk_modulus, rel=1e-14)
    assert declared_moduli(neo_hookean())[0] == pytest.approx(SHEAR)


def _user(energy: str, shear: str, bulk: str, values: dict[str, float]) -> object:
    formula = EnergyFormula.from_document(
        {
            "energy": energy,
            "parameters": dict.fromkeys(values, "MPa"),
            "small_strain": {"shear_modulus": shear, "bulk_modulus": bulk},
        }
    )
    return material(register(formula), values)


def test_an_energy_that_is_not_stress_free_fails_reference() -> None:
    spec = _user(
        "c * (I1 - 3) + k / 2 * (J - 1) ** 2", "2 * c", "k", {"c": 100.0, "k": 400.0}
    )
    report = admit(spec, samples=50)  # type: ignore[arg-type]
    assert not report.check("reference").passed


def test_wrong_declared_moduli_fail_moduli() -> None:
    values = {"c10": 100.0, "c01": 20.0, "bulk_modulus": 400.0}
    spec = _user(
        "c10 * (I1bar - 3) + c01 * (I2bar - 3) + bulk_modulus / 2 * (J - 1) ** 2",
        "c10 + c01",
        "bulk_modulus",
        values,
    )
    report = admit(spec, samples=50)  # type: ignore[arg-type]
    assert not report.check("moduli").passed
    assert report.check("reference").passed


def test_a_negative_shear_stiffness_fails_ellipticity_and_closure() -> None:
    values = {"c10": -120.0, "c01": 30.0, "bulk_modulus": 500.0}
    report = admit(material("mooney_rivlin", values), samples=50)
    assert not report.check("ellipticity").passed
    assert not report.check("closure").passed
    assert not report.passed


def test_an_energy_undefined_in_the_range_fails() -> None:
    values = {**BUILTIN_VALUES["gent"], "jm": 0.3}
    report = admit(material("gent", values), samples=50)
    assert not report.passed
    failed = {check.name for check in report.checks if not check.passed}
    assert failed & {"objectivity", "symmetry", "consistency", "closure"}


@dataclass(frozen=True)
class _Seeded:
    """The Neo-Hookean kernel with one injected defect."""

    defect: str

    @property
    def base(self) -> MaterialKernel:
        return MaterialKernel(neo_hookean())

    def energy(self, deformation: torch.Tensor) -> torch.Tensor:
        energy = self.base.energy(deformation)
        if self.defect == "anisotropic":
            fibre = (deformation[..., :, 0] ** 2).sum(dim=-1)
            energy = energy + 10.0 * (fibre - 1.0) ** 2
        if self.defect == "non-objective":
            energy = (
                energy + 10.0 * (deformation[..., 0, 1] - deformation[..., 1, 0]) ** 2
            )
        return energy

    def stress(self, deformation: torch.Tensor) -> torch.Tensor:
        if self.defect in ("anisotropic", "non-objective"):
            with torch.enable_grad():
                leaf = deformation.detach().requires_grad_(True)
                (stress,) = torch.autograd.grad(self.energy(leaf).sum(), leaf)
            return stress
        stress = self.base.stress(deformation)
        return 1.001 * stress if self.defect == "scaled-stress" else stress

    def tangent(self, deformation: torch.Tensor) -> torch.Tensor:
        tangent = self.base.tangent(deformation)
        if self.defect == "wrong-tangent":
            tangent = tangent * 1.01
        return tangent

    def closure(self, deformation_2d: torch.Tensor) -> torch.Tensor:
        return self.base.closure(deformation_2d)

    def stress_where_defined(self, deformation: torch.Tensor) -> torch.Tensor:
        return self.base.stress_where_defined(deformation)


@pytest.mark.parametrize(
    ("defect", "check"),
    [
        ("anisotropic", "symmetry"),
        ("non-objective", "objectivity"),
        ("scaled-stress", "consistency"),
        ("wrong-tangent", "consistency"),
    ],
)
def test_each_seeded_kernel_defect_is_caught(defect: str, check: str) -> None:
    report = admit(neo_hookean(), samples=50, kernel=_Seeded(defect))
    assert not report.check(check).passed
    assert not report.passed


def test_the_sampled_deformations_and_helpers() -> None:
    generator = torch.Generator().manual_seed(0)
    deformation = deformations_3d(generator, 20, (0.8, 1.2))
    singular = torch.linalg.svdvals(deformation)
    assert float(singular.min()) >= 0.8 - 1e-12 and float(singular.max()) <= 1.2 + 1e-12
    assert bool((torch.linalg.det(deformation) > 0).all())
    planar = deformations_2d(generator, 3, (0.8, 1.2))
    assert planar.shape == (9, 2, 2)
    tangent = isotropic_tangent(2.0, 5.0)
    assert float(tangent[0, 0, 0, 0]) == pytest.approx(5.0 + 4.0 * 2.0 / 3.0)
    condensed = condensed_tangent(tangent[None])
    assert condensed.shape == (1, 2, 2, 2, 2)
    with pytest.raises(ValueError, match="contain 1"):
        admit(neo_hookean(), stretch_range=(1.1, 1.5))
    with pytest.raises(KeyError):
        admit(neo_hookean(), samples=10).check("fem")
