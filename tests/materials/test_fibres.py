"""S26 fibre kernels, symmetry group and admission (ADR 0108)."""

from __future__ import annotations

import math

import pytest
import torch

from piginot.contracts.energy_formula import EnergyFormula
from piginot.materials import register, resolve
from piginot.materials.admission import (
    MaterialKernel,
    admit,
    deformations_3d,
    symmetry_group,
)
from piginot.materials.fem_admission import (
    FIBRE_CASE,
    case_for,
    check_fem,
    minimum_fibre_i4,
)
from tests.materials.helpers import BUILTIN_VALUES, material

F64 = torch.float64
HGO = BUILTIN_VALUES["hgo"]
MATRIX = EnergyFormula.from_document(
    {
        "energy": "shear_modulus / 2 * (I1bar - 3) + bulk_modulus / 2 * (J - 1) ** 2",
        "parameters": {"shear_modulus": "MPa", "bulk_modulus": "MPa"},
        "small_strain": {
            "shear_modulus": "shear_modulus",
            "bulk_modulus": "bulk_modulus",
        },
    }
)


def _matrix() -> object:
    return material(
        register(MATRIX),
        {"shear_modulus": HGO["shear_modulus"], "bulk_modulus": HGO["bulk_modulus"]},
    )


def _closed_form(deformation: torch.Tensor, angle: float) -> torch.Tensor:
    """HGO first Piola stress by hand: matrix plus 2 k1 E exp(k2 E^2) F a a."""

    mu, kappa, k1, k2 = (HGO[n] for n in ("shear_modulus", "bulk_modulus", "k1", "k2"))
    determinant = torch.linalg.det(deformation)
    inverse_t = torch.linalg.inv(deformation).transpose(-2, -1)
    first = (deformation * deformation).sum(dim=(-2, -1))
    stress = (
        mu
        * determinant[..., None, None] ** (-2.0 / 3.0)
        * (deformation - first[..., None, None] / 3.0 * inverse_t)
        + (kappa * (determinant - 1.0) * determinant)[..., None, None] * inverse_t
    )
    for sign in (1.0, -1.0):
        a = torch.tensor(
            (
                math.cos(math.radians(sign * angle)),
                math.sin(math.radians(sign * angle)),
                0.0,
            ),
            dtype=F64,
        )
        stretched = deformation @ a
        strain = torch.clamp((stretched * stretched).sum(dim=-1) - 1.0, min=0.0)
        factor = 2.0 * k1 * strain * torch.exp(k2 * strain**2)
        stress = stress + factor[..., None, None] * torch.einsum(
            "...i,j->...ij", stretched, a
        )
    return stress


def test_the_hgo_stress_equals_its_closed_form() -> None:
    generator = torch.Generator().manual_seed(1)
    deformation = deformations_3d(generator, 500, (0.6, 1.6))
    stress = resolve(material("hgo", HGO)).first_piola(deformation)
    expected = _closed_form(deformation, HGO["fibre_angle"])
    scale = float(expected.abs().max())
    assert float((stress - expected).abs().max()) <= 1e-12 * scale


def test_fibres_carry_nothing_in_compression() -> None:
    hgo = resolve(material("hgo", HGO))
    matrix = resolve(_matrix())  # type: ignore[arg-type]
    generator = torch.Generator().manual_seed(2)
    deformation = deformations_3d(generator, 2000, (0.6, 1.6))
    compressed = torch.ones(deformation.shape[0], dtype=torch.bool)
    for c, s, _ in hgo.directions.values():
        a = torch.tensor((c, s, 0.0), dtype=F64)
        stretched = deformation @ a
        compressed &= (stretched * stretched).sum(dim=-1) <= 1.0
    assert int(compressed.sum()) > 20
    chosen = deformation[compressed]
    difference = hgo.first_piola(chosen) - matrix.first_piola(chosen)
    assert float(difference.abs().max()) <= 1e-12 * HGO["shear_modulus"]


def test_in_plane_fibres_leave_the_closure_root_alone() -> None:
    generator = torch.Generator().manual_seed(3)
    planar = torch.eye(2, dtype=F64) + 0.25 * torch.randn(
        (200, 2, 2), generator=generator, dtype=F64
    )
    planar = planar[torch.linalg.det(planar) > 0.3]
    hgo = resolve(material("hgo", HGO)).solve_f33(planar)
    matrix = resolve(_matrix()).solve_f33(planar)  # type: ignore[arg-type]
    assert float((hgo - matrix).abs().max()) <= 1e-12


@pytest.mark.parametrize(
    ("model", "angle", "rotations"),
    [
        ("hgo", 30.0, ["e1", "e2", "e3"]),
        ("goh", 30.0, ["e1", "e2", "e3"]),
        ("hgo_single", 30.0, ["a_a", "e3", "n_a"]),
        ("hgo_single", 0.0, ["e1", "e2", "e3"]),
        ("hgo_single", 90.0, ["e1", "e2", "e3"]),
    ],
)
def test_the_declared_symmetry_group(
    model: str, angle: float, rotations: list[str]
) -> None:
    group = symmetry_group(
        material(model, {**BUILTIN_VALUES[model], "fibre_angle": angle})
    )
    assert group is not None and sorted(group) == rotations
    for rotation in group.values():
        assert torch.allclose(rotation @ rotation.T, torch.eye(3, dtype=F64))
        assert float(torch.linalg.det(rotation)) == pytest.approx(1.0)


def test_an_isotropic_material_has_no_declared_group() -> None:
    assert symmetry_group(_matrix()) is None  # type: ignore[arg-type]
    assert case_for(_matrix()) != FIBRE_CASE  # type: ignore[arg-type]


@pytest.mark.parametrize("model", ["hgo", "hgo_single", "goh"])
def test_the_fibre_builtins_are_admitted(model: str) -> None:
    report = admit(material(model, BUILTIN_VALUES[model]), samples=200)
    assert report.passed, report.document()
    assert report.check("symmetry").detail["symmetry"] == "fibre-reinforced"


def test_a_fibre_direction_other_than_declared_fails_symmetry() -> None:
    declared = material("hgo_single", {**HGO, "fibre_angle": 30.0})
    actual = MaterialKernel(material("hgo_single", {**HGO, "fibre_angle": 45.0}))
    report = admit(declared, samples=100, kernel=actual)
    assert not report.check("symmetry").passed


def test_a_fibre_term_without_switch_fails_moduli() -> None:
    formula = EnergyFormula.from_document(
        {
            "energy": MATRIX.energy.source
            + " + k1 / (2 * k2) * (exp(k2 * (I4a - 1) ** 2) - 1)",
            "parameters": {
                "shear_modulus": "MPa",
                "bulk_modulus": "MPa",
                "k1": "MPa",
                "k2": "1",
                "fibre_angle": "deg",
            },
            "small_strain": {
                "shear_modulus": "shear_modulus",
                "bulk_modulus": "bulk_modulus",
            },
            "fibres": {"a": "fibre_angle"},
        }
    )
    report = admit(material(register(formula), HGO), samples=100)
    assert not report.check("moduli").passed
    assert report.check("symmetry").passed


def test_softening_fibres_fail_ellipticity() -> None:
    report = admit(material("hgo", {**HGO, "k1": -50.0}), samples=100)
    assert not report.check("ellipticity").passed
    assert not report.passed


def test_the_fem_check_keeps_fibres_in_tension() -> None:
    hgo = material("hgo", HGO)
    assert case_for(hgo) == FIBRE_CASE
    assert minimum_fibre_i4(hgo) >= 1.05
    result = check_fem(hgo)
    assert result.passed, result.detail
    assert result.detail["case"] == FIBRE_CASE


def test_a_field_with_fibres_in_compression_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from piginot.materials import fem_admission

    monkeypatch.setattr(fem_admission, "FIBRE_PRESTRETCH", 0.9)
    result = check_fem(material("hgo", HGO))
    assert not result.passed and "not in tension" in result.detail["error"]
