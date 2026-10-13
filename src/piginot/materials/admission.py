"""The admission gate of a hyperelastic material (S25, ADR 0105 d. 5, ADR 0107).

Before a material model may be trained, its kernel must pass, over a declared
range of principal stretches:

1. ``reference``: ``W(I) = 0`` and ``P(I) = 0``;
2. ``objectivity``: ``W(QF) = W(F)`` and ``P(QF) = Q P(F)`` for rotations;
3. ``symmetry``: the declared material symmetry; isotropy is
   ``W(FQ) = W(F)`` and ``P(FQ) = P(F) Q``;
4. ``consistency``: ``P`` equals the central difference of ``W`` and the
   tangent the central difference of ``P``;
5. ``moduli``: the tangent at ``F = I`` is the isotropic tensor of the
   declared small-strain shear and bulk moduli;
6. ``closure``: the plane-stress equation ``P33 = 0`` has exactly one root,
   which the solver finds, for every in-plane deformation of the range;
7. ``ellipticity``: the condensed plane-stress tangent is strongly elliptic
   (positive acoustic tensor) at those deformations;
8. ``fem``: optimal convergence of the reference FEM solver on a
   manufactured solution (:mod:`piginot.materials.fem_admission`, which needs
   the ``verify`` extra).

Tolerances are relative to the small-strain shear modulus ``mu0`` (stresses,
energies) or to the larger of ``mu0`` and the bulk modulus (tangents).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

import torch

from piginot.contracts import MaterialSpec
from piginot.physics.errors import PhysicsError

REFERENCE_TOLERANCE: Final = 1e-10
OBJECTIVITY_TOLERANCE: Final = 1e-10
SYMMETRY_TOLERANCE: Final = 1e-10
CONSISTENCY_TOLERANCE: Final = 1e-6
MODULI_TOLERANCE: Final = 1e-8
CLOSURE_TOLERANCE: Final = 1e-10
ELLIPTICITY_MARGIN: Final = 1e-6
DEFAULT_STRETCH_RANGE: Final = (0.6, 1.6)
FINITE_DIFFERENCE_STEP: Final = 1e-6
ROOT_SCAN = (math.log(0.05), math.log(20.0), 241)
"""``log F33`` interval and sample count of the root-uniqueness scan."""


class Kernel(Protocol):
    """What the gate evaluates; :func:`material_kernel` builds it."""

    def energy(self, deformation: torch.Tensor) -> torch.Tensor: ...

    def stress(self, deformation: torch.Tensor) -> torch.Tensor: ...

    def tangent(self, deformation: torch.Tensor) -> torch.Tensor: ...

    def closure(self, deformation_2d: torch.Tensor) -> torch.Tensor: ...

    def stress_where_defined(self, deformation: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class MaterialKernel:
    """The :mod:`piginot.physics` functions bound to one material."""

    material: MaterialSpec

    def energy(self, deformation: torch.Tensor) -> torch.Tensor:
        from piginot.physics import energy_density

        return energy_density(deformation, self.material)

    def stress(self, deformation: torch.Tensor) -> torch.Tensor:
        from piginot.physics import first_piola

        return first_piola(deformation, self.material)

    def tangent(self, deformation: torch.Tensor) -> torch.Tensor:
        from piginot.physics import consistent_tangent

        return consistent_tangent(deformation, self.material)

    def closure(self, deformation_2d: torch.Tensor) -> torch.Tensor:
        from piginot.physics import plane_stress_closure

        return plane_stress_closure(deformation_2d, self.material)

    def stress_where_defined(self, deformation: torch.Tensor) -> torch.Tensor:
        """``P`` where ``J`` is admitted and the energy finite, NaN elsewhere."""

        from piginot.materials.registry import is_formula_model, resolve

        determinant = torch.linalg.det(deformation)
        defined = torch.isfinite(determinant) & (determinant > 1e-12)
        if is_formula_model(self.material.model):
            safe = torch.where(
                defined[..., None, None],
                deformation,
                torch.eye(3, dtype=deformation.dtype),
            )
            with torch.no_grad():
                energy = resolve(self.material)._energy(safe)
            defined = defined & torch.isfinite(energy)
        identity = torch.eye(3, dtype=deformation.dtype)
        safe = torch.where(defined[..., None, None], deformation, identity)
        stress = self.stress(safe)
        return torch.where(defined[..., None, None], stress, torch.nan)


def declared_moduli(material: MaterialSpec) -> tuple[float, float]:
    """The declared small-strain shear and bulk moduli of a material."""

    from piginot.materials.registry import formula_for, resolve

    if formula_for(material.model) is not None:
        return resolve(material).moduli
    young = float(material.properties["elastic_modulus"].value)
    poisson = float(material.properties["poisson_ratio"].value)
    return young / (2.0 * (1.0 + poisson)), young / (3.0 * (1.0 - 2.0 * poisson))


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One admission check: its measured value against its threshold."""

    name: str
    measured: float
    threshold: float
    passed: bool
    detail: Mapping[str, Any] = field(default_factory=dict)

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "measured": self.measured,
            "threshold": self.threshold,
            "passed": self.passed,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True, slots=True)
class AdmissionReport:
    """The checks of one material over one stretch range."""

    model: str
    stretch_range: tuple[float, float]
    shear_modulus: float
    bulk_modulus: float
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    def check(self, name: str) -> CheckResult:
        for item in self.checks:
            if item.name == name:
                return item
        raise KeyError(name)

    def document(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "stretch_range": list(self.stretch_range),
            "shear_modulus": self.shear_modulus,
            "bulk_modulus": self.bulk_modulus,
            "checks": [check.document() for check in self.checks],
            "passed": self.passed,
        }


# ------------------------------------------------------------------ samples


def _rotation_3d(generator: torch.Generator, count: int) -> torch.Tensor:
    matrix = torch.randn((count, 3, 3), generator=generator, dtype=torch.float64)
    q: torch.Tensor
    q, r = torch.linalg.qr(matrix)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]
    flip = torch.linalg.det(q) < 0
    q[flip, :, 0] = -q[flip, :, 0]
    return q


def _rotation_2d(angle: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(angle), torch.sin(angle)
    return torch.stack(
        (torch.stack((cos, -sin), dim=-1), torch.stack((sin, cos), dim=-1)), dim=-2
    )


def deformations_3d(
    generator: torch.Generator, count: int, stretch_range: tuple[float, float]
) -> torch.Tensor:
    """``R1 diag(l1, l2, l3) R2`` with principal stretches in the range."""

    low, high = stretch_range
    stretches = low + (high - low) * torch.rand(
        (count, 3), generator=generator, dtype=torch.float64
    )
    return (
        _rotation_3d(generator, count)
        @ torch.diag_embed(stretches)
        @ _rotation_3d(generator, count)
    )


def deformations_2d(
    generator: torch.Generator, per_axis: int, stretch_range: tuple[float, float]
) -> torch.Tensor:
    """A grid of in-plane principal stretches with random rotations."""

    low, high = stretch_range
    values = torch.linspace(low, high, per_axis, dtype=torch.float64)
    first, second = torch.meshgrid(values, values, indexing="ij")
    stretches = torch.stack((first.reshape(-1), second.reshape(-1)), dim=-1)
    angles = (
        2.0
        * math.pi
        * torch.rand((stretches.shape[0], 2), generator=generator, dtype=torch.float64)
    )
    return (
        _rotation_2d(angles[:, 0])
        @ torch.diag_embed(stretches)
        @ _rotation_2d(angles[:, 1])
    )


def _embed(deformation_2d: torch.Tensor, f33: torch.Tensor) -> torch.Tensor:
    from piginot.materials.hyperelastic import embed

    return embed(deformation_2d, f33)


# ------------------------------------------------------------------- checks


def _relative(error: torch.Tensor, scale: float) -> float:
    return float(error.abs().max().item()) / scale


def check_reference(kernel: Kernel, shear: float) -> CheckResult:
    identity = torch.eye(3, dtype=torch.float64)[None]
    energy = kernel.energy(identity)
    stress = kernel.stress(identity)
    measured = max(_relative(energy, shear), _relative(stress, shear))
    return CheckResult(
        "reference", measured, REFERENCE_TOLERANCE, measured <= REFERENCE_TOLERANCE
    )


def check_objectivity(
    kernel: Kernel, deformation: torch.Tensor, rotation: torch.Tensor, shear: float
) -> CheckResult:
    energy = kernel.energy(deformation)
    stress = kernel.stress(deformation)
    rotated = rotation @ deformation
    scale = max(shear, float(energy.abs().max()), float(stress.abs().max()))
    measured = max(
        _relative(kernel.energy(rotated) - energy, scale),
        _relative(kernel.stress(rotated) - rotation @ stress, scale),
    )
    return CheckResult(
        "objectivity",
        measured,
        OBJECTIVITY_TOLERANCE,
        measured <= OBJECTIVITY_TOLERANCE,
    )


def check_isotropy(
    kernel: Kernel, deformation: torch.Tensor, rotation: torch.Tensor, shear: float
) -> CheckResult:
    energy = kernel.energy(deformation)
    stress = kernel.stress(deformation)
    rotated = deformation @ rotation
    scale = max(shear, float(energy.abs().max()), float(stress.abs().max()))
    measured = max(
        _relative(kernel.energy(rotated) - energy, scale),
        _relative(kernel.stress(rotated) - stress @ rotation, scale),
    )
    return CheckResult(
        "symmetry",
        measured,
        SYMMETRY_TOLERANCE,
        measured <= SYMMETRY_TOLERANCE,
        {"symmetry": "isotropic"},
    )


def _central(
    function: Callable[[torch.Tensor], torch.Tensor], deformation: torch.Tensor
) -> torch.Tensor:
    """Central differences of ``function`` with respect to each ``F_kL``."""

    step = FINITE_DIFFERENCE_STEP
    columns = []
    for k in range(3):
        for column in range(3):
            shift = torch.zeros_like(deformation)
            shift[..., k, column] = step
            columns.append(
                (function(deformation + shift) - function(deformation - shift))
                / (2.0 * step)
            )
    stacked = torch.stack(columns, dim=-1)
    return stacked.reshape(*stacked.shape[:-1], 3, 3)


def check_consistency(
    kernel: Kernel, deformation: torch.Tensor, shear: float, bulk: float
) -> CheckResult:
    stress = kernel.stress(deformation)
    numeric_stress = _central(kernel.energy, deformation)
    tangent = kernel.tangent(deformation)
    numeric_tangent = _central(kernel.stress, deformation)
    stress_scale = max(shear, float(stress.abs().max()))
    tangent_scale = max(shear, bulk, float(tangent.abs().max()))
    stress_error = _relative(numeric_stress - stress, stress_scale)
    tangent_error = _relative(numeric_tangent - tangent, tangent_scale)
    measured = max(stress_error, tangent_error)
    return CheckResult(
        "consistency",
        measured,
        CONSISTENCY_TOLERANCE,
        measured <= CONSISTENCY_TOLERANCE,
        {"stress": stress_error, "tangent": tangent_error},
    )


def isotropic_tangent(shear: float, bulk: float) -> torch.Tensor:
    """``lambda d_ij d_kl + mu (d_ik d_jl + d_il d_jk)`` with ``lambda = K - 2mu/3``."""

    identity = torch.eye(3, dtype=torch.float64)
    lame = bulk - 2.0 * shear / 3.0
    return (
        lame * torch.einsum("ij,kl->ijkl", identity, identity)
        + shear * torch.einsum("ik,jl->ijkl", identity, identity)
        + shear * torch.einsum("il,jk->ijkl", identity, identity)
    )


def check_moduli(kernel: Kernel, shear: float, bulk: float) -> CheckResult:
    tangent = kernel.tangent(torch.eye(3, dtype=torch.float64)[None])[0]
    measured = _relative(
        tangent - isotropic_tangent(shear, bulk), max(abs(shear), abs(bulk))
    )
    return CheckResult(
        "moduli",
        measured,
        MODULI_TOLERANCE,
        measured <= MODULI_TOLERANCE,
        {"shear_modulus": shear, "bulk_modulus": bulk},
    )


def _thickness_residual(
    kernel: Kernel, deformation_2d: torch.Tensor, log_f33: torch.Tensor
) -> torch.Tensor:
    """``F33 P33`` on a ``log F33`` grid, NaN where the energy is undefined."""

    count, samples = deformation_2d.shape[0], log_f33.shape[0]
    f33 = torch.exp(log_f33)[None, :].expand(count, samples).reshape(-1)
    planar = deformation_2d[:, None].expand(count, samples, 2, 2).reshape(-1, 2, 2)
    stress = kernel.stress_where_defined(_embed(planar, f33))
    return (f33 * stress[:, 2, 2]).reshape(count, samples)


def check_closure(
    kernel: Kernel, deformation_2d: torch.Tensor, shear: float
) -> tuple[CheckResult, torch.Tensor | None]:
    low, high, samples = ROOT_SCAN
    grid = torch.linspace(low, high, samples, dtype=torch.float64)
    residual = _thickness_residual(kernel, deformation_2d, grid)
    finite = torch.isfinite(residual)
    increasing = torch.ones(deformation_2d.shape[0], dtype=torch.bool)
    roots = torch.zeros(deformation_2d.shape[0], dtype=torch.int64)
    for row in range(residual.shape[0]):
        values = residual[row][finite[row]]
        if values.numel() < 2:
            increasing[row] = False
            continue
        steps = values[1:] - values[:-1]
        increasing[row] = bool((steps > 0).all())
        signs = torch.sign(values)
        roots[row] = int((signs[1:] != signs[:-1]).sum())
    try:
        f33 = kernel.closure(deformation_2d)
        stress = kernel.stress(_embed(deformation_2d, f33))
        solved = _relative(stress[:, 2, 2], shear)
    except PhysicsError as exc:
        return (
            CheckResult(
                "closure",
                math.inf,
                CLOSURE_TOLERANCE,
                False,
                {"error": str(exc)},
            ),
            None,
        )
    unique = int(((roots == 1) & increasing).sum())
    passed = unique == deformation_2d.shape[0] and solved <= CLOSURE_TOLERANCE
    return (
        CheckResult(
            "closure",
            solved,
            CLOSURE_TOLERANCE,
            passed,
            {
                "deformations": int(deformation_2d.shape[0]),
                "unique_increasing_root": unique,
            },
        ),
        f33,
    )


def condensed_tangent(tangent: torch.Tensor) -> torch.Tensor:
    """The plane-stress tangent with ``P33 = 0`` eliminated."""

    denominator = tangent[..., 2, 2, 2, 2]
    left = tangent[..., :2, :2, 2, 2]
    right = tangent[..., 2, 2, :2, :2]
    return tangent[..., :2, :2, :2, :2] - (
        left[..., :, :, None, None]
        * right[..., None, None, :, :]
        / denominator[..., None, None, None, None]
    )


def check_ellipticity(
    kernel: Kernel,
    deformation_2d: torch.Tensor,
    f33: torch.Tensor | None,
    shear: float,
    directions: int = 180,
) -> CheckResult:
    if f33 is None:
        return CheckResult(
            "ellipticity", -math.inf, ELLIPTICITY_MARGIN, False, {"error": "no closure"}
        )
    condensed = condensed_tangent(kernel.tangent(_embed(deformation_2d, f33)))
    angle = torch.linspace(0.0, math.pi, directions + 1, dtype=torch.float64)[:-1]
    normal = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    acoustic = torch.einsum("bijkl,nj,nl->bnik", condensed, normal, normal)
    smallest = torch.linalg.eigvalsh(acoustic)[..., 0]
    measured = float(smallest.min()) / shear
    return CheckResult(
        "ellipticity",
        measured,
        ELLIPTICITY_MARGIN,
        measured >= ELLIPTICITY_MARGIN,
        {"directions": directions},
    )


def admit(
    material: MaterialSpec,
    *,
    stretch_range: tuple[float, float] = DEFAULT_STRETCH_RANGE,
    samples: int = 1000,
    grid: int = 12,
    seed: int = 0,
    kernel: Kernel | None = None,
) -> AdmissionReport:
    """Checks 1-7 of the admission gate for ``material`` over ``stretch_range``.

    ``kernel`` replaces the material's kernel (seeded-defect tests).
    """

    low, high = stretch_range
    if not 0.0 < low < 1.0 < high:
        raise ValueError("the stretch range must contain 1 and be positive")
    shear, bulk = declared_moduli(material)
    evaluator: Kernel = MaterialKernel(material) if kernel is None else kernel
    generator = torch.Generator().manual_seed(seed)
    deformation = deformations_3d(generator, samples, stretch_range)
    rotation = _rotation_3d(generator, samples)
    planar = deformations_2d(generator, grid, stretch_range)
    scale = abs(shear) if shear else 1.0
    checks = [check_reference(evaluator, scale)]
    for check in (check_objectivity, check_isotropy):
        try:
            checks.append(check(evaluator, deformation, rotation, scale))
        except PhysicsError as exc:
            checks.append(_failed(check.__name__[6:], str(exc)))
    try:
        checks.append(check_consistency(evaluator, deformation[:100], scale, bulk))
    except PhysicsError as exc:
        checks.append(_failed("consistency", str(exc)))
    checks.append(check_moduli(evaluator, shear, bulk))
    closure, f33 = check_closure(evaluator, planar, scale)
    checks.append(closure)
    checks.append(check_ellipticity(evaluator, planar, f33, scale))
    return AdmissionReport(
        model=material.model,
        stretch_range=(float(low), float(high)),
        shear_modulus=shear,
        bulk_modulus=bulk,
        checks=tuple(checks),
    )


def _failed(name: str, error: str) -> CheckResult:
    names = {"objectivity": "objectivity", "isotropy": "symmetry"}
    return CheckResult(names.get(name, name), math.inf, 0.0, False, {"error": error})


__all__ = (
    "CLOSURE_TOLERANCE",
    "CONSISTENCY_TOLERANCE",
    "DEFAULT_STRETCH_RANGE",
    "ELLIPTICITY_MARGIN",
    "MODULI_TOLERANCE",
    "OBJECTIVITY_TOLERANCE",
    "REFERENCE_TOLERANCE",
    "SYMMETRY_TOLERANCE",
    "AdmissionReport",
    "CheckResult",
    "Kernel",
    "MaterialKernel",
    "admit",
    "condensed_tangent",
    "declared_moduli",
    "deformations_2d",
    "deformations_3d",
    "isotropic_tangent",
)
