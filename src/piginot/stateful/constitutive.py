"""Constitutive model protocol, stateless adapter, and tangent verification (Feature F0).

Separates local material-point state updates from global equilibrium assembly.
Internal variables live explicitly at material integration points (or in a
declared transfer representation); neural hidden vectors alone are never treated
as verified physical material state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch

from piginot.contracts import MaterialSpec
from piginot.contracts.stateful import StateSchema
from piginot.physics import (
    consistent_tangent,
    energy_density,
    first_piola,
)


@dataclass(frozen=True, slots=True)
class MaterialUpdate:
    """Authoritative local constitutive response for one incremental step."""

    stress: torch.Tensor  # (..., 3, 3) first Piola-Kirchhoff stress P
    state_new: Mapping[str, torch.Tensor]
    algorithmic_tangent: torch.Tensor  # (..., 3, 3, 3, 3) dP/dF_new
    free_energy: torch.Tensor  # (...,) Helmholtz free energy density
    incremental_potential: torch.Tensor  # (...,) incremental variational potential
    dissipation: torch.Tensor  # (...,) discrete step dissipation (>= 0)
    det_f: torch.Tensor  # (...,) det(F_new)
    admissible: bool
    guards: dict[str, bool]


class ConstitutiveModel(Protocol):
    """Differentiable / verifiable local constitutive plugin interface."""

    @property
    def state_schema(self) -> StateSchema: ...

    def initial_state(
        self,
        batch_shape: tuple[int, ...],
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]: ...

    def update(
        self,
        f_old: torch.Tensor,
        f_new: torch.Tensor,
        state_old: Mapping[str, torch.Tensor],
        dt: float,
        *,
        temperature: torch.Tensor | None = None,
    ) -> MaterialUpdate: ...


class StatelessHyperelasticModel:
    """Stateless compatibility adapter wrapping existing S4/S25 hyperelastic laws."""

    def __init__(self, material: MaterialSpec) -> None:
        if not isinstance(material, MaterialSpec):
            raise TypeError("material must be a MaterialSpec")
        self.material = material
        self._schema = StateSchema(model_id=f"stateless:{material.model}", variables=())

    @property
    def state_schema(self) -> StateSchema:
        return self._schema

    def initial_state(
        self,
        batch_shape: tuple[int, ...],
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        del batch_shape, dtype, device
        return {}

    def update(
        self,
        f_old: torch.Tensor,
        f_new: torch.Tensor,
        state_old: Mapping[str, torch.Tensor],
        dt: float,
        *,
        temperature: torch.Tensor | None = None,
    ) -> MaterialUpdate:
        del f_old, temperature
        if state_old:
            raise ValueError("stateless hyperelastic model expects an empty state dict")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive")
        det_f = torch.linalg.det(f_new)
        det_ok = bool((det_f > 0.0).all().item())
        stress = first_piola(f_new, self.material)
        tangent = consistent_tangent(f_new, self.material)
        energy = energy_density(f_new, self.material)
        dissipation = torch.zeros_like(energy)
        finite_ok = bool(
            torch.isfinite(stress).all().item()
            and torch.isfinite(tangent).all().item()
            and torch.isfinite(energy).all().item()
        )
        admissible = det_ok and finite_ok
        return MaterialUpdate(
            stress=stress,
            state_new={},
            algorithmic_tangent=tangent,
            free_energy=energy,
            incremental_potential=energy,
            dissipation=dissipation,
            det_f=det_f,
            admissible=admissible,
            guards={
                "det_f_positive": det_ok,
                "finite_response": finite_ok,
                "nonnegative_dissipation": True,
            },
        )


@dataclass(frozen=True, slots=True)
class TangentCheckResult:
    """Independent finite-difference check of an algorithmic consistent tangent."""

    max_abs_error: float
    relative_error: float
    tangent_norm: float
    passed: bool
    tolerance: float


def verify_algorithmic_tangent(
    model: ConstitutiveModel,
    f_old: torch.Tensor,
    f_new: torch.Tensor,
    state_old: Mapping[str, torch.Tensor],
    dt: float,
    *,
    eps: float = 1e-6,
    rel_tol: float = 1e-4,
    norm_floor: float = 1e-12,
) -> TangentCheckResult:
    """Verify ``dP/dF_new`` against central finite differences at all points."""
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    if not math.isfinite(rel_tol) or rel_tol <= 0.0:
        raise ValueError("rel_tol must be finite and positive")

    with torch.no_grad():
        base = model.update(f_old, f_new, state_old, dt)
        analytic = base.algorithmic_tangent
        fd = torch.zeros_like(analytic)
        for k in range(3):
            for col in range(3):
                delta = torch.zeros_like(f_new)
                delta[..., k, col] = eps
                p_plus = model.update(f_old, f_new + delta, state_old, dt).stress
                p_minus = model.update(f_old, f_new - delta, state_old, dt).stress
                fd[..., :, :, k, col] = (p_plus - p_minus) / (2.0 * eps)
        diff = analytic - fd
        max_abs = float(diff.abs().max().item())
        ref_norm = float(torch.linalg.vector_norm(fd).item())
        err_norm = float(torch.linalg.vector_norm(diff).item())
        rel_err = err_norm / max(ref_norm, norm_floor)
        return TangentCheckResult(
            max_abs_error=max_abs,
            relative_error=rel_err,
            tangent_norm=ref_norm,
            passed=rel_err <= rel_tol,
            tolerance=rel_tol,
        )


__all__ = (
    "ConstitutiveModel",
    "MaterialUpdate",
    "StatelessHyperelasticModel",
    "TangentCheckResult",
    "verify_algorithmic_tangent",
)
