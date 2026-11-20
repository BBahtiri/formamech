"""Finite-strain generalized-Maxwell viscoelasticity with exact incremental potential.

Formulation (generalized-standard-material finite-strain viscoelasticity):

* Ground-state equilibrium free energy ``psi_inf(F)`` uses the verified S4
  compressible Neo-Hookean model.
* ``M`` parallel Maxwell branches with stiffness ratios ``gamma_m > 0`` and
  relaxation times ``tau_m > 0`` carry symmetric viscous Green-Lagrange internal
  strain tensors ``E_v^(m) in R^{3x3}`` initialized to zero:

      psi(F, {E_v^(m)}) = psi_inf(F)
                        + sum_m 0.5 * (E - E_v^(m)) : C_m : (E - E_v^(m))

  where ``E = 0.5 * (F^T F - I)`` is objective (frame-indifferent) and
  ``C_m = gamma_m * C_inf``.
* Implicit backward-Euler evolution ``(E_v^(m)_{n+1} - E_v^(m)_n) / dt = (E_{n+1} - E_v^(m)_{n+1}) / tau_m``
  yields ``alpha_m = 1 / (1 + dt / tau_m)``, an exact closed-form incremental
  variational potential ``W_inc(F_{n+1}; {E_v^(m)_n}, dt)`` whose gradient is
  ``P_{n+1} = dW_inc / dF_{n+1}``, an exact major-symmetric algorithmic tangent
  ``A_{n+1} = d^2 W_inc / dF_{n+1}^2``, and provably nonnegative discrete
  dissipation ``D_step >= 0`` for every finite deformation and ``dt > 0``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import torch

from piginot.contracts import MaterialSpec, Quantity, Unit, canonical_json_bytes
from piginot.contracts.stateful import StateSchema, StateVariableSpec
from piginot.physics.neo_hookean import (
    consistent_tangent as nh_consistent_tangent,
)
from piginot.physics.neo_hookean import (
    energy_density as nh_energy_density,
)
from piginot.physics.neo_hookean import (
    first_piola as nh_first_piola,
)
from piginot.stateful.constitutive import MaterialUpdate

VISCOELASTIC_MODEL_ID: Final = "generalized-maxwell-finite-strain-v1"


@dataclass(frozen=True, slots=True)
class MaxwellBranch:
    """One viscoelastic Maxwell branch ``(stiffness_ratio, relaxation_time)``."""

    stiffness_ratio: float
    relaxation_time: float

    def __post_init__(self) -> None:
        for label in ("stiffness_ratio", "relaxation_time"):
            val = getattr(self, label)
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(float(val))
                or float(val) <= 0.0
            ):
                raise ValueError(f"{label} must be finite and strictly positive")
            object.__setattr__(self, label, float(val))

    def document(self) -> dict[str, float]:
        return {
            "stiffness_ratio": self.stiffness_ratio,
            "relaxation_time": self.relaxation_time,
        }


@dataclass(frozen=True, slots=True)
class GeneralizedMaxwellConfig:
    """Validated parameters of the finite-strain generalized-Maxwell law."""

    equilibrium_modulus_mpa: float
    poisson_ratio: float
    branches: tuple[MaxwellBranch, ...]

    def __post_init__(self) -> None:
        e_inf = self.equilibrium_modulus_mpa
        nu = self.poisson_ratio
        if (
            isinstance(e_inf, bool)
            or not isinstance(e_inf, (int, float))
            or not math.isfinite(float(e_inf))
            or float(e_inf) <= 0.0
        ):
            raise ValueError("equilibrium_modulus_mpa must be finite and positive")
        if (
            isinstance(nu, bool)
            or not isinstance(nu, (int, float))
            or not math.isfinite(float(nu))
            or not (0.0 <= float(nu) < 0.5)
        ):
            raise ValueError("poisson_ratio must lie in [0, 0.5)")
        if not self.branches:
            raise ValueError("branches must contain at least one MaxwellBranch")
        object.__setattr__(self, "equilibrium_modulus_mpa", float(e_inf))
        object.__setattr__(self, "poisson_ratio", float(nu))

    @property
    def mu_inf(self) -> float:
        return self.equilibrium_modulus_mpa / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def lambda_inf(self) -> float:
        return (
            self.equilibrium_modulus_mpa
            * self.poisson_ratio
            / ((1.0 + self.poisson_ratio) * (1.0 - 2.0 * self.poisson_ratio))
        )

    @property
    def instantaneous_modulus_mpa(self) -> float:
        return self.equilibrium_modulus_mpa * (
            1.0 + sum(b.stiffness_ratio for b in self.branches)
        )

    def document(self) -> dict[str, Any]:
        return {
            "model_id": VISCOELASTIC_MODEL_ID,
            "equilibrium_modulus_mpa": self.equilibrium_modulus_mpa,
            "poisson_ratio": self.poisson_ratio,
            "branches": [b.document() for b in self.branches],
        }

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()


@dataclass(frozen=True, slots=True)
class PlaneStressMaterialUpdate:
    """Closed 2D plane-stress incremental response with exact condensed tangent."""

    update_3d: MaterialUpdate
    f33: torch.Tensor  # (...,)
    condensed_tangent_2d: torch.Tensor  # (..., 2, 2, 2, 2)
    closure_residual_max: float
    closure_converged: bool


class GeneralizedMaxwellViscoelastic:
    """Frame-indifferent finite-strain generalized-Maxwell constitutive model."""

    def __init__(self, config: GeneralizedMaxwellConfig) -> None:
        if not isinstance(config, GeneralizedMaxwellConfig):
            raise TypeError("config must be a GeneralizedMaxwellConfig")
        self.config = config
        self._eq_material = MaterialSpec(
            name="gm-equilibrium",
            model="compressible_neo_hookean",
            properties={
                "elastic_modulus": Quantity(
                    config.equilibrium_modulus_mpa, Unit.MPA
                ).to_canonical(),
                "poisson_ratio": Quantity(
                    config.poisson_ratio, Unit.ONE
                ).to_canonical(),
            },
        )
        self._schema = StateSchema(
            model_id=VISCOELASTIC_MODEL_ID,
            variables=(
                StateVariableSpec(
                    name="viscous_strain",
                    shape=(len(config.branches), 3, 3),
                    unit="1",
                    initial_rule="zeros",
                ),
            ),
        )

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
        n_branches = len(self.config.branches)
        return {
            "viscous_strain": torch.zeros(
                (*batch_shape, n_branches, 3, 3), dtype=dtype, device=device
            )
        }

    def _validate_state(
        self, f_new: torch.Tensor, state_old: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if "viscous_strain" not in state_old:
            raise KeyError("state_old is missing required tensor 'viscous_strain'")
        ev_old = state_old["viscous_strain"]
        expected = (*f_new.shape[:-2], len(self.config.branches), 3, 3)
        if tuple(ev_old.shape) != expected:
            raise ValueError(
                f"viscous_strain shape {tuple(ev_old.shape)} != expected {expected}"
            )
        return ev_old

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
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive")
        if f_new.ndim < 2 or tuple(f_new.shape[-2:]) != (3, 3):
            raise ValueError("f_new must have trailing shape (3, 3)")
        ev_old = self._validate_state(f_new, state_old)

        dtype = f_new.dtype
        device = f_new.device
        eye3 = torch.eye(3, dtype=dtype, device=device)

        det_f = torch.linalg.det(f_new)
        det_ok = bool((det_f > 0.0).all().item())
        if not det_ok:
            zeros_p = torch.zeros_like(f_new)
            zeros_a = torch.zeros(
                (*f_new.shape[:-2], 3, 3, 3, 3), dtype=dtype, device=device
            )
            zeros_s = torch.zeros(f_new.shape[:-2], dtype=dtype, device=device)
            return MaterialUpdate(
                stress=zeros_p,
                state_new={"viscous_strain": ev_old.clone()},
                algorithmic_tangent=zeros_a,
                free_energy=zeros_s,
                incremental_potential=zeros_s,
                dissipation=zeros_s,
                det_f=det_f,
                admissible=False,
                guards={
                    "det_f_positive": False,
                    "finite_response": True,
                    "nonnegative_dissipation": True,
                },
            )

        # Equilibrium compressible Neo-Hookean part (verified S4 kernel).
        psi_eq = nh_energy_density(f_new, self._eq_material)
        p_eq = nh_first_piola(f_new, self._eq_material)
        a_eq = nh_consistent_tangent(f_new, self._eq_material)

        # Objective Green-Lagrange strain E = 0.5 * (F^T F - I).
        c_tensor = torch.matmul(f_new.transpose(-2, -1), f_new)
        strain_e = 0.5 * (c_tensor - eye3)  # (..., 3, 3)

        mu_inf = self.config.mu_inf
        lam_inf = self.config.lambda_inf

        gammas = torch.tensor(
            [b.stiffness_ratio for b in self.config.branches],
            dtype=dtype,
            device=device,
        )
        taus = torch.tensor(
            [b.relaxation_time for b in self.config.branches],
            dtype=dtype,
            device=device,
        )
        alphas = 1.0 / (1.0 + dt / taus)  # (M,)
        mu_m = gammas * mu_inf  # (M,)
        lam_m = gammas * lam_inf  # (M,)

        # Expand branch parameters to match (..., M, 1, 1).
        b_shape = (1,) * (f_new.ndim - 2) + (len(self.config.branches),)
        alpha_v = alphas.reshape(*b_shape, 1, 1)
        mu_v = mu_m.reshape(*b_shape)
        lam_v = lam_m.reshape(*b_shape)
        tau_v = taus.reshape(*b_shape)

        # Implicit backward-Euler viscous internal strain update.
        diff_old = strain_e.unsqueeze(-3) - ev_old  # (..., M, 3, 3)
        ev_new = alpha_v * ev_old + (1.0 - alpha_v) * strain_e.unsqueeze(-3)
        delta_ev = ev_new - ev_old  # = (1 - alpha) * diff_old
        e_elastic = strain_e.unsqueeze(-3) - ev_new  # = alpha * diff_old

        # Branch overstresses Q_m = lambda_m * tr(E_e^m) I + 2 mu_m E_e^m.
        tr_el = torch.diagonal(e_elastic, dim1=-2, dim2=-1).sum(dim=-1)  # (..., M)
        q_branches = (
            lam_v[..., None, None] * tr_el[..., None, None] * eye3
            + 2.0 * mu_v[..., None, None] * e_elastic
        )
        s_visc = q_branches.sum(dim=-3)  # (..., 3, 3)
        p_total = p_eq + torch.matmul(f_new, s_visc)

        # Stored Helmholtz free energy at t_{n+1}.
        e_el_sq = (e_elastic * e_elastic).sum(dim=(-2, -1))
        psi_visc = (0.5 * lam_v * tr_el * tr_el + mu_v * e_el_sq).sum(dim=-1)
        free_energy = psi_eq + psi_visc

        # Exact incremental variational energy-dissipation potential W_inc.
        tr_diff = torch.diagonal(diff_old, dim1=-2, dim2=-1).sum(dim=-1)
        diff_sq = (diff_old * diff_old).sum(dim=(-2, -1))
        w_inc_visc = (
            alphas.reshape(*b_shape)
            * (0.5 * lam_v * tr_diff * tr_diff + mu_v * diff_sq)
        ).sum(dim=-1)
        incremental_potential = psi_eq + w_inc_visc

        # Exact discrete step dissipation D_step = sum_m (tau_m/dt + 0.5) dEv : C_m : dEv >= 0.
        tr_dev = torch.diagonal(delta_ev, dim1=-2, dim2=-1).sum(dim=-1)
        dev_sq = (delta_ev * delta_ev).sum(dim=(-2, -1))
        quad_form = lam_v * tr_dev * tr_dev + 2.0 * mu_v * dev_sq
        dissipation = ((tau_v / dt + 0.5) * quad_form).sum(dim=-1)

        # Exact closed-form major-symmetric algorithmic tangent A = dP/dF_new.
        lam_eff = float(torch.sum(alphas * lam_m).item())
        mu_eff = float(torch.sum(alphas * mu_m).item())
        b_tensor = torch.matmul(f_new, f_new.transpose(-2, -1))  # F F^T
        a_visc = (
            torch.einsum("ik,...lj->...ijkl", eye3, s_visc)
            + lam_eff * torch.einsum("...ij,...kl->...ijkl", f_new, f_new)
            + mu_eff * torch.einsum("...il,...kj->...ijkl", f_new, f_new)
            + mu_eff * torch.einsum("...ik,jl->...ijkl", b_tensor, eye3)
        )
        a_total = a_eq + a_visc

        finite_ok = bool(
            torch.isfinite(p_total).all().item()
            and torch.isfinite(a_total).all().item()
            and torch.isfinite(free_energy).all().item()
            and torch.isfinite(dissipation).all().item()
        )
        diss_ok = bool((dissipation >= -1e-14).all().item())
        admissible = det_ok and finite_ok and diss_ok
        return MaterialUpdate(
            stress=p_total,
            state_new={"viscous_strain": ev_new},
            algorithmic_tangent=a_total,
            free_energy=free_energy,
            incremental_potential=incremental_potential,
            dissipation=dissipation,
            det_f=det_f,
            admissible=admissible,
            guards={
                "det_f_positive": det_ok,
                "finite_response": finite_ok,
                "nonnegative_dissipation": diss_ok,
            },
        )

    def plane_stress_update(
        self,
        f2d_old: torch.Tensor,
        f2d_new: torch.Tensor,
        state_old: Mapping[str, torch.Tensor],
        dt: float,
        *,
        f33_guess: torch.Tensor | None = None,
        max_iterations: int = 25,
        tolerance: float = 1e-11,
    ) -> PlaneStressMaterialUpdate:
        """Solve incremental plane-stress closure ``P_33(F_2D, F_33) = 0`` by Newton."""
        if f2d_new.ndim < 2 or tuple(f2d_new.shape[-2:]) != (2, 2):
            raise ValueError("f2d_new must have trailing shape (2, 2)")
        batch = f2d_new.shape[:-2]
        dtype = f2d_new.dtype
        device = f2d_new.device
        f33 = (
            torch.ones(batch, dtype=dtype, device=device)
            if f33_guess is None
            else f33_guess.detach().clone().to(dtype=dtype, device=device)
        )

        def _embed(f2d: torch.Tensor, stretch33: torch.Tensor) -> torch.Tensor:
            f3d = torch.zeros((*batch, 3, 3), dtype=dtype, device=device)
            f3d[..., :2, :2] = f2d
            f3d[..., 2, 2] = stretch33
            return f3d

        f3d_old = _embed(f2d_old, torch.ones(batch, dtype=dtype, device=device))
        converged = False
        residual_max = float("inf")
        last_update: MaterialUpdate | None = None

        for _ in range(max_iterations):
            f3d = _embed(f2d_new, f33)
            upd = self.update(f3d_old, f3d, state_old, dt)
            last_update = upd
            if not upd.admissible:
                break
            p33 = upd.stress[..., 2, 2]
            residual_max = float(p33.abs().max().item())
            if residual_max <= tolerance:
                converged = True
                break
            a3333 = upd.algorithmic_tangent[..., 2, 2, 2, 2]
            if bool((a3333 <= 0.0).any().item()):
                break
            step = p33 / a3333
            # Line-search damping to keep f33 strictly positive.
            alpha = torch.ones_like(f33)
            candidate = f33 - alpha * step
            for _ls in range(8):
                bad = candidate <= 1e-6
                if not bool(bad.any().item()):
                    break
                alpha = torch.where(bad, 0.5 * alpha, alpha)
                candidate = f33 - alpha * step
            f33 = torch.clamp(candidate, min=1e-6)

        assert last_update is not None
        if not converged and last_update.admissible:
            # Final evaluation after last Newton step.
            last_update = self.update(f3d_old, _embed(f2d_new, f33), state_old, dt)
            residual_max = float(last_update.stress[..., 2, 2].abs().max().item())
            converged = residual_max <= tolerance

        t3d = last_update.algorithmic_tangent
        denom = t3d[..., 2, 2, 2, 2]
        left = t3d[..., :2, :2, 2, 2]
        right = t3d[..., 2, 2, :2, :2]
        condensed = t3d[..., :2, :2, :2, :2] - (
            left[..., :, :, None, None]
            * right[..., None, None, :, :]
            / denom[..., None, None, None, None]
        )
        return PlaneStressMaterialUpdate(
            update_3d=last_update,
            f33=f33,
            condensed_tangent_2d=condensed,
            closure_residual_max=residual_max,
            closure_converged=converged and last_update.admissible,
        )


__all__ = (
    "VISCOELASTIC_MODEL_ID",
    "GeneralizedMaxwellConfig",
    "GeneralizedMaxwellViscoelastic",
    "MaxwellBranch",
    "PlaneStressMaterialUpdate",
)
