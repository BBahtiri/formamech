"""Small-strain J2 elastoplasticity with linear/Voce isotropic hardening (M3 / F2).

Provides one unified 3D radial-return constitutive kernel implementing
:class:`~piginot.stateful.constitutive.ConstitutiveModel` with:

* exact closed-form radial return for linear hardening and monotone scalar
  Newton return for linear + Voce saturation hardening,
* exact major/minor-symmetric 3D consistent elastoplastic tangent ``dP/dF``,
* exact 2D **plane-strain** embedding (``eps_33 = 0``),
* exact 2D **plane-stress** closure (``sigma_33 = 0`` via scalar Newton +
  Schur-condensed consistent tangent, matching Simo-Taylor stress-space return
  mapping to machine precision),
* exact Ortiz-Steinmann incremental variational pseudo-potential whose gradient
  equals the updated stress, and
* explicit physical guards (yield-surface admissibility, nonnegative plastic
  multiplier, plastic incompressibility ``tr(eps_p) == 0``, nonnegative
  discrete plastic dissipation).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import torch

from piginot.contracts import canonical_json_bytes
from piginot.contracts.stateful import StateSchema, StateVariableSpec
from piginot.stateful.constitutive import MaterialUpdate, TangentCheckResult
from piginot.stateful.viscoelastic import PlaneStressMaterialUpdate

J2_PLASTICITY_MODEL_ID: Final = "j2-isotropic-small-strain-v1"
_SQRT_3_2: Final = math.sqrt(1.5)
_SQRT_2_3: Final = math.sqrt(2.0 / 3.0)


def _check_finite_number(
    value: object, label: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{label} must be finite")
    if positive and num <= 0.0:
        raise ValueError(f"{label} must be strictly positive")
    if nonnegative and num < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return num


@dataclass(frozen=True, slots=True)
class J2PlasticityConfig:
    """Validated material parameters for small-strain J2 isotropic hardening."""

    youngs_modulus_mpa: float
    poisson_ratio: float
    yield_stress_mpa: float
    hardening_modulus_mpa: float = 0.0
    voce_saturation_mpa: float = 0.0
    voce_exponent: float = 10.0

    def __post_init__(self) -> None:
        e = _check_finite_number(
            self.youngs_modulus_mpa, "youngs_modulus_mpa", positive=True
        )
        nu = _check_finite_number(self.poisson_ratio, "poisson_ratio", nonnegative=True)
        if nu >= 0.5:
            raise ValueError("poisson_ratio must lie in [0, 0.5)")
        sy0 = _check_finite_number(
            self.yield_stress_mpa, "yield_stress_mpa", positive=True
        )
        h_lin = _check_finite_number(
            self.hardening_modulus_mpa, "hardening_modulus_mpa", nonnegative=True
        )
        voce_s = _check_finite_number(
            self.voce_saturation_mpa, "voce_saturation_mpa", nonnegative=True
        )
        voce_d = _check_finite_number(
            self.voce_exponent, "voce_exponent", positive=True
        )
        object.__setattr__(self, "youngs_modulus_mpa", e)
        object.__setattr__(self, "poisson_ratio", nu)
        object.__setattr__(self, "yield_stress_mpa", sy0)
        object.__setattr__(self, "hardening_modulus_mpa", h_lin)
        object.__setattr__(self, "voce_saturation_mpa", voce_s)
        object.__setattr__(self, "voce_exponent", voce_d)

    @property
    def mu(self) -> float:
        return self.youngs_modulus_mpa / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def bulk_modulus(self) -> float:
        return self.youngs_modulus_mpa / (3.0 * (1.0 - 2.0 * self.poisson_ratio))

    @property
    def lame_lambda(self) -> float:
        return (
            self.youngs_modulus_mpa
            * self.poisson_ratio
            / ((1.0 + self.poisson_ratio) * (1.0 - 2.0 * self.poisson_ratio))
        )

    def yield_stress(self, alpha: torch.Tensor) -> torch.Tensor:
        """Current isotropic yield stress ``sigma_y(alpha)``."""
        sy = self.yield_stress_mpa + self.hardening_modulus_mpa * alpha
        if self.voce_saturation_mpa > 0.0:
            sy = sy + self.voce_saturation_mpa * (
                1.0 - torch.exp(-self.voce_exponent * alpha)
            )
        return sy

    def hardening_slope(self, alpha: torch.Tensor) -> torch.Tensor:
        """Isotropic hardening slope ``h(alpha) = d sigma_y / d alpha``."""
        slope = torch.full_like(alpha, self.hardening_modulus_mpa)
        if self.voce_saturation_mpa > 0.0:
            slope = slope + (
                self.voce_saturation_mpa
                * self.voce_exponent
                * torch.exp(-self.voce_exponent * alpha)
            )
        return slope

    def hardening_energy(self, alpha: torch.Tensor) -> torch.Tensor:
        """Stored isotropic hardening free energy ``psi_p(alpha)``."""
        energy = 0.5 * self.hardening_modulus_mpa * alpha * alpha
        if self.voce_saturation_mpa > 0.0:
            energy = energy + self.voce_saturation_mpa * (
                alpha
                + (torch.exp(-self.voce_exponent * alpha) - 1.0) / self.voce_exponent
            )
        return energy

    def document(self) -> dict[str, Any]:
        return {
            "model_id": J2_PLASTICITY_MODEL_ID,
            "youngs_modulus_mpa": self.youngs_modulus_mpa,
            "poisson_ratio": self.poisson_ratio,
            "yield_stress_mpa": self.yield_stress_mpa,
            "hardening_modulus_mpa": self.hardening_modulus_mpa,
            "voce_saturation_mpa": self.voce_saturation_mpa,
            "voce_exponent": self.voce_exponent,
        }

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> J2PlasticityConfig:
        model_id = doc.get("model_id", J2_PLASTICITY_MODEL_ID)
        if model_id != J2_PLASTICITY_MODEL_ID:
            raise ValueError(
                f"expected model_id {J2_PLASTICITY_MODEL_ID!r}, got {model_id!r}"
            )
        return cls(
            youngs_modulus_mpa=float(doc["youngs_modulus_mpa"]),
            poisson_ratio=float(doc["poisson_ratio"]),
            yield_stress_mpa=float(doc["yield_stress_mpa"]),
            hardening_modulus_mpa=float(doc.get("hardening_modulus_mpa", 0.0)),
            voce_saturation_mpa=float(doc.get("voce_saturation_mpa", 0.0)),
            voce_exponent=float(doc.get("voce_exponent", 10.0)),
        )


def _fourth_order_tensors(
    dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(I_ox_I, I_sym, I_dev)`` of shape ``(3, 3, 3, 3)``."""
    eye = torch.eye(3, dtype=dtype, device=device)
    i_ox_i = torch.einsum("ij,kl->ijkl", eye, eye)
    i_sym = 0.5 * (
        torch.einsum("ik,jl->ijkl", eye, eye) + torch.einsum("il,jk->ijkl", eye, eye)
    )
    i_dev = i_sym - (1.0 / 3.0) * i_ox_i
    return i_ox_i, i_sym, i_dev


class J2IsotropicPlasticity:
    """Small-strain J2 elastoplasticity with linear/Voce isotropic hardening."""

    def __init__(self, config: J2PlasticityConfig) -> None:
        if not isinstance(config, J2PlasticityConfig):
            raise TypeError("config must be a J2PlasticityConfig")
        self.config = config
        self._schema = StateSchema(
            model_id=J2_PLASTICITY_MODEL_ID,
            variables=(
                StateVariableSpec(
                    name="plastic_strain",
                    shape=(3, 3),
                    unit="1",
                    initial_rule="zeros",
                ),
                StateVariableSpec(
                    name="eq_plastic_strain",
                    shape=(1,),
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
        return {
            "plastic_strain": torch.zeros(
                (*batch_shape, 3, 3), dtype=dtype, device=device
            ),
            "eq_plastic_strain": torch.zeros(batch_shape, dtype=dtype, device=device),
        }

    def _validate_state(
        self, state_old: Mapping[str, torch.Tensor], batch_shape: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "plastic_strain" not in state_old or "eq_plastic_strain" not in state_old:
            raise ValueError(
                "state_old must contain 'plastic_strain' and 'eq_plastic_strain'"
            )
        eps_p = state_old["plastic_strain"]
        alpha = state_old["eq_plastic_strain"]
        if eps_p.shape != (*batch_shape, 3, 3):
            raise ValueError(
                f"expected plastic_strain shape {(*batch_shape, 3, 3)}, got {tuple(eps_p.shape)}"
            )
        if alpha.shape != batch_shape:
            raise ValueError(
                f"expected eq_plastic_strain shape {batch_shape}, got {tuple(alpha.shape)}"
            )
        return eps_p, alpha

    def _solve_delta_gamma(
        self, q_trial: torch.Tensor, alpha_old: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve scalar radial-return equation ``q_tr - 3*mu*dg - sy(a + dg) = 0``."""
        cfg = self.config
        sy_old = cfg.yield_stress(alpha_old)
        f_trial = q_trial - sy_old
        plastic_mask = f_trial > 0.0

        if cfg.voce_saturation_mpa == 0.0:
            denom = 3.0 * cfg.mu + cfg.hardening_modulus_mpa
            dgamma = torch.where(
                plastic_mask,
                torch.clamp(f_trial / denom, min=0.0),
                torch.zeros_like(q_trial),
            )
            return dgamma, plastic_mask

        # Monotone scalar Newton solve for linear + Voce saturation hardening.
        h0 = cfg.hardening_slope(alpha_old)
        dgamma = torch.where(
            plastic_mask,
            torch.clamp(f_trial / (3.0 * cfg.mu + h0), min=0.0),
            torch.zeros_like(q_trial),
        )
        for _ in range(16):
            a_curr = alpha_old + dgamma
            g = q_trial - 3.0 * cfg.mu * dgamma - cfg.yield_stress(a_curr)
            dg_prime = -(3.0 * cfg.mu + cfg.hardening_slope(a_curr))
            step = g / dg_prime
            dgamma = torch.where(
                plastic_mask,
                torch.clamp(dgamma - step, min=0.0),
                torch.zeros_like(q_trial),
            )
            if not bool(plastic_mask.any().item()):
                break
            max_res = float(
                torch.where(plastic_mask, g.abs(), torch.zeros_like(g)).max().item()
            )
            if max_res <= 1.0e-12 * max(1.0, cfg.yield_stress_mpa):
                break
        return dgamma, plastic_mask

    def update(
        self,
        f_old: torch.Tensor,
        f_new: torch.Tensor,
        state_old: Mapping[str, torch.Tensor],
        dt: float,
        *,
        temperature: torch.Tensor | None = None,
    ) -> MaterialUpdate:
        """Authoritative 3D radial-return update conforming to ``ConstitutiveModel``."""
        del f_old, temperature
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive")
        if f_new.shape[-2:] != (3, 3):
            raise ValueError(f"f_new must end with (3, 3), got {tuple(f_new.shape)}")

        batch_shape = tuple(int(d) for d in f_new.shape[:-2])
        dtype = f_new.dtype
        device = f_new.device
        eps_p_old, alpha_old = self._validate_state(state_old, batch_shape)
        eps_p_old = eps_p_old.to(dtype=dtype, device=device)
        alpha_old = alpha_old.to(dtype=dtype, device=device)

        cfg = self.config
        eye = torch.eye(3, dtype=dtype, device=device)
        eps_new = 0.5 * (f_new + f_new.transpose(-1, -2)) - eye
        eps_e_trial = eps_new - eps_p_old

        tr_e_trial = (
            eps_e_trial[..., 0, 0] + eps_e_trial[..., 1, 1] + eps_e_trial[..., 2, 2]
        )
        p_mean = cfg.bulk_modulus * tr_e_trial
        dev_e_trial = eps_e_trial - (tr_e_trial / 3.0).unsqueeze(-1).unsqueeze(-1) * eye
        s_trial = 2.0 * cfg.mu * dev_e_trial

        # Safe Frobenius norm with smooth zero-floor for autograd stability
        s_sq = (s_trial * s_trial).sum(dim=(-2, -1))
        s_norm = torch.sqrt(torch.clamp(s_sq, min=1.0e-30))
        q_trial = _SQRT_3_2 * s_norm

        dgamma, plastic_mask = self._solve_delta_gamma(q_trial, alpha_old)
        alpha_new = alpha_old + dgamma
        sy_new = cfg.yield_stress(alpha_new)
        h_new = cfg.hardening_slope(alpha_new)

        n_dir = torch.where(
            plastic_mask.unsqueeze(-1).unsqueeze(-1),
            s_trial / s_norm.unsqueeze(-1).unsqueeze(-1),
            torch.zeros_like(s_trial),
        )
        delta_eps_p = (_SQRT_3_2 * dgamma).unsqueeze(-1).unsqueeze(-1) * n_dir
        eps_p_new = eps_p_old + delta_eps_p
        eps_e_new = eps_new - eps_p_new

        theta = torch.where(
            plastic_mask,
            sy_new / torch.clamp(q_trial, min=1.0e-30),
            torch.ones_like(q_trial),
        )
        s_new = theta.unsqueeze(-1).unsqueeze(-1) * s_trial
        stress = p_mean.unsqueeze(-1).unsqueeze(-1) * eye + s_new

        # Exact 3D consistent elastoplastic tangent:
        # C_ep = K (I ox I) + 2*mu*theta * I_dev - 2*mu*(theta - h/(3*mu + h)) (n ox n)
        i_ox_i, _, i_dev = _fourth_order_tensors(dtype, device)
        c_vol = cfg.bulk_modulus * i_ox_i
        n_ox_n = torch.einsum("...ij,...kl->...ijkl", n_dir, n_dir)
        beta = torch.where(
            plastic_mask,
            theta - h_new / (3.0 * cfg.mu + h_new),
            torch.zeros_like(q_trial),
        )
        c_ep = (
            c_vol
            + (2.0 * cfg.mu * theta)[..., None, None, None, None] * i_dev
            - (2.0 * cfg.mu * beta)[..., None, None, None, None] * n_ox_n
        )

        # Free energy, discrete dissipation, and Ortiz-Steinmann incremental potential
        tr_e_new = eps_e_new[..., 0, 0] + eps_e_new[..., 1, 1] + eps_e_new[..., 2, 2]
        dev_e_new = eps_e_new - (tr_e_new / 3.0).unsqueeze(-1).unsqueeze(-1) * eye
        psi_elastic = 0.5 * cfg.bulk_modulus * tr_e_new * tr_e_new + cfg.mu * (
            dev_e_new * dev_e_new
        ).sum(dim=(-2, -1))
        psi_p_old = cfg.hardening_energy(alpha_old)
        psi_p_new = cfg.hardening_energy(alpha_new)
        free_energy = psi_elastic + psi_p_new

        plastic_work = (stress * delta_eps_p).sum(dim=(-2, -1))
        dissipation = plastic_work - (psi_p_new - psi_p_old)
        inc_potential = free_energy + cfg.yield_stress_mpa * dgamma

        det_f = torch.linalg.det(f_new)
        q_new = _SQRT_3_2 * torch.sqrt(
            torch.clamp((s_new * s_new).sum(dim=(-2, -1)), min=0.0)
        )
        yield_res = q_new - sy_new
        yield_tol = 1.0e-7 * max(1.0, cfg.yield_stress_mpa)
        tr_eps_p = eps_p_new[..., 0, 0] + eps_p_new[..., 1, 1] + eps_p_new[..., 2, 2]

        yield_ok = bool((yield_res <= yield_tol).all().item())
        gamma_ok = bool((dgamma >= -1.0e-14).all().item())
        incomp_ok = bool((tr_eps_p.abs() <= 1.0e-10).all().item())
        diss_ok = bool(
            (dissipation >= -1.0e-10 * max(1.0, cfg.yield_stress_mpa)).all().item()
        )
        det_ok = bool((det_f > 0.0).all().item())
        finite_ok = bool(
            torch.isfinite(stress).all().item()
            and torch.isfinite(c_ep).all().item()
            and torch.isfinite(free_energy).all().item()
            and torch.isfinite(dissipation).all().item()
        )
        admissible = (
            yield_ok and gamma_ok and incomp_ok and diss_ok and det_ok and finite_ok
        )
        return MaterialUpdate(
            stress=stress,
            state_new={
                "plastic_strain": eps_p_new,
                "eq_plastic_strain": alpha_new,
            },
            algorithmic_tangent=c_ep,
            free_energy=free_energy,
            incremental_potential=inc_potential,
            dissipation=dissipation,
            det_f=det_f,
            admissible=admissible,
            guards={
                "yield_admissible": yield_ok,
                "plastic_multiplier_nonnegative": gamma_ok,
                "plastic_incompressibility": incomp_ok,
                "nonnegative_dissipation": diss_ok,
                "det_f_positive": det_ok,
                "finite_response": finite_ok,
            },
        )

    @staticmethod
    def _embed_2d_f(f2d: torch.Tensor, f33: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((*f2d.shape[:-2], 3, 3), dtype=f2d.dtype, device=f2d.device)
        out[..., :2, :2] = f2d
        out[..., 2, 2] = f33
        return out

    def plane_strain_update(
        self,
        f2d_old: torch.Tensor,
        f2d_new: torch.Tensor,
        state_old: Mapping[str, torch.Tensor],
        dt: float,
    ) -> PlaneStressMaterialUpdate:
        """2D plane-strain update (``F_33 = 1``, ``eps_33 = 0``)."""
        ones = torch.ones(
            f2d_new.shape[:-2], dtype=f2d_new.dtype, device=f2d_new.device
        )
        f3d_old = self._embed_2d_f(f2d_old, ones)
        f3d_new = self._embed_2d_f(f2d_new, ones)
        upd = self.update(f3d_old, f3d_new, state_old, dt)
        return PlaneStressMaterialUpdate(
            update_3d=upd,
            f33=ones,
            condensed_tangent_2d=upd.algorithmic_tangent[..., :2, :2, :2, :2],
            closure_residual_max=0.0,
            closure_converged=True,
        )

    def plane_stress_update(
        self,
        f2d_old: torch.Tensor,
        f2d_new: torch.Tensor,
        state_old: Mapping[str, torch.Tensor],
        dt: float,
        *,
        f33_guess: torch.Tensor | None = None,
        max_iterations: int = 32,
        tolerance: float = 1.0e-11,
    ) -> PlaneStressMaterialUpdate:
        """Exact 2D plane-stress update (``sigma_33 = 0``) with Schur-condensed tangent.

        Mathematically identical to Simo-Taylor projected stress-space return
        mapping while reusing the exact 3D radial-return kernel and preserving
        differentiability for variational operator training.
        """
        batch_shape = tuple(int(d) for d in f2d_new.shape[:-2])
        dtype = f2d_new.dtype
        device = f2d_new.device
        eps_p_old, _ = self._validate_state(state_old, batch_shape)
        eps_p_old = eps_p_old.to(dtype=dtype, device=device)

        cfg = self.config
        if f33_guess is not None:
            f33 = f33_guess.detach().to(dtype=dtype, device=device).clone()
        else:
            # Exact elastic plane-stress initial guess:
            # eps_33 = -nu/(1-nu)*(eps_11 - eps_p11 + eps_22 - eps_p22) + eps_p33
            e11 = f2d_new.detach()[..., 0, 0] - 1.0 - eps_p_old[..., 0, 0]
            e22 = f2d_new.detach()[..., 1, 1] - 1.0 - eps_p_old[..., 1, 1]
            eps33_init = (
                -(cfg.poisson_ratio / (1.0 - cfg.poisson_ratio)) * (e11 + e22)
                + eps_p_old[..., 2, 2]
            )
            f33 = 1.0 + eps33_init

        f3d_old = self._embed_2d_f(f2d_old.detach(), torch.ones_like(f33))
        f2d_det = f2d_new.detach()
        tol_abs = tolerance * max(1.0, cfg.yield_stress_mpa)
        converged = False
        max_res = float("inf")

        with torch.no_grad():
            for _ in range(max_iterations):
                upd_iter = self.update(
                    f3d_old, self._embed_2d_f(f2d_det, f33), state_old, dt
                )
                s33 = upd_iter.stress[..., 2, 2]
                max_res = float(s33.abs().max().item()) if s33.numel() else 0.0
                if max_res <= tol_abs:
                    converged = True
                    break
                c3333 = upd_iter.algorithmic_tangent[..., 2, 2, 2, 2]
                if not bool((c3333 > 0.0).all().item()):
                    break
                f33 = f33 - s33 / c3333

        # Final graph-connected evaluation (for autograd / variational training)
        if f2d_new.requires_grad:
            # Implicit function theorem: df33 / dF2D = -C_{22,kl} / C_{2222}
            upd_det = self.update(
                f3d_old, self._embed_2d_f(f2d_det, f33), state_old, dt
            )
            c3333_det = upd_det.algorithmic_tangent[..., 2, 2, 2, 2]
            c33_ab = upd_det.algorithmic_tangent[..., 2, 2, :2, :2]
            df33_df2d = -c33_ab / c3333_det.unsqueeze(-1).unsqueeze(-1)
            f33_connected = f33 + (df33_df2d * (f2d_new - f2d_det)).sum(dim=(-2, -1))
            upd = self.update(
                f3d_old, self._embed_2d_f(f2d_new, f33_connected), state_old, dt
            )
        else:
            upd = self.update(f3d_old, self._embed_2d_f(f2d_new, f33), state_old, dt)

        a_full = upd.algorithmic_tangent
        a_2d = a_full[..., :2, :2, :2, :2]
        a_2d_33 = a_full[..., :2, :2, 2, 2]
        a_33_2d = a_full[..., 2, 2, :2, :2]
        a_3333 = a_full[..., 2, 2, 2, 2]
        condensed = (
            a_2d
            - torch.einsum("...ij,...kl->...ijkl", a_2d_33, a_33_2d)
            / a_3333[..., None, None, None, None]
        )

        s33_final = upd.stress[..., 2, 2]
        max_res = float(s33_final.abs().max().item()) if s33_final.numel() else 0.0
        converged = converged or (max_res <= tol_abs)

        return PlaneStressMaterialUpdate(
            update_3d=upd,
            f33=f33,
            condensed_tangent_2d=condensed,
            closure_residual_max=max_res,
            closure_converged=converged,
        )


def verify_plane_stress_tangent(
    model: J2IsotropicPlasticity,
    f2d_old: torch.Tensor,
    f2d_new: torch.Tensor,
    state_old: Mapping[str, torch.Tensor],
    dt: float,
    *,
    eps: float = 1.0e-7,
    rel_tol: float = 1.0e-5,
    norm_floor: float = 1.0e-12,
) -> TangentCheckResult:
    """Verify the 2D Schur-condensed plane-stress tangent against central FD."""
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and strictly positive")
    with torch.no_grad():
        base = model.plane_stress_update(f2d_old, f2d_new, state_old, dt)
        analytic = base.condensed_tangent_2d
        fd = torch.zeros_like(analytic)
        for k in range(2):
            for col in range(2):
                delta = torch.zeros_like(f2d_new)
                delta[..., k, col] = eps
                p_plus = model.plane_stress_update(
                    f2d_old, f2d_new + delta, state_old, dt, f33_guess=base.f33
                ).update_3d.stress[..., :2, :2]
                p_minus = model.plane_stress_update(
                    f2d_old, f2d_new - delta, state_old, dt, f33_guess=base.f33
                ).update_3d.stress[..., :2, :2]
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


def analytical_uniaxial_j2_response(
    config: J2PlasticityConfig,
    axial_strains: Sequence[float],
) -> dict[str, list[float]]:
    """Exact closed-form 1D uniaxial stress-strain history for J2 isotropic hardening.

    Validates both linear hardening and Voce saturation across monotonic loading,
    elastic unloading, and reversed yielding in compression.
    """
    e = config.youngs_modulus_mpa
    eps_p = 0.0
    alpha = 0.0
    stresses: list[float] = []
    plastic_strains: list[float] = []
    eq_plastic_strains: list[float] = []
    dissipations: list[float] = []

    for eps in axial_strains:
        eps_val = float(eps)
        sig_trial = e * (eps_val - eps_p)
        sy_old = float(
            config.yield_stress(torch.tensor(alpha, dtype=torch.float64)).item()
        )
        f_trial = abs(sig_trial) - sy_old
        if f_trial <= 0.0:
            sig = sig_trial
            dgamma = 0.0
            diss = 0.0
        else:
            sign = 1.0 if sig_trial >= 0.0 else -1.0
            # Solve |sig_trial| - E * dgamma - sy(alpha + dgamma) = 0
            if config.voce_saturation_mpa == 0.0:
                dgamma = f_trial / (e + config.hardening_modulus_mpa)
            else:
                dgamma = f_trial / (
                    e
                    + config.hardening_modulus_mpa
                    + config.voce_saturation_mpa
                    * config.voce_exponent
                    * math.exp(-config.voce_exponent * alpha)
                )
                for _ in range(20):
                    a_c = alpha + dgamma
                    sy_c = float(
                        config.yield_stress(
                            torch.tensor(a_c, dtype=torch.float64)
                        ).item()
                    )
                    h_c = float(
                        config.hardening_slope(
                            torch.tensor(a_c, dtype=torch.float64)
                        ).item()
                    )
                    g = abs(sig_trial) - e * dgamma - sy_c
                    if abs(g) <= 1.0e-14 * max(1.0, config.yield_stress_mpa):
                        break
                    dgamma = max(0.0, dgamma + g / (e + h_c))
            psi_old = float(
                config.hardening_energy(torch.tensor(alpha, dtype=torch.float64)).item()
            )
            eps_p += sign * dgamma
            alpha += dgamma
            sig = e * (eps_val - eps_p)
            psi_new = float(
                config.hardening_energy(torch.tensor(alpha, dtype=torch.float64)).item()
            )
            diss = sig * (sign * dgamma) - (psi_new - psi_old)

        stresses.append(sig)
        plastic_strains.append(eps_p)
        eq_plastic_strains.append(alpha)
        dissipations.append(diss)

    return {
        "axial_strain": [float(x) for x in axial_strains],
        "axial_stress_mpa": stresses,
        "plastic_strain": plastic_strains,
        "eq_plastic_strain": eq_plastic_strains,
        "step_dissipation": dissipations,
    }


__all__ = (
    "J2_PLASTICITY_MODEL_ID",
    "J2IsotropicPlasticity",
    "J2PlasticityConfig",
    "analytical_uniaxial_j2_response",
    "verify_plane_stress_tangent",
)
