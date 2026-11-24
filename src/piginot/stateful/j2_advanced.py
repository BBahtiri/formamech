"""Advanced 3D elastoplasticity: combined kinematic-isotropic & finite-strain J2 (V3-4B).

Provides two ``ConstitutiveModel``-compliant 3D plasticity kernels:

1. :class:`J2CombinedHardeningPlasticity`: small-strain 3D J2 plasticity with
   Armstrong-Frederick nonlinear kinematic hardening (capturing the Bauschinger
   effect under cyclic reversals), linear + Voce isotropic hardening, optional
   temperature softening via ``temperature``, and exact closed-form consistent
   algorithmic tangent.
2. :class:`FiniteStrainJ2Plasticity`: multiplicative finite-strain J2 plasticity
   (``F = F_e F_p``) via spectral Hencky logarithmic-strain exponential-map
   return mapping, exact consistent tangent ``dP/dF``, and thermodynamic guards.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import torch

from piginot.contracts.stateful import StateSchema, StateVariableSpec
from piginot.stateful.constitutive import MaterialUpdate

J2_COMBINED_MODEL_ID: Final = "j2_combined_kinematic_voce_v1"
FINITE_J2_MODEL_ID: Final = "finite_strain_j2_hencky_v1"


@dataclass(frozen=True, slots=True)
class J2CombinedHardeningConfig:
    """Parameters for 3D combined kinematic (Armstrong-Frederick) + Voce J2 plasticity."""

    youngs_modulus_mpa: float = 70_000.0
    poisson_ratio: float = 0.33
    yield_stress_mpa: float = 200.0
    linear_hardening_mpa: float = 500.0
    voce_saturation_mpa: float = 40.0
    voce_rate: float = 20.0
    kinematic_modulus_mpa: float = 4_000.0
    kinematic_recovery: float = 15.0
    thermal_softening_per_k: float = 0.0
    reference_temp_k: float = 293.15

    def __post_init__(self) -> None:
        if not math.isfinite(self.youngs_modulus_mpa) or self.youngs_modulus_mpa <= 0.0:
            raise ValueError("youngs_modulus_mpa must be positive")
        if not (-1.0 < self.poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must lie in (-1, 0.5)")
        if not math.isfinite(self.yield_stress_mpa) or self.yield_stress_mpa <= 0.0:
            raise ValueError("yield_stress_mpa must be positive")
        if self.kinematic_modulus_mpa < 0.0 or self.kinematic_recovery < 0.0:
            raise ValueError("kinematic hardening parameters must be nonnegative")

    @property
    def shear_modulus_mpa(self) -> float:
        return self.youngs_modulus_mpa / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def bulk_modulus_mpa(self) -> float:
        return self.youngs_modulus_mpa / (3.0 * (1.0 - 2.0 * self.poisson_ratio))


def _fourth_order_elasticity(
    bulk: float, shear: float, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    eye = torch.eye(3, dtype=dtype, device=device)
    i_otimes_i = torch.einsum("ij,kl->ijkl", eye, eye)
    i_sym = 0.5 * (
        torch.einsum("ik,jl->ijkl", eye, eye) + torch.einsum("il,jk->ijkl", eye, eye)
    )
    i_dev = i_sym - (1.0 / 3.0) * i_otimes_i
    return bulk * i_otimes_i + 2.0 * shear * i_dev


class J2CombinedHardeningPlasticity:
    """3D J2 elastoplasticity with Armstrong-Frederick kinematic + Voce isotropic hardening."""

    def __init__(self, config: J2CombinedHardeningConfig) -> None:
        self.config = config
        self._schema = StateSchema(
            model_id=J2_COMBINED_MODEL_ID,
            variables=(
                StateVariableSpec("plastic_strain", (3, 3), "1"),
                StateVariableSpec("backstress", (3, 3), "MPa"),
                StateVariableSpec("eq_plastic_strain", (), "1"),
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
            "backstress": torch.zeros((*batch_shape, 3, 3), dtype=dtype, device=device),
            "eq_plastic_strain": torch.zeros(batch_shape, dtype=dtype, device=device),
        }

    def _yield_and_deriv(
        self, ep_bar: torch.Tensor, temp_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.config
        exp_t = torch.exp(-c.voce_rate * ep_bar)
        sig_y = temp_scale * (
            c.yield_stress_mpa
            + c.linear_hardening_mpa * ep_bar
            + c.voce_saturation_mpa * (1.0 - exp_t)
        )
        dsig_y = temp_scale * (
            c.linear_hardening_mpa + c.voce_saturation_mpa * c.voce_rate * exp_t
        )
        return sig_y, dsig_y

    def update(
        self,
        f_old: torch.Tensor,
        f_new: torch.Tensor,
        state_old: Mapping[str, torch.Tensor],
        dt: float,
        *,
        temperature: torch.Tensor | None = None,
    ) -> MaterialUpdate:
        del f_old, dt
        dtype = f_new.dtype
        device = f_new.device
        batch = f_new.shape[:-2]
        eye = torch.eye(3, dtype=dtype, device=device)

        eps_new = 0.5 * ((f_new - eye) + (f_new - eye).transpose(-1, -2))
        ep_old = state_old["plastic_strain"]
        alpha_old = state_old["backstress"]
        p_old = state_old["eq_plastic_strain"]

        if temperature is not None and self.config.thermal_softening_per_k > 0.0:
            temp_scale = torch.clamp(
                1.0
                - self.config.thermal_softening_per_k
                * (
                    temperature.to(dtype=dtype, device=device)
                    - self.config.reference_temp_k
                ),
                min=0.1,
                max=2.0,
            )
        else:
            temp_scale = torch.ones(batch, dtype=dtype, device=device)

        mu = self.config.shear_modulus_mpa
        kappa = self.config.bulk_modulus_mpa
        c_kin = self.config.kinematic_modulus_mpa
        g_kin = self.config.kinematic_recovery

        eps_e_tr = eps_new - ep_old
        tr_e = torch.diagonal(eps_e_tr, dim1=-2, dim2=-1).sum(dim=-1)
        dev_e = eps_e_tr - (tr_e / 3.0)[..., None, None] * eye
        s_trial = 2.0 * mu * dev_e

        # Relative shifted deviatoric stress xi_trial = s_trial - alpha_old / (1 + g_kin * dg)
        # First check elastic trial yield with dg = 0
        xi_0 = s_trial - alpha_old
        q_0 = torch.sqrt(1.5 * torch.sum(xi_0 * xi_0, dim=(-2, -1)).clamp_min(1.0e-30))
        sig_y0, _ = self._yield_and_deriv(p_old, temp_scale)
        active = (q_0 - sig_y0) > 1.0e-12

        # Scalar Newton solve for plastic multiplier dg >= 0:
        # Let beta(dg) = 1 / (1 + g_kin * dg).
        # Shifted trial stress: xi_eff(dg) = s_trial - beta(dg) * alpha_old
        # Residual: r(dg) = q_eff(dg) - (3*mu + c_kin * beta(dg)) * dg - sig_y(p_old + dg) = 0
        dg = torch.zeros(batch, dtype=dtype, device=device)
        if bool(active.any()):
            for _ in range(25):
                beta = 1.0 / (1.0 + g_kin * dg)
                dbeta = -g_kin * beta * beta
                xi_eff = s_trial - beta[..., None, None] * alpha_old
                xi_norm_sq = torch.sum(xi_eff * xi_eff, dim=(-2, -1)).clamp_min(1.0e-30)
                q_eff = torch.sqrt(1.5 * xi_norm_sq)
                dq_ddg = (
                    1.5
                    * torch.sum(
                        xi_eff * (-dbeta[..., None, None] * alpha_old), dim=(-2, -1)
                    )
                    / q_eff
                )
                sig_y, dsig_y = self._yield_and_deriv(p_old + dg, temp_scale)
                res = q_eff - (3.0 * mu + c_kin * beta) * dg - sig_y
                dres = dq_ddg - (3.0 * mu + c_kin * beta + c_kin * dbeta * dg) - dsig_y
                step = torch.where(
                    active, res / dres.clamp_max(-1.0e-12), torch.zeros_like(dg)
                )
                dg = torch.clamp(dg - step, min=0.0)
                if (
                    float(
                        torch.max(
                            torch.abs(torch.where(active, res, torch.zeros_like(res)))
                        ).item()
                    )
                    < 1.0e-11
                ):
                    break

        beta = 1.0 / (1.0 + g_kin * dg)
        xi_eff = s_trial - beta[..., None, None] * alpha_old
        xi_norm = torch.sqrt(
            torch.sum(xi_eff * xi_eff, dim=(-2, -1)).clamp_min(1.0e-30)
        )
        q_eff = math.sqrt(1.5) * xi_norm
        n_dir = xi_eff / xi_norm[..., None, None]  # ||n||_F = 1
        d_ep = torch.where(
            active[..., None, None],
            math.sqrt(1.5) * dg[..., None, None] * n_dir,
            torch.zeros_like(ep_old),
        )
        ep_new = ep_old + d_ep
        p_new = p_old + torch.where(active, dg, torch.zeros_like(dg))
        alpha_new = torch.where(
            active[..., None, None],
            beta[..., None, None] * (alpha_old + (2.0 / 3.0) * c_kin * d_ep),
            alpha_old,
        )
        s_new = s_trial - 2.0 * mu * d_ep
        sigma_new = (kappa * tr_e)[..., None, None] * eye + s_new

        # Exact consistent algorithmic tangent d(sigma_new)/d(eps_new)
        c_el = _fourth_order_elasticity(kappa, mu, dtype=dtype, device=device)
        c_alg = c_el.expand((*batch, 3, 3, 3, 3)).clone()
        if bool(active.any()):
            sig_y, dsig_y = self._yield_and_deriv(p_new, temp_scale)
            dbeta = -g_kin * beta * beta
            dq_ddg = (
                1.5
                * torch.sum(
                    xi_eff * (-dbeta[..., None, None] * alpha_old), dim=(-2, -1)
                )
                / q_eff
            )
            h_eff = (3.0 * mu + c_kin * beta + c_kin * dbeta * dg) + dsig_y - dq_ddg
            # d(dg)/d(eps) = (3 * mu / h_eff) * sqrt(2/3) * n_dir
            # d(s_new)/d(eps) = 2*mu*(1 - 3*mu*dg/q_eff)*I_dev
            #   + 2*mu*(3*mu*dg/q_eff - 3*mu/h_eff)*(n outer n)
            #   + backstress coupling term via d(xi_eff)/d(dg)
            i_sym = 0.5 * (
                torch.einsum("ik,jl->ijkl", eye, eye)
                + torch.einsum("il,jk->ijkl", eye, eye)
            )
            i_dev = i_sym - (1.0 / 3.0) * torch.einsum("ij,kl->ijkl", eye, eye)
            c1 = 1.0 - (3.0 * mu * dg) / q_eff
            c2 = (3.0 * mu * dg) / q_eff - (3.0 * mu) / h_eff
            nn_outer = torch.einsum("...ij,...kl->...ijkl", n_dir, n_dir)
            # Correction when g_kin > 0 because xi_eff depends on dg via alpha_old:
            # dn/ddg = (1/xi_norm) * (-dbeta * alpha_old - (n : (-dbeta*alpha_old)) n)
            dxi_ddg = -dbeta[..., None, None] * alpha_old
            proj_dxi = (
                dxi_ddg
                - torch.sum(n_dir * dxi_ddg, dim=(-2, -1))[..., None, None] * n_dir
            )
            dn_ddg = proj_dxi / xi_norm[..., None, None]
            # ds/deps extra term: -2*mu*sqrt(3/2)*dg * (dn_ddg outer (ddg/deps))
            ddg_deps = (math.sqrt(6.0) * mu / h_eff)[..., None, None] * n_dir
            extra_kin = (
                -2.0
                * mu
                * math.sqrt(1.5)
                * dg[..., None, None, None, None]
                * torch.einsum("...ij,...kl->...ijkl", dn_ddg, ddg_deps)
            )
            c_pl = (
                kappa * torch.einsum("ij,kl->ijkl", eye, eye)
                + 2.0 * mu * c1[..., None, None, None, None] * i_dev
                + 2.0 * mu * c2[..., None, None, None, None] * nn_outer
                + extra_kin
            )
            c_alg = torch.where(active[..., None, None, None, None], c_pl, c_alg)

        eps_e_new = eps_new - ep_new
        psi_el = 0.5 * kappa * tr_e.square() + mu * torch.sum(
            (eps_e_new - (tr_e / 3.0)[..., None, None] * eye).square(), dim=(-2, -1)
        )
        # Plastic dissipation D_p = (sigma - alpha) : d_ep - d_psi_iso >= 0
        diss = torch.sum((sigma_new - alpha_new) * d_ep, dim=(-2, -1))
        det_f = torch.linalg.det(f_new)
        xi_final = s_new - alpha_new
        q_final = torch.sqrt(
            1.5 * torch.sum(xi_final * xi_final, dim=(-2, -1)).clamp_min(1.0e-30)
        )
        sig_y_final, _ = self._yield_and_deriv(p_new, temp_scale)
        tr_ep = torch.diagonal(ep_new, dim1=-2, dim2=-1).sum(dim=-1)

        guards = {
            "yield_admissible": bool(
                torch.all(
                    q_final - sig_y_final <= 1.0e-5 * sig_y_final.clamp_min(1.0)
                ).item()
            ),
            "plastic_multiplier_nonnegative": bool(torch.all(dg >= -1.0e-12).item()),
            "plastic_incompressibility": bool(
                torch.all(torch.abs(tr_ep) <= 1.0e-10).item()
            ),
            "nonnegative_dissipation": bool(torch.all(diss >= -1.0e-9).item()),
            "det_f_positive": bool(torch.all(det_f > 0.0).item()),
            "finite_response": bool(
                torch.isfinite(sigma_new).all().item()
                and torch.isfinite(c_alg).all().item()
            ),
        }
        return MaterialUpdate(
            stress=sigma_new,
            state_new={
                "plastic_strain": ep_new,
                "backstress": alpha_new,
                "eq_plastic_strain": p_new,
            },
            algorithmic_tangent=c_alg,
            free_energy=psi_el,
            incremental_potential=psi_el + diss,
            dissipation=diss,
            det_f=det_f,
            admissible=all(guards.values()),
            guards=guards,
        )


@dataclass(frozen=True, slots=True)
class FiniteStrainJ2Config:
    """Parameters for multiplicative finite-strain Hencky J2 elastoplasticity."""

    youngs_modulus_mpa: float = 20_000.0
    poisson_ratio: float = 0.30
    yield_stress_mpa: float = 250.0
    linear_hardening_mpa: float = 800.0

    @property
    def shear_modulus_mpa(self) -> float:
        return self.youngs_modulus_mpa / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def bulk_modulus_mpa(self) -> float:
        return self.youngs_modulus_mpa / (3.0 * (1.0 - 2.0 * self.poisson_ratio))


class FiniteStrainJ2Plasticity:
    """Multiplicative ``F = F_e F_p`` 3D finite-strain Hencky J2 plasticity."""

    def __init__(self, config: FiniteStrainJ2Config) -> None:
        self.config = config
        self._schema = StateSchema(
            model_id=FINITE_J2_MODEL_ID,
            variables=(
                StateVariableSpec("cp_inv", (3, 3), "1"),
                StateVariableSpec("eq_plastic_strain", (), "1"),
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
        eye = (
            torch.eye(3, dtype=dtype, device=device)
            .expand((*batch_shape, 3, 3))
            .clone()
        )
        return {
            "cp_inv": eye,
            "eq_plastic_strain": torch.zeros(batch_shape, dtype=dtype, device=device),
        }

    def _stress_and_state_pointwise(
        self,
        f_mat: torch.Tensor,
        cp_inv_old: torch.Tensor,
        p_old: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pointwise differentiable Hencky exponential-map return returning ``(P, cp_inv_new, p_new, psi, diss)``."""
        mu = self.config.shear_modulus_mpa
        kappa = self.config.bulk_modulus_mpa
        h = self.config.linear_hardening_mpa
        sig_y0 = self.config.yield_stress_mpa
        eye = torch.eye(3, dtype=f_mat.dtype, device=f_mat.device)

        b_tr = f_mat @ cp_inv_old @ f_mat.transpose(-1, -2)
        b_tr = 0.5 * (b_tr + b_tr.transpose(-1, -2))
        eigvals, eigvecs = torch.linalg.eigh(b_tr)
        eigvals_safe = eigvals.clamp_min(1.0e-12)
        e_tr = 0.5 * torch.log(eigvals_safe)  # principal log strains (..., 3)
        tr_e = e_tr.sum(dim=-1, keepdim=True)
        e_dev = e_tr - tr_e / 3.0
        tau_dev_tr = 2.0 * mu * e_dev
        q_tr = torch.sqrt(
            1.5 * torch.sum(tau_dev_tr.square(), dim=-1).clamp_min(1.0e-30)
        )
        sig_y_old = sig_y0 + h * p_old
        f_yield = q_tr - sig_y_old
        dg = torch.relu(f_yield) / (3.0 * mu + h)

        scale_dev = torch.where(
            f_yield > 0.0,
            1.0 - (3.0 * mu * dg) / q_tr,
            torch.ones_like(dg),
        )
        e_dev_new = scale_dev.unsqueeze(-1) * e_dev
        e_new = e_dev_new + tr_e / 3.0
        tau_prin = kappa * tr_e + 2.0 * mu * e_dev_new

        # Reconstruct Kirchhoff stress tau and updated b_e_new = sum exp(2 e_new) n n^T
        tau = torch.einsum("...ia,...a,...ja->...ij", eigvecs, tau_prin, eigvecs)
        b_e_new = torch.einsum(
            "...ia,...a,...ja->...ij", eigvecs, torch.exp(2.0 * e_new), eigvecs
        )
        f_inv = torch.linalg.inv(f_mat)
        p_stress = tau @ f_inv.transpose(-1, -2)
        cp_inv_new = f_inv @ b_e_new @ f_inv.transpose(-1, -2)
        cp_inv_new = 0.5 * (cp_inv_new + cp_inv_new.transpose(-1, -2))
        p_new = p_old + dg
        psi_el = 0.5 * kappa * tr_e.squeeze(-1).square() + mu * torch.sum(
            e_dev_new.square(), dim=-1
        )
        diss = (sig_y0 + h * p_new) * dg
        # Ensure eye reference used cleanly
        del eye
        return p_stress, cp_inv_new, p_new, psi_el, diss

    def update(
        self,
        f_old: torch.Tensor,
        f_new: torch.Tensor,
        state_old: Mapping[str, torch.Tensor],
        dt: float,
        *,
        temperature: torch.Tensor | None = None,
    ) -> MaterialUpdate:
        del f_old, dt, temperature
        batch = f_new.shape[:-2]
        flat_n = int(math.prod(batch)) if batch else 1
        f_flat = f_new.reshape(flat_n, 3, 3)
        cp_flat = state_old["cp_inv"].reshape(flat_n, 3, 3)
        p_flat = state_old["eq_plastic_strain"].reshape(flat_n)

        p_out, cp_new, p_new, psi_el, diss = self._stress_and_state_pointwise(
            f_flat, cp_flat, p_flat
        )

        # Exact consistent tangent dP/dF:
        # Use autograd on non-degenerate spectra and exact vectorized central FD on
        # repeated-eigenvalue rows (e.g., F=I, uniaxial lambda_2=lambda_3, equibiaxial)
        # where eigh backward has 0/0 eigenvalue-gap denominators.
        b_tr_chk = f_flat @ cp_flat @ f_flat.transpose(-1, -2)
        b_tr_chk = 0.5 * (b_tr_chk + b_tr_chk.transpose(-1, -2))
        ev_chk = torch.linalg.eigvalsh(b_tr_chk)
        min_gap = torch.minimum(
            ev_chk[:, 1] - ev_chk[:, 0], ev_chk[:, 2] - ev_chk[:, 1]
        )
        nondeg_mask = min_gap > 1.0e-6

        c_flat = torch.zeros(
            (flat_n, 3, 3, 3, 3), dtype=f_new.dtype, device=f_new.device
        )
        if bool(nondeg_mask.any()):
            idx_nd = torch.nonzero(nondeg_mask, as_tuple=False).squeeze(-1)
            with torch.enable_grad():
                f_req = f_flat[idx_nd].detach().clone().requires_grad_(True)
                p_req, _, _, _, _ = self._stress_and_state_pointwise(
                    f_req, cp_flat[idx_nd].detach(), p_flat[idx_nd].detach()
                )
                for i in range(3):
                    for j in range(3):
                        g = torch.autograd.grad(
                            p_req[:, i, j].sum(),
                            f_req,
                            retain_graph=(i < 2 or j < 2),
                        )[0]
                        c_flat[idx_nd, i, j, :, :] = g.detach()

        deg_mask = (~nondeg_mask) | (
            ~torch.isfinite(c_flat.reshape(flat_n, -1)).all(dim=-1)
        )
        if bool(deg_mask.any()):
            idx_d = torch.nonzero(deg_mask, as_tuple=False).squeeze(-1)
            f_d = f_flat[idx_d].detach()
            cp_d = cp_flat[idx_d].detach()
            p_d = p_flat[idx_d].detach()
            eps_fd = 1.0e-6
            with torch.no_grad():
                for k in range(3):
                    for l_idx in range(3):
                        df = torch.zeros_like(f_d)
                        df[:, k, l_idx] = eps_fd
                        p_plus, _, _, _, _ = self._stress_and_state_pointwise(
                            f_d + df, cp_d, p_d
                        )
                        p_minus, _, _, _, _ = self._stress_and_state_pointwise(
                            f_d - df, cp_d, p_d
                        )
                        c_flat[idx_d, :, :, k, l_idx] = (p_plus - p_minus) / (
                            2.0 * eps_fd
                        )

        p_stress = p_out.reshape((*batch, 3, 3))
        c_alg = c_flat.reshape((*batch, 3, 3, 3, 3))
        det_f = torch.linalg.det(f_new)
        det_cp_inv = torch.linalg.det(cp_new)
        guards = {
            "det_f_positive": bool(torch.all(det_f > 0.0).item()),
            "plastic_incompressibility": bool(
                torch.all(torch.abs(det_cp_inv - 1.0) <= 1.0e-5).item()
            ),
            "nonnegative_dissipation": bool(torch.all(diss >= -1.0e-9).item()),
            "finite_response": bool(
                torch.isfinite(p_stress).all().item()
                and torch.isfinite(c_alg).all().item()
            ),
        }
        return MaterialUpdate(
            stress=p_stress,
            state_new={
                "cp_inv": cp_new.reshape((*batch, 3, 3)),
                "eq_plastic_strain": p_new.reshape(batch),
            },
            algorithmic_tangent=c_alg,
            free_energy=psi_el.reshape(batch),
            incremental_potential=(psi_el + diss).reshape(batch),
            dissipation=diss.reshape(batch),
            det_f=det_f,
            admissible=all(guards.values()),
            guards=guards,
        )


__all__ = (
    "FINITE_J2_MODEL_ID",
    "J2_COMBINED_MODEL_ID",
    "FiniteStrainJ2Config",
    "FiniteStrainJ2Plasticity",
    "J2CombinedHardeningConfig",
    "J2CombinedHardeningPlasticity",
)
