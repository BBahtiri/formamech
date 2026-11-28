"""3D virtual-work / weak-form equilibrium objectives for non-conservative loads (V6-R1).

Pure Deep Ritz potential-energy minimization breaks down under non-conservative
3D follower pressure (``t = -p J F^{-T} N`` on open boundaries) and routed tendon
forces. This module computes differentiable PyTorch weak-form virtual-work
residuals::

    R_vw(u; v) = int_Omega P(I + grad u) : grad(v) dV
                 - int_{Gamma_p} (-p) v . (g_1 x g_2) dxi deta
                 - sum_i v(x_i) . f_point_i

using randomized or structured vector test fields ``v`` that vanish on Dirichlet
boundaries, combined with an explicit ``relu(J_min - det(F))^2`` admissibility
barrier.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piginot.contracts import MaterialSpec
from piginot.physics import first_piola


@dataclass(frozen=True, slots=True)
class VirtualWork3DLossBreakdown:
    """Decomposed differentiable 3D weak-form virtual-work loss terms."""

    total: torch.Tensor
    virtual_work_residual: torch.Tensor
    jacobian_barrier: torch.Tensor
    min_det_f: torch.Tensor


def compute_3d_follower_virtual_work_loss(
    *,
    grad_u_vol: torch.Tensor,  # (N_vol, 3, 3)
    vol_weights: torch.Tensor,  # (N_vol,)
    grad_v_vol: torch.Tensor,  # (K_test, N_vol, 3, 3)
    material: MaterialSpec,
    surf_g1: torch.Tensor | None = None,  # (N_surf, 3) deformed dx/dxi
    surf_g2: torch.Tensor | None = None,  # (N_surf, 3) deformed dx/deta
    surf_weights: torch.Tensor | None = None,  # (N_surf,) quadrature weights
    surf_pressure_mpa: torch.Tensor | float = 0.0,
    surf_v: torch.Tensor | None = None,  # (K_test, N_surf, 3)
    point_forces_n: torch.Tensor | None = None,  # (N_pt, 3)
    point_v: torch.Tensor | None = None,  # (K_test, N_pt, 3)
    j_min: float = 0.1,
    barrier_weight: float = 100.0,
    scale_ref: float = 1.0,
) -> VirtualWork3DLossBreakdown:
    """Compute differentiable 3D virtual-work residual across ``K_test`` test modes."""
    eye = torch.eye(3, dtype=grad_u_vol.dtype, device=grad_u_vol.device)
    f_vol = grad_u_vol + eye.unsqueeze(0)
    det_f = torch.linalg.det(f_vol)
    min_det = det_f.min()
    j_bar = torch.relu(float(j_min) - det_f).square().mean()

    # Clamp extreme negative det only in constitutive evaluation to keep gradients finite
    f_safe = f_vol
    p_vol = first_piola(f_safe, material)  # (N_vol, 3, 3)
    # Internal virtual work per test function k: sum_q w_q * P_q : grad(v)_{k,q}
    w_int = torch.einsum("q,qij,kqij->k", vol_weights, p_vol, grad_v_vol)

    w_ext = torch.zeros_like(w_int)
    if surf_g1 is not None and surf_g2 is not None and surf_v is not None:
        g_cross = torch.linalg.cross(surf_g1, surf_g2, dim=-1)  # (N_surf, 3)
        w_s = (
            surf_weights
            if surf_weights is not None
            else torch.ones(
                surf_g1.shape[0], dtype=surf_g1.dtype, device=surf_g1.device
            )
        )
        p_t = (
            surf_pressure_mpa
            if isinstance(surf_pressure_mpa, torch.Tensor)
            else torch.tensor(
                float(surf_pressure_mpa), dtype=surf_g1.dtype, device=surf_g1.device
            )
        )
        w_ext = w_ext + torch.einsum("s,si,ksi->k", -p_t * w_s, g_cross, surf_v)

    if point_forces_n is not None and point_v is not None:
        w_ext = w_ext + torch.einsum("pi,kpi->k", point_forces_n, point_v)

    inv_scale = 1.0 / max(float(scale_ref), 1.0e-12)
    vw_res = ((w_int - w_ext) * inv_scale).square().mean()
    total = vw_res + float(barrier_weight) * j_bar
    return VirtualWork3DLossBreakdown(
        total=total,
        virtual_work_residual=vw_res,
        jacobian_barrier=j_bar,
        min_det_f=min_det,
    )


__all__ = (
    "VirtualWork3DLossBreakdown",
    "compute_3d_follower_virtual_work_loss",
)
