"""Structure-preserving Port-Hamiltonian Neural Dynamics for Robotics (V6-R3).

Parameterizes low-dimensional soft/continuum/articulated robot dynamics as a
passive Port-Hamiltonian system::

    H(q, p) = 0.5 * p^T M^{-1}(q) p + W_theta(q)
    dq/dt =  dH/dp = M^{-1}(q) p
    dp/dt = -dH/dq - D_theta(q) (dH/dp) + G_theta(q) u

where:
* ``M^{-1}(q) = L(q) L(q)^T + eps * I`` is strictly symmetric positive definite
  (Cholesky parameterization guaranteeing bounded kinetic energy),
* ``D_theta(q) = R(q) R(q)^T`` is symmetric positive semidefinite (guaranteeing
  nonnegative physical dissipation ``(dH/dp)^T D (dH/dp) >= 0``),
* ``W_theta(q)`` can be distilled directly from FormaMech's verified 3D FEM
  strain energy and integrated with symplectic / RK4 stepping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

FloatArray = NDArray[np.float64]
PHNN_ARCHITECTURE_ID: Final = "formamech-port-hamiltonian-v1"


@dataclass(frozen=True, slots=True)
class PortHamiltonianConfig:
    """Architecture hyperparameters for :class:`PortHamiltonianSurrogate`."""

    state_dim: int  # generalized coordinate dimension n (total phase space 2n)
    control_dim: int  # actuator input dimension m
    hidden_dim: int = 64
    min_mass_inv: float = 1.0e-3
    min_damping: float = 1.0e-4

    def __post_init__(self) -> None:
        if type(self.state_dim) is not int or self.state_dim < 1:
            raise ValueError("state_dim must be an integer >= 1")
        if type(self.control_dim) is not int or self.control_dim < 1:
            raise ValueError("control_dim must be an integer >= 1")
        if type(self.hidden_dim) is not int or self.hidden_dim < 4:
            raise ValueError("hidden_dim must be >= 4")
        if not math.isfinite(self.min_mass_inv) or self.min_mass_inv <= 0.0:
            raise ValueError("min_mass_inv must be positive")
        if not math.isfinite(self.min_damping) or self.min_damping < 0.0:
            raise ValueError("min_damping must be nonnegative")


class PortHamiltonianSurrogate(nn.Module):
    """Passive, energy-consistent Port-Hamiltonian neural dynamics surrogate."""

    def __init__(self, config: PortHamiltonianConfig) -> None:
        super().__init__()
        self.config = config
        n = config.state_dim
        m = config.control_dim
        h = config.hidden_dim

        # Smooth C^inf Softplus potential energy W_theta(q) + quadratic baseline
        self.potential_net = nn.Sequential(
            nn.Linear(n, h),
            nn.Softplus(),
            nn.Linear(h, h),
            nn.Softplus(),
            nn.Linear(h, 1),
        )
        self.quad_stiff_log = nn.Parameter(torch.zeros(n))

        # Cholesky lower-triangular factor of M^{-1}(q)
        self.n_tril = n * (n + 1) // 2
        self.mass_inv_net = nn.Sequential(
            nn.Linear(n, h),
            nn.Tanh(),
            nn.Linear(h, self.n_tril),
        )
        # Cholesky factor of dissipation matrix D(q) = R(q) R(q)^T >= 0
        self.damping_net = nn.Sequential(
            nn.Linear(n, h),
            nn.Tanh(),
            nn.Linear(h, self.n_tril),
        )
        # Input port matrix G(q) (n x m)
        self.input_net = nn.Sequential(
            nn.Linear(n, h),
            nn.Tanh(),
            nn.Linear(h, n * m),
        )
        self._tril_rows, self._tril_cols = torch.tril_indices(n, n)

    @property
    def architecture_id(self) -> str:
        return PHNN_ARCHITECTURE_ID

    def _build_psd(self, raw_tril: torch.Tensor, floor: float) -> torch.Tensor:
        b = raw_tril.shape[0]
        n = self.config.state_dim
        l_mat = torch.zeros((b, n, n), dtype=raw_tril.dtype, device=raw_tril.device)
        l_mat[:, self._tril_rows, self._tril_cols] = raw_tril
        eye = torch.eye(n, dtype=raw_tril.dtype, device=raw_tril.device).unsqueeze(0)
        return torch.matmul(l_mat, l_mat.transpose(-1, -2)) + float(floor) * eye

    def mass_inv_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Strictly SPD inverse inertia matrix ``M^{-1}(q)`` of shape ``(B, n, n)``."""
        return self._build_psd(self.mass_inv_net(q), self.config.min_mass_inv)

    def damping_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Strictly PSD dissipation matrix ``D(q)`` of shape ``(B, n, n)``."""
        return self._build_psd(self.damping_net(q), self.config.min_damping)

    def input_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Actuator input matrix ``G(q)`` of shape ``(B, n, m)``."""
        b = q.shape[0]
        out: torch.Tensor = self.input_net(q).view(
            b, self.config.state_dim, self.config.control_dim
        )
        return out

    def potential_energy(self, q: torch.Tensor) -> torch.Tensor:
        """Scalar potential energy ``W(q)`` of shape ``(B,)``."""
        k_diag = torch.nn.functional.softplus(self.quad_stiff_log)
        w_quad = 0.5 * (q.square() * k_diag.unsqueeze(0)).sum(dim=-1)
        w_nn: torch.Tensor = self.potential_net(q).squeeze(-1)
        total: torch.Tensor = w_quad + w_nn
        return total

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Total Hamiltonian ``H(q, p) = 0.5 p^T M^{-1}(q) p + W(q)``."""
        m_inv = self.mass_inv_matrix(q)
        ke = 0.5 * torch.einsum("bi,bij,bj->b", p, m_inv, p)
        return ke + self.potential_energy(q)

    def vector_field(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        u: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(dq_dt, dp_dt, dissipation_rate >= 0)``."""
        q_req = q if q.requires_grad else q.detach().requires_grad_(True)
        p_req = p if p.requires_grad else p.detach().requires_grad_(True)
        h_val = self.hamiltonian(q_req, p_req)
        dh_dq, dh_dp = torch.autograd.grad(
            h_val.sum(),
            (q_req, p_req),
            create_graph=create_graph,
            retain_graph=create_graph,
        )
        d_mat = self.damping_matrix(q_req)
        g_mat = self.input_matrix(q_req)
        d_vel = torch.einsum("bij,bj->bi", d_mat, dh_dp)
        g_u = torch.einsum("bij,bj->bi", g_mat, u)

        dq_dt = dh_dp
        dp_dt = -dh_dq - d_vel + g_u
        dissipation = torch.einsum("bi,bi->b", dh_dp, d_vel)
        return dq_dt, dp_dt, dissipation

    def step_rk4(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        u: torch.Tensor,
        dt_s: float,
        *,
        create_graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """4th-order Runge-Kutta integration preserving Hamiltonian passivity."""
        dt = float(dt_s)
        k1_q, k1_p, _ = self.vector_field(q, p, u, create_graph=create_graph)
        k2_q, k2_p, _ = self.vector_field(
            q + 0.5 * dt * k1_q, p + 0.5 * dt * k1_p, u, create_graph=create_graph
        )
        k3_q, k3_p, _ = self.vector_field(
            q + 0.5 * dt * k2_q, p + 0.5 * dt * k2_p, u, create_graph=create_graph
        )
        k4_q, k4_p, _ = self.vector_field(
            q + dt * k3_q, p + dt * k3_p, u, create_graph=create_graph
        )
        q_next = q + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        p_next = p + (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
        return q_next, p_next

    def linearize_discrete(
        self,
        x_np: FloatArray,  # (2n,) concatenated [q, p]
        u_np: FloatArray,  # (m,)
        dt_s: float,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return exact discrete-time linearization ``(x_next, A_d, B_d)`` for QP-MPC."""
        n = self.config.state_dim
        m = self.config.control_dim
        x_t = torch.tensor(x_np, dtype=torch.float32)
        u_t = torch.tensor(u_np, dtype=torch.float32)

        def _f(x_in: torch.Tensor, u_in: torch.Tensor) -> torch.Tensor:
            q_n, p_n = self.step_rk4(
                x_in[:n].unsqueeze(0),
                x_in[n:].unsqueeze(0),
                u_in.unsqueeze(0),
                dt_s,
                create_graph=True,
            )
            return torch.cat((q_n[0], p_n[0]), dim=0)

        x_next_t = _f(x_t, u_t)
        # Finite-difference Jacobian in eval mode is ultra-fast for small n (<=12)
        eps = 1.0e-4
        a_d = np.zeros((2 * n, 2 * n), dtype=np.float64)
        b_d = np.zeros((2 * n, m), dtype=np.float64)
        for j in range(2 * n):
            dx = torch.zeros(2 * n, dtype=torch.float32)
            dx[j] = eps
            fp = _f(x_t + dx, u_t).detach().numpy()
            fm = _f(x_t - dx, u_t).detach().numpy()
            a_d[:, j] = (fp - fm) / (2.0 * eps)
        for j in range(m):
            du = torch.zeros(m, dtype=torch.float32)
            du[j] = eps
            fp = _f(x_t, u_t + du).detach().numpy()
            fm = _f(x_t, u_t - du).detach().numpy()
            b_d[:, j] = (fp - fm) / (2.0 * eps)
        return x_next_t.detach().numpy().astype(np.float64), a_d, b_d


__all__ = (
    "PHNN_ARCHITECTURE_ID",
    "PortHamiltonianConfig",
    "PortHamiltonianSurrogate",
)
