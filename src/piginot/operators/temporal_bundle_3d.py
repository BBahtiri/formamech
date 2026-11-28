"""Temporally Bundled & Velocity-Residual 3D Neural Operator (V6-R3).

Combines three late-2026 rollout stabilization mechanisms inside FormaMech's
pure-PyTorch 3D operator ecosystem:

1. **Temporal Bundling (TB-Transolver):** predicts ``bundle_steps = K`` future
   velocity-residual frames simultaneously in one forward pass, reducing
   autoregressive rollout depth by ``K``x.
2. **Velocity-Based Residual Integration (PGRD):** updates state via symplectic
   / semi-implicit velocity residuals ``v_{t+1} = v_base + delta_v``,
   ``x_{t+1} = x_t + dt * v_{t+1}`` instead of raw position jumps.
3. **Stop-Gradient Residual Isolation:** optional ``stop_grad_residual=True``
   blocks noisy off-distribution residual Jacobians during multi-step BPTT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from piginot.operators.surface_operator_3d import (
    Surface3DConfig,
    Surface3DEncoding,
    Surface3DEnergyOperator,
)

TB_OPERATOR_3D_ARCHITECTURE_ID: Final = "formamech-tb-surface-3d-velocity-v1"


@dataclass(frozen=True, slots=True)
class TemporalBundleConfig:
    """Hyperparameters for :class:`TemporallyBundledOperator3D`."""

    surface_config: Surface3DConfig
    bundle_steps: int = 4
    control_dim: int = 4
    dt_s: float = 0.01

    def __post_init__(self) -> None:
        if type(self.bundle_steps) is not int or self.bundle_steps < 1:
            raise ValueError("bundle_steps must be an integer >= 1")
        if type(self.control_dim) is not int or self.control_dim < 1:
            raise ValueError("control_dim must be an integer >= 1")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")


class TemporallyBundledOperator3D(nn.Module):
    """Multi-step temporally bundled velocity-residual 3D dynamics operator."""

    def __init__(self, config: TemporalBundleConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = Surface3DEnergyOperator(config.surface_config)
        d = config.surface_config.hidden_dim
        self.state_control_proj = nn.Sequential(
            nn.Linear(6 + config.control_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.bundle_head = nn.Sequential(
            nn.Linear(d + 3, d),
            nn.GELU(),
            nn.Linear(d, 3 * config.bundle_steps),
        )
        last = self.bundle_head[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(last.bias)

    @property
    def architecture_id(self) -> str:
        return TB_OPERATOR_3D_ARCHITECTURE_ID

    def encode_surface(
        self,
        centroids: torch.Tensor,
        normals: torch.Tensor,
        area_weights: torch.Tensor,
        features: torch.Tensor,
        surface_mask: torch.Tensor | None = None,
    ) -> Surface3DEncoding:
        return self.backbone.encode_surface(
            centroids, normals, area_weights, features, surface_mask=surface_mask
        )

    def predict_velocity_bundle(
        self,
        *,
        encoding: Surface3DEncoding,
        query_xyz: torch.Tensor,  # (B, Q, 3)
        current_u: torch.Tensor,  # (B, Q, 3)
        current_v: torch.Tensor,  # (B, Q, 3)
        control_u: torch.Tensor,  # (B, C)
        parameters: torch.Tensor,  # (B, P)
        stop_grad_residual: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict bundled velocity residuals ``(B, K, Q, 3)`` and integrated ``u``."""
        b, q, _ = query_xyz.shape
        k_steps = self.config.bundle_steps
        base_u = self.backbone.displacement(
            encoding=encoding, query_xyz=query_xyz, parameters=parameters
        )
        ctrl_exp = control_u.unsqueeze(1).expand(b, q, -1)
        state_in = torch.cat((current_u, current_v, ctrl_exp), dim=-1)
        h_state = self.state_control_proj(state_in)
        dv_flat = self.bundle_head(torch.cat((h_state, base_u), dim=-1))
        dv = dv_flat.view(b, q, k_steps, 3).permute(0, 2, 1, 3)  # (B, K, Q, 3)
        if stop_grad_residual:
            dv = dv.detach()

        # Semi-implicit velocity-residual rollout across the K bundled substeps
        dt = float(self.config.dt_s)
        u_cur = current_u
        v_cur = current_v
        u_seq: list[torch.Tensor] = []
        v_seq: list[torch.Tensor] = []
        for s_idx in range(k_steps):
            v_cur = v_cur + dv[:, s_idx]
            u_cur = u_cur + dt * v_cur
            u_seq.append(u_cur)
            v_seq.append(v_cur)
        return torch.stack(u_seq, dim=1), torch.stack(v_seq, dim=1)


__all__ = (
    "TB_OPERATOR_3D_ARCHITECTURE_ID",
    "TemporalBundleConfig",
    "TemporallyBundledOperator3D",
)
