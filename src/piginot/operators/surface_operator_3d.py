"""Linear-complexity O(N * S) 3D surface-token neural operator foundation (Phase 6).

Avoids cubic O(N^3) voxel SDF grids and quadratic O(N^2) point-cloud attention
by using Transolver++/LinearNO-style ``PhysicsSliceAttention`` with explicit
double-backward-safe scaled dot-product attention (supporting arbitrary-order
autograd for Deep Ritz / energy-form training on CPU and GPU).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

SURFACE_3D_ARCHITECTURE_ID: Final = "formamech-surface-3d-slice-energy-v1"


@dataclass(frozen=True, slots=True)
class Surface3DConfig:
    """Architecture hyperparameters for :class:`Surface3DEnergyOperator`."""

    parameter_dim: int = 2
    surface_feature_dim: int = 6
    hidden_dim: int = 32
    slice_count: int = 8
    num_heads: int = 4
    num_blocks: int = 2
    positional_frequencies: int = 3

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if self.slice_count < 1:
            raise ValueError("slice_count must be >= 1")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if self.positional_frequencies < 1:
            raise ValueError("positional_frequencies must be >= 1")


@dataclass(frozen=True, slots=True)
class Surface3DEncoding:
    """Reusable compressed physics-slice representation of one 3D CAD surface."""

    slice_tokens: torch.Tensor  # (B, S, D)
    origin: torch.Tensor  # (B, 3)
    extent: torch.Tensor  # (B, 3)


class _ExplicitMultiHeadAttention(nn.Module):
    """Pure-matmul multi-head attention supporting higher-order autograd everywhere."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        b, nq, _ = query.shape
        nk = key.shape[1]
        h, d = self.num_heads, self.head_dim
        q = self.q_proj(query).view(b, nq, h, d).transpose(1, 2)
        k = self.k_proj(key).view(b, nk, h, d).transpose(1, 2)
        v = self.v_proj(value).view(b, nk, h, d).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, nq, h * d)
        projected: torch.Tensor = self.out_proj(out)
        return projected


class PhysicsSliceAttention(nn.Module):
    """Transolver++-style O(N * S) linear-complexity physics-slice attention block."""

    def __init__(self, hidden_dim: int, slice_count: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slice_count = slice_count
        self.slice_weights = nn.Linear(hidden_dim, slice_count)
        self.slice_attn = _ExplicitMultiHeadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        measure_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project ``(B, N, D)`` onto ``S`` slices, attend in ``O(S^2)``, and deslice.

        Returns ``(updated_tokens (B, N, D), slice_tokens (B, S, D))``.
        """
        logits = self.slice_weights(tokens)
        pi = torch.softmax(logits, dim=-1)
        weighted_pi = (
            pi * measure_weights.unsqueeze(-1) if measure_weights is not None else pi
        )
        denom = weighted_pi.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        slices = torch.einsum("bns,bnd->bsd", weighted_pi, tokens) / denom.transpose(
            1, 2
        )
        attn_out = self.slice_attn(slices, slices, slices)
        slices = self.norm1(slices + attn_out)
        slices = self.norm2(slices + self.ffn(slices))
        desliced = torch.einsum("bns,bsd->bnd", pi, slices)
        return tokens + desliced, slices


def _fourier_encode_3d(coords: torch.Tensor, frequencies: int) -> torch.Tensor:
    """Deterministic multi-frequency sinusoidal encoding of normalized 3D coords."""
    parts: list[torch.Tensor] = [coords]
    for k in range(frequencies):
        scale = (2.0**k) * math.pi
        parts.append(torch.sin(scale * coords))
        parts.append(torch.cos(scale * coords))
    return torch.cat(parts, dim=-1)


class Surface3DEnergyOperator(nn.Module):
    """Mesh-free 3D surface-conditioned neural operator with O(N * S) slice attention."""

    def __init__(self, config: Surface3DConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.surface_in = nn.Sequential(
            nn.Linear(6 + config.surface_feature_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.encoder_blocks = nn.ModuleList(
            [
                PhysicsSliceAttention(d, config.slice_count, config.num_heads)
                for _ in range(config.num_blocks)
            ]
        )
        pos_dim = 3 * (1 + 2 * config.positional_frequencies)
        self.param_proj = nn.Sequential(
            nn.Linear(config.parameter_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.query_in = nn.Sequential(
            nn.Linear(pos_dim + d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.query_cross_attn = _ExplicitMultiHeadAttention(d, config.num_heads)
        self.query_norm = nn.LayerNorm(d)
        self.decoder_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 3),
        )
        last = self.decoder_head[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(last.bias)

    @property
    def architecture_id(self) -> str:
        return SURFACE_3D_ARCHITECTURE_ID

    def encode_surface(
        self,
        centroids: torch.Tensor,
        normals: torch.Tensor,
        area_weights: torch.Tensor,
        features: torch.Tensor,
    ) -> Surface3DEncoding:
        """Encode 3D surface triangles ``(B, M, ...)`` into ``S`` physics slices."""
        origin = centroids.amin(dim=1)
        extent = (centroids.amax(dim=1) - origin).clamp_min(1.0e-6)
        norm_c = (centroids - origin.unsqueeze(1)) / extent.unsqueeze(1)
        x = torch.cat((norm_c, normals, features), dim=-1)
        h = self.surface_in(x)
        slices = h[:, : self.config.slice_count, :]
        for block in self.encoder_blocks:
            h, slices = block(h, measure_weights=area_weights)
        return Surface3DEncoding(slice_tokens=slices, origin=origin, extent=extent)

    def displacement(
        self,
        *,
        encoding: Surface3DEncoding,
        query_xyz: torch.Tensor,
        parameters: torch.Tensor,
        lifting_offset: torch.Tensor | None = None,
        lifting_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict 3D displacement field ``(B, Q, 3)`` at arbitrary interior query points."""
        norm_q = (query_xyz - encoding.origin.unsqueeze(1)) / encoding.extent.unsqueeze(
            1
        )
        pos = _fourier_encode_3d(norm_q, self.config.positional_frequencies)
        p_emb = (
            self.param_proj(parameters).unsqueeze(1).expand(-1, query_xyz.shape[1], -1)
        )
        q_tok = self.query_in(torch.cat((pos, p_emb), dim=-1))
        cross = self.query_cross_attn(
            q_tok, encoding.slice_tokens, encoding.slice_tokens
        )
        q_tok = self.query_norm(q_tok + cross)
        raw_u: torch.Tensor = self.decoder_head(q_tok)
        if lifting_scale is not None:
            raw_u = raw_u * lifting_scale
        if lifting_offset is not None:
            raw_u = raw_u + lifting_offset
        return raw_u

    def displacement_and_gradient(
        self,
        *,
        encoding: Surface3DEncoding,
        query_xyz: torch.Tensor,
        parameters: torch.Tensor,
        lifting_offset: torch.Tensor | None = None,
        lifting_scale: torch.Tensor | None = None,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(u (B, Q, 3), grad_u (B, Q, 3, 3))`` via exact PyTorch autograd."""
        pts = query_xyz.detach().requires_grad_(True)
        u = self.displacement(
            encoding=encoding,
            query_xyz=pts,
            parameters=parameters,
            lifting_offset=lifting_offset,
            lifting_scale=lifting_scale,
        )
        grads: list[torch.Tensor] = []
        for comp in range(3):
            g = torch.autograd.grad(
                u[..., comp].sum(),
                pts,
                create_graph=create_graph,
                retain_graph=create_graph or comp < 2,
            )[0]
            grads.append(g)
        grad_u = torch.stack(grads, dim=-2)
        return u, grad_u


__all__ = (
    "SURFACE_3D_ARCHITECTURE_ID",
    "PhysicsSliceAttention",
    "Surface3DConfig",
    "Surface3DEncoding",
    "Surface3DEnergyOperator",
)
