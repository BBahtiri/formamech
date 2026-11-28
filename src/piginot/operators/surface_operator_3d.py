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
    decoder_blocks: int = 1
    adaptive_temperature: bool = False  # V6-R5 Transolver++ locally adaptive slicing
    anchor_count: int | None = None  # V6-R5 AB-UPT anchor-token subsampling
    low_rank_query_dim: int | None = None  # V6-R5 LRQ-Solver low-rank query compression
    decode_chunk_size: int | None = None  # V6-R5 bounded-memory 100K-1M query decoding

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if self.slice_count < 1:
            raise ValueError("slice_count must be >= 1")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if self.positional_frequencies < 1:
            raise ValueError("positional_frequencies must be >= 1")
        if self.decoder_blocks < 1:
            raise ValueError("decoder_blocks must be >= 1")
        if self.anchor_count is not None and self.anchor_count < 1:
            raise ValueError("anchor_count must be >= 1 when set")
        if self.low_rank_query_dim is not None and (
            self.low_rank_query_dim < 1 or self.low_rank_query_dim > self.hidden_dim
        ):
            raise ValueError("low_rank_query_dim must lie in [1, hidden_dim]")
        if self.decode_chunk_size is not None and self.decode_chunk_size < 1:
            raise ValueError("decode_chunk_size must be >= 1 when set")


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

    def __init__(
        self,
        hidden_dim: int,
        slice_count: int,
        num_heads: int,
        *,
        adaptive_temperature: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slice_count = slice_count
        self.adaptive_temperature = adaptive_temperature
        self.slice_weights = nn.Linear(hidden_dim, slice_count)
        self.temp_proj: nn.Linear | None = (
            nn.Linear(hidden_dim, 1) if adaptive_temperature else None
        )
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
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project ``(B, N, D)`` onto ``S`` slices, attend in ``O(S^2)``, and deslice.

        Supports optional ``token_mask`` of shape ``(B, N)`` (1.0 valid, 0.0 padded)
        for batching heterogeneous 3D meshes with zero padding contamination.

        Returns ``(updated_tokens (B, N, D), slice_tokens (B, S, D))``.
        """
        logits = self.slice_weights(tokens)
        if self.temp_proj is not None:
            # Transolver++ locally adaptive positive temperature preventing slice collapse
            tau = torch.nn.functional.softplus(self.temp_proj(tokens)) + 0.25
            logits = logits / tau
        pi = torch.softmax(logits, dim=-1)
        if token_mask is not None:
            mask_f = token_mask.to(dtype=tokens.dtype).unsqueeze(-1)
            pi = pi * mask_f
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
        out_tokens = tokens + desliced
        if token_mask is not None:
            out_tokens = out_tokens * token_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return out_tokens, slices


def _fourier_encode_3d(coords: torch.Tensor, frequencies: int) -> torch.Tensor:
    """Deterministic multi-frequency sinusoidal encoding of normalized 3D coords."""
    parts: list[torch.Tensor] = [coords]
    for k in range(frequencies):
        scale = (2.0**k) * math.pi
        parts.append(torch.sin(scale * coords))
        parts.append(torch.cos(scale * coords))
    return torch.cat(parts, dim=-1)


class _DecoderCrossBlock(nn.Module):
    """Double-backward-safe residual slice cross-attention + FFN decoder block."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.cross = _ExplicitMultiHeadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, q_tok: torch.Tensor, slice_tokens: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.norm1(
            q_tok + self.cross(q_tok, slice_tokens, slice_tokens)
        )
        out = self.norm2(out + self.ffn(out))
        return out


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
                PhysicsSliceAttention(
                    d,
                    config.slice_count,
                    config.num_heads,
                    adaptive_temperature=config.adaptive_temperature,
                )
                for _ in range(config.num_blocks)
            ]
        )
        pos_dim = 3 * (1 + 2 * config.positional_frequencies)
        self.param_proj = nn.Sequential(
            nn.Linear(config.parameter_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        if config.low_rank_query_dim is not None:
            r_dim = config.low_rank_query_dim
            self.query_in: nn.Sequential = nn.Sequential(
                nn.Linear(pos_dim + d, r_dim),
                nn.GELU(),
                nn.Linear(r_dim, d),
            )
        else:
            self.query_in = nn.Sequential(
                nn.Linear(pos_dim + d, d),
                nn.GELU(),
                nn.Linear(d, d),
            )
        self.query_cross_attn = _ExplicitMultiHeadAttention(d, config.num_heads)
        self.query_norm = nn.LayerNorm(d)
        self.extra_decoder_blocks = nn.ModuleList(
            [
                _DecoderCrossBlock(d, config.num_heads)
                for _ in range(config.decoder_blocks - 1)
            ]
        )
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
        surface_mask: torch.Tensor | None = None,
    ) -> Surface3DEncoding:
        """Encode 3D surface triangles ``(B, M, ...)`` into ``S`` physics slices.

        When ``surface_mask`` (shape ``(B, M)``, 1 valid / 0 padded) is provided,
        bounding-box normalization and slice aggregation strictly ignore padded
        entries so heterogeneous 3D meshes batch with zero cross-contamination.
        """
        if surface_mask is not None:
            valid = surface_mask.to(dtype=torch.bool).unsqueeze(-1)
            inf_pos = torch.full_like(centroids, float("inf"))
            inf_neg = torch.full_like(centroids, float("-inf"))
            origin = torch.where(valid, centroids, inf_pos).amin(dim=1)
            upper = torch.where(valid, centroids, inf_neg).amax(dim=1)
            extent = (upper - origin).clamp_min(1.0e-6)
            norm_c = (centroids - origin.unsqueeze(1)) / extent.unsqueeze(1)
            mask_f = surface_mask.to(dtype=centroids.dtype).unsqueeze(-1)
            norm_c = norm_c * mask_f
            normals_eff = normals * mask_f
            features_eff = features * mask_f
            weights_eff = area_weights * surface_mask.to(dtype=area_weights.dtype)
        else:
            origin = centroids.amin(dim=1)
            extent = (centroids.amax(dim=1) - origin).clamp_min(1.0e-6)
            norm_c = (centroids - origin.unsqueeze(1)) / extent.unsqueeze(1)
            normals_eff = normals
            features_eff = features
            weights_eff = area_weights
        x = torch.cat((norm_c, normals_eff, features_eff), dim=-1)
        # V6-R5 AB-UPT deterministic anchor-token subsampling for huge 3D surfaces
        if (
            self.config.anchor_count is not None
            and x.shape[1] > self.config.anchor_count
        ):
            anchor_idx = torch.linspace(
                0,
                x.shape[1] - 1,
                steps=self.config.anchor_count,
                device=x.device,
            ).long()
            x = x[:, anchor_idx, :]
            weights_eff = weights_eff[:, anchor_idx]
            if surface_mask is not None:
                surface_mask = surface_mask[:, anchor_idx]
        h = self.surface_in(x)
        if surface_mask is not None:
            h = h * surface_mask.to(dtype=h.dtype).unsqueeze(-1)
        slices = h[:, : self.config.slice_count, :]
        for block in self.encoder_blocks:
            h, slices = block(
                h,
                measure_weights=weights_eff,
                token_mask=surface_mask,
            )
        return Surface3DEncoding(slice_tokens=slices, origin=origin, extent=extent)

    def _decode_chunk(
        self,
        *,
        encoding: Surface3DEncoding,
        query_xyz: torch.Tensor,
        p_emb_single: torch.Tensor,
    ) -> torch.Tensor:
        norm_q = (query_xyz - encoding.origin.unsqueeze(1)) / encoding.extent.unsqueeze(
            1
        )
        pos = _fourier_encode_3d(norm_q, self.config.positional_frequencies)
        p_emb = p_emb_single.expand(-1, query_xyz.shape[1], -1)
        q_tok = self.query_in(torch.cat((pos, p_emb), dim=-1))
        cross = self.query_cross_attn(
            q_tok, encoding.slice_tokens, encoding.slice_tokens
        )
        q_tok = self.query_norm(q_tok + cross)
        for dec_block in self.extra_decoder_blocks:
            q_tok = dec_block(q_tok, encoding.slice_tokens)
        raw_u: torch.Tensor = self.decoder_head(q_tok)
        return raw_u

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
        p_emb_single = self.param_proj(parameters).unsqueeze(1)
        chunk = self.config.decode_chunk_size
        if chunk is not None and query_xyz.shape[1] > chunk:
            parts: list[torch.Tensor] = []
            for start in range(0, query_xyz.shape[1], chunk):
                end = min(start + chunk, query_xyz.shape[1])
                parts.append(
                    self._decode_chunk(
                        encoding=encoding,
                        query_xyz=query_xyz[:, start:end, :],
                        p_emb_single=p_emb_single,
                    )
                )
            raw_u = torch.cat(parts, dim=1)
        else:
            raw_u = self._decode_chunk(
                encoding=encoding,
                query_xyz=query_xyz,
                p_emb_single=p_emb_single,
            )
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
