"""Vector-BC-Conditioned Coupled Multiphysics 3D Neural Operator & Guarded Rollout (V5-5).

Solves both the AeroZoo boundary-condition shift failure mode and full-field
thermal-mechanical coupling in linear ``O((N_s + N_q) * S)`` complexity:

* **32-Channel Continuous Vector-BC Surface Tokens:** encodes per-facet spatial
  coordinates, normals, area fractions, continuous mechanical Dirichlet/traction
  vectors, thermal Dirichlet/flux/convective ``(h, T_amb)`` maps, multi-material
  ``(k, E, alpha)`` descriptors, and local volumetric heat-source density.
* **Interior Volumetric Power-Source Slice Tokens:** cross-attends 3D chiplet /
  Joule volumetric heat-source locations directly alongside surface BC slices.
* **Joint Coupled Multi-Field Decoder:** predicts full 3D temperature ``T(x)``
  and displacement ``u(x)`` in a single forward pass.
* **Guarded 3D Transient Stepper (:class:`GuardedTransient3DStepper`):** tracks
  discrete PDE thermal residual at every rollout step and falls back to exact
  3D FEM automatically whenever drift exceeds threshold (zero unbounded drift).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
from torch import nn

from piginot.geometry.mesh3d import CanonicalMesh3D
from piginot.operators.surface_operator_3d import (
    PhysicsSliceAttention,
    _DecoderCrossBlock,
    _fourier_encode_3d,
)

FloatArray = NDArray[np.float64]

MULTIPHYSICS_3D_ARCHITECTURE_ID: Final = "formamech-multiphysics-vector-bc-3d-v1"
MULTIPHYSICS_SURFACE_FEATURE_DIM: Final = 32
MULTIPHYSICS_SOURCE_FEATURE_DIM: Final = 8


@dataclass(frozen=True, slots=True)
class VectorBcSurfaceSpec3D:
    """Continuous spatially varying mechanical + thermal boundary specification on a 3D mesh."""

    dirichlet_face_masks: Mapping[
        str, tuple[float, float, float]
    ]  # tag -> (mx, my, mz)
    dirichlet_face_values: Mapping[
        str, tuple[float, float, float]
    ]  # tag -> (ux, uy, uz)
    traction_face_vectors: Mapping[
        str, tuple[float, float, float]
    ]  # tag -> (tx, ty, tz)
    temp_dirichlet_faces: Mapping[str, float]  # tag -> T_0 (K)
    heat_flux_faces: Mapping[str, float]  # tag -> q_n (W/mm^2)
    convection_faces: Mapping[str, tuple[float, float]]  # tag -> (h, T_amb)
    material_k_e_alpha: tuple[float, float, float] = (0.15, 1000.0, 1.2e-5)
    volumetric_power_points_xyz: FloatArray | None = None  # (N_q, 3)
    volumetric_power_values: FloatArray | None = None  # (N_q,) W/mm^3


def build_vector_bc_surface_tokens_3d(
    mesh: CanonicalMesh3D,
    bc_spec: VectorBcSurfaceSpec3D,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build 32-channel continuous vector-BC surface tokens + 8-channel power-source tokens.

    Returns ``(centroids, normals, areas, surface_features (1, M, 32), source_tokens (1, Nq, 8))``.
    """
    centroids, normals, areas = mesh.surface_geometry()
    c_np = np.asarray(centroids, dtype=np.float64)
    n_np = np.asarray(normals, dtype=np.float64)
    a_np = np.asarray(areas, dtype=np.float64)
    m = int(c_np.shape[0])

    lo = np.min(c_np, axis=0)
    hi = np.max(c_np, axis=0)
    span = np.maximum(hi - lo, 1.0e-6)
    c_norm = 2.0 * (c_np - lo[None, :]) / span[None, :] - 1.0
    total_area = max(float(np.sum(a_np)), 1.0e-12)
    area_frac = a_np / total_area

    feats = np.zeros((m, MULTIPHYSICS_SURFACE_FEATURE_DIM), dtype=np.float64)
    feats[:, 0:3] = c_norm
    feats[:, 3:6] = n_np
    feats[:, 6] = np.log1p(area_frac * float(m))

    # Mechanical Dirichlet masks & values
    for tag, mask_xyz in bc_spec.dirichlet_face_masks.items():
        if tag in mesh.semantic_face_tags:
            tids = list(mesh.semantic_face_tags[tag])
            feats[tids, 7:10] = np.asarray(mask_xyz, dtype=np.float64)
    for tag, val_xyz in bc_spec.dirichlet_face_values.items():
        if tag in mesh.semantic_face_tags:
            tids = list(mesh.semantic_face_tags[tag])
            feats[tids, 10:13] = np.asarray(val_xyz, dtype=np.float64)
    # Continuous traction vectors
    for tag, trac_xyz in bc_spec.traction_face_vectors.items():
        if tag in mesh.semantic_face_tags:
            tids = list(mesh.semantic_face_tags[tag])
            feats[tids, 13:16] = np.asarray(trac_xyz, dtype=np.float64)

    # Thermal Dirichlet, heat flux, convection
    for tag, t_val in bc_spec.temp_dirichlet_faces.items():
        if tag in mesh.semantic_face_tags:
            tids = list(mesh.semantic_face_tags[tag])
            feats[tids, 16] = 1.0
            feats[tids, 17] = (float(t_val) - 293.15) / 100.0
    for tag, q_val in bc_spec.heat_flux_faces.items():
        if tag in mesh.semantic_face_tags:
            tids = list(mesh.semantic_face_tags[tag])
            feats[tids, 18] = float(q_val)
    for tag, (h_val, t_amb) in bc_spec.convection_faces.items():
        if tag in mesh.semantic_face_tags:
            tids = list(mesh.semantic_face_tags[tag])
            feats[tids, 19] = math.log1p(max(0.0, float(h_val) * 1.0e3))
            feats[tids, 20] = (float(t_amb) - 293.15) / 100.0

    # Multi-material property channels
    k_val, e_val, a_val = bc_spec.material_k_e_alpha
    feats[:, 21] = math.log1p(max(0.0, float(k_val) * 100.0))
    feats[:, 22] = math.log1p(max(0.0, float(e_val))) / 10.0
    feats[:, 23] = float(a_val) * 1.0e5

    # Geometric lever-arm channels from clamped centroid to each facet
    clamped_mask = np.sum(feats[:, 7:10], axis=1) > 0.0
    clamp_cent = (
        np.mean(c_norm[clamped_mask], axis=0)
        if np.any(clamped_mask)
        else np.zeros(3, dtype=np.float64)
    )
    feats[:, 24:27] = c_norm - clamp_cent[None, :]
    feats[:, 27] = np.linalg.norm(feats[:, 24:27], axis=1)

    # Volumetric heat source tokens + nearest-source summary channel
    if (
        bc_spec.volumetric_power_points_xyz is not None
        and bc_spec.volumetric_power_values is not None
        and len(bc_spec.volumetric_power_points_xyz) > 0
    ):
        q_pts = np.asarray(bc_spec.volumetric_power_points_xyz, dtype=np.float64)
        q_vals = np.asarray(bc_spec.volumetric_power_values, dtype=np.float64).reshape(
            -1
        )
        q_norm = 2.0 * (q_pts - lo[None, :]) / span[None, :] - 1.0
        src_tok = np.zeros(
            (len(q_pts), MULTIPHYSICS_SOURCE_FEATURE_DIM), dtype=np.float64
        )
        src_tok[:, 0:3] = q_norm
        src_tok[:, 3] = q_vals
        src_tok[:, 4] = feats[0, 21]
        src_tok[:, 5] = feats[0, 22]
        src_tok[:, 6] = feats[0, 23]
        src_tok[:, 7] = 1.0  # active source indicator
        # Project power-weighted centroid onto surface channels 28:32
        w_q = np.abs(q_vals) + 1.0e-12
        q_cent = np.sum(q_norm * w_q[:, None], axis=0) / float(np.sum(w_q))
        feats[:, 28] = float(np.mean(q_vals))
        feats[:, 29:32] = c_norm - q_cent[None, :]
    else:
        src_tok = np.zeros((1, MULTIPHYSICS_SOURCE_FEATURE_DIM), dtype=np.float64)

    return (
        torch.as_tensor(c_np, dtype=dtype, device=device).unsqueeze(0),
        torch.as_tensor(n_np, dtype=dtype, device=device).unsqueeze(0),
        torch.as_tensor(a_np, dtype=dtype, device=device).unsqueeze(0),
        torch.as_tensor(feats, dtype=dtype, device=device).unsqueeze(0),
        torch.as_tensor(src_tok, dtype=dtype, device=device).unsqueeze(0),
    )


@dataclass(frozen=True, slots=True)
class Multiphysics3DConfig:
    """Architecture hyperparameters for :class:`Multiphysics3DOperator`."""

    parameter_dim: int = 2
    surface_feature_dim: int = MULTIPHYSICS_SURFACE_FEATURE_DIM
    source_feature_dim: int = MULTIPHYSICS_SOURCE_FEATURE_DIM
    hidden_dim: int = 32
    slice_count: int = 8
    num_blocks: int = 2
    num_heads: int = 4
    positional_frequencies: int = 3
    decoder_blocks: int = 1


@dataclass(frozen=True, slots=True)
class Multiphysics3DEncoding:
    """Joint surface-BC + interior power-source compressed slice tokens."""

    joint_slices: torch.Tensor  # (B, S_surf + S_src + 2, D)
    bc_context: torch.Tensor  # (B, D) sharp active mechanical + thermal BC summary
    origin: torch.Tensor  # (B, 3)
    extent: torch.Tensor  # (B, 3)


class Multiphysics3DOperator(nn.Module):
    """Vector-BC-conditioned linear-complexity 3D neural operator predicting ``(T, u)``."""

    def __init__(self, config: Multiphysics3DConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.surface_in = nn.Sequential(
            nn.Linear(config.surface_feature_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.source_in = nn.Sequential(
            nn.Linear(config.source_feature_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.surface_blocks = nn.ModuleList(
            [
                PhysicsSliceAttention(d, config.slice_count, config.num_heads)
                for _ in range(config.num_blocks)
            ]
        )
        self.source_slice = PhysicsSliceAttention(
            d, max(2, config.slice_count // 2), config.num_heads
        )
        # Dedicated active-BC summary projectors preventing dilution of localized
        # traction pads / thermal ports across large unloaded 3D CAD surfaces
        self.active_mech_bc_proj = nn.Sequential(
            nn.Linear(config.surface_feature_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.active_therm_bc_proj = nn.Sequential(
            nn.Linear(config.surface_feature_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        # Separate FiLM bilinear operator modulations for thermal & mechanical fields
        # (prevents gradient-scale interference between K temperature and mm displacement)
        self.bc_film_temp = nn.Linear(d, 2 * d)
        self.bc_film_disp = nn.Linear(d, 2 * d)
        coord_dim = 3 * (1 + 2 * config.positional_frequencies)
        self.query_in = nn.Sequential(
            nn.Linear(coord_dim + config.parameter_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.decoder_blocks = nn.ModuleList(
            [
                _DecoderCrossBlock(d, config.num_heads)
                for _ in range(config.decoder_blocks)
            ]
        )
        # Joint coupled heads: delta-temperature (1, scaled) + 3D displacement (3)
        self.temp_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.disp_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))

    def encode(
        self,
        centroids: torch.Tensor,
        areas: torch.Tensor,
        surface_features: torch.Tensor,
        source_tokens: torch.Tensor,
    ) -> Multiphysics3DEncoding:
        origin = centroids.min(dim=1).values
        extent = (centroids.max(dim=1).values - origin).clamp_min(1.0e-6)
        tok = self.surface_in(surface_features)
        slices = tok[:, : self.config.slice_count, :]
        for blk in self.surface_blocks:
            tok, slices = blk(tok, measure_weights=areas)
        src_h = self.source_in(source_tokens)
        _, src_slices = self.source_slice(src_h)

        # Sharp localized active-BC pooling (weights proportional to traction & thermal BC activity)
        trac_mag = torch.linalg.norm(surface_features[:, :, 13:16], dim=-1)  # (B, M)
        mech_w = (trac_mag + 1.0e-6) * areas.clamp_min(1.0e-8)
        mech_w = mech_w / mech_w.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        mech_summary = torch.einsum("bm,bmf->bf", mech_w, surface_features)
        mech_tok = self.active_mech_bc_proj(mech_summary).unsqueeze(1)  # (B, 1, D)

        therm_act = (
            surface_features[:, :, 16].abs()
            + surface_features[:, :, 18].abs()
            + surface_features[:, :, 19].abs()
        )
        therm_w = (therm_act + 1.0e-6) * areas.clamp_min(1.0e-8)
        therm_w = therm_w / therm_w.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        therm_summary = torch.einsum("bm,bmf->bf", therm_w, surface_features)
        therm_tok = self.active_therm_bc_proj(therm_summary).unsqueeze(1)  # (B, 1, D)

        joint = torch.cat((slices, src_slices, mech_tok, therm_tok), dim=1)
        bc_ctx = mech_tok.squeeze(1) + therm_tok.squeeze(1) + src_slices.mean(dim=1)
        return Multiphysics3DEncoding(
            joint_slices=joint, bc_context=bc_ctx, origin=origin, extent=extent
        )

    def predict_fields(
        self,
        encoding: Multiphysics3DEncoding,
        query_xyz: torch.Tensor,
        parameters: torch.Tensor,
        *,
        reference_temp_k: float = 293.15,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict coupled ``(temperature_k (B, N), displacement_mm (B, N, 3))``."""
        norm_xyz = (
            2.0
            * (query_xyz - encoding.origin.unsqueeze(1))
            / encoding.extent.unsqueeze(1)
            - 1.0
        )
        q_enc = _fourier_encode_3d(norm_xyz, self.config.positional_frequencies)
        p_exp = parameters.unsqueeze(1).expand(-1, query_xyz.shape[1], -1)
        q_base = self.query_in(torch.cat((q_enc, p_exp), dim=-1))
        for blk in self.decoder_blocks:
            q_base = blk(q_base, encoding.joint_slices)
        g_t, b_t = self.bc_film_temp(encoding.bc_context).chunk(2, dim=-1)
        g_u, b_u = self.bc_film_disp(encoding.bc_context).chunk(2, dim=-1)
        q_t = q_base * (1.0 + g_t.unsqueeze(1)) + b_t.unsqueeze(1)
        q_u = q_base * (1.0 + g_u.unsqueeze(1)) + b_u.unsqueeze(1)
        delta_t: torch.Tensor = self.temp_head(q_t).squeeze(-1) * 10.0
        disp: torch.Tensor = self.disp_head(q_u)
        return reference_temp_k + delta_t, disp


@dataclass(frozen=True, slots=True)
class GuardedTransientStepRecord:
    """Per-step residual audit for guarded 3D transient multiphysics rollout."""

    step_index: int
    relative_pde_residual: float
    used_fem_fallback: bool
    max_temperature_k: float


@dataclass(frozen=True, slots=True)
class GuardedTransientRolloutResult:
    """Full transient rollout trajectory with zero-unbounded-drift FEM fallback log."""

    temperature_trajectory_k: tuple[FloatArray, ...]
    step_records: tuple[GuardedTransientStepRecord, ...]
    fem_fallback_count: int
    max_recorded_residual: float


class GuardedTransient3DStepper:
    """Autoregressive 3D thermal stepper with PDE residual guard & automatic FEM fallback."""

    def __init__(
        self,
        *,
        neural_step_fn: Callable[[FloatArray, int], FloatArray],
        exact_fem_step_fn: Callable[[FloatArray, int], FloatArray],
        lhs_matrix: csr_matrix,
        rhs_matrix: csr_matrix,
        rhs_load: FloatArray,
        free_dofs: NDArray[np.int64],
        max_relative_residual: float = 0.05,
    ) -> None:
        self._neural_step = neural_step_fn
        self._exact_fem_step = exact_fem_step_fn
        self._lhs = lhs_matrix
        self._rhs_mat = rhs_matrix
        self._rhs_load = rhs_load
        self._free = free_dofs
        self._max_res = float(max_relative_residual)

    def _step_residual(self, t_prev: FloatArray, t_pred: FloatArray) -> float:
        res_full = self._lhs @ t_pred - (self._rhs_mat @ t_prev + self._rhs_load)
        rhs_full = self._rhs_mat @ t_prev + self._rhs_load
        denom = max(float(np.linalg.norm(rhs_full[self._free])), 1.0e-12)
        return float(np.linalg.norm(res_full[self._free])) / denom

    def rollout(
        self, initial_temperature_k: FloatArray, *, num_steps: int
    ) -> GuardedTransientRolloutResult:
        """Roll out ``num_steps`` forward, triggering exact FEM whenever drift exceeds threshold."""
        t_curr = np.asarray(initial_temperature_k, dtype=np.float64).copy()
        traj: list[FloatArray] = [t_curr.copy()]
        records: list[GuardedTransientStepRecord] = []
        fallbacks = 0
        max_res = 0.0

        for step in range(1, num_steps + 1):
            t_cand = np.asarray(self._neural_step(t_curr, step), dtype=np.float64)
            res_cand = (
                self._step_residual(t_curr, t_cand)
                if np.all(np.isfinite(t_cand))
                else float("inf")
            )
            if not math.isfinite(res_cand) or res_cand > self._max_res:
                t_next = np.asarray(
                    self._exact_fem_step(t_curr, step), dtype=np.float64
                )
                res_final = self._step_residual(t_curr, t_next)
                used_fb = True
                fallbacks += 1
            else:
                t_next = t_cand
                res_final = res_cand
                used_fb = False
            max_res = max(max_res, res_final)
            records.append(
                GuardedTransientStepRecord(
                    step_index=step,
                    relative_pde_residual=res_final,
                    used_fem_fallback=used_fb,
                    max_temperature_k=float(np.max(t_next)),
                )
            )
            t_curr = t_next
            traj.append(t_curr.copy())

        return GuardedTransientRolloutResult(
            temperature_trajectory_k=tuple(traj),
            step_records=tuple(records),
            fem_fallback_count=fallbacks,
            max_recorded_residual=max_res,
        )


__all__ = (
    "MULTIPHYSICS_3D_ARCHITECTURE_ID",
    "MULTIPHYSICS_SOURCE_FEATURE_DIM",
    "MULTIPHYSICS_SURFACE_FEATURE_DIM",
    "GuardedTransient3DStepper",
    "GuardedTransientRolloutResult",
    "GuardedTransientStepRecord",
    "Multiphysics3DConfig",
    "Multiphysics3DEncoding",
    "Multiphysics3DOperator",
    "VectorBcSurfaceSpec3D",
    "build_vector_bc_surface_tokens_3d",
)
