"""3D Dirichlet lifting, volumetric Deep Ritz / multi-fidelity Surface3DEnergyOperator training, serving, and qualification (M5 / F1).

Connects :class:`~piginot.geometry.mesh3d.CanonicalMesh3D`,
:class:`~piginot.operators.surface_operator_3d.Surface3DEnergyOperator`,
:mod:`piginot.physics` exact 3x3 finite-strain kernels, and
:mod:`piginot.fem.backend_3d` (``TorchFemBackend3D`` / ``ScikitFemBackend3D``)
into an end-to-end 3D solid mechanics neural-operator pipeline with:

* Exact 3D essential boundary lifting (:class:`PlanarClampLifting3D` and
  :class:`SoftMinFaceLifting3D`),
* Volumetric 3D Deep Ritz potential energy training (label-free or
  explicitly disclosed ``multi-fidelity-label-efficient``),
* Tamper-checked ``safetensors`` 3D bundle persistence
  (``formamech.surface-3d-bundle-v1``),
* Fast, certified, and warm-started exact 3D ``TET10`` serving,
* Held-out unseen 3D load qualification (capped at ``HOLDOUT_VERIFIED``) and
  honest break-even query-count ``N*`` reporting.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from piginot import __version__
from piginot.contracts import MaterialSpec, Quantity, Unit, canonical_json_bytes
from piginot.fem.backend_3d import (
    DirichletBC3D,
    FemRequest3D,
    FemResult3D,
    TorchFemBackend3D,
    TractionBC3D,
    _cauchy_von_mises_3d,
    _project_cell_scalar_to_nodes_3d,
    _tet_quadrature,
    assemble_surface_tractions_3d,
    face_node_indices_3d,
)
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D
from piginot.operators.surface_operator_3d import (
    SURFACE_3D_ARCHITECTURE_ID,
    Surface3DConfig,
    Surface3DEncoding,
    Surface3DEnergyOperator,
)
from piginot.physics import energy_density, first_piola, reference_modulus

FloatArray = NDArray[np.float64]
SURFACE_3D_BUNDLE_SCHEMA: Final = "formamech.surface-3d-bundle-v1"
SURFACE_3D_QUALIFICATION_SCHEMA: Final = "formamech.surface-3d-qualification-v1"
PredictMode3D = Literal["fast", "certified", "exact"]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_doc(document: object) -> str:
    return _sha256_bytes(canonical_json_bytes(document))


# ---------------------------------------------------------------------------
# 3D Exact Dirichlet Lifting
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlanarClampLifting3D:
    """Smooth exact 3D zero-clamp lifting on a planar boundary ``x_axis == plane_coord``."""

    axis: Literal["x", "y", "z"]
    plane_coord: float
    extent_mm: float
    displacement_scale_mm: float = 1.0

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y", "z"):
            raise ValueError("axis must be 'x', 'y', or 'z'")
        if not math.isfinite(self.extent_mm) or self.extent_mm <= 0.0:
            raise ValueError("extent_mm must be finite and positive")
        if (
            not math.isfinite(self.displacement_scale_mm)
            or self.displacement_scale_mm <= 0.0
        ):
            raise ValueError("displacement_scale_mm must be finite and positive")

    def scale_and_offset(
        self, query_xyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        axis_idx = {"x": 0, "y": 1, "z": 2}[self.axis]
        coord = query_xyz[..., axis_idx]
        s = self.displacement_scale_mm * ((coord - self.plane_coord) / self.extent_mm)
        scale = s.unsqueeze(-1).expand_as(query_xyz)
        offset = torch.zeros_like(query_xyz)
        return scale, offset

    def document(self) -> dict[str, Any]:
        return {
            "kind": "planar-clamp-3d",
            "axis": self.axis,
            "plane_coord": self.plane_coord,
            "extent_mm": self.extent_mm,
            "displacement_scale_mm": self.displacement_scale_mm,
        }


@dataclass(frozen=True, slots=True)
class SoftMinFaceLifting3D:
    """General 3D zero-clamp lifting vanishing on arbitrary clamped anchor nodes."""

    anchor_points_mm: tuple[tuple[float, float, float], ...]
    length_scale_mm: float
    displacement_scale_mm: float = 1.0

    def __post_init__(self) -> None:
        if not self.anchor_points_mm:
            raise ValueError("anchor_points_mm must be non-empty")
        if not math.isfinite(self.length_scale_mm) or self.length_scale_mm <= 0.0:
            raise ValueError("length_scale_mm must be finite and positive")

    def scale_and_offset(
        self, query_xyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchors = query_xyz.new_tensor(self.anchor_points_mm)  # (A, 3)
        diff = query_xyz.unsqueeze(-2) - anchors.view(1, 1, -1, 3)  # (B, Q, A, 3)
        dist_sq = (diff * diff).sum(dim=-1)  # (B, Q, A)
        min_sq = dist_sq.amin(dim=-1)  # (B, Q)
        eps = 1.0e-10
        dist = torch.sqrt(min_sq + eps) - math.sqrt(eps)
        s = self.displacement_scale_mm * (dist / (dist + self.length_scale_mm))
        scale = s.unsqueeze(-1).expand_as(query_xyz)
        offset = torch.zeros_like(query_xyz)
        return scale, offset

    def document(self) -> dict[str, Any]:
        return {
            "kind": "softmin-face-3d",
            "anchor_points_mm": [list(p) for p in self.anchor_points_mm],
            "length_scale_mm": self.length_scale_mm,
            "displacement_scale_mm": self.displacement_scale_mm,
        }


Lifting3D = PlanarClampLifting3D | SoftMinFaceLifting3D


def lifting_from_document(doc: Mapping[str, Any]) -> Lifting3D:
    kind = str(doc["kind"])
    if kind == "planar-clamp-3d":
        return PlanarClampLifting3D(
            axis=doc["axis"],
            plane_coord=float(doc["plane_coord"]),
            extent_mm=float(doc["extent_mm"]),
            displacement_scale_mm=float(doc.get("displacement_scale_mm", 1.0)),
        )
    if kind == "softmin-face-3d":
        return SoftMinFaceLifting3D(
            anchor_points_mm=tuple(
                (float(p[0]), float(p[1]), float(p[2])) for p in doc["anchor_points_mm"]
            ),
            length_scale_mm=float(doc["length_scale_mm"]),
            displacement_scale_mm=float(doc.get("displacement_scale_mm", 1.0)),
        )
    raise ValueError(f"unsupported 3D lifting kind {kind!r}")


def lifted_displacement_and_gradient(
    operator: Surface3DEnergyOperator,
    *,
    encoding: Surface3DEncoding,
    query_xyz: torch.Tensor,
    parameters: torch.Tensor,
    lifting: Lifting3D,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact ``(u, grad_u)`` differentiating through BOTH operator and 3D lifting."""
    pts = query_xyz.detach().requires_grad_(True)
    scale, offset = lifting.scale_and_offset(pts)
    u = operator.displacement(
        encoding=encoding,
        query_xyz=pts,
        parameters=parameters,
        lifting_offset=offset,
        lifting_scale=scale,
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


# ---------------------------------------------------------------------------
# 3D Surface Tokenization & Volumetric Quadrature
# ---------------------------------------------------------------------------


def build_surface_3d_tensors(
    mesh: CanonicalMesh3D,
    *,
    clamped_face_tags: Sequence[str],
    loaded_face_tags: Sequence[str],
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract ``(centroids, normals, areas, features)`` tensors ``(1, M, ...)``."""
    centroids_t, normals_t, areas_t = mesh.surface_geometry()
    centroids = torch.tensor(centroids_t, dtype=dtype, device=device)
    normals = torch.tensor(normals_t, dtype=dtype, device=device)
    areas = torch.tensor(areas_t, dtype=dtype, device=device)
    m_tris = centroids.shape[0]

    clamp_tri_set: set[int] = set()
    for tag in clamped_face_tags:
        clamp_tri_set.update(mesh.semantic_face_tags.get(tag, ()))
    load_tri_set: set[int] = set()
    for tag in loaded_face_tags:
        load_tri_set.update(mesh.semantic_face_tags.get(tag, ()))

    clamp_pts = centroids[sorted(clamp_tri_set)] if clamp_tri_set else centroids[:1]
    bbox_diag = float(
        torch.linalg.vector_norm(
            centroids.amax(dim=0) - centroids.amin(dim=0)
        ).clamp_min(1.0)
    )
    feats = torch.zeros((m_tris, 6), dtype=dtype, device=device)
    for idx in range(m_tris):
        is_c = 1.0 if idx in clamp_tri_set else 0.0
        is_l = 1.0 if idx in load_tri_set else 0.0
        is_f = 1.0 if (is_c == 0.0 and is_l == 0.0) else 0.0
        dist_c = float(
            torch.linalg.vector_norm(clamp_pts - centroids[idx], dim=-1).min().item()
        )
        feats[idx, 0] = is_c
        feats[idx, 1] = is_c
        feats[idx, 2] = is_c
        feats[idx, 3] = is_l
        feats[idx, 4] = is_f
        feats[idx, 5] = dist_c / bbox_diag

    return (
        centroids.unsqueeze(0),
        normals.unsqueeze(0),
        areas.unsqueeze(0),
        feats.unsqueeze(0),
    )


def _volumetric_quadrature_3d(
    mesh: CanonicalMesh3D,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute exact tetrahedral Gauss points ``(Q_tot, 3)`` and volume weights ``(Q_tot,)``."""
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    xi_pts, w_ref = _tet_quadrature(mesh.cell_type)
    # Corner vertices determine straight-sided affine map for quadrature points
    v0 = nodes[cells[:, 0]]
    v1 = nodes[cells[:, 1]]
    v2 = nodes[cells[:, 2]]
    v3 = nodes[cells[:, 3]]
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0
    det_j = np.einsum("ei,ei->e", np.cross(e1, e2), e3)  # 6 * volume
    q_pts_list: list[FloatArray] = []
    q_wts_list: list[FloatArray] = []
    for k in range(xi_pts.shape[0]):
        xi, eta, zeta = float(xi_pts[k, 0]), float(xi_pts[k, 1]), float(xi_pts[k, 2])
        pts_k = v0 + xi * e1 + eta * e2 + zeta * e3
        wts_k = det_j * float(w_ref[k])
        q_pts_list.append(pts_k)
        q_wts_list.append(wts_k)
    q_pts_np = np.concatenate(q_pts_list, axis=0)
    q_wts_np = np.concatenate(q_wts_list, axis=0)
    total_vol = float(np.sum(q_wts_np))
    return (
        torch.tensor(q_pts_np, dtype=dtype, device=device),
        torch.tensor(q_wts_np, dtype=dtype, device=device),
        total_vol,
    )


# ---------------------------------------------------------------------------
# Training Case, Config, Bundle & Trainer
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadCase3D:
    """One 3D load/parameter training or evaluation case on a fixed 3D mesh."""

    case_id: str
    parameter_vector: tuple[float, ...]
    traction_bcs: tuple[TractionBC3D, ...]
    fem_result: FemResult3D | None = None


@dataclass(frozen=True, slots=True)
class Surface3DTrainingConfig:
    """Hyperparameters for volumetric Deep Ritz / multi-fidelity 3D operator training."""

    epochs: int = 120
    learning_rate: float = 3.0e-3
    barrier_weight: float = 500.0
    barrier_threshold: float = 0.2
    supervision_weight: float = 25.0
    quadrature_subsample: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not math.isfinite(self.barrier_weight) or self.barrier_weight <= 0.0:
            raise ValueError("barrier_weight must be positive")
        if not math.isfinite(self.supervision_weight) or self.supervision_weight < 0.0:
            raise ValueError("supervision_weight must be nonnegative")


@dataclass(frozen=True, slots=True)
class Surface3DTrainingSummary:
    """Auditable training record for a 3D ``Surface3DEnergyOperator``."""

    epochs: int
    n_train_cases: int
    n_parameters: int
    initial_loss: float
    final_loss: float
    loss_history: tuple[float, ...]
    min_det_f_seen: float
    train_wall_time_s: float
    training_mode: str
    claim_level: str = "NOT_VERIFIED"

    def document(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "n_train_cases": self.n_train_cases,
            "n_parameters": self.n_parameters,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "loss_history": list(self.loss_history),
            "min_det_f_seen": self.min_det_f_seen,
            "train_wall_time_s": self.train_wall_time_s,
            "training_mode": self.training_mode,
            "claim_level": self.claim_level,
        }


@dataclass(frozen=True, slots=True)
class Surface3DBundle:
    """In-memory handle for a trained 3D ``Surface3DEnergyOperator`` bundle."""

    mesh: CanonicalMesh3D
    material: MaterialSpec
    clamped_face_tags: tuple[str, ...]
    loaded_face_tags: tuple[str, ...]
    lifting: Lifting3D
    operator: Surface3DEnergyOperator
    training_summary: Surface3DTrainingSummary
    claim_level: str = "NOT_VERIFIED"

    @property
    def bundle_hash(self) -> str:
        return _sha256_doc(
            {
                "schema": SURFACE_3D_BUNDLE_SCHEMA,
                "architecture_id": self.operator.architecture_id,
                "mesh_id": self.mesh.mesh_id,
                "material": self.material.name,
                "clamped_face_tags": list(self.clamped_face_tags),
                "loaded_face_tags": list(self.loaded_face_tags),
                "lifting": self.lifting.document(),
                "config": asdict(self.operator.config),
                "training_mode": self.training_summary.training_mode,
            }
        )


def train_surface_3d_operator(
    mesh: CanonicalMesh3D,
    material: MaterialSpec,
    *,
    clamped_face_tags: Sequence[str],
    loaded_face_tags: Sequence[str],
    lifting: Lifting3D,
    train_cases: Sequence[LoadCase3D],
    operator_config: Surface3DConfig | None = None,
    training_config: Surface3DTrainingConfig | None = None,
    dtype: torch.dtype = torch.float64,
) -> Surface3DBundle:
    """Train ``Surface3DEnergyOperator`` via 3D volumetric Deep Ritz (+ optional FEM anchors)."""
    if not train_cases:
        raise ValueError("train_cases must be non-empty")
    t0 = time.perf_counter()
    nu_prop = material.properties.get("poisson_ratio")
    if nu_prop is not None:
        mesh.validate_incompressibility_locking(float(nu_prop.value))

    t_cfg = Surface3DTrainingConfig() if training_config is None else training_config
    torch.manual_seed(t_cfg.seed)

    p_dim = len(train_cases[0].parameter_vector)
    op_cfg = (
        Surface3DConfig(parameter_dim=p_dim)
        if operator_config is None
        else operator_config
    )
    if op_cfg.parameter_dim != p_dim:
        raise ValueError(
            f"operator_config.parameter_dim={op_cfg.parameter_dim} != case dim {p_dim}"
        )

    operator = Surface3DEnergyOperator(op_cfg).to(dtype=dtype)
    n_model_params = sum(p.numel() for p in operator.parameters())

    s_cent, s_norm, s_area, s_feat = build_surface_3d_tensors(
        mesh,
        clamped_face_tags=clamped_face_tags,
        loaded_face_tags=loaded_face_tags,
        dtype=dtype,
    )
    q_pts_all, q_wts_all, total_vol = _volumetric_quadrature_3d(mesh, dtype=dtype)
    nodes_t = torch.tensor(mesh.nodes, dtype=dtype).unsqueeze(0)  # (1, N, 3)
    node_scale, node_offset = lifting.scale_and_offset(nodes_t)
    e_ref = reference_modulus(material)
    energy_scale = max(e_ref * total_vol, 1.0e-12)

    # Pre-assemble batched external traction forces and optional FEM targets
    any_supervised = any(c.fem_result is not None for c in train_cases)
    n_b = len(train_cases)
    p_batch = torch.tensor(
        [list(c.parameter_vector) for c in train_cases], dtype=dtype
    )  # (B, P)
    f_ext_batch = torch.stack(
        [
            torch.tensor(
                assemble_surface_tractions_3d(mesh, c.traction_bcs), dtype=dtype
            )
            for c in train_cases
        ],
        dim=0,
    )  # (B, N, 3)
    sup_indices = [i for i, c in enumerate(train_cases) if c.fem_result is not None]
    u_sup_batch: torch.Tensor | None = None
    u_scale_batch: torch.Tensor | None = None
    if sup_indices:
        u_sup_list = [
            torch.tensor(
                train_cases[i].fem_result.nodal_displacement,  # type: ignore[union-attr]
                dtype=dtype,
            )
            for i in sup_indices
        ]
        u_sup_batch = torch.stack(u_sup_list, dim=0)  # (B_sup, N, 3)
        u_scale_batch = torch.linalg.vector_norm(
            u_sup_batch.reshape(len(sup_indices), -1), dim=-1
        ).clamp_min(1.0e-6)

    optimizer = torch.optim.Adam(operator.parameters(), lr=t_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(t_cfg.epochs, 1), eta_min=t_cfg.learning_rate * 0.05
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(t_cfg.seed)

    n_q_tot = q_pts_all.shape[0]
    eye3 = torch.eye(3, dtype=dtype)
    nodes_batch = nodes_t.expand(n_b, -1, -1)
    node_scale_batch = node_scale.expand(n_b, -1, -1)
    node_offset_batch = node_offset.expand(n_b, -1, -1)
    loss_history: list[float] = []
    min_det_seen = float("inf")

    for _ in range(t_cfg.epochs):
        optimizer.zero_grad(set_to_none=True)
        # Optional uniform volumetric quadrature subsampling for large TET10 meshes
        if (
            t_cfg.quadrature_subsample is not None
            and t_cfg.quadrature_subsample < n_q_tot
        ):
            idx = torch.randperm(n_q_tot, generator=gen)[: t_cfg.quadrature_subsample]
            q_pts_1 = q_pts_all[idx].unsqueeze(0)
            q_wts = q_wts_all[idx] * (
                float(n_q_tot) / float(t_cfg.quadrature_subsample)
            )
        else:
            q_pts_1 = q_pts_all.unsqueeze(0)
            q_wts = q_wts_all

        # Encode surface ONCE per epoch and broadcast physics slices across all B cases
        enc_1 = operator.encode_surface(s_cent, s_norm, s_area, s_feat)
        enc_b = Surface3DEncoding(
            slice_tokens=enc_1.slice_tokens.expand(n_b, -1, -1),
            origin=enc_1.origin.expand(n_b, -1),
            extent=enc_1.extent.expand(n_b, -1),
        )
        q_pts_b = q_pts_1.expand(n_b, -1, -1)
        _, grad_u = lifted_displacement_and_gradient(
            operator,
            encoding=enc_b,
            query_xyz=q_pts_b,
            parameters=p_batch,
            lifting=lifting,
            create_graph=True,
        )  # (B, Q, 3, 3)
        f_def = eye3 + grad_u
        det_f = torch.linalg.det(f_def)  # (B, Q)
        min_det_seen = min(min_det_seen, float(det_f.detach().min().item()))
        admissible = det_f > 1.0e-5
        safe_f = torch.where(admissible[..., None, None], f_def, eye3.expand_as(f_def))
        w_dens = energy_density(safe_f, material)  # (B, Q)
        w_dens = torch.where(admissible, w_dens, torch.zeros_like(w_dens))
        strain_e = torch.sum(w_dens * q_wts.unsqueeze(0), dim=-1)  # (B,)

        u_nodes = operator.displacement(
            encoding=enc_b,
            query_xyz=nodes_batch,
            parameters=p_batch,
            lifting_offset=node_offset_batch,
            lifting_scale=node_scale_batch,
        )  # (B, N, 3)
        ext_work = torch.sum(f_ext_batch * u_nodes, dim=(-2, -1))  # (B,)
        pi_norm = torch.mean((strain_e - ext_work) / energy_scale)
        barrier = torch.mean(torch.relu(t_cfg.barrier_threshold - det_f).square())
        loss = pi_norm + t_cfg.barrier_weight * barrier

        if u_sup_batch is not None and u_scale_batch is not None:
            u_pred_sup = u_nodes[sup_indices]
            diff_norms = torch.linalg.vector_norm(
                (u_pred_sup - u_sup_batch).reshape(len(sup_indices), -1), dim=-1
            )
            rel_l2 = torch.sqrt((diff_norms / u_scale_batch) ** 2 + 1.0e-8)
            sup_loss = torch.mean(rel_l2 + 0.5 * rel_l2.square())
            loss = loss + t_cfg.supervision_weight * sup_loss

        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(operator.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(float(loss.detach().item()))

    mode_str = (
        "multi-fidelity-label-efficient"
        if any_supervised
        else "variational-energy-label-free"
    )
    summary = Surface3DTrainingSummary(
        epochs=t_cfg.epochs,
        n_train_cases=len(train_cases),
        n_parameters=n_model_params,
        initial_loss=loss_history[0],
        final_loss=loss_history[-1],
        loss_history=tuple(loss_history),
        min_det_f_seen=min_det_seen,
        train_wall_time_s=time.perf_counter() - t0,
        training_mode=mode_str,
        claim_level="NOT_VERIFIED",
    )
    return Surface3DBundle(
        mesh=mesh,
        material=material,
        clamped_face_tags=tuple(clamped_face_tags),
        loaded_face_tags=tuple(loaded_face_tags),
        lifting=lifting,
        operator=operator,
        training_summary=summary,
        claim_level="NOT_VERIFIED",
    )


# ---------------------------------------------------------------------------
# Persistence (safetensors + SHA-256 manifest)
# ---------------------------------------------------------------------------


def save_surface_3d_bundle(
    bundle: Surface3DBundle, output_dir: Path | str
) -> dict[str, Any]:
    """Save a tamper-checked 3D ``Surface3DEnergyOperator`` bundle."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.detach().cpu().contiguous()
        for k, v in bundle.operator.state_dict().items()
    }
    weights_bytes = save_safetensors(state)
    (out / "weights.safetensors").write_bytes(weights_bytes)

    mesh_doc = {
        "schema": "formamech.canonical-mesh-3d-v1",
        "mesh_id": bundle.mesh.mesh_id,
        "cell_type": bundle.mesh.cell_type.value,
        "unit": bundle.mesh.unit.value,
        "nodes": [list(p) for p in bundle.mesh.nodes],
        "cells": [list(c) for c in bundle.mesh.cells],
        "surface_triangles": [list(t) for t in bundle.mesh.surface_triangles],
        "semantic_face_tags": {
            k: list(v) for k, v in bundle.mesh.semantic_face_tags.items()
        },
        "metadata": dict(bundle.mesh.metadata),
    }
    mesh_bytes = canonical_json_bytes(mesh_doc) + b"\n"
    (out / "mesh3d.json").write_bytes(mesh_bytes)

    mat_doc = {
        "name": bundle.material.name,
        "model": bundle.material.model,
        "properties": {
            k: {"value": float(v.value), "unit": v.unit.value}
            for k, v in bundle.material.properties.items()
        },
    }
    manifest: dict[str, Any] = {
        "schema": SURFACE_3D_BUNDLE_SCHEMA,
        "version": __version__,
        "architecture_id": SURFACE_3D_ARCHITECTURE_ID,
        "bundle_hash": bundle.bundle_hash,
        "claim_level": bundle.claim_level,
        "weights_file": "weights.safetensors",
        "weights_sha256": _sha256_bytes(weights_bytes),
        "mesh_file": "mesh3d.json",
        "mesh_sha256": _sha256_bytes(mesh_bytes),
        "dtype": str(next(bundle.operator.parameters()).dtype),
        "operator_config": asdict(bundle.operator.config),
        "material": mat_doc,
        "clamped_face_tags": list(bundle.clamped_face_tags),
        "loaded_face_tags": list(bundle.loaded_face_tags),
        "lifting": bundle.lifting.document(),
        "training_summary": bundle.training_summary.document(),
    }
    (out / "bundle3d.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def load_surface_3d_bundle(bundle_dir: Path | str) -> Surface3DBundle:
    """Verify SHA-256 hashes and reload a 3D ``Surface3DEnergyOperator`` bundle."""
    root = Path(bundle_dir)
    manifest_path = root / "bundle3d.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"bundle3d.json not found in {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SURFACE_3D_BUNDLE_SCHEMA:
        raise ValueError(f"invalid 3D bundle schema {manifest.get('schema')!r}")

    w_bytes = (root / str(manifest["weights_file"])).read_bytes()
    if _sha256_bytes(w_bytes) != manifest["weights_sha256"]:
        raise ValueError("3D bundle weights.safetensors SHA-256 mismatch")
    m_bytes = (root / str(manifest["mesh_file"])).read_bytes()
    if _sha256_bytes(m_bytes) != manifest["mesh_sha256"]:
        raise ValueError("3D bundle mesh3d.json SHA-256 mismatch")

    m_raw = json.loads(m_bytes.decode("utf-8"))
    mesh = CanonicalMesh3D(
        nodes=tuple((float(p[0]), float(p[1]), float(p[2])) for p in m_raw["nodes"]),
        cells=tuple(tuple(int(i) for i in c) for c in m_raw["cells"]),
        cell_type=CellType3D(m_raw["cell_type"]),
        surface_triangles=tuple(
            (int(t[0]), int(t[1]), int(t[2])) for t in m_raw["surface_triangles"]
        ),
        semantic_face_tags={
            str(k): tuple(int(i) for i in v)
            for k, v in m_raw["semantic_face_tags"].items()
        },
        unit=Unit(m_raw.get("unit", "mm")),
        metadata=dict(m_raw.get("metadata", {})),
    )

    mat_raw = manifest["material"]
    material = MaterialSpec(
        name=str(mat_raw["name"]),
        model=str(mat_raw["model"]),
        properties={
            str(k): Quantity(float(v["value"]), Unit(str(v["unit"]))).to_canonical()
            for k, v in mat_raw["properties"].items()
        },
    )
    dtype = torch.float32 if manifest.get("dtype") == "torch.float32" else torch.float64
    op_cfg = Surface3DConfig(**manifest["operator_config"])
    operator = Surface3DEnergyOperator(op_cfg).to(dtype=dtype)
    operator.load_state_dict(dict(load_safetensors(w_bytes)), strict=True)

    ts = manifest["training_summary"]
    summary = Surface3DTrainingSummary(
        epochs=int(ts["epochs"]),
        n_train_cases=int(ts["n_train_cases"]),
        n_parameters=int(ts["n_parameters"]),
        initial_loss=float(ts["initial_loss"]),
        final_loss=float(ts["final_loss"]),
        loss_history=tuple(float(v) for v in ts["loss_history"]),
        min_det_f_seen=float(ts["min_det_f_seen"]),
        train_wall_time_s=float(ts["train_wall_time_s"]),
        training_mode=str(ts["training_mode"]),
        claim_level=str(ts.get("claim_level", "NOT_VERIFIED")),
    )
    claim_level = str(manifest.get("claim_level", "NOT_VERIFIED"))
    if claim_level not in ("NOT_VERIFIED", "DEVELOPMENT_VERIFIED", "HOLDOUT_VERIFIED"):
        raise ValueError(f"unsupported claim_level {claim_level!r}")
    return Surface3DBundle(
        mesh=mesh,
        material=material,
        clamped_face_tags=tuple(str(t) for t in manifest["clamped_face_tags"]),
        loaded_face_tags=tuple(str(t) for t in manifest["loaded_face_tags"]),
        lifting=lifting_from_document(manifest["lifting"]),
        operator=operator,
        training_summary=summary,
        claim_level=claim_level,
    )


# ---------------------------------------------------------------------------
# Serving, Warm-Start Exact Mode, Qualification & Honest Break-Even N*
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Prediction3DReport:
    """3D neural-operator prediction + optional FEM certification / warm-start solve."""

    mode: PredictMode3D
    case_id: str
    nodal_displacement: FloatArray
    nodal_von_mises_mpa: FloatArray
    max_displacement_mm: float
    peak_von_mises_mpa: float
    min_det_f: float
    inference_wall_time_s: float
    fem_result: FemResult3D | None = None
    displacement_rel_l2_error: float | None = None
    peak_von_mises_rel_error: float | None = None
    max_nodal_displacement_error_mm: float | None = None

    def summary_document(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "case_id": self.case_id,
            "max_displacement_mm": self.max_displacement_mm,
            "peak_von_mises_mpa": self.peak_von_mises_mpa,
            "min_det_f": self.min_det_f,
            "inference_wall_time_s": self.inference_wall_time_s,
            "displacement_rel_l2_error": self.displacement_rel_l2_error,
            "peak_von_mises_rel_error": self.peak_von_mises_rel_error,
            "max_nodal_displacement_error_mm": self.max_nodal_displacement_error_mm,
            "fem_converged": None
            if self.fem_result is None
            else self.fem_result.converged,
            "fem_newton_iterations": None
            if self.fem_result is None
            else self.fem_result.total_newton_iterations,
            "fem_wall_time_s": None
            if self.fem_result is None
            else self.fem_result.wall_time_s,
        }


def predict_3d_bundle(
    bundle: Surface3DBundle,
    case: LoadCase3D,
    *,
    mode: PredictMode3D = "fast",
    backend: TorchFemBackend3D | None = None,
    cached_encoding: Surface3DEncoding | None = None,
) -> Prediction3DReport:
    """Predict 3D displacement & von Mises stress (``fast`` / ``certified`` / ``exact``)."""
    t0 = time.perf_counter()
    dtype = next(bundle.operator.parameters()).dtype
    mesh = bundle.mesh
    nodes_np = np.asarray(mesh.nodes, dtype=np.float64)
    cells_np = np.asarray(mesh.cells, dtype=np.int64)
    nodes_t = torch.tensor(nodes_np, dtype=dtype).unsqueeze(0)
    centroids_np = nodes_np[cells_np[:, :4]].mean(axis=1)  # (E, 3)
    centroids_t = torch.tensor(centroids_np, dtype=dtype).unsqueeze(0)

    if cached_encoding is None:
        s_c, s_n, s_a, s_f = build_surface_3d_tensors(
            mesh,
            clamped_face_tags=bundle.clamped_face_tags,
            loaded_face_tags=bundle.loaded_face_tags,
            dtype=dtype,
        )
        with torch.no_grad():
            encoding = bundle.operator.encode_surface(s_c, s_n, s_a, s_f)
    else:
        encoding = cached_encoding

    p_t = torch.tensor([list(case.parameter_vector)], dtype=dtype)
    n_scale, n_offset = bundle.lifting.scale_and_offset(nodes_t)

    with torch.no_grad():
        u_pred_t = bundle.operator.displacement(
            encoding=encoding,
            query_xyz=nodes_t,
            parameters=p_t,
            lifting_offset=n_offset,
            lifting_scale=n_scale,
        )[0]
    _, grad_c = lifted_displacement_and_gradient(
        bundle.operator,
        encoding=encoding,
        query_xyz=centroids_t,
        parameters=p_t,
        lifting=bundle.lifting,
        create_graph=False,
    )
    eye3 = torch.eye(3, dtype=dtype)
    f_c = eye3 + grad_c[0]
    det_c = torch.linalg.det(f_c)
    min_det = float(det_c.min().item())
    safe_f = torch.where((det_c > 1.0e-6)[:, None, None], f_c, eye3.expand_as(f_c))
    with torch.no_grad():
        p_c = first_piola(safe_f, bundle.material).cpu().numpy()
    f_c_np = safe_f.cpu().numpy()
    vm_cells = _cauchy_von_mises_3d(p_c, f_c_np)

    v0 = nodes_np[cells_np[:, 0]]
    e1 = nodes_np[cells_np[:, 1]] - v0
    e2 = nodes_np[cells_np[:, 2]] - v0
    e3 = nodes_np[cells_np[:, 3]] - v0
    cell_vols = np.einsum("ei,ei->e", np.cross(e1, e2), e3) / 6.0
    vm_nodes = _project_cell_scalar_to_nodes_3d(mesh, vm_cells, cell_vols)

    u_pred_np = u_pred_t.cpu().numpy()
    # Enforce exact zero on clamped face nodes for numerical cleanliness
    clamp_nids = face_node_indices_3d(mesh, bundle.clamped_face_tags)
    u_pred_np[list(clamp_nids), :] = 0.0
    inf_time = time.perf_counter() - t0

    if mode not in ("fast", "certified", "exact"):
        raise ValueError(
            f"unsupported 3D prediction mode {mode!r}; expected 'fast', 'certified', or 'exact'"
        )
    if mode == "fast":
        return Prediction3DReport(
            mode="fast",
            case_id=case.case_id,
            nodal_displacement=u_pred_np,
            nodal_von_mises_mpa=vm_nodes,
            max_displacement_mm=float(np.max(np.linalg.norm(u_pred_np, axis=1))),
            peak_von_mises_mpa=float(np.max(vm_nodes)),
            min_det_f=min_det,
            inference_wall_time_s=inf_time,
        )

    solver = TorchFemBackend3D() if backend is None else backend
    req = FemRequest3D(
        request_id=f"3d-{mode}-{case.case_id}",
        mesh=mesh,
        material=bundle.material,
        dirichlet_bcs=(
            DirichletBC3D(
                name="clamp",
                face_tags=bundle.clamped_face_tags,
                components=("x", "y", "z"),
                values_mm=(0.0, 0.0, 0.0),
            ),
        ),
        traction_bcs=case.traction_bcs,
        load_steps=3,
    )
    if mode == "exact":
        fem_res = solver.solve(req, initial_displacement=u_pred_np)
        if not fem_res.converged:
            fem_res = solver.solve(req)
    else:
        fem_res = case.fem_result if case.fem_result is not None else solver.solve(req)

    u_ref_norm = max(float(np.linalg.norm(fem_res.nodal_displacement)), 1.0e-12)
    diff_u = u_pred_np - fem_res.nodal_displacement
    rel_u = float(np.linalg.norm(diff_u)) / u_ref_norm
    max_u_err = float(np.max(np.linalg.norm(diff_u, axis=1)))
    rel_vm = abs(float(np.max(vm_nodes)) - fem_res.peak_von_mises_mpa) / max(
        fem_res.peak_von_mises_mpa, 1.0e-12
    )

    served_u = fem_res.nodal_displacement if mode == "exact" else u_pred_np
    served_vm = fem_res.nodal_von_mises_mpa if mode == "exact" else vm_nodes
    return Prediction3DReport(
        mode=mode,
        case_id=case.case_id,
        nodal_displacement=served_u,
        nodal_von_mises_mpa=served_vm,
        max_displacement_mm=float(np.max(np.linalg.norm(served_u, axis=1))),
        peak_von_mises_mpa=float(np.max(served_vm)),
        min_det_f=fem_res.min_det_f if mode == "exact" else min_det,
        inference_wall_time_s=inf_time,
        fem_result=fem_res,
        displacement_rel_l2_error=rel_u,
        peak_von_mises_rel_error=rel_vm,
        max_nodal_displacement_error_mm=max_u_err,
    )


def compute_3d_break_even(
    *,
    train_wall_time_s: float,
    cold_fem_s_per_query: float,
    fast_surrogate_s_per_query: float,
    warm_fem_s_per_query: float | None = None,
) -> dict[str, float | None]:
    """Compute honest amortized break-even query counts ``N*`` vs. cold 3D FEM."""
    saving_fast = cold_fem_s_per_query - fast_surrogate_s_per_query
    n_star_fast = (
        math.ceil(train_wall_time_s / saving_fast) if saving_fast > 1.0e-9 else None
    )
    n_star_warm: float | None = None
    warm_speedup: float | None = None
    if warm_fem_s_per_query is not None and warm_fem_s_per_query > 0.0:
        warm_speedup = cold_fem_s_per_query / warm_fem_s_per_query
        saving_warm = cold_fem_s_per_query - warm_fem_s_per_query
        if saving_warm > 1.0e-9:
            n_star_warm = float(math.ceil(train_wall_time_s / saving_warm))
    return {
        "train_wall_time_s": train_wall_time_s,
        "cold_fem_s_per_query": cold_fem_s_per_query,
        "fast_surrogate_s_per_query": fast_surrogate_s_per_query,
        "warm_fem_s_per_query": warm_fem_s_per_query,
        "fast_speedup_vs_cold_fem": (
            cold_fem_s_per_query / max(fast_surrogate_s_per_query, 1.0e-9)
        ),
        "warm_speedup_vs_cold_fem": warm_speedup,
        "break_even_n_star_fast": None if n_star_fast is None else float(n_star_fast),
        "break_even_n_star_warm": n_star_warm,
    }


def qualify_3d_bundle(
    bundle: Surface3DBundle,
    holdout_cases: Sequence[LoadCase3D],
    *,
    max_mean_displacement_rel_l2: float = 0.15,
    max_mean_peak_vm_rel_error: float = 0.25,
    backend: TorchFemBackend3D | None = None,
) -> dict[str, Any]:
    """Qualify a 3D bundle across unseen held-out load cases and compute break-even ``N*``."""
    if not holdout_cases:
        raise ValueError("holdout_cases must be non-empty")
    solver = TorchFemBackend3D() if backend is None else backend
    dtype = next(bundle.operator.parameters()).dtype
    s_c, s_n, s_a, s_f = build_surface_3d_tensors(
        bundle.mesh,
        clamped_face_tags=bundle.clamped_face_tags,
        loaded_face_tags=bundle.loaded_face_tags,
        dtype=dtype,
    )
    with torch.no_grad():
        cached_enc = bundle.operator.encode_surface(s_c, s_n, s_a, s_f)

    u_errs: list[float] = []
    vm_errs: list[float] = []
    fast_times: list[float] = []
    cold_times: list[float] = []
    warm_times: list[float] = []
    cold_iters: list[int] = []
    warm_iters: list[int] = []
    min_det = float("inf")
    records: list[dict[str, Any]] = []

    for case in holdout_cases:
        # Cold reference solve (or reuse precomputed cold result)
        req = FemRequest3D(
            request_id=f"qual-cold-{case.case_id}",
            mesh=bundle.mesh,
            material=bundle.material,
            dirichlet_bcs=(
                DirichletBC3D(
                    name="clamp",
                    face_tags=bundle.clamped_face_tags,
                    components=("x", "y", "z"),
                    values_mm=(0.0, 0.0, 0.0),
                ),
            ),
            traction_bcs=case.traction_bcs,
            load_steps=3,
        )
        cold_res = case.fem_result if case.fem_result is not None else solver.solve(req)
        cold_times.append(cold_res.wall_time_s)
        cold_iters.append(cold_res.total_newton_iterations)

        cert = predict_3d_bundle(
            bundle,
            LoadCase3D(
                case_id=case.case_id,
                parameter_vector=case.parameter_vector,
                traction_bcs=case.traction_bcs,
                fem_result=cold_res,
            ),
            mode="certified",
            backend=solver,
            cached_encoding=cached_enc,
        )
        exact_warm = predict_3d_bundle(
            bundle,
            case,
            mode="exact",
            backend=solver,
            cached_encoding=cached_enc,
        )
        fast_times.append(cert.inference_wall_time_s)
        if exact_warm.fem_result is not None:
            warm_times.append(
                cert.inference_wall_time_s + exact_warm.fem_result.wall_time_s
            )
            warm_iters.append(exact_warm.fem_result.total_newton_iterations)
        assert cert.displacement_rel_l2_error is not None
        assert cert.peak_von_mises_rel_error is not None
        u_errs.append(cert.displacement_rel_l2_error)
        vm_errs.append(cert.peak_von_mises_rel_error)
        min_det = min(min_det, cert.min_det_f)
        records.append(
            {
                "case_id": case.case_id,
                "parameters": list(case.parameter_vector),
                "displacement_rel_l2_error": cert.displacement_rel_l2_error,
                "peak_von_mises_rel_error": cert.peak_von_mises_rel_error,
                "max_nodal_displacement_error_mm": cert.max_nodal_displacement_error_mm,
                "min_det_f": cert.min_det_f,
                "fast_wall_time_s": cert.inference_wall_time_s,
                "cold_fem_wall_time_s": cold_res.wall_time_s,
                "cold_fem_newton_iters": cold_res.total_newton_iterations,
                "warm_fem_wall_time_s": None
                if exact_warm.fem_result is None
                else exact_warm.fem_result.wall_time_s,
                "warm_fem_newton_iters": None
                if exact_warm.fem_result is None
                else exact_warm.fem_result.total_newton_iterations,
            }
        )

    mean_u = float(np.mean(u_errs))
    mean_vm = float(np.mean(vm_errs))
    passed = (
        mean_u <= max_mean_displacement_rel_l2
        and mean_vm <= max_mean_peak_vm_rel_error
        and min_det > 0.0
    )
    break_even = compute_3d_break_even(
        train_wall_time_s=bundle.training_summary.train_wall_time_s,
        cold_fem_s_per_query=float(np.mean(cold_times)),
        fast_surrogate_s_per_query=float(np.mean(fast_times)),
        warm_fem_s_per_query=float(np.mean(warm_times)) if warm_times else None,
    )
    doc: dict[str, Any] = {
        "schema": SURFACE_3D_QUALIFICATION_SCHEMA,
        "bundle_hash": bundle.bundle_hash,
        "passed": passed,
        "claim_level": "HOLDOUT_VERIFIED" if passed else "NOT_VERIFIED",
        "holdout_count": len(holdout_cases),
        "mean_displacement_rel_l2_error": mean_u,
        "max_displacement_rel_l2_error": float(np.max(u_errs)),
        "mean_peak_von_mises_rel_error": mean_vm,
        "max_peak_von_mises_rel_error": float(np.max(vm_errs)),
        "min_det_f": min_det,
        "mean_cold_newton_iters": float(np.mean(cold_iters)),
        "mean_warm_newton_iters": float(np.mean(warm_iters)) if warm_iters else None,
        "break_even": break_even,
        "cases": records,
    }
    doc["manifest_hash"] = _sha256_doc(doc)
    return doc


__all__ = (
    "SURFACE_3D_BUNDLE_SCHEMA",
    "SURFACE_3D_QUALIFICATION_SCHEMA",
    "Lifting3D",
    "LoadCase3D",
    "PlanarClampLifting3D",
    "PredictMode3D",
    "Prediction3DReport",
    "SoftMinFaceLifting3D",
    "Surface3DBundle",
    "Surface3DTrainingConfig",
    "Surface3DTrainingSummary",
    "build_surface_3d_tensors",
    "compute_3d_break_even",
    "lifted_displacement_and_gradient",
    "lifting_from_document",
    "load_surface_3d_bundle",
    "predict_3d_bundle",
    "qualify_3d_bundle",
    "save_surface_3d_bundle",
    "train_surface_3d_operator",
)
