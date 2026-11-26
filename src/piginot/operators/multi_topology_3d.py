"""Multi-topology 3D surface encoders & two-tier multi-genus OOD detection (V4-4).

Extends FormaMech's 16-channel GALE surface representation with 4
topology-intrinsic channels (20 channels total):

* Local medial-thickness proxy (distance along inward surface normal to nearest
  opposing boundary facet),
* Two normalized surface-Laplacian spectral harmonic coordinates (Fiedler
  eigenvectors ``phi_1, phi_2`` capturing global genus-invariant connectivity),
* Normalized surface genus indicator ``(2 - chi) / 8``.

This lets a single :class:`Surface3DEnergyOperator` operate across genus-0,
genus-1, and genus-2+ 3D solids while :class:`MultiTopologyOODDetector3D`
enforces fail-closed topology & Mahalanobis admissibility across multiple
trained Euler characteristics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, diags  # type: ignore[import-untyped]
from scipy.sparse.csgraph import connected_components  # type: ignore[import-untyped]
from scipy.sparse.linalg import (  # type: ignore[import-untyped]
    ArpackNoConvergence,
    eigsh,
)

from piginot.geometry.mesh3d import CanonicalMesh3D
from piginot.lifting.adf_3d import build_adf_lifting_from_mesh_faces
from piginot.operators.geometry_3d import (
    SURFACE_3D_GALE_FEATURE_DIM,
    GeometryDescriptor3D,
    GeometryOOD3DStatus,
    GeometryOODReport3D,
    build_gale_surface_3d_tensors,
    compute_geometry_descriptor_3d,
)
from piginot.operators.surface_operator_3d import (
    Surface3DConfig,
    Surface3DEncoding,
    Surface3DEnergyOperator,
)

FloatArray = NDArray[np.float64]

MULTI_TOPOLOGY_FEATURE_DIM: Final = SURFACE_3D_GALE_FEATURE_DIM + 4  # 20
MULTI_TOPOLOGY_OOD_SCHEMA: Final = "formamech.multi-topology-ood-3d-v1"


def _surface_spectral_coords(
    centroids: FloatArray,
    triangles: NDArray[np.int64],
) -> tuple[FloatArray, FloatArray]:
    """Compute first two non-trivial surface-graph Laplacian spectral coordinates."""
    m = int(centroids.shape[0])
    if m < 4:
        z = np.zeros(m, dtype=np.float64)
        return z, z.copy()

    # Build triangle adjacency via shared undirected edges
    edge_to_tris: dict[tuple[int, int], list[int]] = {}
    for t_idx in range(m):
        a, b, c = (
            int(triangles[t_idx, 0]),
            int(triangles[t_idx, 1]),
            int(triangles[t_idx, 2]),
        )
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_to_tris.setdefault(key, []).append(t_idx)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for tri_list in edge_to_tris.values():
        if len(tri_list) >= 2:
            t0, t1 = tri_list[0], tri_list[1]
            dist = float(np.linalg.norm(centroids[t0] - centroids[t1]))
            w = 1.0 / max(dist, 1.0e-6)
            rows.extend((t0, t1))
            cols.extend((t1, t0))
            vals.extend((w, w))

    if not rows:
        z = np.zeros(m, dtype=np.float64)
        return z, z.copy()

    adj = coo_matrix((vals, (rows, cols)), shape=(m, m)).tocsr()
    n_comp, _ = connected_components(adj, directed=False)
    idx1 = int(n_comp)
    idx2 = int(n_comp) + 1
    if m <= idx1:
        z = np.zeros(m, dtype=np.float64)
        return z, z.copy()

    deg = np.asarray(adj.sum(axis=1), dtype=np.float64).reshape(-1)
    inv_sqrt_d = np.where(deg > 1.0e-12, 1.0 / np.sqrt(np.maximum(deg, 1.0e-12)), 0.0)
    d_inv = diags(inv_sqrt_d, format="csr")
    lap = (
        diags(np.ones(m, dtype=np.float64), format="csr") - d_inv @ adj @ d_inv
    ).tocsr()

    k_req = min(m - 1, idx2 + 1)
    if m <= 128 or k_req >= m - 1:
        evals, evecs = np.linalg.eigh(lap.toarray())
        order = np.argsort(evals)
        phi1 = (
            evecs[:, order[idx1]].copy() if idx1 < m else np.zeros(m, dtype=np.float64)
        )
        phi2 = (
            evecs[:, order[idx2]].copy() if idx2 < m else np.zeros(m, dtype=np.float64)
        )
    else:
        # Deterministic starting vector for reproducible ARPACK convergence
        v0 = np.sin(np.arange(1, m + 1, dtype=np.float64) * 1.618033988749895) + 0.5
        try:
            evals, evecs = eigsh(lap, k=k_req, sigma=-1.0e-4, which="LM", v0=v0)
            order = np.argsort(evals)
            phi1 = np.asarray(evecs[:, order[idx1]], dtype=np.float64).copy()
            phi2 = (
                np.asarray(evecs[:, order[idx2]], dtype=np.float64).copy()
                if idx2 < len(order)
                else np.zeros(m, dtype=np.float64)
            )
        except (ArpackNoConvergence, RuntimeError):
            evals, evecs = np.linalg.eigh(lap.toarray())
            order = np.argsort(evals)
            phi1 = evecs[:, order[idx1]].copy()
            phi2 = (
                evecs[:, order[idx2]].copy()
                if idx2 < m
                else np.zeros(m, dtype=np.float64)
            )

    # Translation-invariant canonical 2D Procrustes alignment + normalization to [-1, 1].
    # On symmetric 3D CAD solids (e.g., bolted mounting lugs), the 2nd & 3rd Laplacian
    # eigenvalues are nearly degenerate and can swap or rotate in their 2D eigenspace
    # across unstructured remeshes. Orthogonal Procrustes alignment against canonical
    # geometric directions makes (phi1, phi2) invariant to both sign flips AND 2D
    # eigenspace rotations/swaps while preserving the exact Fiedler harmonic subspace.
    c_centered = centroids - np.mean(centroids, axis=0, keepdims=True)
    ref_dirs = np.column_stack(
        (
            c_centered @ np.array([1.0, 0.31, 0.13], dtype=np.float64),
            c_centered @ np.array([-0.29, 1.0, 0.37], dtype=np.float64),
        )
    )  # (M, 2)
    phi_mat = np.column_stack((phi1, phi2))  # (M, 2)
    if float(np.linalg.norm(phi_mat)) > 1.0e-12:
        u_p, _, vt_p = np.linalg.svd(phi_mat.T @ ref_dirs, full_matrices=False)
        rot_2x2 = u_p @ vt_p
        phi_aligned = phi_mat @ rot_2x2
        phi1 = phi_aligned[:, 0].copy()
        phi2 = phi_aligned[:, 1].copy()

    for idx_a, arr in enumerate((phi1, phi2)):
        if float(np.dot(arr, ref_dirs[:, idx_a])) < 0.0:
            arr *= -1.0
        scale = max(float(np.max(np.abs(arr))), 1.0e-12)
        arr /= scale
    return phi1, phi2


def _local_thickness_proxy(centroids: FloatArray, normals: FloatArray) -> FloatArray:
    """Estimate normalized local wall thickness by cone-aligned opposing-facet distance."""
    m = int(centroids.shape[0])
    if m == 0:
        return np.zeros(0, dtype=np.float64)
    bbox_diag = float(np.linalg.norm(centroids.max(axis=0) - centroids.min(axis=0)))
    diag = max(bbox_diag, 1.0e-6)
    thick = np.ones(m, dtype=np.float64)
    chunk = 256
    for start in range(0, m, chunk):
        end = min(start + chunk, m)
        c_blk = centroids[start:end]  # (C, 3)
        n_blk = normals[start:end]  # (C, 3)
        diff = centroids[None, :, :] - c_blk[:, None, :]  # (C, M, 3)
        dist = np.linalg.norm(diff, axis=-1)  # (C, M)
        # Opposing normal facets lying inside a narrow cone along -n
        n_dot = n_blk @ normals.T  # (C, M)
        inward_proj = -np.einsum("ci,cmi->cm", n_blk, diff)  # positive along -n
        cos_cone = inward_proj / np.maximum(dist, 1.0e-12)
        opposing = (
            (n_dot < -0.25)
            & (inward_proj > 1.0e-6)
            & (cos_cone > 0.85)
            & (dist > 1.0e-6)
        )
        masked_d = np.where(opposing, dist, np.inf)
        min_d = np.min(masked_d, axis=1)
        thick[start:end] = np.where(np.isfinite(min_d), min_d / diag, 1.0)
    return thick


def build_multi_topology_surface_tensors(
    mesh: CanonicalMesh3D,
    *,
    clamped_face_tags: Sequence[str] = (),
    loaded_face_tags: Sequence[str] = (),
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build 20-channel multi-topology surface tensors ``(1, M, ...)`` for any genus."""
    c_t, n_t, a_t, gale_f = build_gale_surface_3d_tensors(
        mesh,
        clamped_face_tags=clamped_face_tags,
        loaded_face_tags=loaded_face_tags,
        dtype=dtype,
        device=device,
    )
    # Compute surface centroids & unit normals in float64 to preserve precision
    nodes_f64 = np.asarray(mesh.nodes, dtype=np.float64)
    tris = np.asarray(mesh.surface_triangles, dtype=np.int64)
    if tris.shape[0] > 0:
        p0, p1, p2 = nodes_f64[tris[:, 0]], nodes_f64[tris[:, 1]], nodes_f64[tris[:, 2]]
        c_np = (p0 + p1 + p2) / 3.0
        cross = np.cross(p1 - p0, p2 - p0)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        n_np = cross / np.maximum(norms, 1.0e-15)
    else:
        c_np = np.zeros((0, 3), dtype=np.float64)
        n_np = np.zeros((0, 3), dtype=np.float64)

    desc = compute_geometry_descriptor_3d(
        mesh, clamped_face_tags=clamped_face_tags, loaded_face_tags=loaded_face_tags
    )

    thick = _local_thickness_proxy(c_np, n_np)
    phi1, phi2 = _surface_spectral_coords(c_np, tris)
    genus_norm = max(
        0.0, min(1.0, (2.0 - float(desc.surface_euler_characteristic)) / 8.0)
    )
    genus_col = np.full(c_np.shape[0], genus_norm, dtype=np.float64)

    extra = np.column_stack((thick, phi1, phi2, genus_col))
    extra_t = torch.tensor(extra, dtype=dtype, device=device).unsqueeze(0)
    feats = torch.cat((gale_f, extra_t), dim=-1)
    return c_t, n_t, a_t, feats


def _fit_mahalanobis_envelope(
    mat: FloatArray,
    *,
    threshold_multiplier: float,
    min_threshold: float,
    ridge: float,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...], float]:
    n_s, dim = mat.shape
    mu = mat.mean(axis=0)
    std_floor = np.maximum(1.0e-2 * np.abs(mu), 1.0e-3)
    std = np.maximum(mat.std(axis=0), std_floor)
    z = (mat - mu[None, :]) / std[None, :]
    cov = np.cov(z, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    shrink = max(float(ridge), min(0.85, float(dim) / float(dim + n_s)))
    cov_reg = (1.0 - shrink) * cov + shrink * np.eye(dim, dtype=np.float64)
    inv_z = np.linalg.inv(cov_reg)
    inv_std = np.diag(1.0 / std)
    precision = (inv_std @ inv_z @ inv_std) / float(dim)
    precision = 0.5 * (precision + precision.T)

    dists = [
        math.sqrt(max(float((row - mu) @ precision @ (row - mu)), 0.0)) for row in mat
    ]
    thresh = max(float(min_threshold), float(threshold_multiplier) * max(dists))
    return (
        tuple(float(v) for v in mu),
        tuple(tuple(float(x) for x in r) for r in precision),
        thresh,
    )


@dataclass(frozen=True, slots=True)
class MultiTopologyOODDetector3D:
    """Two-tier fail-closed OOD detector supporting multiple trained Euler topologies."""

    allowed_eulers: tuple[int, ...]
    mean: tuple[float, ...]
    inv_cov: tuple[tuple[float, ...], ...]
    threshold: float
    train_count: int
    feature_dim: int
    allow_unseen_topology: bool = False
    per_euler_envelopes: tuple[
        tuple[int, tuple[float, ...], tuple[tuple[float, ...], ...], float], ...
    ] = ()

    @classmethod
    def fit(
        cls,
        descriptors: Sequence[GeometryDescriptor3D],
        *,
        threshold_multiplier: float = 1.8,
        min_threshold: float = 1.5,
        ridge: float = 1.0e-2,
        allow_unseen_topology: bool = False,
    ) -> MultiTopologyOODDetector3D:
        if len(descriptors) < 2:
            raise ValueError("MultiTopologyOODDetector3D.fit requires >= 2 geometries")
        if not math.isfinite(threshold_multiplier) or threshold_multiplier <= 0.0:
            raise ValueError("threshold_multiplier must be finite and positive")
        if not math.isfinite(min_threshold) or min_threshold <= 0.0:
            raise ValueError("min_threshold must be finite and positive")
        if not math.isfinite(ridge) or not (0.0 < ridge < 1.0):
            raise ValueError("ridge must be finite and lie in (0, 1)")

        eulers = tuple(sorted({d.surface_euler_characteristic for d in descriptors}))
        rows = [list(d.vector()) for d in descriptors]
        mat = np.asarray(rows, dtype=np.float64)
        if not np.all(np.isfinite(mat)):
            raise ValueError("training geometry descriptors must contain finite values")
        _, dim = mat.shape
        mu_t, prec_t, thresh = _fit_mahalanobis_envelope(
            mat,
            threshold_multiplier=threshold_multiplier,
            min_threshold=min_threshold,
            ridge=ridge,
        )

        # Fit per-Euler-stratum envelopes when a topology class has >= 2 samples
        per_euler: list[
            tuple[int, tuple[float, ...], tuple[tuple[float, ...], ...], float]
        ] = []
        for chi in eulers:
            sub_rows = [
                list(d.vector())
                for d in descriptors
                if d.surface_euler_characteristic == chi
            ]
            if len(sub_rows) >= 2:
                sub_mu, sub_prec, sub_thresh = _fit_mahalanobis_envelope(
                    np.asarray(sub_rows, dtype=np.float64),
                    threshold_multiplier=threshold_multiplier,
                    min_threshold=min_threshold,
                    ridge=ridge,
                )
                per_euler.append((chi, sub_mu, sub_prec, sub_thresh))

        return cls(
            allowed_eulers=eulers,
            mean=mu_t,
            inv_cov=prec_t,
            threshold=thresh,
            train_count=len(descriptors),
            feature_dim=dim,
            allow_unseen_topology=allow_unseen_topology,
            per_euler_envelopes=tuple(per_euler),
        )

    def score(self, descriptor: GeometryDescriptor3D) -> GeometryOODReport3D:
        chi = descriptor.surface_euler_characteristic
        unseen = chi not in self.allowed_eulers
        if unseen and not self.allow_unseen_topology:
            return GeometryOODReport3D(
                mesh_id=descriptor.mesh_id,
                status=GeometryOOD3DStatus.TOPOLOGY_REFUSED,
                mahalanobis_distance=float("inf"),
                threshold=self.threshold,
                expected_euler_characteristic=self.allowed_eulers[0],
                observed_euler_characteristic=chi,
                reason=(
                    f"Euler characteristic chi={chi} not in trained topology set "
                    f"{list(self.allowed_eulers)}"
                ),
            )

        arr = np.asarray(descriptor.vector(), dtype=np.float64).reshape(-1)
        if arr.size != self.feature_dim or not np.all(np.isfinite(arr)):
            return GeometryOODReport3D(
                mesh_id=descriptor.mesh_id,
                status=GeometryOOD3DStatus.OUT_OF_DISTRIBUTION,
                mahalanobis_distance=float("inf"),
                threshold=self.threshold,
                expected_euler_characteristic=(
                    self.allowed_eulers[0] if unseen else chi
                ),
                observed_euler_characteristic=chi,
                reason=(
                    f"non-finite or mismatched descriptor dimension "
                    f"(got {arr.size}, expected {self.feature_dim})"
                ),
            )

        # Select per-Euler envelope if available for this topology, else global envelope
        mu_use = self.mean
        prec_use = self.inv_cov
        thresh_use = self.threshold
        for e_chi, e_mu, e_prec, e_thresh in self.per_euler_envelopes:
            if e_chi == chi:
                mu_use = e_mu
                prec_use = e_prec
                thresh_use = e_thresh
                break

        mu = np.asarray(mu_use, dtype=np.float64)
        prec = np.asarray(prec_use, dtype=np.float64)
        diff = arr - mu
        quad = float(diff @ prec @ diff)
        dist = math.sqrt(max(quad, 0.0)) if math.isfinite(quad) else float("inf")
        if not math.isfinite(dist) or dist > thresh_use:
            return GeometryOODReport3D(
                mesh_id=descriptor.mesh_id,
                status=GeometryOOD3DStatus.OUT_OF_DISTRIBUTION,
                mahalanobis_distance=dist,
                threshold=thresh_use,
                expected_euler_characteristic=(
                    self.allowed_eulers[0] if unseen else chi
                ),
                observed_euler_characteristic=chi,
                reason=f"Mahalanobis distance {dist:.3f} > threshold {thresh_use:.3f}",
            )
        return GeometryOODReport3D(
            mesh_id=descriptor.mesh_id,
            status=GeometryOOD3DStatus.IN_DISTRIBUTION,
            mahalanobis_distance=dist,
            threshold=thresh_use,
            expected_euler_characteristic=self.allowed_eulers[0] if unseen else chi,
            observed_euler_characteristic=chi,
            reason=(
                "accepted inside global envelope (unseen topology allowed)"
                if unseen
                else "accepted inside multi-topology envelope"
            ),
        )

    def document(self) -> dict[str, Any]:
        return {
            "schema": MULTI_TOPOLOGY_OOD_SCHEMA,
            "allowed_eulers": list(self.allowed_eulers),
            "threshold": float(self.threshold),
            "train_count": self.train_count,
            "feature_dim": self.feature_dim,
            "allow_unseen_topology": self.allow_unseen_topology,
        }


class MultiTopologyOperator3D:
    """Shared 20-channel 3D surface neural operator guarded by multi-genus OOD."""

    def __init__(
        self,
        *,
        parameter_dim: int = 2,
        hidden_dim: int = 32,
        slice_count: int = 6,
        num_blocks: int = 2,
        num_heads: int = 4,
        positional_frequencies: int = 3,
        decoder_blocks: int = 1,
        ood_detector: MultiTopologyOODDetector3D | None = None,
    ) -> None:
        self.config = Surface3DConfig(
            parameter_dim=parameter_dim,
            surface_feature_dim=MULTI_TOPOLOGY_FEATURE_DIM,
            hidden_dim=hidden_dim,
            slice_count=slice_count,
            num_blocks=num_blocks,
            num_heads=num_heads,
            positional_frequencies=positional_frequencies,
            decoder_blocks=decoder_blocks,
        )
        self.operator = Surface3DEnergyOperator(self.config)
        self.ood_detector = ood_detector

    def fit_ood(
        self,
        meshes: Sequence[tuple[CanonicalMesh3D, Sequence[str], Sequence[str]]],
    ) -> MultiTopologyOODDetector3D:
        descs = [
            compute_geometry_descriptor_3d(
                m, clamped_face_tags=c_tags, loaded_face_tags=l_tags
            )
            for m, c_tags, l_tags in meshes
        ]
        self.ood_detector = MultiTopologyOODDetector3D.fit(descs)
        return self.ood_detector

    def encode_mesh(
        self,
        mesh: CanonicalMesh3D,
        *,
        clamped_face_tags: Sequence[str] = (),
        loaded_face_tags: Sequence[str] = (),
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> tuple[Surface3DEncoding, GeometryOODReport3D | None]:
        desc = compute_geometry_descriptor_3d(
            mesh, clamped_face_tags=clamped_face_tags, loaded_face_tags=loaded_face_tags
        )
        ood = None if self.ood_detector is None else self.ood_detector.score(desc)
        c_t, n_t, a_t, f_t = build_multi_topology_surface_tensors(
            mesh,
            clamped_face_tags=clamped_face_tags,
            loaded_face_tags=loaded_face_tags,
            dtype=dtype,
            device=device,
        )
        enc = self.operator.encode_surface(c_t, n_t, a_t, f_t)
        return enc, ood

    def predict_nodal_displacement(
        self,
        mesh: CanonicalMesh3D,
        *,
        parameters: Sequence[float],
        clamped_face_tags: Sequence[str] = (),
        loaded_face_tags: Sequence[str] = (),
        enforce_ood: bool = True,
        apply_adf_lifting: bool = False,
        displacement_scale_mm: float = 1.0,
    ) -> tuple[FloatArray | None, GeometryOODReport3D | None]:
        """Predict 3D nodal displacement, refusing fail-closed if OOD/unseen topology."""
        params_list = [float(v) for v in parameters]
        if len(params_list) != self.config.parameter_dim:
            raise ValueError(
                f"parameters length {len(params_list)} must equal "
                f"parameter_dim={self.config.parameter_dim}"
            )
        if any(not math.isfinite(v) for v in params_list):
            raise ValueError("parameters must contain only finite numbers")

        desc = compute_geometry_descriptor_3d(
            mesh, clamped_face_tags=clamped_face_tags, loaded_face_tags=loaded_face_tags
        )
        if enforce_ood and self.ood_detector is None:
            return None, GeometryOODReport3D(
                mesh_id=desc.mesh_id,
                status=GeometryOOD3DStatus.OUT_OF_DISTRIBUTION,
                mahalanobis_distance=float("inf"),
                threshold=0.0,
                expected_euler_characteristic=desc.surface_euler_characteristic,
                observed_euler_characteristic=desc.surface_euler_characteristic,
                reason="fail-closed refusal: no MultiTopologyOODDetector3D fitted",
            )
        ood = None if self.ood_detector is None else self.ood_detector.score(desc)
        if enforce_ood and ood is not None and not ood.accepted:
            return None, ood

        self.operator.eval()
        first_param = next(self.operator.parameters())
        dtype = first_param.dtype
        device = first_param.device
        with torch.no_grad():
            c_t, n_t, a_t, f_t = build_multi_topology_surface_tensors(
                mesh,
                clamped_face_tags=clamped_face_tags,
                loaded_face_tags=loaded_face_tags,
                dtype=dtype,
                device=device,
            )
            enc = self.operator.encode_surface(c_t, n_t, a_t, f_t)
            pts = torch.tensor(mesh.nodes, dtype=dtype, device=device).unsqueeze(0)
            p_vec = torch.tensor([params_list], dtype=dtype, device=device)
            lift_scale: torch.Tensor | None = None
            lift_offset: torch.Tensor | None = None
            if apply_adf_lifting and clamped_face_tags:
                lift = build_adf_lifting_from_mesh_faces(
                    mesh,
                    clamped_face_tags,
                    displacement_scale_mm=float(displacement_scale_mm),
                )
                lift_scale, lift_offset = lift.scale_and_offset(pts)
            u_t = self.operator.displacement(
                encoding=enc,
                query_xyz=pts,
                parameters=p_vec,
                lifting_offset=lift_offset,
                lifting_scale=lift_scale,
            )
            return u_t[0].detach().cpu().numpy().astype(np.float64), ood

    def summary_document(self) -> Mapping[str, Any]:
        return {
            "surface_feature_dim": MULTI_TOPOLOGY_FEATURE_DIM,
            "hidden_dim": self.config.hidden_dim,
            "slice_count": self.config.slice_count,
            "ood_detector": (
                None if self.ood_detector is None else self.ood_detector.document()
            ),
        }


__all__ = (
    "MULTI_TOPOLOGY_FEATURE_DIM",
    "MULTI_TOPOLOGY_OOD_SCHEMA",
    "MultiTopologyOODDetector3D",
    "MultiTopologyOperator3D",
    "build_multi_topology_surface_tensors",
)
