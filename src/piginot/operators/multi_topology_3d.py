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
from scipy.sparse.linalg import eigsh  # type: ignore[import-untyped]

from piginot.geometry.mesh3d import CanonicalMesh3D
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
    deg = np.asarray(adj.sum(axis=1), dtype=np.float64).reshape(-1)
    inv_sqrt_d = np.where(deg > 1.0e-12, 1.0 / np.sqrt(np.maximum(deg, 1.0e-12)), 0.0)
    d_inv = diags(inv_sqrt_d, format="csr")
    lap = (
        diags(np.ones(m, dtype=np.float64), format="csr") - d_inv @ adj @ d_inv
    ).tocsr()

    if m <= 128:
        evals, evecs = np.linalg.eigh(lap.toarray())
        order = np.argsort(evals)
        phi1 = evecs[:, order[1]] if m > 1 else np.zeros(m, dtype=np.float64)
        phi2 = evecs[:, order[2]] if m > 2 else np.zeros(m, dtype=np.float64)
    else:
        evals, evecs = eigsh(lap, k=3, sigma=-1.0e-4, which="LM")
        order = np.argsort(evals)
        phi1 = np.asarray(evecs[:, order[1]], dtype=np.float64)
        phi2 = np.asarray(evecs[:, order[2]], dtype=np.float64)

    # Canonical sign disambiguation + normalization to [-1, 1]
    for arr in (phi1, phi2):
        # Deterministic sign via correlation with x+y+z centroid projection
        proj = centroids @ np.array([1.0, 0.7, 0.3], dtype=np.float64)
        if float(np.dot(arr, proj)) < 0.0:
            arr *= -1.0
        scale = max(float(np.max(np.abs(arr))), 1.0e-12)
        arr /= scale
    return phi1, phi2


def _local_thickness_proxy(centroids: FloatArray, normals: FloatArray) -> FloatArray:
    """Estimate normalized local wall thickness by opposing-normal facet distance."""
    m = int(centroids.shape[0])
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
        # Opposing normal facets behind current facet along -n
        n_dot = n_blk @ normals.T  # (C, M)
        inward_proj = -np.einsum("ci,cmi->cm", n_blk, diff)  # positive along -n
        opposing = (n_dot < -0.25) & (inward_proj > 1.0e-6) & (dist > 1.0e-6)
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
    c_np = c_t[0].detach().cpu().numpy().astype(np.float64)
    n_np = n_t[0].detach().cpu().numpy().astype(np.float64)
    tris = np.asarray(mesh.surface_triangles, dtype=np.int64)
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
        eulers = tuple(sorted({d.surface_euler_characteristic for d in descriptors}))
        rows = [list(d.vector()) for d in descriptors]
        mat = np.asarray(rows, dtype=np.float64)
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
            math.sqrt(max(float((row - mu) @ precision @ (row - mu)), 0.0))
            for row in mat
        ]
        thresh = max(float(min_threshold), float(threshold_multiplier) * max(dists))
        return cls(
            allowed_eulers=eulers,
            mean=tuple(float(v) for v in mu),
            inv_cov=tuple(tuple(float(x) for x in r) for r in precision),
            threshold=thresh,
            train_count=len(descriptors),
            feature_dim=dim,
            allow_unseen_topology=allow_unseen_topology,
        )

    def score(self, descriptor: GeometryDescriptor3D) -> GeometryOODReport3D:
        chi = descriptor.surface_euler_characteristic
        if chi not in self.allowed_eulers:
            status = (
                GeometryOOD3DStatus.OUT_OF_DISTRIBUTION
                if self.allow_unseen_topology
                else GeometryOOD3DStatus.TOPOLOGY_REFUSED
            )
            return GeometryOODReport3D(
                mesh_id=descriptor.mesh_id,
                status=status,
                mahalanobis_distance=float("inf"),
                threshold=self.threshold,
                expected_euler_characteristic=self.allowed_eulers[0],
                observed_euler_characteristic=chi,
                reason=(
                    f"Euler characteristic chi={chi} not in trained topology set "
                    f"{list(self.allowed_eulers)}"
                ),
            )
        arr = np.asarray(descriptor.vector(), dtype=np.float64)
        mu = np.asarray(self.mean, dtype=np.float64)
        prec = np.asarray(self.inv_cov, dtype=np.float64)
        diff = arr - mu
        dist = math.sqrt(max(float(diff @ prec @ diff), 0.0))
        if dist > self.threshold:
            return GeometryOODReport3D(
                mesh_id=descriptor.mesh_id,
                status=GeometryOOD3DStatus.OUT_OF_DISTRIBUTION,
                mahalanobis_distance=dist,
                threshold=self.threshold,
                expected_euler_characteristic=chi,
                observed_euler_characteristic=chi,
                reason=f"Mahalanobis distance {dist:.3f} > threshold {self.threshold:.3f}",
            )
        return GeometryOODReport3D(
            mesh_id=descriptor.mesh_id,
            status=GeometryOOD3DStatus.IN_DISTRIBUTION,
            mahalanobis_distance=dist,
            threshold=self.threshold,
            expected_euler_characteristic=chi,
            observed_euler_characteristic=chi,
            reason="accepted inside multi-topology envelope",
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
        ood_detector: MultiTopologyOODDetector3D | None = None,
    ) -> None:
        self.config = Surface3DConfig(
            parameter_dim=parameter_dim,
            surface_feature_dim=MULTI_TOPOLOGY_FEATURE_DIM,
            hidden_dim=hidden_dim,
            slice_count=slice_count,
            num_blocks=num_blocks,
            num_heads=num_heads,
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
    ) -> tuple[FloatArray | None, GeometryOODReport3D | None]:
        """Predict 3D nodal displacement, refusing fail-closed if OOD/unseen topology."""
        self.operator.eval()
        dtype = next(self.operator.parameters()).dtype
        with torch.no_grad():
            enc, ood = self.encode_mesh(
                mesh,
                clamped_face_tags=clamped_face_tags,
                loaded_face_tags=loaded_face_tags,
                dtype=dtype,
            )
            if enforce_ood and ood is not None and not ood.accepted:
                return None, ood
            pts = torch.tensor(mesh.nodes, dtype=dtype).unsqueeze(0)
            p_vec = torch.tensor(
                [list(parameters[: self.config.parameter_dim])], dtype=dtype
            )
            u_t = self.operator.displacement(
                encoding=enc, query_xyz=pts, parameters=p_vec
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
