"""Geometry-aware 3D GALE surface tokenization and 3D OOD detection (V3-2).

Extracts 16-channel Geometry-Aware Latent Embedding (GALE) surface features
directly from any :class:`~piginot.geometry.mesh3d.CanonicalMesh3D` — combining
boundary roles, normalized area, local normal curvature proxies, multi-scale
ball-neighborhood statistics, and essential/loaded geodesic proxies — plus a
``TRAIN``-fitted Mahalanobis + slice-latent :class:`GeometryOODDetector3D`.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray

from piginot.contracts import canonical_json_bytes
from piginot.geometry.mesh3d import CanonicalMesh3D

FloatArray = NDArray[np.float64]
SURFACE_3D_GALE_FEATURE_DIM: Final = 16
GEOMETRY_OOD_3D_SCHEMA: Final = "formamech.geometry-ood-3d-v1"
GEOMETRY_DESCRIPTOR_3D_FIELDS: Final[tuple[str, ...]] = (
    "volume_mm3",
    "surface_area_mm2",
    "compactness",
    "extent_x_mm",
    "extent_y_mm",
    "extent_z_mm",
    "inertia_eig_0",
    "inertia_eig_1",
    "inertia_eig_2",
    "clamped_area_fraction",
    "loaded_area_fraction",
)


class GeometryOOD3DStatus(StrEnum):
    """Fail-closed 3D geometry OOD gate verdicts."""

    IN_DISTRIBUTION = "IN_DISTRIBUTION"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    TOPOLOGY_REFUSED = "TOPOLOGY_REFUSED"


@dataclass(frozen=True, slots=True)
class GeometryDescriptor3D:
    """Intrinsic 3D volumetric & surface geometric descriptor vector."""

    mesh_id: str
    volume_mm3: float
    surface_area_mm2: float
    compactness: float
    extent_mm: tuple[float, float, float]
    inertia_eigenvalues: tuple[float, float, float]
    clamped_area_fraction: float
    loaded_area_fraction: float
    surface_euler_characteristic: int

    def vector(self) -> tuple[float, ...]:
        return (
            self.volume_mm3,
            self.surface_area_mm2,
            self.compactness,
            self.extent_mm[0],
            self.extent_mm[1],
            self.extent_mm[2],
            self.inertia_eigenvalues[0],
            self.inertia_eigenvalues[1],
            self.inertia_eigenvalues[2],
            self.clamped_area_fraction,
            self.loaded_area_fraction,
        )

    def document(self) -> dict[str, Any]:
        return {
            "mesh_id": self.mesh_id,
            "volume_mm3": float(self.volume_mm3),
            "surface_area_mm2": float(self.surface_area_mm2),
            "compactness": float(self.compactness),
            "extent_mm": [float(v) for v in self.extent_mm],
            "inertia_eigenvalues": [float(v) for v in self.inertia_eigenvalues],
            "clamped_area_fraction": float(self.clamped_area_fraction),
            "loaded_area_fraction": float(self.loaded_area_fraction),
            "surface_euler_characteristic": int(self.surface_euler_characteristic),
        }


def compute_surface_euler_characteristic(mesh: CanonicalMesh3D) -> int:
    """Compute boundary surface mesh Euler characteristic ``V - E + F``."""
    verts: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for a, b, c in mesh.surface_triangles:
        verts.update((a, b, c))
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((u, v) if u < v else (v, u))
    return len(verts) - len(edges) + len(mesh.surface_triangles)


def compute_geometry_descriptor_3d(
    mesh: CanonicalMesh3D,
    *,
    clamped_face_tags: Sequence[str] = (),
    loaded_face_tags: Sequence[str] = (),
) -> GeometryDescriptor3D:
    """Compute canonical intrinsic 3D volume, inertia, and surface descriptors."""
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    # Volume from corner vertices (valid for TET4 and TET10)
    p0 = nodes[cells[:, 0]]
    p1 = nodes[cells[:, 1]]
    p2 = nodes[cells[:, 2]]
    p3 = nodes[cells[:, 3]]
    vols = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0
    total_vol = float(np.sum(np.maximum(vols, 0.0)))

    centroids_t, _, areas_t = mesh.surface_geometry()
    centroids = np.asarray(centroids_t, dtype=np.float64)
    areas = np.asarray(areas_t, dtype=np.float64)
    total_area = float(np.sum(areas))
    compactness = (
        (total_area**3) / max(36.0 * math.pi * (total_vol**2), 1.0e-12)
        if total_vol > 0.0
        else 0.0
    )

    lo = nodes.min(axis=0)
    hi = nodes.max(axis=0)
    ext = hi - lo

    # Area-weighted surface covariance eigenvalues (rotation-invariant shape signature)
    w = areas / max(total_area, 1.0e-12)
    mean_c = np.sum(centroids * w[:, None], axis=0)
    rel = centroids - mean_c[None, :]
    cov = np.einsum("m,mi,mj->ij", w, rel, rel)
    eigs = np.sort(np.maximum(np.linalg.eigvalsh(0.5 * (cov + cov.T)), 0.0))

    clamped_tris: set[int] = set()
    for tag in clamped_face_tags:
        clamped_tris.update(mesh.semantic_face_tags.get(tag, ()))
    loaded_tris: set[int] = set()
    for tag in loaded_face_tags:
        loaded_tris.update(mesh.semantic_face_tags.get(tag, ()))

    clamped_area = float(sum(areas[i] for i in clamped_tris))
    loaded_area = float(sum(areas[i] for i in loaded_tris))
    denom_a = max(total_area, 1.0e-12)

    return GeometryDescriptor3D(
        mesh_id=mesh.mesh_id,
        volume_mm3=total_vol,
        surface_area_mm2=total_area,
        compactness=float(compactness),
        extent_mm=(float(ext[0]), float(ext[1]), float(ext[2])),
        inertia_eigenvalues=(float(eigs[0]), float(eigs[1]), float(eigs[2])),
        clamped_area_fraction=clamped_area / denom_a,
        loaded_area_fraction=loaded_area / denom_a,
        surface_euler_characteristic=compute_surface_euler_characteristic(mesh),
    )


def build_gale_surface_3d_tensors(
    mesh: CanonicalMesh3D,
    *,
    clamped_face_tags: Sequence[str],
    loaded_face_tags: Sequence[str],
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build 16-channel GALE surface tensors ``(centroids, normals, areas, features)``.

    Channels (16 total):
    ``[0..2]`` BC roles (clamped, loaded, free),
    ``[3..5]`` bbox-centered normalized coordinates,
    ``[6]`` normalized triangle area,
    ``[7]`` discrete local normal-variation curvature proxy,
    ``[8..10]`` small-ball (r = 0.12*diag) GALE mean normal,
    ``[11..13]`` medium-ball (r = 0.28*diag) GALE local centroid offset,
    ``[14]`` normalized distance to clamped surface,
    ``[15]`` normalized distance to loaded surface.
    """
    centroids_t, normals_t, areas_t = mesh.surface_geometry()
    c_np = np.asarray(centroids_t, dtype=np.float64)  # (M, 3)
    n_np = np.asarray(normals_t, dtype=np.float64)  # (M, 3)
    a_np = np.asarray(areas_t, dtype=np.float64)  # (M,)
    m = c_np.shape[0]

    lo = c_np.min(axis=0)
    hi = c_np.max(axis=0)
    center = 0.5 * (lo + hi)
    extent = np.maximum(hi - lo, 1.0e-6)
    diag = max(float(np.linalg.norm(extent)), 1.0e-6)

    clamped_mask = np.zeros(m, dtype=bool)
    for tag in clamped_face_tags:
        for tid in mesh.semantic_face_tags.get(tag, ()):
            clamped_mask[tid] = True
    loaded_mask = np.zeros(m, dtype=bool)
    for tag in loaded_face_tags:
        for tid in mesh.semantic_face_tags.get(tag, ()):
            loaded_mask[tid] = True
    free_mask = ~(clamped_mask | loaded_mask)

    # Pairwise Euclidean distance matrix on surface triangles (chunked for large M)
    r_small = 0.12 * diag
    r_med = 0.28 * diag
    curv = np.zeros(m, dtype=np.float64)
    mean_norm_small = np.zeros((m, 3), dtype=np.float64)
    mean_off_med = np.zeros((m, 3), dtype=np.float64)

    chunk = 512
    for start in range(0, m, chunk):
        end = min(start + chunk, m)
        diff = c_np[start:end, None, :] - c_np[None, :, :]  # (C, M, 3)
        dist = np.linalg.norm(diff, axis=-1)  # (C, M)
        # Small ball (Gaussian-weighted fallback guarantees non-empty neighborhood)
        w_s = np.exp(-0.5 * (dist / max(r_small, 1.0e-6)) ** 2) * (
            dist <= 2.0 * r_small
        )
        w_s_sum = np.maximum(w_s.sum(axis=1, keepdims=True), 1.0e-12)
        w_s_norm = w_s / w_s_sum
        avg_n = w_s_norm @ n_np  # (C, 3)
        avg_n_norm = np.maximum(np.linalg.norm(avg_n, axis=1, keepdims=True), 1.0e-12)
        avg_n_unit = avg_n / avg_n_norm
        mean_norm_small[start:end] = avg_n_unit
        dots = np.clip(np.sum(n_np[start:end] * avg_n_unit, axis=1), -1.0, 1.0)
        curv[start:end] = 1.0 - dots

        # Medium ball GALE local centroid displacement
        w_m = np.exp(-0.5 * (dist / max(r_med, 1.0e-6)) ** 2) * (dist <= 2.0 * r_med)
        w_m_sum = np.maximum(w_m.sum(axis=1, keepdims=True), 1.0e-12)
        avg_c = (w_m / w_m_sum) @ c_np
        mean_off_med[start:end] = (c_np[start:end] - avg_c) / diag

    def _min_dist(target_mask: NDArray[np.bool_]) -> FloatArray:
        if not np.any(target_mask):
            return np.ones(m, dtype=np.float64)
        targets = c_np[target_mask]
        out = np.empty(m, dtype=np.float64)
        for start in range(0, m, chunk):
            end = min(start + chunk, m)
            d = np.linalg.norm(c_np[start:end, None, :] - targets[None, :, :], axis=-1)
            out[start:end] = d.min(axis=1) / diag
        return out

    d_clamp = _min_dist(clamped_mask)
    d_load = _min_dist(loaded_mask)

    feats = np.zeros((m, SURFACE_3D_GALE_FEATURE_DIM), dtype=np.float64)
    feats[:, 0] = clamped_mask.astype(np.float64)
    feats[:, 1] = loaded_mask.astype(np.float64)
    feats[:, 2] = free_mask.astype(np.float64)
    feats[:, 3:6] = (c_np - center[None, :]) / extent[None, :]
    feats[:, 6] = a_np / max(float(np.mean(a_np)), 1.0e-12)
    feats[:, 7] = curv
    feats[:, 8:11] = mean_norm_small
    feats[:, 11:14] = mean_off_med
    feats[:, 14] = d_clamp
    feats[:, 15] = d_load

    return (
        torch.tensor(c_np, dtype=dtype, device=device).unsqueeze(0),
        torch.tensor(n_np, dtype=dtype, device=device).unsqueeze(0),
        torch.tensor(a_np, dtype=dtype, device=device).unsqueeze(0),
        torch.tensor(feats, dtype=dtype, device=device).unsqueeze(0),
    )


def pad_and_stack_surface_3d_batch(
    items: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad heterogeneous ``(1, M_g, ...)`` surface items into a masked ``(B, M_max, ...)`` batch."""
    if not items:
        raise ValueError("items must be non-empty")
    max_m = max(int(c.shape[1]) for c, _, _, _ in items)
    b = len(items)
    dtype = items[0][0].dtype
    device = items[0][0].device
    f_dim = int(items[0][3].shape[-1])

    c_out = torch.zeros((b, max_m, 3), dtype=dtype, device=device)
    n_out = torch.zeros((b, max_m, 3), dtype=dtype, device=device)
    a_out = torch.zeros((b, max_m), dtype=dtype, device=device)
    f_out = torch.zeros((b, max_m, f_dim), dtype=dtype, device=device)
    mask = torch.zeros((b, max_m), dtype=dtype, device=device)

    for i, (c, n, a, f) in enumerate(items):
        m_i = int(c.shape[1])
        c_out[i, :m_i] = c[0]
        n_out[i, :m_i] = n[0]
        a_out[i, :m_i] = a[0]
        f_out[i, :m_i] = f[0]
        mask[i, :m_i] = 1.0
    return c_out, n_out, a_out, f_out, mask


@dataclass(frozen=True, slots=True)
class GeometryOODReport3D:
    """Auditable 3D geometry OOD & topology admissibility verdict."""

    mesh_id: str
    status: GeometryOOD3DStatus
    mahalanobis_distance: float
    threshold: float
    expected_euler_characteristic: int
    observed_euler_characteristic: int
    reason: str

    @property
    def accepted(self) -> bool:
        return self.status is GeometryOOD3DStatus.IN_DISTRIBUTION

    def document(self) -> dict[str, Any]:
        return {
            "schema": GEOMETRY_OOD_3D_SCHEMA,
            "mesh_id": self.mesh_id,
            "status": self.status.value,
            "accepted": self.accepted,
            "mahalanobis_distance": float(self.mahalanobis_distance),
            "threshold": float(self.threshold),
            "expected_euler_characteristic": self.expected_euler_characteristic,
            "observed_euler_characteristic": self.observed_euler_characteristic,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class GeometryOODDetector3D:
    """``TRAIN``-fitted 3D geometric + latent slice Mahalanobis OOD gate."""

    mean: tuple[float, ...]
    inv_cov: tuple[tuple[float, ...], ...]
    threshold: float
    expected_euler_characteristic: int
    train_count: int
    feature_dim: int

    @classmethod
    def fit(
        cls,
        descriptors: Sequence[GeometryDescriptor3D],
        *,
        latent_vectors: Sequence[Sequence[float]] | None = None,
        threshold_multiplier: float = 1.6,
        min_threshold: float = 3.5,
        ridge: float = 1.0e-3,
    ) -> GeometryOODDetector3D:
        if len(descriptors) < 2:
            raise ValueError("GeometryOODDetector3D.fit requires >= 2 TRAIN geometries")
        eulers = {d.surface_euler_characteristic for d in descriptors}
        if len(eulers) != 1:
            raise ValueError(
                f"TRAIN 3D geometries must share Euler characteristic, got {sorted(eulers)}"
            )
        rows: list[list[float]] = []
        for idx, desc in enumerate(descriptors):
            vec = list(desc.vector())
            if latent_vectors is not None:
                vec.extend(float(v) for v in latent_vectors[idx])
            rows.append(vec)
        mat = np.asarray(rows, dtype=np.float64)
        n_samples, dim = mat.shape
        mu = mat.mean(axis=0)
        # Floor std relative to feature magnitude so fixed dimensions tolerate mesh-float noise
        std_floor = np.maximum(1.0e-2 * np.abs(mu), 1.0e-3)
        std = np.maximum(mat.std(axis=0), std_floor)
        z = (mat - mu[None, :]) / std[None, :]
        cov = np.cov(z, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=np.float64)
        # Shrink sample covariance toward identity in small-sample (N <= D) regimes
        shrink = max(float(ridge), min(0.85, float(dim) / float(dim + n_samples)))
        cov_reg = (1.0 - shrink) * cov + shrink * np.eye(dim, dtype=np.float64)
        inv_z = np.linalg.inv(cov_reg)
        # Dimension-normalized precision matrix: E[d_in] ~ O(1) regardless of D
        inv_std = np.diag(1.0 / std)
        precision = (inv_std @ inv_z @ inv_std) / float(dim)
        precision = 0.5 * (precision + precision.T)

        train_dists = [
            math.sqrt(max(float((row - mu) @ precision @ (row - mu)), 0.0))
            for row in mat
        ]
        thresh = max(
            float(min_threshold), float(threshold_multiplier) * max(train_dists)
        )
        return cls(
            mean=tuple(float(v) for v in mu),
            inv_cov=tuple(tuple(float(x) for x in r) for r in precision),
            threshold=thresh,
            expected_euler_characteristic=next(iter(eulers)),
            train_count=len(descriptors),
            feature_dim=int(mat.shape[1]),
        )

    def score(
        self,
        descriptor: GeometryDescriptor3D,
        *,
        latent_vector: Sequence[float] | None = None,
    ) -> GeometryOODReport3D:
        if (
            descriptor.surface_euler_characteristic
            != self.expected_euler_characteristic
        ):
            return GeometryOODReport3D(
                mesh_id=descriptor.mesh_id,
                status=GeometryOOD3DStatus.TOPOLOGY_REFUSED,
                mahalanobis_distance=float("inf"),
                threshold=self.threshold,
                expected_euler_characteristic=self.expected_euler_characteristic,
                observed_euler_characteristic=descriptor.surface_euler_characteristic,
                reason=(
                    f"3D surface Euler characteristic mismatch: expected "
                    f"{self.expected_euler_characteristic}, got "
                    f"{descriptor.surface_euler_characteristic}"
                ),
            )
        vec = list(descriptor.vector())
        if latent_vector is not None:
            vec.extend(float(v) for v in latent_vector)
        if len(vec) != self.feature_dim:
            raise ValueError(
                f"3D OOD vector dim {len(vec)} != fitted dim {self.feature_dim}"
            )
        arr = np.asarray(vec, dtype=np.float64)
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
                expected_euler_characteristic=self.expected_euler_characteristic,
                observed_euler_characteristic=descriptor.surface_euler_characteristic,
                reason=f"Mahalanobis distance {dist:.3f} > threshold {self.threshold:.3f}",
            )
        return GeometryOODReport3D(
            mesh_id=descriptor.mesh_id,
            status=GeometryOOD3DStatus.IN_DISTRIBUTION,
            mahalanobis_distance=dist,
            threshold=self.threshold,
            expected_euler_characteristic=self.expected_euler_characteristic,
            observed_euler_characteristic=descriptor.surface_euler_characteristic,
            reason="in-distribution",
        )

    @property
    def detector_hash(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()

    def document(self) -> dict[str, Any]:
        return {
            "schema": GEOMETRY_OOD_3D_SCHEMA,
            "mean": list(self.mean),
            "inv_cov": [list(r) for r in self.inv_cov],
            "threshold": float(self.threshold),
            "expected_euler_characteristic": int(self.expected_euler_characteristic),
            "train_count": int(self.train_count),
            "feature_dim": int(self.feature_dim),
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> GeometryOODDetector3D:
        return cls(
            mean=tuple(float(v) for v in doc["mean"]),
            inv_cov=tuple(tuple(float(x) for x in r) for r in doc["inv_cov"]),
            threshold=float(doc["threshold"]),
            expected_euler_characteristic=int(doc["expected_euler_characteristic"]),
            train_count=int(doc["train_count"]),
            feature_dim=int(doc["feature_dim"]),
        )


__all__ = (
    "GEOMETRY_DESCRIPTOR_3D_FIELDS",
    "GEOMETRY_OOD_3D_SCHEMA",
    "SURFACE_3D_GALE_FEATURE_DIM",
    "GeometryDescriptor3D",
    "GeometryOODDetector3D",
    "GeometryOODReport3D",
    "GeometryOOD3DStatus",
    "build_gale_surface_3d_tensors",
    "compute_geometry_descriptor_3d",
    "compute_surface_euler_characteristic",
    "pad_and_stack_surface_3d_batch",
)
