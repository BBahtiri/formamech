"""Physics-regime transition guards for 3D neural surrogates (V5-1).

Prevents silent catastrophic extrapolations when a query stays geometrically
close to training parts yet crosses a physical regime boundary:

1. **Spatial BC-Layout Shift Guard** (AeroZoo failure mode): detects shifted
   clamped/loaded patch locations, orientations, or lever arms even when total
   loaded area fraction is unchanged.
2. **Elastic-to-Plastic Yield-Onset Guard:** flags when an elastic surrogate's
   unsmoothed element peak von Mises stress crosses ``safety_factor * sigma_y``.
3. **Buckling / Tangent Coercivity Proximity Guard:** flags when linearized
   buckling margin ``lambda_crit < min_buckling_margin`` or tangent stiffness
   loses positive definiteness (catching sudden structural bifurcations).
4. **Contact-Gap Closure & Jacobian Folding Guard:** flags predicted closure
   between declared clearance face pairs or ``min det(F) <= floor``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from piginot.contracts import MaterialSpec, canonical_json_bytes
from piginot.fem.backend_3d import DirichletBC3D, TractionBC3D, face_node_indices_3d
from piginot.geometry.mesh3d import CanonicalMesh3D

FloatArray = NDArray[np.float64]

REGIME_GUARD_3D_SCHEMA: Final = "formamech.regime-guard-3d-v1"
BC_LAYOUT_OOD_SCHEMA: Final = "formamech.bc-layout-ood-3d-v1"


def _sha256_doc(doc: object) -> str:
    return hashlib.sha256(canonical_json_bytes(doc)).hexdigest()


class RegimeFlag(StrEnum):
    """Canonical physical regime-transition triggers requiring FEM fallback or refusal."""

    NONFINITE_FIELD = "NONFINITE_FIELD"
    JACOBIAN_FOLDED = "JACOBIAN_FOLDED"
    BC_LAYOUT_SHIFT = "BC_LAYOUT_SHIFT"
    YIELD_REGIME_CROSSED = "YIELD_REGIME_CROSSED"
    BUCKLING_PROXIMITY = "BUCKLING_PROXIMITY"
    CONTACT_GAP_CLOSED = "CONTACT_GAP_CLOSED"


@dataclass(frozen=True, slots=True)
class BcLayoutDescriptor3D:
    """Spatial moments of clamped and loaded boundary patches on a 3D solid."""

    mesh_id: str
    clamped_centroid_norm: tuple[float, float, float]
    loaded_centroid_norm: tuple[float, float, float]
    lever_vector_norm: tuple[float, float, float]
    clamped_mean_normal: tuple[float, float, float]
    loaded_mean_normal: tuple[float, float, float]
    clamped_area_fraction: float
    loaded_area_fraction: float
    traction_direction: tuple[float, float, float]

    def vector(self) -> tuple[float, ...]:
        return (
            *self.clamped_centroid_norm,
            *self.loaded_centroid_norm,
            *self.lever_vector_norm,
            *self.clamped_mean_normal,
            *self.loaded_mean_normal,
            self.clamped_area_fraction,
            self.loaded_area_fraction,
            *self.traction_direction,
        )

    def document(self) -> dict[str, Any]:
        return {
            "mesh_id": self.mesh_id,
            "clamped_centroid_norm": list(self.clamped_centroid_norm),
            "loaded_centroid_norm": list(self.loaded_centroid_norm),
            "lever_vector_norm": list(self.lever_vector_norm),
            "clamped_mean_normal": list(self.clamped_mean_normal),
            "loaded_mean_normal": list(self.loaded_mean_normal),
            "clamped_area_fraction": float(self.clamped_area_fraction),
            "loaded_area_fraction": float(self.loaded_area_fraction),
            "traction_direction": list(self.traction_direction),
        }


def compute_bc_layout_descriptor_3d(
    mesh: CanonicalMesh3D,
    *,
    clamped_face_tags: Sequence[str],
    loaded_face_tags: Sequence[str],
    traction_mpa: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> BcLayoutDescriptor3D:
    """Compute bbox-normalized spatial moments of clamped & loaded face patches."""
    centroids_t, normals_t, areas_t = mesh.surface_geometry()
    c_np = np.asarray(centroids_t, dtype=np.float64)
    n_np = np.asarray(normals_t, dtype=np.float64)
    a_np = np.asarray(areas_t, dtype=np.float64)
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    lo = nodes.min(axis=0)
    hi = nodes.max(axis=0)
    ext = np.maximum(hi - lo, 1.0e-6)
    total_area = max(float(np.sum(a_np)), 1.0e-12)

    def _patch_stats(
        tags: Sequence[str],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
        tids: list[int] = []
        for tag in tags:
            tids.extend(mesh.semantic_face_tags.get(tag, ()))
        if not tids:
            return (0.5, 0.5, 0.5), (0.0, 0.0, 0.0), 0.0
        idx = np.asarray(sorted(set(tids)), dtype=np.int64)
        w = a_np[idx]
        w_sum = max(float(np.sum(w)), 1.0e-12)
        c_mean = np.sum(c_np[idx] * w[:, None], axis=0) / w_sum
        c_norm = (c_mean - lo) / ext
        n_mean = np.sum(n_np[idx] * w[:, None], axis=0) / w_sum
        n_mag = float(np.linalg.norm(n_mean))
        if n_mag > 1.0e-12:
            n_mean = n_mean / n_mag
        return (
            (float(c_norm[0]), float(c_norm[1]), float(c_norm[2])),
            (float(n_mean[0]), float(n_mean[1]), float(n_mean[2])),
            w_sum / total_area,
        )

    c_cent, c_norm, c_frac = _patch_stats(clamped_face_tags)
    l_cent, l_norm, l_frac = _patch_stats(loaded_face_tags)
    lever = (
        l_cent[0] - c_cent[0],
        l_cent[1] - c_cent[1],
        l_cent[2] - c_cent[2],
    )
    t_arr = np.asarray(traction_mpa, dtype=np.float64)
    t_mag = float(np.linalg.norm(t_arr))
    t_dir = (
        (float(t_arr[0] / t_mag), float(t_arr[1] / t_mag), float(t_arr[2] / t_mag))
        if t_mag > 1.0e-12
        else (0.0, 0.0, 0.0)
    )
    return BcLayoutDescriptor3D(
        mesh_id=mesh.mesh_id,
        clamped_centroid_norm=c_cent,
        loaded_centroid_norm=l_cent,
        lever_vector_norm=lever,
        clamped_mean_normal=c_norm,
        loaded_mean_normal=l_norm,
        clamped_area_fraction=c_frac,
        loaded_area_fraction=l_frac,
        traction_direction=t_dir,
    )


@dataclass(frozen=True, slots=True)
class BcLayoutOODDetector3D:
    """Spatial BC-layout OOD detector catching AeroZoo-style load/support shifts."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    threshold: float
    train_count: int

    @classmethod
    def fit(
        cls,
        descriptors: Sequence[BcLayoutDescriptor3D],
        *,
        std_floor: float = 0.05,
        threshold_multiplier: float = 1.5,
        min_threshold: float = 0.75,
    ) -> BcLayoutOODDetector3D:
        if not descriptors:
            raise ValueError("BcLayoutOODDetector3D.fit requires >= 1 descriptor")
        mat = np.asarray([d.vector() for d in descriptors], dtype=np.float64)
        mu = mat.mean(axis=0)
        sd = np.maximum(mat.std(axis=0), float(std_floor))
        dists = np.sqrt(np.mean(np.square((mat - mu[None, :]) / sd[None, :]), axis=1))
        thresh = max(
            float(min_threshold), float(threshold_multiplier) * float(np.max(dists))
        )
        return cls(
            mean=tuple(float(v) for v in mu),
            std=tuple(float(v) for v in sd),
            threshold=thresh,
            train_count=len(descriptors),
        )

    def distance(self, descriptor: BcLayoutDescriptor3D) -> float:
        arr = np.asarray(descriptor.vector(), dtype=np.float64)
        mu = np.asarray(self.mean, dtype=np.float64)
        sd = np.asarray(self.std, dtype=np.float64)
        return float(np.sqrt(np.mean(np.square((arr - mu) / sd))))

    def is_in_distribution(
        self, descriptor: BcLayoutDescriptor3D
    ) -> tuple[bool, float]:
        d = self.distance(descriptor)
        return d <= self.threshold, d

    def document(self) -> dict[str, Any]:
        return {
            "schema": BC_LAYOUT_OOD_SCHEMA,
            "mean": list(self.mean),
            "std": list(self.std),
            "threshold": float(self.threshold),
            "train_count": self.train_count,
        }


@dataclass(frozen=True, slots=True)
class ContactGapPairSpec:
    """Two face tag groups whose deformed minimum distance must stay above clearance."""

    name: str
    face_tags_a: tuple[str, ...]
    face_tags_b: tuple[str, ...]
    min_clearance_mm: float = 0.05


@dataclass(frozen=True, slots=True)
class PhysicsRegimeGuardConfig3D:
    """Thresholds governing 3D physics-regime transition guards."""

    min_det_f_floor: float = 1.0e-5
    yield_stress_mpa: float | None = None
    yield_safety_factor: float = 0.90
    min_buckling_margin: float | None = None
    contact_gap_pairs: tuple[ContactGapPairSpec, ...] = ()
    bc_layout_detector: BcLayoutOODDetector3D | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_det_f_floor) or self.min_det_f_floor <= 0.0:
            raise ValueError("min_det_f_floor must be positive")
        if self.yield_stress_mpa is not None and (
            not math.isfinite(self.yield_stress_mpa) or self.yield_stress_mpa <= 0.0
        ):
            raise ValueError("yield_stress_mpa must be positive when set")
        if not math.isfinite(self.yield_safety_factor) or not (
            0.0 < self.yield_safety_factor <= 1.0
        ):
            raise ValueError("yield_safety_factor must lie in (0, 1]")
        if self.min_buckling_margin is not None and (
            not math.isfinite(self.min_buckling_margin)
            or self.min_buckling_margin <= 0.0
        ):
            raise ValueError("min_buckling_margin must be positive when set")


@dataclass(frozen=True, slots=True)
class RegimeGuardVerdict:
    """Auditable physics-regime guard decision for one 3D surrogate evaluation."""

    passed: bool
    triggered_flags: tuple[RegimeFlag, ...]
    min_det_f: float | None
    bc_layout_distance: float | None
    peak_von_mises_mpa: float | None
    yield_utilization: float | None
    buckling_load_factor: float | None
    min_contact_gap_mm: float | None
    reason: str

    @property
    def verdict_sha256(self) -> str:
        return _sha256_doc(self.document())

    def document(self) -> dict[str, Any]:
        return {
            "schema": REGIME_GUARD_3D_SCHEMA,
            "passed": self.passed,
            "triggered_flags": [f.value for f in self.triggered_flags],
            "min_det_f": self.min_det_f,
            "bc_layout_distance": self.bc_layout_distance,
            "peak_von_mises_mpa": self.peak_von_mises_mpa,
            "yield_utilization": self.yield_utilization,
            "buckling_load_factor": self.buckling_load_factor,
            "min_contact_gap_mm": self.min_contact_gap_mm,
            "reason": self.reason,
        }


class PhysicsRegimeGuard3D:
    """Fast pre- and post-inference physics-regime monitor for 3D surrogates."""

    def __init__(self, config: PhysicsRegimeGuardConfig3D) -> None:
        self.config = config

    def evaluate(
        self,
        mesh: CanonicalMesh3D,
        *,
        nodal_displacement: FloatArray | None = None,
        min_det_f: float | None = None,
        element_peak_von_mises_mpa: FloatArray | float | None = None,
        clamped_face_tags: Sequence[str] = (),
        loaded_face_tags: Sequence[str] = (),
        traction_mpa: tuple[float, float, float] = (0.0, 0.0, 0.0),
        material: MaterialSpec | None = None,
        dirichlet_bcs: Sequence[DirichletBC3D] = (),
        traction_bcs: Sequence[TractionBC3D] = (),
        buckling_load_factor_override: float | None = None,
    ) -> RegimeGuardVerdict:
        """Evaluate all configured physics-regime monitors and return a verdict."""
        flags: list[RegimeFlag] = []
        reasons: list[str] = []

        if nodal_displacement is not None and not np.all(
            np.isfinite(nodal_displacement)
        ):
            flags.append(RegimeFlag.NONFINITE_FIELD)
            reasons.append("non-finite nodal displacement field")

        if min_det_f is not None:
            if not math.isfinite(min_det_f) or min_det_f <= self.config.min_det_f_floor:
                flags.append(RegimeFlag.JACOBIAN_FOLDED)
                reasons.append(
                    f"min_det_f={min_det_f:.3e} <= floor {self.config.min_det_f_floor:.3e}"
                )

        # 1) Spatial BC-layout shift guard (AeroZoo failure mode)
        bc_dist: float | None = None
        if self.config.bc_layout_detector is not None:
            desc = compute_bc_layout_descriptor_3d(
                mesh,
                clamped_face_tags=clamped_face_tags,
                loaded_face_tags=loaded_face_tags,
                traction_mpa=traction_mpa,
            )
            ok_bc, bc_dist = self.config.bc_layout_detector.is_in_distribution(desc)
            if not ok_bc:
                flags.append(RegimeFlag.BC_LAYOUT_SHIFT)
                reasons.append(
                    f"spatial BC-layout distance {bc_dist:.3f} > "
                    f"threshold {self.config.bc_layout_detector.threshold:.3f}"
                )

        # 2) Elastic-to-plastic yield onset guard using unsmoothed hotspot peak
        peak_vm: float | None = None
        yield_util: float | None = None
        if element_peak_von_mises_mpa is not None:
            arr_vm = np.asarray(element_peak_von_mises_mpa, dtype=np.float64)
            if not np.all(np.isfinite(arr_vm)):
                flags.append(RegimeFlag.NONFINITE_FIELD)
                reasons.append("non-finite von Mises stress field")
            elif arr_vm.size > 0:
                peak_vm = float(np.max(arr_vm))
                if self.config.yield_stress_mpa is not None:
                    yield_util = peak_vm / float(self.config.yield_stress_mpa)
                    if yield_util >= self.config.yield_safety_factor:
                        flags.append(RegimeFlag.YIELD_REGIME_CROSSED)
                        reasons.append(
                            f"unsmoothed peak von Mises {peak_vm:.2f} MPa >= "
                            f"{self.config.yield_safety_factor:.2f} * sigma_y "
                            f"({self.config.yield_stress_mpa:.2f} MPa)"
                        )

        # 3) Buckling / bifurcation proximity guard
        buckling_lam: float | None = buckling_load_factor_override
        if self.config.min_buckling_margin is not None:
            if (
                buckling_lam is None
                and material is not None
                and dirichlet_bcs
                and traction_bcs
            ):
                from piginot.fem.buckling_3d import solve_3d_linear_buckling

                b_res = solve_3d_linear_buckling(
                    mesh,
                    material,
                    dirichlet_bcs=dirichlet_bcs,
                    traction_bcs=traction_bcs,
                    num_modes=1,
                )
                if b_res.load_factors:
                    buckling_lam = float(b_res.load_factors[0])
            if buckling_lam is not None and (
                not math.isfinite(buckling_lam)
                or buckling_lam < self.config.min_buckling_margin
            ):
                flags.append(RegimeFlag.BUCKLING_PROXIMITY)
                reasons.append(
                    f"buckling load factor {buckling_lam:.3f} < "
                    f"min_buckling_margin {self.config.min_buckling_margin:.3f}"
                )

        # 4) Contact-gap closure guard on deformed surface pairs
        min_gap: float | None = None
        if (
            self.config.contact_gap_pairs
            and nodal_displacement is not None
            and np.all(np.isfinite(nodal_displacement))
        ):
            def_nodes = np.asarray(mesh.nodes, dtype=np.float64) + np.asarray(
                nodal_displacement, dtype=np.float64
            ).reshape(-1, 3)
            for pair in self.config.contact_gap_pairs:
                idx_a = face_node_indices_3d(mesh, pair.face_tags_a)
                idx_b = face_node_indices_3d(mesh, pair.face_tags_b)
                if not idx_a or not idx_b:
                    continue
                pts_a = def_nodes[list(idx_a)]
                pts_b = def_nodes[list(idx_b)]
                dists = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=-1)
                pair_min = float(np.min(dists))
                min_gap = pair_min if min_gap is None else min(min_gap, pair_min)
                if pair_min <= pair.min_clearance_mm:
                    flags.append(RegimeFlag.CONTACT_GAP_CLOSED)
                    reasons.append(
                        f"contact gap {pair.name!r} closed to {pair_min:.4f} mm "
                        f"<= clearance {pair.min_clearance_mm:.4f} mm"
                    )

        passed = len(flags) == 0
        return RegimeGuardVerdict(
            passed=passed,
            triggered_flags=tuple(flags),
            min_det_f=min_det_f,
            bc_layout_distance=bc_dist,
            peak_von_mises_mpa=peak_vm,
            yield_utilization=yield_util,
            buckling_load_factor=buckling_lam,
            min_contact_gap_mm=min_gap,
            reason="all physics-regime guards passed" if passed else "; ".join(reasons),
        )


__all__ = (
    "BC_LAYOUT_OOD_SCHEMA",
    "REGIME_GUARD_3D_SCHEMA",
    "BcLayoutDescriptor3D",
    "BcLayoutOODDetector3D",
    "ContactGapPairSpec",
    "PhysicsRegimeGuard3D",
    "PhysicsRegimeGuardConfig3D",
    "RegimeFlag",
    "RegimeGuardVerdict",
    "compute_bc_layout_descriptor_3d",
)
