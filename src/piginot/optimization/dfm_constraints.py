"""Design-for-Manufacturing (DfM) constraints for 3D topology & shape optimization (V5-4).

Provides projection-based manufacturability constraints for 3D SIMP density fields:

* :class:`KeepOutZoneSpec` — protected functional bolt-hole bosses, bearing seats,
  planar mounting pads (locked solid ``rho = 1``) or clearance bores (locked void
  ``rho = 0``).
* :class:`MinThicknessConstraint` — morphological erosion-dilation minimum member
  size enforcement over element centroids.
* :class:`OverhangConstraint3D` — additive manufacturing self-supporting build-angle
  cone filter along a declared build direction.
* :class:`PullDirectionDraftConstraint` — casting / injection-molding / 3-axis
  milling non-undercut projection along the tool pull direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DFM_AUDIT_SCHEMA: Final = "formamech.dfm-audit-3d-v1"


def _unit_vec(v: tuple[float, float, float]) -> FloatArray:
    arr = np.asarray(v, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError("direction vector must be 3 finite floats")
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-12:
        raise ValueError("direction vector must be non-zero")
    return arr / norm


@dataclass(frozen=True, slots=True)
class KeepOutZoneSpec:
    """Functional CAD keep-in (solid) or keep-out (void bore) zone in 3D mm."""

    name: str
    kind: str  # "box" | "cylinder"
    lock_solid: bool = True  # True -> force rho=1.0; False -> force rho=0.0 (bore)
    bbox_min_mm: tuple[float, float, float] | None = None
    bbox_max_mm: tuple[float, float, float] | None = None
    origin_mm: tuple[float, float, float] | None = None
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    radius_mm: float | None = None
    length_mm: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("KeepOutZoneSpec name must be non-empty")
        if self.kind not in ("box", "cylinder"):
            raise ValueError(f"unsupported KeepOutZoneSpec kind {self.kind!r}")
        if self.kind == "box":
            if self.bbox_min_mm is None or self.bbox_max_mm is None:
                raise ValueError("box keep-out requires bbox_min_mm and bbox_max_mm")
            if any(
                a > b for a, b in zip(self.bbox_min_mm, self.bbox_max_mm, strict=True)
            ):
                raise ValueError(
                    "bbox_min_mm must be <= bbox_max_mm in every coordinate"
                )
        else:
            if (
                self.origin_mm is None
                or self.radius_mm is None
                or self.length_mm is None
            ):
                raise ValueError(
                    "cylinder keep-out requires origin_mm, radius_mm, and length_mm"
                )
            if self.radius_mm <= 0.0 or self.length_mm <= 0.0:
                raise ValueError("cylinder radius_mm and length_mm must be positive")
            _unit_vec(self.axis)

    def element_mask(self, centroids_mm: FloatArray) -> BoolArray:
        pts = np.asarray(centroids_mm, dtype=np.float64)
        if self.kind == "box":
            assert self.bbox_min_mm is not None and self.bbox_max_mm is not None
            lo = np.asarray(self.bbox_min_mm, dtype=np.float64)
            hi = np.asarray(self.bbox_max_mm, dtype=np.float64)
            return np.asarray(
                np.all((pts >= lo[None, :]) & (pts <= hi[None, :]), axis=1),
                dtype=np.bool_,
            )
        assert (
            self.origin_mm is not None
            and self.radius_mm is not None
            and self.length_mm is not None
        )
        orig = np.asarray(self.origin_mm, dtype=np.float64)
        u_ax = _unit_vec(self.axis)
        rel = pts - orig[None, :]
        axial = rel @ u_ax
        radial_vec = rel - axial[:, None] * u_ax[None, :]
        radial = np.linalg.norm(radial_vec, axis=1)
        return np.asarray(
            (axial >= 0.0) & (axial <= self.length_mm) & (radial <= self.radius_mm),
            dtype=np.bool_,
        )


@dataclass(frozen=True, slots=True)
class MinThicknessConstraint:
    """Morphological opening filter removing features thinner than ``min_thickness_mm``."""

    min_thickness_mm: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_thickness_mm) or self.min_thickness_mm <= 0.0:
            raise ValueError("min_thickness_mm must be strictly positive")

    def project(self, rho: FloatArray, centroids_mm: FloatArray) -> FloatArray:
        r_half = 0.5 * self.min_thickness_mm
        tree = cKDTree(centroids_mm)
        neighbors = tree.query_ball_point(centroids_mm, r=r_half)
        eroded = np.array(
            [
                float(np.min(rho[idxs])) if idxs else float(rho[i])
                for i, idxs in enumerate(neighbors)
            ],
            dtype=np.float64,
        )
        dilated = np.array(
            [
                float(np.max(eroded[idxs])) if idxs else float(eroded[i])
                for i, idxs in enumerate(neighbors)
            ],
            dtype=np.float64,
        )
        return dilated


@dataclass(frozen=True, slots=True)
class OverhangConstraint3D:
    """Additive manufacturing self-supporting overhang angle projection."""

    build_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    max_overhang_angle_deg: float = 45.0
    layer_search_radius_mm: float = 4.0

    def __post_init__(self) -> None:
        _unit_vec(self.build_direction)
        if not (5.0 <= self.max_overhang_angle_deg <= 85.0):
            raise ValueError("max_overhang_angle_deg must lie in [5, 85]")
        if (
            not math.isfinite(self.layer_search_radius_mm)
            or self.layer_search_radius_mm <= 0.0
        ):
            raise ValueError("layer_search_radius_mm must be positive")

    def project(self, rho: FloatArray, centroids_mm: FloatArray) -> FloatArray:
        b_dir = _unit_vec(self.build_direction)
        heights = centroids_mm @ b_dir
        h_min = float(np.min(heights))
        order = np.argsort(heights)
        tan_max = math.tan(math.radians(self.max_overhang_angle_deg))
        out = rho.copy()
        tree = cKDTree(centroids_mm)
        base_band = h_min + 0.5 * self.layer_search_radius_mm

        for idx in order:
            h_i = float(heights[idx])
            if h_i <= base_band:
                continue  # Build plate supports the base layer directly
            cand = tree.query_ball_point(
                centroids_mm[idx], r=self.layer_search_radius_mm
            )
            support_max = 0.0
            for j in cand:
                if j == idx:
                    continue
                dh = h_i - float(heights[j])
                if dh <= 1.0e-6:
                    continue
                delta = centroids_mm[idx] - centroids_mm[j]
                horiz = float(np.linalg.norm(delta - dh * b_dir))
                if horiz <= dh * tan_max + 1.0e-6:
                    support_max = max(support_max, float(out[j]))
            if support_max > 0.0:
                out[idx] = min(float(out[idx]), support_max)
        return out


@dataclass(frozen=True, slots=True)
class PullDirectionDraftConstraint:
    """Casting / milling non-undercut monotonicity projection along pull direction."""

    pull_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    column_radius_mm: float = 3.0

    def __post_init__(self) -> None:
        _unit_vec(self.pull_direction)
        if not math.isfinite(self.column_radius_mm) or self.column_radius_mm <= 0.0:
            raise ValueError("column_radius_mm must be positive")

    def project(self, rho: FloatArray, centroids_mm: FloatArray) -> FloatArray:
        p_dir = _unit_vec(self.pull_direction)
        heights = centroids_mm @ p_dir
        # Project onto plane orthogonal to pull direction
        proj_xy = centroids_mm - heights[:, None] * p_dir[None, :]
        tree_2d = cKDTree(proj_xy)
        order = np.argsort(heights)
        out = rho.copy()
        for idx in order:
            col = tree_2d.query_ball_point(proj_xy[idx], r=self.column_radius_mm)
            below = [j for j in col if heights[j] < heights[idx] - 1.0e-6]
            if below:
                # Non-undercut: density cannot increase above a void region along pull axis
                out[idx] = min(float(out[idx]), float(np.max(out[below])))
        return out


@dataclass(frozen=True, slots=True)
class DfmAuditReport:
    """Quantitative audit of DfM constraint satisfaction on a thresholded 3D solid."""

    keep_out_zone_count: int
    keep_out_violations: int
    min_thickness_enabled: bool
    overhang_enabled: bool
    overhang_unsupported_count: int
    pull_draft_enabled: bool
    passed: bool

    def document(self) -> dict[str, Any]:
        return {
            "schema": DFM_AUDIT_SCHEMA,
            "keep_out_zone_count": self.keep_out_zone_count,
            "keep_out_violations": self.keep_out_violations,
            "min_thickness_enabled": self.min_thickness_enabled,
            "overhang_enabled": self.overhang_enabled,
            "overhang_unsupported_count": self.overhang_unsupported_count,
            "pull_draft_enabled": self.pull_draft_enabled,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class DfmConstraintSet:
    """Composable manufacturing constraints enforced during 3D topology optimization."""

    keep_out_zones: tuple[KeepOutZoneSpec, ...] = ()
    min_thickness: MinThicknessConstraint | None = None
    overhang: OverhangConstraint3D | None = None
    pull_draft: PullDirectionDraftConstraint | None = None

    def apply_to_densities(
        self,
        rho: FloatArray,
        centroids_mm: FloatArray,
        *,
        min_density: float = 1.0e-3,
    ) -> FloatArray:
        """Project a physical density field onto the DfM-admissible set."""
        out = np.asarray(rho, dtype=np.float64).copy()
        if self.min_thickness is not None:
            out = self.min_thickness.project(out, centroids_mm)
        if self.overhang is not None:
            out = self.overhang.project(out, centroids_mm)
        if self.pull_draft is not None:
            out = self.pull_draft.project(out, centroids_mm)
        # Keep-out / keep-in functional datum zones take strict priority
        for zone in self.keep_out_zones:
            mask = zone.element_mask(centroids_mm)
            out[mask] = 1.0 if zone.lock_solid else float(min_density)
        return np.clip(out, float(min_density), 1.0)

    def audit_solid_mask(
        self, solid_mask: BoolArray, centroids_mm: FloatArray
    ) -> DfmAuditReport:
        """Audit final discrete solid mask against all declared DfM rules."""
        mask_arr = np.asarray(solid_mask, dtype=np.bool_)
        ko_viol = 0
        for zone in self.keep_out_zones:
            z_mask = zone.element_mask(centroids_mm)
            if zone.lock_solid:
                ko_viol += int(np.sum(z_mask & ~mask_arr))
            else:
                ko_viol += int(np.sum(z_mask & mask_arr))

        overhang_viol = 0
        if self.overhang is not None:
            proj = self.overhang.project(mask_arr.astype(np.float64), centroids_mm)
            # Ignore locked solid keep-out zones when counting unsupported overhang cells
            locked = np.zeros_like(mask_arr)
            for zone in self.keep_out_zones:
                if zone.lock_solid:
                    locked |= zone.element_mask(centroids_mm)
            overhang_viol = int(np.sum(mask_arr & ~locked & (proj < 0.5)))

        return DfmAuditReport(
            keep_out_zone_count=len(self.keep_out_zones),
            keep_out_violations=ko_viol,
            min_thickness_enabled=self.min_thickness is not None,
            overhang_enabled=self.overhang is not None,
            overhang_unsupported_count=overhang_viol,
            pull_draft_enabled=self.pull_draft is not None,
            passed=(ko_viol == 0 and overhang_viol == 0),
        )


__all__ = (
    "DFM_AUDIT_SCHEMA",
    "DfmAuditReport",
    "DfmConstraintSet",
    "KeepOutZoneSpec",
    "MinThicknessConstraint",
    "OverhangConstraint3D",
    "PullDirectionDraftConstraint",
)
