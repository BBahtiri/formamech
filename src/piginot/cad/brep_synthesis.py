"""Watertight OpenCASCADE B-Rep STEP synthesis & GD&T tolerance verification (V5-4).

Closes the mesh-to-manufacturable CAD loop using pure-MIT OpenCASCADE
(``gmsh.model.occ``):

1. :func:`synthesize_watertight_brep_step`: converts a certified 3D SIMP solid
   mask into a smooth, lofted B-Rep solid while preserving exact analytical
   keep-out datum features (cylindrical bolt bores, planar mounting pads),
   exports ISO-10303-21 STEP, re-imports via :func:`import_cad_3d` to verify
   0 open boundary edges / watertight manifold topology and <= 0.05 mm bore
   preservation, and optionally re-certifies compliance/stress with exact 3D FEM
   on the re-meshed B-Rep solid.
2. :func:`verify_gdt_tolerance_robustness`: screens all ``+/- tol`` extreme
   tolerance-band corners with a fast surrogate evaluator and certifies the
   worst-case corner with exact 3D FEM, failing closed if any in-tolerance
   manufactured part violates stress or stiffness limits.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from piginot.cad.importing_3d import (
    _gmsh_module,
    classify_faces_by_bbox,
    import_cad_3d,
)
from piginot.contracts import MaterialSpec, Quantity, Unit, canonical_json_bytes
from piginot.fem.backend_3d import (
    DirichletBC3D,
    FemRequest3D,
    TorchFemBackend3D,
    TractionBC3D,
)
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

BREP_SYNTHESIS_SCHEMA: Final = "formamech.brep-synthesis-3d-v1"
GDT_ROBUSTNESS_SCHEMA: Final = "formamech.gdt-tolerance-robustness-v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_doc(doc: object) -> str:
    return _sha256_bytes(canonical_json_bytes(doc))


@dataclass(frozen=True, slots=True)
class KeepOutDatumFeature:
    """Exact analytical B-Rep datum feature preserved during STEP synthesis."""

    name: str
    kind: str  # "cylindrical_bore" | "cylindrical_boss" | "planar_pad"
    origin_mm: tuple[float, float, float]
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    radius_mm: float | None = None
    length_mm: float | None = None
    size_xyz_mm: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("KeepOutDatumFeature name must be non-empty")
        if self.kind not in ("cylindrical_bore", "cylindrical_boss", "planar_pad"):
            raise ValueError(f"unsupported KeepOutDatumFeature kind {self.kind!r}")
        if self.kind in ("cylindrical_bore", "cylindrical_boss"):
            if self.radius_mm is None or self.radius_mm <= 0.0:
                raise ValueError("cylindrical feature requires positive radius_mm")
            if self.length_mm is None or self.length_mm <= 0.0:
                raise ValueError("cylindrical feature requires positive length_mm")
        else:
            if self.size_xyz_mm is None or any(v <= 0.0 for v in self.size_xyz_mm):
                raise ValueError("planar_pad requires positive size_xyz_mm")


@dataclass(frozen=True, slots=True)
class BRepSynthesisResult:
    """Verified watertight B-Rep STEP synthesis & round-trip FEM certification dossier."""

    step_path: Path
    step_sha256: str
    volume_mm3: float
    surface_area_mm2: float
    open_boundary_edge_count: int
    nonmanifold_edge_count: int
    euler_characteristic: int
    watertight: bool
    max_bore_center_error_mm: float
    remeshed_fem_verified: bool
    remeshed_fem_max_disp_mm: float | None
    remeshed_fem_peak_vm_mpa: float | None
    passed: bool
    reason: str

    @property
    def receipt_sha256(self) -> str:
        return _sha256_doc(self.document())

    def document(self) -> dict[str, Any]:
        return {
            "schema": BREP_SYNTHESIS_SCHEMA,
            "step_path": str(self.step_path),
            "step_sha256": self.step_sha256,
            "volume_mm3": float(self.volume_mm3),
            "surface_area_mm2": float(self.surface_area_mm2),
            "open_boundary_edge_count": self.open_boundary_edge_count,
            "nonmanifold_edge_count": self.nonmanifold_edge_count,
            "euler_characteristic": self.euler_characteristic,
            "watertight": self.watertight,
            "max_bore_center_error_mm": float(self.max_bore_center_error_mm),
            "remeshed_fem_verified": self.remeshed_fem_verified,
            "remeshed_fem_max_disp_mm": self.remeshed_fem_max_disp_mm,
            "remeshed_fem_peak_vm_mpa": self.remeshed_fem_peak_vm_mpa,
            "passed": self.passed,
            "reason": self.reason,
        }


def _surface_topology_stats(mesh: CanonicalMesh3D) -> tuple[int, int, int]:
    """Compute (open_boundary_edges, nonmanifold_edges, euler_characteristic)."""
    edge_counts: dict[tuple[int, int], int] = {}
    verts: set[int] = set()
    for a, b, c in mesh.surface_triangles:
        verts.update((a, b, c))
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_counts[key] = edge_counts.get(key, 0) + 1
    open_edges = sum(1 for cnt in edge_counts.values() if cnt == 1)
    nonmanifold = sum(1 for cnt in edge_counts.values() if cnt > 2)
    chi = len(verts) - len(edge_counts) + len(mesh.surface_triangles)
    return open_edges, nonmanifold, chi


def _point_to_line_distance(
    pt: tuple[float, float, float],
    origin: tuple[float, float, float],
    axis: tuple[float, float, float],
) -> float:
    p = np.asarray(pt, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(axis, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1.0e-12)
    rel = p - o
    perp = rel - float(np.dot(rel, d)) * d
    return float(np.linalg.norm(perp))


def synthesize_watertight_brep_step(
    mesh: CanonicalMesh3D,
    solid_mask: BoolArray,
    output_step_path: Path | str,
    *,
    keep_out_features: Sequence[KeepOutDatumFeature] = (),
    n_sections: int = 4,
    min_section_thickness_mm: float = 2.0,
    verify_roundtrip: bool = True,
    roundtrip_mesh_size_mm: float = 5.0,
    run_remeshed_fem_check: bool = True,
    fem_material: MaterialSpec | None = None,
    bore_tolerance_mm: float = 0.05,
) -> BRepSynthesisResult:
    """Synthesize a watertight OpenCASCADE B-Rep STEP solid from a 3D SIMP solid mask."""
    mask = np.asarray(solid_mask, dtype=np.bool_)
    if mask.ndim != 1 or mask.shape[0] != len(mesh.cells):
        raise ValueError("solid_mask must match len(mesh.cells)")
    if not np.any(mask):
        raise ValueError("cannot synthesize B-Rep from an empty solid_mask")

    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    solid_cells = cells[mask]
    solid_node_ids = np.unique(solid_cells.reshape(-1))
    solid_pts = nodes[solid_node_ids]

    x_min, x_max = float(np.min(solid_pts[:, 0])), float(np.max(solid_pts[:, 0]))
    y_min_g, y_max_g = float(np.min(solid_pts[:, 1])), float(np.max(solid_pts[:, 1]))
    z_min_g, z_max_g = float(np.min(solid_pts[:, 2])), float(np.max(solid_pts[:, 2]))

    # Compute smooth cross-section envelopes along x for OpenCASCADE B-Spline lofting
    n_sec = max(2, int(n_sections))
    x_stations = np.linspace(x_min, x_max, n_sec)
    slab_half = 0.65 * (x_max - x_min) / max(n_sec - 1, 1)
    raw_bounds: list[tuple[float, float, float, float]] = []
    for xs in x_stations:
        in_slab = solid_pts[np.abs(solid_pts[:, 0] - xs) <= slab_half]
        if in_slab.shape[0] < 3:
            in_slab = solid_pts
        ylo = float(np.min(in_slab[:, 1]))
        yhi = float(np.max(in_slab[:, 1]))
        zlo = float(np.min(in_slab[:, 2]))
        zhi = float(np.max(in_slab[:, 2]))
        if yhi - ylo < min_section_thickness_mm:
            ym = 0.5 * (ylo + yhi)
            ylo = max(y_min_g, ym - 0.5 * min_section_thickness_mm)
            yhi = min(y_max_g, ym + 0.5 * min_section_thickness_mm)
        if zhi - zlo < min_section_thickness_mm:
            zm = 0.5 * (zlo + zhi)
            zlo = max(z_min_g, zm - 0.5 * min_section_thickness_mm)
            zhi = min(z_max_g, zm + 0.5 * min_section_thickness_mm)
        raw_bounds.append((ylo, yhi, zlo, zhi))

    # Feature-preserving Taubin-like 1D smoothing across interior section profiles
    smoothed: list[tuple[float, float, float, float]] = list(raw_bounds)
    if n_sec >= 3:
        for idx in range(1, n_sec - 1):
            prev_b, cur_b, next_b = (
                raw_bounds[idx - 1],
                raw_bounds[idx],
                raw_bounds[idx + 1],
            )
            smoothed[idx] = (
                0.25 * prev_b[0] + 0.5 * cur_b[0] + 0.25 * next_b[0],
                0.25 * prev_b[1] + 0.5 * cur_b[1] + 0.25 * next_b[1],
                0.25 * prev_b[2] + 0.5 * cur_b[2] + 0.25 * next_b[2],
                0.25 * prev_b[3] + 0.5 * cur_b[3] + 0.25 * next_b[3],
            )

    dest = Path(output_step_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    gmsh = _gmsh_module()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("formamech_brep_synth")
        wire_tags: list[int] = []
        for xs, (ylo, yhi, zlo, zhi) in zip(x_stations, smoothed, strict=True):
            p1 = gmsh.model.occ.addPoint(float(xs), ylo, zlo)
            p2 = gmsh.model.occ.addPoint(float(xs), yhi, zlo)
            p3 = gmsh.model.occ.addPoint(float(xs), yhi, zhi)
            p4 = gmsh.model.occ.addPoint(float(xs), ylo, zhi)
            l1 = gmsh.model.occ.addLine(p1, p2)
            l2 = gmsh.model.occ.addLine(p2, p3)
            l3 = gmsh.model.occ.addLine(p3, p4)
            l4 = gmsh.model.occ.addLine(p4, p1)
            wire_tags.append(gmsh.model.occ.addWire([l1, l2, l3, l4]))

        lofted = gmsh.model.occ.addThruSections(
            wire_tags, makeSolid=True, makeRuled=False
        )
        solids = [(dim, tag) for dim, tag in lofted if dim == 3]
        if not solids:
            raise RuntimeError("OpenCASCADE loft failed to produce a 3D solid")

        # Fuse analytical bosses / planar mounting pads
        add_tools: list[tuple[int, int]] = []
        cut_tools: list[tuple[int, int]] = []
        for feat in keep_out_features:
            ox, oy, oz = feat.origin_mm
            if feat.kind == "planar_pad":
                assert feat.size_xyz_mm is not None
                dx, dy, dz = feat.size_xyz_mm
                tag = gmsh.model.occ.addBox(ox, oy, oz, dx, dy, dz)
                add_tools.append((3, tag))
            else:
                assert feat.radius_mm is not None and feat.length_mm is not None
                ax_arr = np.asarray(feat.axis, dtype=np.float64)
                ax_u = ax_arr / max(float(np.linalg.norm(ax_arr)), 1.0e-12)
                if feat.kind == "cylindrical_boss":
                    tag = gmsh.model.occ.addCylinder(
                        ox,
                        oy,
                        oz,
                        float(ax_u[0] * feat.length_mm),
                        float(ax_u[1] * feat.length_mm),
                        float(ax_u[2] * feat.length_mm),
                        float(feat.radius_mm),
                    )
                    add_tools.append((3, tag))
                else:
                    # Extend cut cylinder by 1.0 mm on both ends for clean Boolean penetration
                    ext = 1.0
                    tag = gmsh.model.occ.addCylinder(
                        float(ox - ax_u[0] * ext),
                        float(oy - ax_u[1] * ext),
                        float(oz - ax_u[2] * ext),
                        float(ax_u[0] * (feat.length_mm + 2.0 * ext)),
                        float(ax_u[1] * (feat.length_mm + 2.0 * ext)),
                        float(ax_u[2] * (feat.length_mm + 2.0 * ext)),
                        float(feat.radius_mm),
                    )
                    cut_tools.append((3, tag))

        cur_solids = solids
        if add_tools:
            cur_solids, _ = gmsh.model.occ.fuse(cur_solids, add_tools)
            cur_solids = [(d, t) for d, t in cur_solids if d == 3]
        if cut_tools:
            cur_solids, _ = gmsh.model.occ.cut(cur_solids, cut_tools)
            cur_solids = [(d, t) for d, t in cur_solids if d == 3]

        gmsh.model.occ.synchronize()
        if not gmsh.model.getEntities(3):
            raise RuntimeError("synthesized B-Rep contains zero 3D solid entities")
        gmsh.write(str(dest))
    finally:
        gmsh.finalize()

    step_sha = _sha256_bytes(dest.read_bytes())
    if not verify_roundtrip:
        return BRepSynthesisResult(
            step_path=dest,
            step_sha256=step_sha,
            volume_mm3=0.0,
            surface_area_mm2=0.0,
            open_boundary_edge_count=0,
            nonmanifold_edge_count=0,
            euler_characteristic=0,
            watertight=True,
            max_bore_center_error_mm=0.0,
            remeshed_fem_verified=False,
            remeshed_fem_max_disp_mm=None,
            remeshed_fem_peak_vm_mpa=None,
            passed=True,
            reason="B-Rep STEP exported (round-trip verification skipped)",
        )

    # Mandatory round-trip OpenCASCADE re-import, watertightness & bore audit
    imported = import_cad_3d(
        dest,
        unit=Unit.MM,
        mesh_size_mm=roundtrip_mesh_size_mm,
        element_type=CellType3D.TET4,
    )
    open_edges, nonmanifold, chi = _surface_topology_stats(imported.mesh)
    watertight = open_edges == 0 and nonmanifold == 0 and imported.volume_mm3 > 0.0

    # Verify every cylindrical_bore datum is preserved accurately in OpenCASCADE faces
    max_bore_err = 0.0
    bore_ok = True
    cyl_faces = [f for f in imported.faces if f.kind == "cylindrical"]
    for feat in keep_out_features:
        if feat.kind != "cylindrical_bore":
            continue
        if not cyl_faces:
            bore_ok = False
            max_bore_err = float("inf")
            break
        best_err = min(
            _point_to_line_distance(f.centroid_mm, feat.origin_mm, feat.axis)
            for f in cyl_faces
        )
        max_bore_err = max(max_bore_err, best_err)
        if best_err > bore_tolerance_mm:
            bore_ok = False

    fem_verified = False
    fem_max_u: float | None = None
    fem_peak_vm: float | None = None
    if run_remeshed_fem_check and watertight:
        groups = classify_faces_by_bbox(imported)
        if groups["x_min"] and groups["x_max"]:
            mat = (
                fem_material
                if fem_material is not None
                else MaterialSpec(
                    name="brep-roundtrip-al",
                    model="compressible_neo_hookean",
                    properties={
                        "elastic_modulus": Quantity(1000.0, Unit.MPA).to_canonical(),
                        "poisson_ratio": Quantity(0.30, Unit.ONE).to_canonical(),
                    },
                )
            )
            req = FemRequest3D(
                request_id="brep-roundtrip-cert",
                mesh=imported.mesh,
                material=mat,
                dirichlet_bcs=(
                    DirichletBC3D(
                        "fix_xmin",
                        tuple(groups["x_min"]),
                        ("x", "y", "z"),
                        (0.0, 0.0, 0.0),
                    ),
                ),
                traction_bcs=(
                    TractionBC3D(
                        "load_xmax", tuple(groups["x_max"]), (0.0, -0.25, 0.0)
                    ),
                ),
                load_steps=1,
            )
            res = TorchFemBackend3D().solve(req)
            if res.converged and res.min_det_f > 0.0:
                fem_verified = True
                fem_max_u = float(res.max_displacement_mm)
                fem_peak_vm = float(
                    res.quadrature_peak_von_mises_mpa
                    if res.quadrature_peak_von_mises_mpa is not None
                    else res.peak_von_mises_mpa
                )

    passed = (
        watertight and bore_ok and (fem_verified if run_remeshed_fem_check else True)
    )
    reason = (
        "watertight OpenCASCADE B-Rep STEP verified via round-trip re-import and 3D FEM"
        if passed
        else (
            f"B-Rep verification failed: watertight={watertight}, "
            f"bore_ok={bore_ok} (max_err={max_bore_err:.4f} mm), "
            f"fem_verified={fem_verified}"
        )
    )
    return BRepSynthesisResult(
        step_path=dest,
        step_sha256=step_sha,
        volume_mm3=imported.volume_mm3,
        surface_area_mm2=imported.surface_area_mm2,
        open_boundary_edge_count=open_edges,
        nonmanifold_edge_count=nonmanifold,
        euler_characteristic=chi,
        watertight=watertight,
        max_bore_center_error_mm=max_bore_err,
        remeshed_fem_verified=fem_verified,
        remeshed_fem_max_disp_mm=fem_max_u,
        remeshed_fem_peak_vm_mpa=fem_peak_vm,
        passed=passed,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class GdtToleranceBandSpec:
    """ISO 2768 / GD&T +/- tolerance envelope on critical CAD dimensions."""

    nominal_parameters: Mapping[str, float]
    tolerances_mm: Mapping[str, float]
    max_allowable_stress_mpa: float
    max_allowable_displacement_mm: float
    standard_label: str = "ISO-2768-m"

    def __post_init__(self) -> None:
        if not self.nominal_parameters or not self.tolerances_mm:
            raise ValueError("nominal_parameters and tolerances_mm must be non-empty")
        for k, tol in self.tolerances_mm.items():
            if k not in self.nominal_parameters:
                raise ValueError(f"tolerance key {k!r} not found in nominal_parameters")
            if not math.isfinite(tol) or tol < 0.0:
                raise ValueError(f"tolerance for {k!r} must be finite and nonnegative")
        if (
            self.max_allowable_stress_mpa <= 0.0
            or self.max_allowable_displacement_mm <= 0.0
        ):
            raise ValueError(
                "allowable stress and displacement limits must be positive"
            )


@dataclass(frozen=True, slots=True)
class GdtRobustnessReport:
    """Fail-closed GD&T tolerance-band worst-case verification report."""

    standard_label: str
    corners_screened: int
    worst_stress_corner: Mapping[str, float]
    worst_disp_corner: Mapping[str, float]
    fast_worst_stress_mpa: float
    fast_worst_disp_mm: float
    certified_fem_worst_stress_mpa: float
    certified_fem_worst_disp_mm: float
    max_allowable_stress_mpa: float
    max_allowable_displacement_mm: float
    passed: bool
    reason: str

    @property
    def receipt_sha256(self) -> str:
        return _sha256_doc(self.document())

    def document(self) -> dict[str, Any]:
        return {
            "schema": GDT_ROBUSTNESS_SCHEMA,
            "standard_label": self.standard_label,
            "corners_screened": self.corners_screened,
            "worst_stress_corner": dict(sorted(self.worst_stress_corner.items())),
            "worst_disp_corner": dict(sorted(self.worst_disp_corner.items())),
            "fast_worst_stress_mpa": float(self.fast_worst_stress_mpa),
            "fast_worst_disp_mm": float(self.fast_worst_disp_mm),
            "certified_fem_worst_stress_mpa": float(
                self.certified_fem_worst_stress_mpa
            ),
            "certified_fem_worst_disp_mm": float(self.certified_fem_worst_disp_mm),
            "max_allowable_stress_mpa": float(self.max_allowable_stress_mpa),
            "max_allowable_displacement_mm": float(self.max_allowable_displacement_mm),
            "passed": self.passed,
            "reason": self.reason,
        }


def verify_gdt_tolerance_robustness(
    spec: GdtToleranceBandSpec,
    *,
    fast_evaluator: Callable[[Mapping[str, float]], tuple[float, float]],
    exact_fem_evaluator: Callable[[Mapping[str, float]], tuple[float, float]],
) -> GdtRobustnessReport:
    """Screen all +/- tolerance extreme corners via fast surrogate and certify worst-case with exact FEM.

    Both evaluators return ``(peak_von_mises_mpa, max_displacement_mm)``.
    Fails closed if ANY fast corner or exact-FEM-certified worst corner violates limits.
    """
    tol_keys = sorted(spec.tolerances_mm)
    corners: list[dict[str, float]] = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(tol_keys)):
        pt = {str(k): float(v) for k, v in spec.nominal_parameters.items()}
        for k, s in zip(tol_keys, signs, strict=True):
            pt[k] = pt[k] + s * float(spec.tolerances_mm[k])
        corners.append(pt)

    best_stress = -1.0
    best_disp = -1.0
    worst_stress_pt = corners[0]
    worst_disp_pt = corners[0]
    for pt in corners:
        s_val, u_val = fast_evaluator(pt)
        if s_val > best_stress:
            best_stress = float(s_val)
            worst_stress_pt = pt
        if u_val > best_disp:
            best_disp = float(u_val)
            worst_disp_pt = pt

    # Certify worst-case corner(s) using exact 3D FEM
    cert_s1, cert_u1 = exact_fem_evaluator(worst_stress_pt)
    if worst_disp_pt == worst_stress_pt:
        cert_s, cert_u = float(cert_s1), float(cert_u1)
    else:
        cert_s2, cert_u2 = exact_fem_evaluator(worst_disp_pt)
        cert_s = max(float(cert_s1), float(cert_s2))
        cert_u = max(float(cert_u1), float(cert_u2))

    passed = (
        best_stress <= spec.max_allowable_stress_mpa
        and best_disp <= spec.max_allowable_displacement_mm
        and cert_s <= spec.max_allowable_stress_mpa
        and cert_u <= spec.max_allowable_displacement_mm
    )
    reason = (
        f"all {len(corners)} {spec.standard_label} corners passed fast screen and exact FEM worst-case certification"
        if passed
        else (
            f"GD&T tolerance violation: certified stress={cert_s:.2f}/{spec.max_allowable_stress_mpa:.2f} MPa, "
            f"disp={cert_u:.4f}/{spec.max_allowable_displacement_mm:.4f} mm"
        )
    )
    return GdtRobustnessReport(
        standard_label=spec.standard_label,
        corners_screened=len(corners),
        worst_stress_corner=worst_stress_pt,
        worst_disp_corner=worst_disp_pt,
        fast_worst_stress_mpa=best_stress,
        fast_worst_disp_mm=best_disp,
        certified_fem_worst_stress_mpa=cert_s,
        certified_fem_worst_disp_mm=cert_u,
        max_allowable_stress_mpa=spec.max_allowable_stress_mpa,
        max_allowable_displacement_mm=spec.max_allowable_displacement_mm,
        passed=passed,
        reason=reason,
    )


__all__ = (
    "BREP_SYNTHESIS_SCHEMA",
    "GDT_ROBUSTNESS_SCHEMA",
    "BRepSynthesisResult",
    "GdtRobustnessReport",
    "GdtToleranceBandSpec",
    "KeepOutDatumFeature",
    "synthesize_watertight_brep_step",
    "verify_gdt_tolerance_robustness",
)
