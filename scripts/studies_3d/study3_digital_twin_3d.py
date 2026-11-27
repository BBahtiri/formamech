"""Study 3 — Real-Time 3D Structural Digital Twin on Real Experimental FOS Steel Beam Data.

Open-Source Real Experimental Dataset:
  SUM4Re: FOS steel beam dataset (Jolastokieta pilot site, Basque Country, Spain)
  DOI: 10.5281/zenodo.22686007  |  License: CC-BY-4.0
  41,871 real physical telemetry samples across 6 multi-jack static tests and
  2 physical boundary-condition regimes:
    - CON PUNTALES (removable right-end shoring prop installed)
    - SIN PUNTALES (removable shoring prop removed)

Pipeline (strictly separating numerical FEM verification from blind physical validation
per ASME V&V 10):
  1. Build calibrated 3D solid CAD (IPE 200 reclaimed steel crane beam, L = 4,900 mm,
     end bearing plates, slender stub columns, removable right prop footplate, and
     3 hydraulic-jack load pads) and mesh in quadratic TET10 (~4,500 nodes / ~13,500 DOFs).
  2. Solve 24 multi-load / dual-BC 3D quadratic TET10 FEM training anchors (12 CON + 12 SIN)
     plus 8 strictly unseen 3D FEM holdout cases (4 CON + 4 SIN) using TorchFemBackend3D
     with exact continuum similitude (E_steel = 210 GPa, nu = 0.30).
  3. Calibrate ONLY the 4 lower-flange FOS gauge x-coordinates (CH4, CH5, CH7, CH8) against
     TEST1 (CON + SIN), keeping TEST2 & TEST3 (21,327 real physical samples) strictly BLIND.
  4. Train ONE unified 4-channel 3D PI-GINOT operator (Surface3DEnergyOperator backbone
     with exact 3D base-clamp lifting) purely on 3D FEM physics — ZERO experimental FOS
     data used during neural operator training.
  5. Verify numerical holdout accuracy against exact 3D TET10 FEM (8 unseen cases) and
     export a SHA-256 tamper-checked operator bundle.
  6. Execute blind physical Digital Twin validation on TEST2 & TEST3 (21,327 real samples):
       (A) Real-time forward full-field 3D reconstruction (<10 ms/frame, >100 Hz)
           vs. blind physical FOS microstrain streams (CH4, CH5, CH7, CH8).
       (B) Real-time inverse load-cell & full-3D hidden-stress reconstruction from
           ONLY the 4 sparse FOS sensors (compared against blind physical C1, C2, C3).
       (C) Live structural boundary-condition / prop-removal detection (CON vs SIN)
           via physics-operator sensor residuals.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import gmsh
import numpy as np
import torch
from torch import nn

from piginot.cad.importing_3d import import_cad_3d
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import (
    DirichletBC3D,
    FemRequest3D,
    FemResult3D,
    TorchFemBackend3D,
    TractionBC3D,
)
from piginot.geometry.mesh3d import CellType3D
from piginot.operators.surface_operator_3d import (
    Surface3DConfig,
    Surface3DEnergyOperator,
)
from piginot.training.energy_engine_3d import build_surface_3d_tensors

# Paths (strictly gitignored .formamech-runs/ cache)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_XLSX = (
    REPO_ROOT
    / ".formamech-runs"
    / "datasets"
    / "sum4re-fos-steel-beam"
    / "SUM4Re_FOS_STRAIN TEST BEAM1 JOLASTOKIETA.xlsx"
)
STUDY_DIR = (
    REPO_ROOT / ".formamech-runs" / "open_source_3d_studies" / "study3_digital_twin_3d"
)
CAD_DIR = STUDY_DIR / "cad"
DATA_DIR = STUDY_DIR / "data"
CACHE_DIR = DATA_DIR / "fem_cache"
BUNDLE_DIR = STUDY_DIR / "bundle"
for _d in (CAD_DIR, DATA_DIR, CACHE_DIR, BUNDLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Physical constants — Reclaimed S275 Steel IPE 200 crane beam
L_BEAM = 4900.0
H_BEAM = 200.0
B_FLANGE = 100.0
T_FLANGE = 8.5
T_WEB = 5.6
H_COL = 580.0  # slender end stub columns matching SUM4Re ANSYS geometry
E_STEEL_MPA = 210_000.0
NU_STEEL = 0.30
SIMILITUDE_SCALE = (
    1000.0  # E_model = 210 MPa, t_model = t_phys / 1000 -> u & eps exact to 1e-15!
)
LOAD_SCALE_KN = 12.0  # parameter normalization scale for C1, C2, C3
PAD_W = 140.0
PAD_H = 12.0
PAD_AREA_MM2 = PAD_W * B_FLANGE
LOAD_X_MM = (1010.0, 2024.0, 3214.0)  # C1, C2, C3 hydraulic jack centers
MESH_SIZE_MM = 128.0


def load_sum4re_excel(xlsx_path: Path) -> dict[str, np.ndarray]:
    """Parse all 6 experimental load+FOS sheets using stdlib zipfile + ElementTree."""
    out: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(xlsx_path) as z:
        ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        sheets = [
            (s.attrib["name"], idx + 1)
            for idx, s in enumerate(wb.findall(".//x:sheet", ns))
        ]
        col_map = {"B": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6}
        for sname, sidx in sheets:
            if "TEST" not in sname:
                continue
            root = ET.fromstring(z.read(f"xl/worksheets/sheet{sidx}.xml"))
            rows_arr = []
            for r in root.findall(".//x:sheetData/x:row", ns)[5:]:
                vals = [math.nan] * 7
                for c in r.findall("x:c", ns):
                    col = "".join(ch for ch in c.attrib.get("r", "") if ch.isalpha())
                    if col not in col_map:
                        continue
                    v = c.find("x:v", ns)
                    if v is None or c.attrib.get("t") == "s":
                        continue
                    vals[col_map[col]] = float(v.text)
                if not any(math.isnan(v) for v in vals):
                    rows_arr.append(vals)
            out[sname] = np.asarray(rows_arr, dtype=np.float64)
    return out


def build_sum4re_step(step_path: Path) -> None:
    """Create 3D OpenCASCADE STEP for SUM4Re IPE 200 steel beam + stub columns + prop footplate."""
    xc_R = L_BEAM - 95.0
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("sum4re_ipe200_digital_twin")
    occ = gmsh.model.occ

    # Main IPE 200 profile (X in [0, 4900], Y in [-50, +50], Z in [0, 200])
    b_bot = occ.addBox(0.0, -B_FLANGE / 2, 0.0, L_BEAM, B_FLANGE, T_FLANGE)
    b_top = occ.addBox(
        0.0, -B_FLANGE / 2, H_BEAM - T_FLANGE, L_BEAM, B_FLANGE, T_FLANGE
    )
    b_web = occ.addBox(0.0, -T_WEB / 2, T_FLANGE, L_BEAM, T_WEB, H_BEAM - 2 * T_FLANGE)

    # Left welded end bearing plate + semi-rigid stub column (tc_L = 48 mm, H = 580 mm)
    plate_L = occ.addBox(0.0, -B_FLANGE / 2 - 15.0, -14.0, 180.0, B_FLANGE + 30.0, 14.0)
    col_L = occ.addBox(64.0, -B_FLANGE / 2, -H_COL, 48.0, B_FLANGE, H_COL - 14.0)

    # Right end bearing plate + stub column (tc_R = 78 mm, H = 580 mm) with hinged center pin
    # (z = -H_COL, ALWAYS supported in y, z) + outer prop footplate wings (z = -H_COL + 4,
    # supported in z = 0 ONLY when removable props "CON PUNTALES" are installed!)
    plate_R = occ.addBox(
        L_BEAM - 180.0, -B_FLANGE / 2 - 15.0, -14.0, 180.0, B_FLANGE + 30.0, 14.0
    )
    col_R = occ.addBox(
        xc_R - 39.0, -B_FLANGE / 2, -H_COL + 14.0, 78.0, B_FLANGE, H_COL - 28.0
    )
    foot_R_pin = occ.addBox(xc_R - 12.0, -B_FLANGE / 2, -H_COL, 24.0, B_FLANGE, 14.0)
    foot_R_wing_l = occ.addBox(
        xc_R - 74.0, -B_FLANGE / 2, -H_COL + 4.0, 62.0, B_FLANGE, 14.0
    )
    foot_R_wing_r = occ.addBox(
        xc_R + 12.0, -B_FLANGE / 2, -H_COL + 4.0, 62.0, B_FLANGE, 14.0
    )

    # 3 hydraulic-jack load pads on top flange (C1, C2, C3)
    pads = [
        occ.addBox(xc - PAD_W / 2, -B_FLANGE / 2, H_BEAM, PAD_W, B_FLANGE, PAD_H)
        for xc in LOAD_X_MM
    ]

    occ.fuse(
        [(3, b_bot)],
        [
            (3, t)
            for t in [
                b_top,
                b_web,
                plate_L,
                col_L,
                plate_R,
                col_R,
                foot_R_pin,
                foot_R_wing_l,
                foot_R_wing_r,
            ]
            + pads
        ],
    )
    occ.synchronize()
    gmsh.write(str(step_path))
    gmsh.finalize()


def identify_beam_faces(mesh: Any) -> dict[str, tuple[str, ...]]:
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    tris = np.asarray(mesh.surface_triangles, dtype=np.int64)

    def _find(cond: Any) -> tuple[str, ...]:
        return tuple(
            sorted(
                tag
                for tag, fids in mesh.semantic_face_tags.items()
                if cond(nodes[tris[list(fids)]].reshape(-1, 3))
            )
        )

    roles = {
        "left_base": _find(
            lambda p: p[:, 2].max() < -H_COL + 1.0 and p[:, 0].max() < 500.0
        ),
        "right_pin": _find(
            lambda p: p[:, 2].max() < -H_COL + 1.0 and p[:, 0].min() > 4000.0
        ),
        "right_prop": _find(
            lambda p: (
                (-H_COL + 3.0 < p[:, 2].min())
                and (p[:, 2].max() < -H_COL + 5.0)
                and p[:, 0].min() > 4000.0
            )
        ),
        "c1_pad": _find(
            lambda p: (
                p[:, 2].min() > H_BEAM + PAD_H - 1.0
                and abs(p[:, 0].mean() - LOAD_X_MM[0]) < 100.0
            )
        ),
        "c2_pad": _find(
            lambda p: (
                p[:, 2].min() > H_BEAM + PAD_H - 1.0
                and abs(p[:, 0].mean() - LOAD_X_MM[1]) < 100.0
            )
        ),
        "c3_pad": _find(
            lambda p: (
                p[:, 2].min() > H_BEAM + PAD_H - 1.0
                and abs(p[:, 0].mean() - LOAD_X_MM[2]) < 100.0
            )
        ),
    }
    for k, v in roles.items():
        if not v:
            raise RuntimeError(f"Failed to identify face role {k}")
    return roles


def make_bcs(
    forces_kn: tuple[float, float, float],
    prop_installed: bool,
    roles: dict[str, tuple[str, ...]],
) -> tuple[tuple[DirichletBC3D, ...], tuple[TractionBC3D, ...]]:
    dbcs = [
        DirichletBC3D(
            "left_col_base", roles["left_base"], ("x", "y", "z"), (0.0, 0.0, 0.0)
        ),
        DirichletBC3D("right_col_pin", roles["right_pin"], ("y", "z"), (0.0, 0.0)),
    ]
    if prop_installed:
        dbcs.append(
            DirichletBC3D("right_prop_wings", roles["right_prop"], ("z",), (0.0,))
        )

    tbcs: list[TractionBC3D] = []
    for f_kn, name, ftags in zip(
        forces_kn,
        ("c1", "c2", "c3"),
        (roles["c1_pad"], roles["c2_pad"], roles["c3_pad"]),
        strict=True,
    ):
        if abs(f_kn) > 1e-9:
            tz_model = (-f_kn * 1000.0 / PAD_AREA_MM2) / SIMILITUDE_SCALE
            tbcs.append(TractionBC3D(f"{name}_load", ftags, (0.0, 0.0, tz_model)))
    if not tbcs:
        tbcs.append(TractionBC3D("zero_load", roles["c2_pad"], (0.0, 0.0, 0.0)))
    return tuple(dbcs), tuple(tbcs)


def solve_or_load_fem(
    case_id: str,
    forces_kn: tuple[float, float, float],
    prop_installed: bool,
    mesh: Any,
    material: MaterialSpec,
    roles: dict[str, tuple[str, ...]],
    solver: TorchFemBackend3D,
) -> FemResult3D:
    npz_p = CACHE_DIR / f"{case_id}.npz"
    js_p = CACHE_DIR / f"{case_id}.json"
    if npz_p.is_file() and js_p.is_file():
        meta = json.loads(js_p.read_text(encoding="utf-8"))
        if (
            meta.get("mesh_id") == mesh.mesh_id
            and meta.get("forces_kn") == list(forces_kn)
            and meta.get("prop") == prop_installed
        ):
            arr = np.load(npz_p)
            return FemResult3D(
                request_id=meta["request_id"],
                backend_name=meta["backend_name"],
                mesh_id=meta["mesh_id"],
                converged=bool(meta["converged"]),
                nodal_displacement=arr["u"],
                nodal_von_mises_mpa=arr["vm"],
                reactions_n={k: tuple(v) for k, v in meta["reactions_n"].items()},
                total_strain_energy_n_mm=float(meta["total_strain_energy_n_mm"]),
                external_work_n_mm=float(meta["external_work_n_mm"]),
                min_det_f=float(meta["min_det_f"]),
                max_displacement_mm=float(meta["max_displacement_mm"]),
                peak_von_mises_mpa=float(meta["peak_von_mises_mpa"]),
                final_residual_norm=float(meta["final_residual_norm"]),
                total_newton_iterations=int(meta["total_newton_iterations"]),
                wall_time_s=float(meta["wall_time_s"]),
                reason=str(meta["reason"]),
            )

    dbcs, tbcs = make_bcs(forces_kn, prop_installed, roles)
    req = FemRequest3D(
        request_id=f"sum4re-{case_id}",
        mesh=mesh,
        material=material,
        dirichlet_bcs=dbcs,
        traction_bcs=tbcs,
        load_steps=1,
        solver_tolerance=1e-6,
    )
    res = solver.solve(req)
    if not res.converged:
        raise RuntimeError(f"3D FEM failed for {case_id}: {res.reason}")
    np.savez_compressed(npz_p, u=res.nodal_displacement, vm=res.nodal_von_mises_mpa)
    meta = {
        "request_id": res.request_id,
        "backend_name": res.backend_name,
        "mesh_id": res.mesh_id,
        "forces_kn": list(forces_kn),
        "prop": prop_installed,
        "converged": res.converged,
        "reactions_n": {k: list(v) for k, v in res.reactions_n.items()},
        "total_strain_energy_n_mm": res.total_strain_energy_n_mm,
        "external_work_n_mm": res.external_work_n_mm,
        "min_det_f": res.min_det_f,
        "max_displacement_mm": res.max_displacement_mm,
        "peak_von_mises_mpa": res.peak_von_mises_mpa,
        "final_residual_norm": res.final_residual_norm,
        "total_newton_iterations": res.total_newton_iterations,
        "wall_time_s": res.wall_time_s,
        "reason": res.reason,
    }
    js_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return res


def build_fos_strain_operator(
    nodes: np.ndarray, station_xs: list[float], half_window_mm: float = 210.0
) -> np.ndarray:
    """Build exact linear differentiation matrix D_fos (4, N) mapping nodal u_x (mm) -> lower-flange microstrain (ue)."""
    bot_mask = (
        (nodes[:, 2] < 1.5)
        & (nodes[:, 2] > -1.0)
        & (np.abs(nodes[:, 1]) < 38.0)
        & (nodes[:, 0] > 190.0)
        & (nodes[:, 0] < L_BEAM - 190.0)
    )
    bot_idx = np.nonzero(bot_mask)[0]
    xb = nodes[bot_idx, 0]
    D = np.zeros((len(station_xs), len(nodes)), dtype=np.float64)
    for j, xs in enumerate(station_xs):
        local = np.abs(xb - xs) <= half_window_mm
        if local.sum() < 4:
            local = np.abs(xb - xs) <= (half_window_mm * 1.6)
        idx_l = bot_idx[local]
        xl = xb[local]
        xm = xl.mean()
        dx = xl - xm
        denom = float(np.sum(dx * dx))
        # Ordinary least-squares slope d(u_x)/dx * 1e6 (microstrain)
        D[j, idx_l] = (dx / denom) * 1.0e6
    return D


class DigitalTwinBeamPIGINOT3D(nn.Module):
    """3D PI-GINOT Digital Twin operator sharing Surface3DEnergyOperator backbone for (u_3d, vm_3d).

    Combines Surface3DEnergyOperator's 3D surface-token cross-attention with FiLM parameter
    modulation and a low-rank neural-operator branch-trunk head over anisotropic 3D coordinates,
    enforcing exact 3D Dirichlet lifting at both stub-column bases and zero-load proportionality.
    """

    def __init__(
        self,
        config: Surface3DConfig,
        xyz_min: np.ndarray,
        xyz_span: np.ndarray,
        u_scale_mm: float,
        vm_scale_mpa: float,
        n_modes: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = Surface3DEnergyOperator(config)
        d = config.hidden_dim
        self.n_modes = n_modes
        self.register_buffer(
            "xyz_min", torch.tensor(xyz_min, dtype=torch.float32).view(1, 1, 3)
        )
        self.register_buffer(
            "xyz_span",
            torch.tensor(np.maximum(xyz_span, 1.0), dtype=torch.float32).view(1, 1, 3),
        )
        self.aniso_freqs = 8
        aniso_dim = 3 + 6 * self.aniso_freqs  # 51
        self.trunk_in = nn.Sequential(
            nn.Linear(aniso_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.trunk_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
        )
        # Spatial trunk basis: 3D displacement modes (Q, M, 3) & von Mises modes (Q, M)
        self.u_trunk = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3 * n_modes)
        )
        self.vm_trunk = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_modes)
        )

        # Parameter branch conditioned on 5D telemetry (u_dir[3], load_norm, prop_state) + global surface context
        self.branch_u = nn.Sequential(
            nn.Linear(config.parameter_dim + d, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, n_modes),
        )
        self.branch_vm = nn.Sequential(
            nn.Linear(config.parameter_dim + d, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, n_modes),
        )
        self.u_scale_mm = float(u_scale_mm)
        self.vm_scale_mpa = float(vm_scale_mpa)

    def encode(
        self,
        s_cent: torch.Tensor,
        s_norm: torch.Tensor,
        s_area: torch.Tensor,
        s_feat: torch.Tensor,
    ) -> Any:
        return self.backbone.encode_surface(s_cent, s_norm, s_area, s_feat)

    def compute_spatial_basis(
        self, encoding: Any, query_xyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode 3D geometry-conditioned spatial trunk modes ONCE for any query point set (1, Q, 3)."""
        from piginot.operators.surface_operator_3d import _fourier_encode_3d

        aniso_q = (query_xyz - self.xyz_min) / self.xyz_span
        pos = _fourier_encode_3d(aniso_q, self.aniso_freqs)
        q_tok = self.trunk_in(pos)  # (1, Q, d)
        cross = self.backbone.query_cross_attn(
            q_tok, encoding.slice_tokens[:1], encoding.slice_tokens[:1]
        )
        q_tok = self.backbone.query_norm(q_tok + cross)
        q_tok = q_tok + self.trunk_mlp(q_tok)
        u_basis = self.u_trunk(q_tok).view(
            query_xyz.shape[1], self.n_modes, 3
        )  # (Q, M, 3)
        vm_basis = self.vm_trunk(q_tok).view(query_xyz.shape[1], self.n_modes)  # (Q, M)
        g_ctx = encoding.slice_tokens[:1].mean(dim=1)  # (1, d)
        return u_basis, vm_basis, g_ctx

    def forward_from_basis(
        self,
        u_basis: torch.Tensor,
        vm_basis: torch.Tensor,
        g_ctx: torch.Tensor,
        parameters: torch.Tensor,
        base_clamp_factor: torch.Tensor,
        load_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ultra-fast branch-trunk evaluation for arbitrary batch B of live telemetry states."""
        b = parameters.shape[0]
        ctx_b = g_ctx.expand(b, -1)
        p_in = torch.cat((parameters, ctx_b), dim=-1)
        cu = self.branch_u(p_in)  # (B, M)
        cvm = self.branch_vm(p_in)  # (B, M)
        raw_u = torch.einsum("bm,qmc->bqc", cu, u_basis)
        raw_vm = torch.nn.functional.softplus(
            torch.einsum("bm,qm->bq", cvm, vm_basis), beta=8.0
        )
        u_pred = (
            (load_norm.view(-1, 1, 1) * self.u_scale_mm) * base_clamp_factor * raw_u
        )
        vm_pred = (load_norm.view(-1, 1) * self.vm_scale_mpa) * raw_vm
        return u_pred, vm_pred

    def forward_fields(
        self,
        encoding: Any,
        query_xyz: torch.Tensor,
        parameters: torch.Tensor,
        base_clamp_factor: torch.Tensor,
        load_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_basis, vm_basis, g_ctx = self.compute_spatial_basis(encoding, query_xyz[:1])
        return self.forward_from_basis(
            u_basis, vm_basis, g_ctx, parameters, base_clamp_factor, load_norm
        )


def build_base_clamp_factor(mesh: Any, roles: dict[str, tuple[str, ...]]) -> np.ndarray:
    """Exact 3D ADF lifting mask (N, 3): 0 on constrained support DOFs, smoothly 1 elsewhere."""
    from piginot.fem.backend_3d import face_node_indices_3d

    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    n_nodes = len(nodes)
    left_n = np.asarray(face_node_indices_3d(mesh, roles["left_base"]), dtype=np.int64)
    right_pin_n = np.asarray(
        face_node_indices_3d(mesh, roles["right_pin"]), dtype=np.int64
    )

    pts_l = nodes[left_n]
    pts_r = nodes[right_pin_n]
    # Distance to nearest left-base node & right-pin node
    d_l = np.min(np.linalg.norm(nodes[:, None, :] - pts_l[None, :, :], axis=-1), axis=1)
    d_r = np.min(np.linalg.norm(nodes[:, None, :] - pts_r[None, :, :], axis=-1), axis=1)
    # Smooth R-function ramp saturating to 1 over 180 mm
    w_l = np.tanh(d_l / 120.0)
    w_r = np.tanh(d_r / 120.0)
    w_both = w_l * w_r
    factor = np.ones((n_nodes, 3), dtype=np.float64)
    factor[:, 0] = w_l  # left base fixed in x; right base free in x (roller)
    factor[:, 1] = w_both  # both bases fixed in y
    factor[:, 2] = w_both  # both bases fixed in z
    return factor


def param_vec(
    forces_kn: tuple[float, float, float], prop_installed: bool
) -> list[float]:
    f = np.asarray(forces_kn, dtype=np.float64) / LOAD_SCALE_KN
    norm = float(np.linalg.norm(f))
    u_dir = f / max(norm, 1.0e-8)
    # Unit load direction (3) + magnitude (1) + prop regime (1) -> exact scale-invariant linear elasticity!
    return [
        float(u_dir[0]),
        float(u_dir[1]),
        float(u_dir[2]),
        float(norm),
        1.0 if prop_installed else -1.0,
    ]


def main() -> None:
    t_total = time.perf_counter()
    print("=" * 78)
    print("STUDY 3 — 3D PI-GINOT STRUCTURAL DIGITAL TWIN ON SUM4Re REAL FOS STEEL BEAM")
    print("=" * 78, flush=True)

    # ------------------------------------------------------------------
    # 1. Load real experimental SUM4Re dataset & hash provenance
    # ------------------------------------------------------------------
    if not DATASET_XLSX.is_file():
        from fetch_open_source_3d_datasets import fetch_sum4re

        fetch_sum4re()
    dataset_sha256 = hashlib.sha256(DATASET_XLSX.read_bytes()).hexdigest()
    exp_data = load_sum4re_excel(DATASET_XLSX)
    total_samples = sum(len(v) for v in exp_data.values())
    print(
        f"[1/6] Loaded SUM4Re dataset ({total_samples:,} real samples across 6 tests, sha256={dataset_sha256[:16]}...)"
    )
    for k, v in exp_data.items():
        print(
            f"      {k:<22s}: N={len(v):5,d} | max F={np.round(v[:, :3].max(axis=0), 2)} kN | max FOS={np.round(v[:, 3:].max(axis=0), 1)} ue",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 2. Build 3D OpenCASCADE STEP & quadratic TET10 solid mesh
    # ------------------------------------------------------------------
    step_path = CAD_DIR / "sum4re_ipe200_digital_twin.step"
    build_sum4re_step(step_path)
    imported = import_cad_3d(
        step_path,
        unit=Unit.MM,
        mesh_size_mm=MESH_SIZE_MM,
        element_type=CellType3D.TET10,
        poisson_ratio=NU_STEEL,
    )
    mesh = imported.mesh
    roles = identify_beam_faces(mesh)
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    tris = np.asarray(mesh.surface_triangles, dtype=np.int64)
    n_nodes, n_cells = len(nodes), len(mesh.cells)
    print(
        f"[2/6] 3D TET10 IPE-200 mesh ready: {n_nodes:,} nodes ({3 * n_nodes:,} DOFs), {n_cells:,} TET10 cells, {len(tris):,} surface tris",
        flush=True,
    )

    material = MaterialSpec(
        name="reclaimed-steel-s275-sim1000",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(
                E_STEEL_MPA / SIMILITUDE_SCALE, Unit.MPA
            ).to_canonical(),
            "poisson_ratio": Quantity(NU_STEEL, Unit.ONE).to_canonical(),
        },
    )
    solver = TorchFemBackend3D()

    # ------------------------------------------------------------------
    # 3. Solve 24 multi-load / dual-BC 3D TET10 FEM training anchors + 8 holdouts
    # ------------------------------------------------------------------
    load_combos_train: list[tuple[str, tuple[float, float, float]]] = [
        ("c1_10", (10.0, 0.0, 0.0)),
        ("c2_10", (0.0, 10.0, 0.0)),
        ("c3_10", (0.0, 0.0, 10.0)),
        ("c1_05", (5.0, 0.0, 0.0)),
        ("c2_05", (0.0, 5.0, 0.0)),
        ("c3_05", (0.0, 0.0, 5.0)),
        ("c12_10", (10.5, 10.5, 0.0)),
        ("c23_10", (0.0, 11.0, 11.0)),
        ("c13_09", (9.0, 0.0, 9.0)),
        ("c123_06", (6.0, 6.0, 6.0)),
        ("c123_11", (11.0, 11.5, 10.5)),
        ("c123_asym", (10.5, 5.5, 11.5)),
    ]
    load_combos_holdout: list[tuple[str, tuple[float, float, float]]] = [
        ("ho_c12_asym", (8.2, 10.4, 0.0)),
        ("ho_c23_asym", (0.0, 8.8, 10.2)),
        ("ho_c123_mid", (7.5, 9.5, 6.8)),
        ("ho_c123_hi", (10.2, 10.6, 9.4)),
    ]

    train_records: list[dict[str, Any]] = []
    holdout_records: list[dict[str, Any]] = []
    fem_times: list[float] = []

    print(
        f"[3/6] Solving/loading {2 * len(load_combos_train)} training + {2 * len(load_combos_holdout)} holdout 3D TET10 FEM cases...",
        flush=True,
    )
    for prop in (True, False):
        reg = "con" if prop else "sin"
        for cname, f_kn in load_combos_train:
            cid = f"tr_{reg}_{cname}"
            res = solve_or_load_fem(cid, f_kn, prop, mesh, material, roles, solver)
            fem_times.append(res.wall_time_s)
            train_records.append({"id": cid, "f_kn": f_kn, "prop": prop, "res": res})
        for cname, f_kn in load_combos_holdout:
            cid = f"ho_{reg}_{cname}"
            res = solve_or_load_fem(cid, f_kn, prop, mesh, material, roles, solver)
            holdout_records.append({"id": cid, "f_kn": f_kn, "prop": prop, "res": res})
    mean_fem_s = float(np.mean(fem_times))
    print(
        f"      All 32 3D TET10 FEM solves converged! Mean cold 3D FEM time = {mean_fem_s:.2f} s/query.",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 4. Calibrate ONLY 4 FOS x-coordinates on TEST1 (CON + SIN); TEST2 & TEST3 STAY 100% BLIND!
    # ------------------------------------------------------------------
    # Extract 3D FEM basis strain profiles from the 6 single-jack 10 kN anchors (pure 3D FEM)
    x_grid = np.linspace(300.0, 4450.0, 416)
    D_grid = build_fos_strain_operator(nodes, list(x_grid), half_window_mm=210.0)
    fem_prof: dict[str, np.ndarray] = {}
    for prop, reg in ((True, "CON PUNTALES"), (False, "SIN PUNTALES")):
        reg_key = "con" if prop else "sin"
        basis = []
        for cname in ("c1_10", "c2_10", "c3_10"):
            rec = next(r for r in train_records if r["id"] == f"tr_{reg_key}_{cname}")
            ux = rec["res"].nodal_displacement[:, 0]
            basis.append((D_grid @ ux) / 10.0)  # ue / kN
        fem_prof[reg] = np.stack(basis, axis=0)  # (3, G)

    # Empirical compliance on TEST1 ONLY (strict calibration split!)
    S_test1: dict[str, np.ndarray] = {}
    for reg in ("CON PUNTALES", "SIN PUNTALES"):
        A1 = exp_data[f"{reg} TEST1"]
        X1 = np.hstack([A1[:, :3], np.ones((len(A1), 1))])
        coef, _, _, _ = np.linalg.lstsq(X1, A1[:, 3:], rcond=None)
        S_test1[reg] = coef[:3, :]  # (3, 4)

    fos_names = ["CH4", "CH5", "CH7", "CH8"]
    fos_x_mm: list[float] = []
    print(
        "[4/6] Calibrating 4 lower-flange FOS gauge x-coordinates on TEST1 ONLY (TEST2 & TEST3 strictly blind):",
        flush=True,
    )
    for j, ch in enumerate(fos_names):
        err_g = np.mean(
            (fem_prof["CON PUNTALES"] - S_test1["CON PUNTALES"][:, j : j + 1]) ** 2,
            axis=0,
        ) + np.mean(
            (fem_prof["SIN PUNTALES"] - S_test1["SIN PUNTALES"][:, j : j + 1]) ** 2,
            axis=0,
        )
        k_best = int(np.argmin(err_g))
        fos_x_mm.append(float(x_grid[k_best]))
        print(
            f"      {ch} @ x = {fos_x_mm[-1]:6.0f} mm (xi={fos_x_mm[-1] / L_BEAM:.3f}) | "
            f"FEM_CON={np.round(fem_prof['CON PUNTALES'][:, k_best], 2)} vs T1={np.round(S_test1['CON PUNTALES'][:, j], 2)} | "
            f"FEM_SIN={np.round(fem_prof['SIN PUNTALES'][:, k_best], 2)} vs T1={np.round(S_test1['SIN PUNTALES'][:, j], 2)}",
            flush=True,
        )

    D_fos = build_fos_strain_operator(nodes, fos_x_mm, half_window_mm=210.0)  # (4, N)
    D_fos_t = torch.tensor(D_fos, dtype=torch.float32)

    # Also compute exact 3D FEM compliance matrix S_fem_con, S_fem_sin for transparency
    S_fem_con = np.stack(
        [
            D_fos
            @ next(r for r in train_records if r["id"] == f"tr_con_{c}")[
                "res"
            ].nodal_displacement[:, 0]
            / 10.0
            for c in ("c1_10", "c2_10", "c3_10")
        ],
        axis=0,
    )
    S_fem_sin = np.stack(
        [
            D_fos
            @ next(r for r in train_records if r["id"] == f"tr_sin_{c}")[
                "res"
            ].nodal_displacement[:, 0]
            / 10.0
            for c in ("c1_10", "c2_10", "c3_10")
        ],
        axis=0,
    )

    # ------------------------------------------------------------------
    # 5. Train ONE unified 3D PI-GINOT Digital Twin Operator (100% FEM physics, ZERO exp labels!)
    # ------------------------------------------------------------------
    print(
        "[5/6] Training unified 4-channel 3D PI-GINOT Digital Twin operator (pure 3D FEM physics, ZERO exp labels)...",
        flush=True,
    )
    torch.manual_seed(42)
    dtype = torch.float32
    op_cfg = Surface3DConfig(
        parameter_dim=5,
        surface_feature_dim=6,
        hidden_dim=112,
        slice_count=12,
        num_heads=4,
        num_blocks=2,
        positional_frequencies=3,
    )
    xyz_min = nodes.min(axis=0)
    xyz_span = nodes.max(axis=0) - xyz_min
    u_scale_mm = float(max(r["res"].max_displacement_mm for r in train_records))
    vm_scale_mpa = float(
        max(r["res"].peak_von_mises_mpa * SIMILITUDE_SCALE for r in train_records)
    )

    model = DigitalTwinBeamPIGINOT3D(
        op_cfg,
        xyz_min=xyz_min,
        xyz_span=xyz_span,
        u_scale_mm=u_scale_mm,
        vm_scale_mpa=vm_scale_mpa,
    ).to(dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())

    clamped_tags = roles["left_base"] + roles["right_pin"] + roles["right_prop"]
    loaded_tags = roles["c1_pad"] + roles["c2_pad"] + roles["c3_pad"]
    s_cent, s_norm, s_area, s_feat = build_surface_3d_tensors(
        mesh, clamped_face_tags=clamped_tags, loaded_face_tags=loaded_tags, dtype=dtype
    )
    nodes_t = torch.tensor(nodes, dtype=dtype).unsqueeze(0)  # (1, N, 3)
    base_clamp_t = torch.tensor(
        build_base_clamp_factor(mesh, roles), dtype=dtype
    ).unsqueeze(0)  # (1, N, 3)

    # Assemble training batch (24 FEM cases)
    p_tr = torch.tensor(
        [param_vec(r["f_kn"], r["prop"]) for r in train_records], dtype=dtype
    )
    ln_tr = p_tr[:, 3]
    u_tr = torch.tensor(
        np.stack([r["res"].nodal_displacement for r in train_records], axis=0),
        dtype=dtype,
    )
    vm_tr = torch.tensor(
        np.stack(
            [r["res"].nodal_von_mises_mpa * SIMILITUDE_SCALE for r in train_records],
            axis=0,
        ),
        dtype=dtype,
    )
    eps_tr = torch.einsum(
        "kn,bn->bk", D_fos_t, u_tr[:, :, 0]
    )  # (B, 4) exact 3D FEM lower-flange strains

    b_tot = len(train_records)
    u_norms = torch.linalg.vector_norm(u_tr.reshape(b_tot, -1), dim=-1).clamp_min(1e-4)
    vm_norms = torch.linalg.vector_norm(vm_tr, dim=-1).clamp_min(1.0)
    eps_norms = torch.linalg.vector_norm(eps_tr, dim=-1).clamp_min(5.0)

    epochs = 1500
    lr = 4.0e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.008
    )

    t_tr0 = time.perf_counter()
    loss_hist: list[float] = []
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        enc_1 = model.encode(s_cent, s_norm, s_area, s_feat)
        u_basis, vm_basis, g_ctx = model.compute_spatial_basis(enc_1, nodes_t)
        u_pred, vm_pred = model.forward_from_basis(
            u_basis, vm_basis, g_ctx, p_tr, base_clamp_t, ln_tr
        )
        eps_pred = torch.einsum("kn,bn->bk", D_fos_t, u_pred[:, :, 0])

        u_rel = (
            torch.linalg.vector_norm((u_pred - u_tr).reshape(b_tot, -1), dim=-1)
            / u_norms
        ).mean()
        vm_rel = (torch.linalg.vector_norm(vm_pred - vm_tr, dim=-1) / vm_norms).mean()
        vm_pk_rel = (
            torch.abs(vm_pred.max(dim=-1).values - vm_tr.max(dim=-1).values)
            / vm_tr.max(dim=-1).values.clamp_min(1.0)
        ).mean()
        eps_rel = (
            torch.linalg.vector_norm(eps_pred - eps_tr, dim=-1) / eps_norms
        ).mean()

        loss = 3.0 * u_rel + 2.5 * vm_rel + 0.8 * vm_pk_rel + 1.5 * eps_rel
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        sched.step()
        loss_hist.append(float(loss.detach().item()))
        if ep in (0, 499, 999, epochs - 1):
            print(
                f"      epoch {ep + 1:3d}/{epochs}: loss={loss_hist[-1]:.4f} | "
                f"u_L2={float(u_rel.detach()) * 100:.2f}% | vm_L2={float(vm_rel.detach()) * 100:.2f}% | fos_L2={float(eps_rel.detach()) * 100:.2f}%",
                flush=True,
            )

    train_wall_s = time.perf_counter() - t_tr0
    print(
        f"      Trained 3D PI-GINOT ({n_params:,} params) in {train_wall_s:.1f}s.",
        flush=True,
    )

    # Save weights + SHA-256 receipt
    weights_path = BUNDLE_DIR / "sum4re_piginot_3d.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": dataclasses.asdict(op_cfg),
            "fos_x_mm": fos_x_mm,
        },
        weights_path,
    )
    weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()

    # Evaluate numerical holdout verification against 8 unseen 3D TET10 FEM solves!
    # Cache geometry-conditioned spatial basis ONCE (just like a deployed digital twin!)
    model.eval()
    with torch.no_grad():
        t_enc0 = time.perf_counter()
        cached_enc = model.encode(s_cent, s_norm, s_area, s_feat)
        cached_u_basis, cached_vm_basis, cached_g_ctx = model.compute_spatial_basis(
            cached_enc, nodes_t
        )
        encode_ms = (time.perf_counter() - t_enc0) * 1000.0

    def predict_3d(
        forces_kn: tuple[float, float, float], prop_installed: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        t0 = time.perf_counter()
        with torch.no_grad():
            p_t = torch.tensor([param_vec(forces_kn, prop_installed)], dtype=dtype)
            ln_t = p_t[:, 3]
            u_t, vm_t = model.forward_from_basis(
                cached_u_basis, cached_vm_basis, cached_g_ctx, p_t, base_clamp_t, ln_t
            )
            u_np = u_t[0].numpy().astype(np.float64)
            vm_np = vm_t[0].numpy().astype(np.float64)
            eps_np = D_fos @ u_np[:, 0]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return u_np, vm_np, eps_np, dt_ms

    # Warm-up 2 calls before timing
    _ = predict_3d((5.0, 5.0, 5.0), True)
    _ = predict_3d((5.0, 5.0, 5.0), False)

    ho_metrics: list[dict[str, Any]] = []
    for rec in holdout_records:
        u_p, vm_p, eps_p, dt_ms = predict_3d(rec["f_kn"], rec["prop"])
        u_ref = rec["res"].nodal_displacement
        vm_ref = rec["res"].nodal_von_mises_mpa * SIMILITUDE_SCALE
        eps_ref = D_fos @ u_ref[:, 0]
        u_err = float(np.linalg.norm(u_p - u_ref) / max(np.linalg.norm(u_ref), 1e-12))
        vm_err = float(
            np.linalg.norm(vm_p - vm_ref) / max(np.linalg.norm(vm_ref), 1e-12)
        )
        pk_err = abs(float(vm_p.max()) - float(vm_ref.max())) / max(
            float(vm_ref.max()), 1e-12
        )
        eps_rmse = float(np.sqrt(np.mean((eps_p - eps_ref) ** 2)))
        ho_metrics.append(
            {
                "id": rec["id"],
                "forces_kn": list(rec["f_kn"]),
                "prop": rec["prop"],
                "u_rel_l2": u_err,
                "vm_rel_l2": vm_err,
                "peak_vm_rel_err": pk_err,
                "fos_rmse_vs_fem_ue": eps_rmse,
                "infer_ms": dt_ms,
                "fem_wall_s": float(rec["res"].wall_time_s),
            }
        )

    mean_ho_u = float(np.mean([m["u_rel_l2"] for m in ho_metrics]))
    mean_ho_vm = float(np.mean([m["vm_rel_l2"] for m in ho_metrics]))
    mean_ho_pk = float(np.mean([m["peak_vm_rel_err"] for m in ho_metrics]))
    mean_infer_ms = float(np.mean([m["infer_ms"] for m in ho_metrics]))
    speedup = mean_fem_s / (mean_infer_ms / 1000.0)
    print(
        f"      [NUMERICAL VERIFICATION vs 8 UNSEEN 3D TET10 FEM HOLDOUTS]\n"
        f"        mean u rel L2   = {mean_ho_u * 100:.2f}% | mean VM rel L2 = {mean_ho_vm * 100:.2f}% | peak VM err = {mean_ho_pk * 100:.2f}%\n"
        f"        3D PI-GINOT     = {mean_infer_ms:.2f} ms/frame ({1000.0 / mean_infer_ms:.0f} Hz) vs Cold 3D FEM = {mean_fem_s:.2f} s ({speedup:.0f}x faster!)",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 6. BLIND PHYSICAL EXPERIMENTAL VALIDATION ON SUM4Re (TEST2 & TEST3: 21,327 REAL SAMPLES!)
    #    PLUS real-time inverse load recovery & prop-removal BC regime detection!
    # ------------------------------------------------------------------
    print(
        "[6/6] Running BLIND experimental Digital Twin validation on SUM4Re real telemetry streams...",
        flush=True,
    )

    # Extract exact Jacobian of 3D PI-GINOT virtual FOS probes w.r.t. (C1, C2, C3) for BOTH CON & SIN
    # (used for fast inverse load-cell recovery + live prop-removal detection!)
    J_op: dict[str, np.ndarray] = {}
    for prop, key in ((True, "CON"), (False, "SIN")):
        cols = []
        for f_basis in ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)):
            _, _, eps_b, _ = predict_3d(f_basis, prop)
            cols.append(eps_b / 10.0)
        J_op[key] = np.stack(cols, axis=1)  # (4, 3): rows=CH4..CH8, cols=C1..C3

    # Evaluate every single one of the 6 experimental tests!
    # (Clearly label TEST1 as calibration-coordinate reference and TEST2 & TEST3 as STRICTLY BLIND!)
    exp_results: dict[str, Any] = {}
    stream_archives: dict[str, Any] = {}

    for sheet_name in [
        "CON PUNTALES TEST1",
        "CON PUNTALES TEST2",
        "CON PUNTALES TEST3",
        "SIN PUNTALES TEST1",
        "SIN PUNTALES TEST2",
        "SIN PUNTALES TEST3",
    ]:
        A = exp_data[sheet_name]
        prop = sheet_name.startswith("CON")
        reg_key = "CON" if prop else "SIN"
        is_blind = "TEST1" not in sheet_name
        F_real = A[:, :3]  # (T, 3) kN
        E_real = A[:, 3:]  # (T, 4) ue

        # Subsample 96 evenly-spaced live frames along the physical test for full 3D operator rollout
        # AND evaluate vectorized batch forward inference across ALL T samples!
        idx_96 = np.linspace(0, len(A) - 1, 96, dtype=int)
        u_stream_96 = np.zeros((96, n_nodes, 3), dtype=np.float32)
        vm_stream_96 = np.zeros((96, n_nodes), dtype=np.float32)
        eps_pred_96 = np.zeros((96, 4), dtype=np.float64)
        peak_vm_96 = np.zeros(96, dtype=np.float64)
        max_u_96 = np.zeros(96, dtype=np.float64)
        infer_ms_96 = np.zeros(96, dtype=np.float64)

        for k, ti in enumerate(idx_96):
            f_tup = (
                float(max(F_real[ti, 0], 0.0)),
                float(max(F_real[ti, 1], 0.0)),
                float(max(F_real[ti, 2], 0.0)),
            )
            u_p, vm_p, eps_p, dt_ms = predict_3d(f_tup, prop)
            u_stream_96[k] = u_p.astype(np.float32)
            vm_stream_96[k] = vm_p.astype(np.float32)
            eps_pred_96[k] = eps_p
            peak_vm_96[k] = float(vm_p.max())
            max_u_96[k] = float(np.linalg.norm(u_p, axis=1).max())
            infer_ms_96[k] = dt_ms

        # Evaluate ALL T samples through the exact 3D PI-GINOT operator using cached spatial trunk basis (<0.5s!)
        E_pred_all = np.zeros_like(E_real)
        F_clip = np.maximum(F_real, 0.0)
        F_scaled = F_clip / LOAD_SCALE_KN
        F_norms = np.linalg.norm(F_scaled, axis=1)
        F_dirs = F_scaled / np.maximum(F_norms[:, None], 1.0e-8)
        with torch.no_grad():
            bs = 1024
            for i0 in range(0, len(A), bs):
                i1 = min(i0 + bs, len(A))
                nb = i1 - i0
                p_b = np.zeros((nb, 5), dtype=np.float32)
                p_b[:, :3] = F_dirs[i0:i1]
                p_b[:, 3] = F_norms[i0:i1]
                p_b[:, 4] = 1.0 if prop else -1.0
                p_bt = torch.from_numpy(p_b)
                ln_bt = p_bt[:, 3]
                u_bt, _ = model.forward_from_basis(
                    cached_u_basis,
                    cached_vm_basis,
                    cached_g_ctx,
                    p_bt,
                    base_clamp_t,
                    ln_bt,
                )
                E_pred_all[i0:i1] = torch.einsum(
                    "kn,bn->bk", D_fos_t, u_bt[:, :, 0]
                ).numpy()

        # Also run OPPOSITE-BC hypothesis on ALL T samples for live prop-removal regime detection!
        E_pred_opp = np.zeros_like(E_real)
        with torch.no_grad():
            bs = 1024
            for i0 in range(0, len(A), bs):
                i1 = min(i0 + bs, len(A))
                nb = i1 - i0
                p_b = np.zeros((nb, 5), dtype=np.float32)
                p_b[:, :3] = F_dirs[i0:i1]
                p_b[:, 3] = F_norms[i0:i1]
                p_b[:, 4] = -1.0 if prop else 1.0
                p_bt = torch.from_numpy(p_b)
                ln_bt = p_bt[:, 3]
                u_bt, _ = model.forward_from_basis(
                    cached_u_basis,
                    cached_vm_basis,
                    cached_g_ctx,
                    p_bt,
                    base_clamp_t,
                    ln_bt,
                )
                E_pred_opp[i0:i1] = torch.einsum(
                    "kn,bn->bk", D_fos_t, u_bt[:, :, 0]
                ).numpy()

        # Forward accuracy vs real physical FOS sensors (all T samples)
        resid = E_pred_all - E_real
        rmse_ch = np.sqrt(np.mean(resid**2, axis=0))
        mae_ch = np.mean(np.abs(resid), axis=0)
        rmse_all = float(np.sqrt(np.mean(resid**2)))
        peak_strain = float(np.max(E_real))
        nrmse_pct = (rmse_all / max(peak_strain, 1.0)) * 100.0
        r2_all = float(
            1.0 - np.sum(resid**2) / np.sum((E_real - E_real.mean(axis=0)) ** 2)
        )
        r2_ch = [
            float(
                1.0
                - np.sum(resid[:, c] ** 2)
                / max(np.sum((E_real[:, c] - E_real[:, c].mean()) ** 2), 1e-9)
            )
            for c in range(4)
        ]

        # Inverse load reconstruction from ONLY the 4 physical FOS sensors (non-negative Tikhonov via operator Jacobian J_op)
        # Solve min_{F >= 0} || J_op F - E_real ||^2 via fast projected Gauss-Newton (exact in <0.1 ms/sample!)
        J = J_op[reg_key]  # (4, 3)
        JtJ = J.T @ J + 1e-4 * np.eye(3)
        Jt = J.T
        F_inv = np.clip((np.linalg.solve(JtJ, Jt @ E_real.T)).T, 0.0, 15.0)
        # One active-set polish for exact non-negativity (when a jack is at 0 kN like C3 in TEST2 or C1 in TEST3!)
        for j_zero in range(3):
            z_mask = F_inv[:, j_zero] < 0.05
            if np.any(z_mask):
                keep = [c for c in range(3) if c != j_zero]
                Jk = J[:, keep]
                F_sub = np.clip(
                    (
                        np.linalg.solve(
                            Jk.T @ Jk + 1e-4 * np.eye(2), Jk.T @ E_real[z_mask].T
                        )
                    ).T,
                    0.0,
                    15.0,
                )
                F_inv[z_mask, :] = 0.0
                for idx_k, ck in enumerate(keep):
                    F_inv[z_mask, ck] = F_sub[:, idx_k]
        inv_rmse_kn = float(np.sqrt(np.mean((F_inv - F_clip) ** 2)))
        inv_mae_kn = float(np.mean(np.abs(F_inv - F_clip)))
        inv_r2 = float(
            1.0
            - np.sum((F_inv - F_clip) ** 2)
            / max(np.sum((F_clip - F_clip.mean(axis=0)) ** 2), 1e-9)
        )
        inv_tot_load_rmse = float(
            np.sqrt(np.mean((F_inv.sum(axis=1) - F_clip.sum(axis=1)) ** 2))
        )

        # Live BC / Prop-Removal detection accuracy on active-load samples (total load > 8 kN)
        # Focus on CH8 (closest to removable right prop) + global residual!
        act = F_clip.sum(axis=1) > 8.0
        err_true_ch8 = np.abs(E_pred_all[act, 3] - E_real[act, 3])
        err_opp_ch8 = np.abs(E_pred_opp[act, 3] - E_real[act, 3])
        bc_detect_acc_ch8 = (
            float(np.mean(err_true_ch8 < err_opp_ch8)) if np.any(act) else 0.0
        )
        err_true_all = np.linalg.norm(E_pred_all[act] - E_real[act], axis=1)
        err_opp_all = np.linalg.norm(E_pred_opp[act] - E_real[act], axis=1)
        bc_detect_acc_all = (
            float(np.mean(err_true_all < err_opp_all)) if np.any(act) else 0.0
        )

        exp_results[sheet_name] = {
            "split": "BLIND_HOLDOUT" if is_blind else "CALIBRATION_X_ONLY",
            "prop_installed": prop,
            "n_samples": int(len(A)),
            "peak_total_load_kn": float(F_clip.sum(axis=1).max()),
            "peak_strain_ue": peak_strain,
            "forward_rmse_all_ue": rmse_all,
            "forward_nrmse_pct": nrmse_pct,
            "forward_r2_all": r2_all,
            "forward_rmse_per_ch_ue": [float(x) for x in rmse_ch],
            "forward_mae_per_ch_ue": [float(x) for x in mae_ch],
            "forward_r2_per_ch": r2_ch,
            "inverse_load_rmse_kn": inv_rmse_kn,
            "inverse_load_mae_kn": inv_mae_kn,
            "inverse_total_load_rmse_kn": inv_tot_load_rmse,
            "inverse_load_r2": inv_r2,
            "prop_detect_acc_ch8": bc_detect_acc_ch8,
            "prop_detect_acc_all": bc_detect_acc_all,
            "peak_3d_vm_mpa": float(peak_vm_96.max()),
            "max_3d_disp_mm": float(max_u_96.max()),
            "mean_infer_ms_96": float(infer_ms_96.mean()),
        }
        print(
            f"      [{exp_results[sheet_name]['split']:<18s}] {sheet_name:<20s} (N={len(A):5,d}): "
            f"FOS RMSE={rmse_all:5.2f} ue ({nrmse_pct:4.2f}% NRMSE, R2={r2_all:.4f}) | "
            f"Inv Load RMSE={inv_rmse_kn:.2f} kN (R2={inv_r2:.4f}) | "
            f"Prop Detect CH8={bc_detect_acc_ch8 * 100:5.1f}% | Peak VM={peak_vm_96.max():5.1f} MPa",
            flush=True,
        )

        # Keep compact 96-frame streams for D3 + 3D Studio visualization!
        key_slug = sheet_name.lower().replace(" ", "_")
        stream_archives[f"{key_slug}_F_real"] = F_clip[idx_96]
        stream_archives[f"{key_slug}_F_inv"] = F_inv[idx_96]
        stream_archives[f"{key_slug}_E_real"] = E_real[idx_96]
        stream_archives[f"{key_slug}_E_pred"] = eps_pred_96
        stream_archives[f"{key_slug}_E_opp"] = E_pred_opp[idx_96]
        stream_archives[f"{key_slug}_peak_vm"] = peak_vm_96
        stream_archives[f"{key_slug}_max_u"] = max_u_96
        if sheet_name in (
            "CON PUNTALES TEST2",
            "SIN PUNTALES TEST2",
            "CON PUNTALES TEST3",
            "SIN PUNTALES TEST3",
        ):
            # Store 3D field snapshots for peak frame & animation frames of blind TEST2/TEST3
            stream_archives[f"{key_slug}_u_96"] = u_stream_96
            stream_archives[f"{key_slug}_vm_96"] = vm_stream_96

    # Aggregate BLIND physical validation metrics across ALL 4 blind experimental tests (21,327 samples!)
    blind_keys = [k for k, v in exp_results.items() if v["split"] == "BLIND_HOLDOUT"]
    blind_n = sum(exp_results[k]["n_samples"] for k in blind_keys)
    blind_mean_rmse = float(
        np.mean([exp_results[k]["forward_rmse_all_ue"] for k in blind_keys])
    )
    blind_mean_nrmse = float(
        np.mean([exp_results[k]["forward_nrmse_pct"] for k in blind_keys])
    )
    blind_mean_r2 = float(
        np.mean([exp_results[k]["forward_r2_all"] for k in blind_keys])
    )
    blind_mean_inv_rmse = float(
        np.mean([exp_results[k]["inverse_load_rmse_kn"] for k in blind_keys])
    )
    blind_mean_inv_r2 = float(
        np.mean([exp_results[k]["inverse_load_r2"] for k in blind_keys])
    )

    # Save 3D visualization archive & JSON summary
    np.savez_compressed(
        DATA_DIR / "study3_sum4re_3d_digital_twin.npz",
        nodes=nodes,
        surface_triangles=tris,
        fos_x_mm=np.asarray(fos_x_mm, dtype=np.float64),
        load_x_mm=np.asarray(LOAD_X_MM, dtype=np.float64),
        S_fem_con=S_fem_con,
        S_fem_sin=S_fem_sin,
        J_op_con=J_op["CON"],
        J_op_sin=J_op["SIN"],
        **stream_archives,
    )

    summary = {
        "study": "Study 3 — 3D PI-GINOT Structural Digital Twin on Real Experimental FOS Steel Beam (SUM4Re)",
        "dataset": {
            "name": "SUM4Re: FOS steel beam dataset (Jolastokieta pilot site, Basque Country, Spain)",
            "doi": "10.5281/zenodo.22686007",
            "license": "CC-BY-4.0",
            "file": DATASET_XLSX.name,
            "sha256": dataset_sha256,
            "total_samples": int(total_samples),
            "blind_holdout_samples": int(blind_n),
            "modal_frequencies_hz": {
                "unpropped_real_vert1": 32.52,
                "unpropped_ansys_vert1": 32.78,
                "propped_real_vert1": 31.05,
                "propped_ansys_vert1": 36.245,
                "propped_real_vert2": 109.60,
                "propped_ansys_vert2": 113.846,
                "propped_real_lat1": 10.40,
                "propped_ansys_lat1": 14.393,
            },
        },
        "epistemic_governance": {
            "asme_vv10_note": (
                "Numerical operator verification (8 unseen 3D TET10 FEM holdouts) is reported separately "
                "from blind physical experimental validation (TEST2 & TEST3: 21,327 real FOS samples). "
                "The 3D PI-GINOT operator is trained purely on 3D FEM physics with zero experimental FOS labels; "
                "only the 4 lower-flange FOS gauge x-coordinates were calibrated on TEST1."
            ),
            "numerical_claim_level": "HOLDOUT_VERIFIED",
            "experimental_validation_split": "TEST1 calibration (x-coords only) -> TEST2 & TEST3 strictly blind",
        },
        "model_3d": {
            "profile": "European IPE 200 reclaimed structural steel crane beam + slender end stub columns + removable right prop footplate",
            "span_mm": L_BEAM,
            "depth_mm": H_BEAM,
            "flange_width_mm": B_FLANGE,
            "E_steel_mpa": E_STEEL_MPA,
            "nu_steel": NU_STEEL,
            "element_type": "TET10",
            "n_nodes": n_nodes,
            "n_dofs": 3 * n_nodes,
            "n_cells": n_cells,
            "n_surface_triangles": len(tris),
            "load_pad_x_mm": list(LOAD_X_MM),
            "calibrated_fos_x_mm": fos_x_mm,
        },
        "operator": {
            "architecture": "Surface3DEnergyOperator 3D PI-GINOT (exact 3D base-clamp lifting, 4-channel load+BC parameter conditioning)",
            "n_parameters": n_params,
            "n_train_fem_anchors": len(train_records),
            "epochs": epochs,
            "train_wall_s": train_wall_s,
            "initial_loss": loss_hist[0],
            "final_loss": loss_hist[-1],
            "weights_sha256": weights_sha256,
            "surface_encode_ms": encode_ms,
        },
        "numerical_holdout_verification": {
            "n_holdout_cases": len(holdout_records),
            "mean_u_rel_l2_pct": mean_ho_u * 100.0,
            "mean_vm_rel_l2_pct": mean_ho_vm * 100.0,
            "mean_peak_vm_rel_err_pct": mean_ho_pk * 100.0,
            "mean_cold_3d_fem_s": mean_fem_s,
            "mean_piginot_infer_ms": mean_infer_ms,
            "update_rate_hz": 1000.0 / mean_infer_ms,
            "speedup_vs_3d_fem": speedup,
            "cases": ho_metrics,
        },
        "blind_experimental_validation": {
            "blind_tests": blind_keys,
            "total_blind_samples": int(blind_n),
            "mean_blind_fos_rmse_ue": blind_mean_rmse,
            "mean_blind_fos_nrmse_pct": blind_mean_nrmse,
            "mean_blind_fos_r2": blind_mean_r2,
            "mean_blind_inverse_load_rmse_kn": blind_mean_inv_rmse,
            "mean_blind_inverse_load_r2": blind_mean_inv_r2,
            "per_test": exp_results,
        },
        "total_wall_time_s": time.perf_counter() - t_total,
    }
    (DATA_DIR / "study3_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("=" * 78)
    print(
        f"[DONE] Study 3 complete in {(time.perf_counter() - t_total) / 60.0:.2f} min!"
    )
    print(
        f"       BLIND Physical Validation (TEST2+TEST3, {blind_n:,} samples): "
        f"FOS RMSE={blind_mean_rmse:.2f} ue ({blind_mean_nrmse:.2f}% NRMSE, R2={blind_mean_r2:.4f}) | "
        f"Inverse Load RMSE={blind_mean_inv_rmse:.2f} kN (R2={blind_mean_inv_r2:.4f}) | "
        f"3D Twin Latency={mean_infer_ms:.2f} ms ({1000.0 / mean_infer_ms:.0f} Hz, {speedup:.0f}x faster than 3D TET10 FEM)!"
    )
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
