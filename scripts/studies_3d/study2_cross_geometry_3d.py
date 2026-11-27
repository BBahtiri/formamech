"""Study 2 (Upgraded) — Cross-Geometry 3D Mechanics PI-GINOT Across Three Real Open-Source 3D Geometry Families.

Trains ONE single higher-capacity 20-channel ``MultiTopologyOperator3D``
(832,203 parameters: H=128, S=24, num_blocks=3, F=5, decoder_blocks=3,
Procrustes-stabilized Fiedler harmonics, exact 3D ADF Dirichlet lifting,
Buckingham-Pi dimensionless continuum-mechanics scaling + load-path cross-section
invariants, and MultiTopologyOODDetector3D) jointly across THREE genuinely
different real open-source 3D mechanical geometry families (210 total real 3D CAD parts):

  1. 3D Double-Arm Support Brackets   (HaomingLuo/AgentFEM-Bracket-Elasticity, CC-BY-4.0)
     - ALL 128 TRAIN / 12 DEV / 16 BLIND HOLDOUT geometries (5,574 surface tris, Genus 4, chi=-6)
  2. 3D Solid-FE 90-deg Pipe Elbows   (HaomingLuo/AgentFEM-Pipe-Components, CC-BY-4.0)
     - 14 TRAIN / 4 DEV / 6 BLIND HOLDOUT (2x denser 4,800-tri Q2-subdivided curved surface, Genus 1, chi=0)
  3. 3D Solid-FE Straight Pipes       (HaomingLuo/AgentFEM-Pipe-Components, CC-BY-4.0)
     - 14 TRAIN / 4 DEV / 6 BLIND HOLDOUT (2x denser 4,800-tri Q2-subdivided curved surface, Genus 1, chi=0)
  +  3.2 m Industrial Power-Piping OOD Elbows (35,280 DOFs, E=190 GPa, 4,992-tri Q2-subdivided surface):
     - 2 ADAPT / 4 BLIND TEST for Level 2 Fail-Closed OOD Audit & Level 3 Budgeted 3D LoRA Transfer.

ZERO geometry-family IDs or CAD template parameters are passed to the network.
All 27 conditioning features are strictly dimensionless universal geometric and
constitutive invariants (aspect ratios, slenderness, compactness, Poisson's ratio,
lever arm, principal inertia ratios, and clamped-to-loaded cross-section bands).

Epistemic scope (ASME V&V 10): Numerical surrogate verification against blind
held-out external 3D FEM geometries — not physical coupon/hardware testing.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors_file
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from piginot.bounds.conformal_3d import (
    Conformal3DCalibrator,
    Conformal3DDevSample,
    verify_as6983_holdout_coverage,
)
from piginot.contracts import canonical_json_bytes
from piginot.geometry.mesh3d import (
    CanonicalMesh3D,
    CellType3D,
    _tet_signed_volume,
    make_brick_tet4_mesh,
)
from piginot.lifting.adf_3d import (
    GeneralFaceADFLifting3D,
    build_adf_lifting_from_mesh_faces,
)
from piginot.operators.geometry_3d import (
    GeometryDescriptor3D,
    compute_geometry_descriptor_3d,
    pad_and_stack_surface_3d_batch,
)
from piginot.operators.low_rank_adapter_3d import (
    LowRankAdapter3DConfig,
    extract_lora_3d_state_dict,
    inject_lora_3d,
)
from piginot.operators.multi_topology_3d import (
    MultiTopologyOODDetector3D,
    MultiTopologyOperator3D,
    build_multi_topology_surface_tensors,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACKET_DIR = REPO_ROOT / ".formamech-runs" / "datasets" / "agentfem-bracket-3d"
PIPE_DIR = REPO_ROOT / ".formamech-runs" / "datasets" / "agentfem-pipe-3d"
OUT_DIR = (
    REPO_ROOT
    / ".formamech-runs"
    / "open_source_3d_studies"
    / "study2_cross_geometry_3d"
)
CACHE_DIR = OUT_DIR / "surface_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOAD_FORCE_N = 1000.0
AL_DENSITY_KG_M3 = 2700.0
PARAM_DIM = 27

# Upgraded PI-GINOT architecture hyperparameters
HIDDEN_DIM = 128
SLICE_COUNT = 24
NUM_BLOCKS = 3
NUM_HEADS = 4
POS_FREQS = 5
DECODER_BLOCKS = 3

# Dense 3D query counts per geometry per step (total = 1,024 3D nodes/geometry/step)
Q_UNI = 832
Q_HOT = 128
Q_LOAD = 64
Q_TOTAL = Q_UNI + Q_HOT + Q_LOAD


def _orient_closed_surface_outward(
    pts_mm: np.ndarray, tris: np.ndarray
) -> tuple[np.ndarray, float]:
    """Ensure consistent manifold triangle winding & positive outward enclosed volume."""
    n_f = len(tris)
    edge_adj: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for fi in range(n_f):
        a, b, c = int(tris[fi, 0]), int(tris[fi, 1]), int(tris[fi, 2])
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_adj.setdefault(key, []).append((fi, u, v))

    neighbors: list[list[tuple[int, bool]]] = [[] for _ in range(n_f)]
    for entries in edge_adj.values():
        if len(entries) == 2:
            (f1, u1, v1), (f2, u2, v2) = entries
            same_dir = (u1 == u2) and (v1 == v2)
            neighbors[f1].append((f2, same_dir))
            neighbors[f2].append((f1, same_dir))

    oriented = tris.copy()
    visited = np.zeros(n_f, dtype=bool)
    for start in range(n_f):
        if visited[start]:
            continue
        comp_faces: list[int] = []
        flip_flag = np.zeros(n_f, dtype=bool)
        q: deque[int] = deque([start])
        visited[start] = True
        while q:
            cur = q.popleft()
            comp_faces.append(cur)
            for nb, must_flip_rel in neighbors[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    flip_flag[nb] = flip_flag[cur] ^ must_flip_rel
                    q.append(nb)
        for fi in comp_faces:
            if flip_flag[fi]:
                oriented[fi, 1], oriented[fi, 2] = oriented[fi, 2], oriented[fi, 1]

    p0 = pts_mm[oriented[:, 0]]
    p1 = pts_mm[oriented[:, 1]]
    p2 = pts_mm[oriented[:, 2]]
    signed_vol = float(np.sum(np.einsum("ij,ij->i", p0, np.cross(p1, p2))) / 6.0)
    if signed_vol < 0.0:
        oriented = oriented[:, [0, 2, 1]]
        signed_vol = -signed_vol
    return oriented, signed_vol


def _universal_physics_invariants(
    desc: GeometryDescriptor3D,
    *,
    volume_mm3: float,
    e_mpa: float,
    nu: float,
    clamped_centroid_mm: np.ndarray,
    loaded_centroid_mm: np.ndarray,
    surface_centroids_mm: np.ndarray,
) -> tuple[list[float], float]:
    """Compute 27 universal dimensionless Buckingham-Pi + load-path cross-section invariants (ZERO family IDs!)."""
    ex, ey, ez = desc.extent_mm
    l_diag = max(math.sqrt(ex * ex + ey * ey + ez * ez), 1.0)
    area = max(float(desc.surface_area_mm2), 1.0)
    vol = max(float(volume_mm3), 1.0)
    h_eq = max(vol / area, 0.1)
    compact = (area**3) / max(36.0 * math.pi * (vol**2), 1.0e-12)
    axis = loaded_centroid_mm - clamped_centroid_mm
    lever = max(float(np.linalg.norm(axis)), 1.0)
    e_ax = axis / lever

    # Buckingham-Pi slender-elastic displacement scale: F * L_diag / (E * h_eq^2)
    u_scale_mm = float(LOAD_FORCE_N * l_diag / max(e_mpa * h_eq * h_eq, 1.0e-9))

    e0, e1, e2 = desc.inertia_eigenvalues
    inv_l2 = 1.0 / (l_diag * l_diag)
    c_mean = surface_centroids_mm.mean(axis=0)
    r_mid = float(
        np.std(
            np.linalg.norm((surface_centroids_mm - c_mean[None, :]) / l_diag, axis=1)
        )
    )

    # Universal 12 base invariants
    p_vec: list[float] = [
        ex / l_diag,
        ey / l_diag,
        ez / l_diag,
        math.log10(max(h_eq / l_diag, 1.0e-4)) + 2.0,
        math.log10(max(compact, 1.0)) / 3.0,
        (float(nu) - 0.30) / 0.10,
        lever / l_diag,
        float(desc.clamped_area_fraction) - float(desc.loaded_area_fraction),
        math.sqrt(max(e0 * inv_l2, 1.0e-12)),
        math.sqrt(max(e1 * inv_l2, 1.0e-12)),
        math.sqrt(max(e2 * inv_l2, 1.0e-12)),
        r_mid,
    ]

    # + 15 universal load-path cross-section invariants along clamped -> loaded span (3 bands x 5 features)
    # Captures arch bow, waist taper, wall thickness, and elbow bend curvature identically for ANY 3D solid!
    rel = surface_centroids_mm - clamped_centroid_mm[None, :]
    s_proj = (rel @ e_ax) / lever
    perp = rel - (rel @ e_ax)[:, None] * e_ax[None, :]
    for b0, b1 in ((0.15, 0.35), (0.40, 0.60), (0.65, 0.85)):
        mask = (s_proj >= b0) & (s_proj <= b1)
        bp = perp[mask] if int(np.sum(mask)) >= 6 else perp
        r_p = np.linalg.norm(bp, axis=1) / l_diag
        cov2 = np.cov(bp.T) / (l_diag * l_diag)
        ev = np.sort(np.maximum(np.linalg.eigvalsh(0.5 * (cov2 + cov2.T)), 0.0))
        p_vec.extend(
            [
                float(r_p.mean()),
                float(np.percentile(r_p, 10)),
                float(np.percentile(r_p, 90)),
                math.sqrt(float(ev[1])),
                math.sqrt(float(ev[2])),
            ]
        )
    assert len(p_vec) == PARAM_DIM
    return p_vec, u_scale_mm


@dataclass
class MechanicsSample3D:
    sample_id: str
    family: str
    dataset_source: str
    split: str
    mesh: CanonicalMesh3D
    descriptor: GeometryDescriptor3D
    volume_mm3: float
    e_mpa: float
    nu: float
    u_scale_mm: float
    p_vec: list[float]
    query_xyz_mm: np.ndarray
    u_true_mm: np.ndarray
    clamped_node_ids: np.ndarray
    loaded_node_ids: np.ndarray
    surf_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    lifting: GeneralFaceADFLifting3D
    n_sc: torch.Tensor
    n_off: torch.Tensor
    q_xyz_t: torch.Tensor
    u_true_t: torch.Tensor
    top_hot_idx: torch.Tensor
    loaded_idx_fixed: torch.Tensor
    euler_chi: int
    dofs: int


def _cached_surface_tensors(
    cache_key: str,
    mesh: CanonicalMesh3D,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cfile = CACHE_DIR / f"{cache_key}.npz"
    if cfile.is_file():
        with np.load(cfile) as z:
            return (
                torch.tensor(z["c"], dtype=dtype),
                torch.tensor(z["n"], dtype=dtype),
                torch.tensor(z["a"], dtype=dtype),
                torch.tensor(z["f"], dtype=dtype),
            )
    c_t, n_t, a_t, f_t = build_multi_topology_surface_tensors(
        mesh, clamped_face_tags=("clamped",), loaded_face_tags=("loaded",), dtype=dtype
    )
    np.savez_compressed(
        cfile,
        c=c_t.cpu().numpy(),
        n=n_t.cpu().numpy(),
        a=a_t.cpu().numpy(),
        f=f_t.cpu().numpy(),
    )
    return c_t, n_t, a_t, f_t


def _fixed_size_index(ids: np.ndarray, k: int) -> torch.Tensor:
    if len(ids) >= k:
        step = np.linspace(0, len(ids) - 1, k, dtype=np.int64)
        return torch.tensor(ids[step], dtype=torch.long)
    rep = np.resize(ids, k)
    return torch.tensor(rep, dtype=torch.long)


def _load_bracket_sample(
    zf: zipfile.ZipFile,
    design_dir: str,
    split: str,
    dtype: torch.dtype,
) -> MechanicsSample3D:
    with np.load(io.BytesIO(zf.read(f"{design_dir}/fields.npz"))) as d:
        pts_mm = np.asarray(d["points"], dtype=np.float64) * 1000.0
        u_raw_mm = np.asarray(d["displacement"], dtype=np.float64) * 1000.0
        tris_raw = np.asarray(d["triangles"], dtype=np.int64)
    summary = json.loads(zf.read(f"{design_dir}/summary.json").decode("utf-8"))
    phys_load = float(summary.get("load_N", 1000.0))
    u_mm = u_raw_mm * (LOAD_FORCE_N / max(phys_load, 1.0e-12))
    vol_mm3 = float(summary["mass_kg"]) / AL_DENSITY_KG_M3 * 1.0e9
    e_mpa = 70000.0
    nu = 0.33

    tris_out, signed_v = _orient_closed_surface_outward(pts_mm, tris_raw)
    if signed_v > 0.5 * vol_mm3:
        vol_mm3 = signed_v

    z_min, z_max = float(pts_mm[:, 2].min()), float(pts_mm[:, 2].max())
    clamp_t = [
        i
        for i, (a, b, c) in enumerate(tris_out)
        if np.all(pts_mm[[a, b, c], 2] <= z_min + 0.25)
    ]
    load_t = [
        i
        for i, (a, b, c) in enumerate(tris_out)
        if np.all(pts_mm[[a, b, c], 2] >= z_max - 0.25)
    ]
    used_t = set(clamp_t) | set(load_t)
    free_t = [i for i in range(len(tris_out)) if i not in used_t]

    i0 = 0
    i1 = int(np.argmax(np.linalg.norm(pts_mm - pts_mm[i0], axis=1)))
    e01 = pts_mm[i1] - pts_mm[i0]
    i2 = int(
        np.argmax(np.linalg.norm(np.cross(pts_mm - pts_mm[i0], e01[None, :]), axis=1))
    )
    n012 = np.cross(e01, pts_mm[i2] - pts_mm[i0])
    i3 = int(np.argmax(np.abs((pts_mm - pts_mm[i0]) @ n012)))
    v4 = _tet_signed_volume(
        tuple(pts_mm[i0]), tuple(pts_mm[i1]), tuple(pts_mm[i2]), tuple(pts_mm[i3])
    )
    tet0 = (i0, i1, i2, i3) if v4 > 0.0 else (i0, i2, i1, i3)

    mesh = CanonicalMesh3D(
        nodes=tuple((float(p[0]), float(p[1]), float(p[2])) for p in pts_mm),
        cells=(tet0,),
        cell_type=CellType3D.TET4,
        surface_triangles=tuple((int(a), int(b), int(c)) for a, b, c in tris_out),
        semantic_face_tags={
            "clamped": tuple(clamp_t),
            "loaded": tuple(load_t),
            "free": tuple(free_t),
        },
        metadata={"sample_id": design_dir, "family": "bracket"},
    )
    raw_desc = compute_geometry_descriptor_3d(
        mesh, clamped_face_tags=("clamped",), loaded_face_tags=("loaded",)
    )
    area_s = raw_desc.surface_area_mm2
    compact_exact = (area_s**3) / max(36.0 * math.pi * (vol_mm3**2), 1.0e-12)
    desc = GeometryDescriptor3D(
        mesh_id=raw_desc.mesh_id,
        volume_mm3=vol_mm3,
        surface_area_mm2=area_s,
        compactness=compact_exact,
        extent_mm=raw_desc.extent_mm,
        inertia_eigenvalues=raw_desc.inertia_eigenvalues,
        clamped_area_fraction=raw_desc.clamped_area_fraction,
        loaded_area_fraction=raw_desc.loaded_area_fraction,
        surface_euler_characteristic=raw_desc.surface_euler_characteristic,
    )

    clamped_ids = np.nonzero(pts_mm[:, 2] <= z_min + 0.25)[0].astype(np.int64)
    loaded_ids = np.nonzero(pts_mm[:, 2] >= z_max - 0.25)[0].astype(np.int64)
    c_surf_mm = np.asarray(mesh.surface_geometry()[0], dtype=np.float64)
    p_vec, u_scale_mm = _universal_physics_invariants(
        desc,
        volume_mm3=vol_mm3,
        e_mpa=e_mpa,
        nu=nu,
        clamped_centroid_mm=pts_mm[clamped_ids].mean(axis=0),
        loaded_centroid_mm=pts_mm[loaded_ids].mean(axis=0),
        surface_centroids_mm=c_surf_mm,
    )

    sid = f"bracket_{design_dir.split('/')[-1]}"
    surf = _cached_surface_tensors(sid, mesh, dtype)
    lift = build_adf_lifting_from_mesh_faces(
        mesh, ("clamped",), displacement_scale_mm=u_scale_mm
    )
    q_t = torch.tensor(pts_mm, dtype=dtype).unsqueeze(0)
    n_sc, n_off = lift.scale_and_offset(q_t)
    u_t = torch.tensor(u_mm, dtype=dtype)
    umag = torch.linalg.vector_norm(u_t, dim=-1)
    top_hot = torch.topk(umag, k=min(Q_HOT, len(umag))).indices

    return MechanicsSample3D(
        sample_id=sid,
        family="bracket",
        dataset_source="HaomingLuo/AgentFEM-Bracket-Elasticity",
        split=split,
        mesh=mesh,
        descriptor=desc,
        volume_mm3=vol_mm3,
        e_mpa=e_mpa,
        nu=nu,
        u_scale_mm=u_scale_mm,
        p_vec=p_vec,
        query_xyz_mm=pts_mm,
        u_true_mm=u_mm,
        clamped_node_ids=clamped_ids,
        loaded_node_ids=loaded_ids,
        surf_tensors=surf,
        lifting=lift,
        n_sc=n_sc,
        n_off=n_off,
        q_xyz_t=q_t,
        u_true_t=u_t,
        top_hot_idx=top_hot,
        loaded_idx_fixed=_fixed_size_index(loaded_ids, Q_LOAD),
        euler_chi=int(desc.surface_euler_characteristic),
        dofs=int(summary.get("dofs", 3 * len(pts_mm))),
    )


def _load_pipe_sample(
    comp_dir: Path,
    family: str,
    split: str,
    dtype: torch.dtype,
) -> MechanicsSample3D:
    """Load real 3D pipe component with 2x denser Q2-center-subdivided curved surface mesh."""
    meta = json.loads((comp_dir / "metadata.json").read_text(encoding="utf-8"))
    spec = meta["spec"]
    kind = str(spec["kind"])
    e_mpa = float(spec["young"]) * 1.0e-6
    nu = float(spec["poisson"])

    with np.load(comp_dir / "mesh.npz") as m:
        pts_m = m["points"] if "points" in m else m["coordinates"]
        pts_q1_mm = np.asarray(pts_m, dtype=np.float64) * 1000.0
        cells_basix = np.asarray(m["cells"], dtype=np.int64)
    hex_std = cells_basix[:, [0, 1, 3, 2, 4, 5, 7, 6]]

    # Exact Q2 solid-FE coordinates & displacement field (clamped inlet port + F_z = -1000 N at outlet)
    with np.load(comp_dir / "component.npz") as c_npz:
        xyz_mm = np.asarray(c_npz["xyz"], dtype=np.float64) * 1000.0
        s_mat = np.asarray(c_npz["S"], dtype=np.float64)
        c_mat = np.asarray(c_npz["C"], dtype=np.float64)
    f_ext = np.array([0.0, 0.0, -LOAD_FORCE_N, 0.0, 0.0, 0.0], dtype=np.float64)
    q_out = np.linalg.solve(s_mat[6:, 6:], f_ext)
    q_full = np.concatenate([np.zeros(6, dtype=np.float64), q_out])
    u_mm = (c_mat @ q_full).reshape(-1, 3) * 1000.0

    # Build positive volume TET4s on Q1 nodes + 2x denser Q2-face-center-subdivided curved boundary surface!
    # Snapping each boundary quad center onto the exact Q2 node doubles surface triangle density
    # (2,400 -> 4,800 tris; 2,496 -> 4,992 for power elbows) while preserving exact chi = 0 (Genus 1)!
    tree_q2 = cKDTree(xyz_mm)
    nodes_ext: list[tuple[float, float, float]] = [
        (float(p[0]), float(p[1]), float(p[2])) for p in pts_q1_mm
    ]

    tets: list[tuple[int, int, int, int]] = []
    face_map: dict[
        tuple[int, int, int, int], list[tuple[tuple[int, int, int, int], np.ndarray]]
    ] = {}
    hex_faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
    ]
    for h in hex_std:
        n0, n1, n2, n3, n4, n5, n6, n7 = [int(x) for x in h]
        h_cent = pts_q1_mm[[n0, n1, n2, n3, n4, n5, n6, n7]].mean(axis=0)
        for a, b, c, d in (
            (n0, n1, n2, n6),
            (n0, n2, n3, n6),
            (n0, n3, n7, n6),
            (n0, n7, n4, n6),
            (n0, n4, n5, n6),
            (n0, n5, n1, n6),
        ):
            v = _tet_signed_volume(
                tuple(pts_q1_mm[a]),
                tuple(pts_q1_mm[b]),
                tuple(pts_q1_mm[c]),
                tuple(pts_q1_mm[d]),
            )
            if v < 0.0:
                tets.append((a, c, b, d))
            elif v > 0.0:
                tets.append((a, b, c, d))
        for fa, fb, fc, fd in hex_faces:
            quad = (int(h[fa]), int(h[fb]), int(h[fc]), int(h[fd]))
            key_s = sorted(quad)
            key = (key_s[0], key_s[1], key_s[2], key_s[3])
            face_map.setdefault(key, []).append((quad, h_cent))

    b_quads = [entries[0] for entries in face_map.values() if len(entries) == 1]
    q_centers_approx = np.asarray(
        [pts_q1_mm[list(quad)].mean(axis=0) for quad, _ in b_quads], dtype=np.float64
    )
    _, q2_center_ids = tree_q2.query(q_centers_approx, k=1)

    surf_tris: list[tuple[int, int, int]] = []
    for (quad, h_cent), q2_cid in zip(b_quads, q2_center_ids.tolist(), strict=True):
        qa, qb, qc, qd = quad
        pc_q2 = xyz_mm[int(q2_cid)]
        mid_idx = len(nodes_ext)
        nodes_ext.append((float(pc_q2[0]), float(pc_q2[1]), float(pc_q2[2])))
        for u_i, v_i in ((qa, qb), (qb, qc), (qc, qd), (qd, qa)):
            pa, pb = pts_q1_mm[u_i], pts_q1_mm[v_i]
            tri = (u_i, v_i, mid_idx)
            nrm = np.cross(pb - pa, pc_q2 - pa)
            if float(np.dot(nrm, ((pa + pb + pc_q2) / 3.0) - h_cent)) < 0.0:
                tri = (u_i, mid_idx, v_i)
            surf_tris.append(tri)

    nodes_ext_np = np.asarray(nodes_ext, dtype=np.float64)
    x_min, x_max = float(nodes_ext_np[:, 0].min()), float(nodes_ext_np[:, 0].max())
    y_max = float(nodes_ext_np[:, 1].max())
    tol = max(
        1.0e-3 * float(np.linalg.norm(nodes_ext_np.max(0) - nodes_ext_np.min(0))), 0.25
    )
    inlet_tids, outlet_tids, free_tids = [], [], []
    for ti, (a, b, c) in enumerate(surf_tris):
        tri_pts = nodes_ext_np[[a, b, c]]
        if np.all(np.abs(tri_pts[:, 0] - x_min) <= tol):
            inlet_tids.append(ti)
        elif kind == "straight" and np.all(np.abs(tri_pts[:, 0] - x_max) <= tol):
            outlet_tids.append(ti)
        elif kind == "elbow" and np.all(np.abs(tri_pts[:, 1] - y_max) <= tol):
            outlet_tids.append(ti)
        else:
            free_tids.append(ti)

    mesh = CanonicalMesh3D(
        nodes=tuple(nodes_ext),
        cells=tuple(tets),
        cell_type=CellType3D.TET4,
        surface_triangles=tuple(surf_tris),
        semantic_face_tags={
            "clamped": tuple(inlet_tids),
            "loaded": tuple(outlet_tids),
            "free": tuple(free_tids),
        },
        metadata={
            "sample_id": comp_dir.name,
            "family": family,
            "surface_subdivision": "q2_face_center_2x",
        },
    )
    desc = compute_geometry_descriptor_3d(
        mesh, clamped_face_tags=("clamped",), loaded_face_tags=("loaded",)
    )

    q2_x_min, q2_x_max, q2_y_max = (
        float(xyz_mm[:, 0].min()),
        float(xyz_mm[:, 0].max()),
        float(xyz_mm[:, 1].max()),
    )
    clamped_ids = np.nonzero(np.abs(xyz_mm[:, 0] - q2_x_min) <= tol)[0].astype(np.int64)
    if kind == "straight":
        loaded_ids = np.nonzero(np.abs(xyz_mm[:, 0] - q2_x_max) <= tol)[0].astype(
            np.int64
        )
    else:
        loaded_ids = np.nonzero(np.abs(xyz_mm[:, 1] - q2_y_max) <= tol)[0].astype(
            np.int64
        )

    c_surf_mm = np.asarray(mesh.surface_geometry()[0], dtype=np.float64)
    p_vec, u_scale_mm = _universal_physics_invariants(
        desc,
        volume_mm3=desc.volume_mm3,
        e_mpa=e_mpa,
        nu=nu,
        clamped_centroid_mm=xyz_mm[clamped_ids].mean(axis=0),
        loaded_centroid_mm=xyz_mm[loaded_ids].mean(axis=0),
        surface_centroids_mm=c_surf_mm,
    )

    surf = _cached_surface_tensors(f"{comp_dir.name}_q2dense", mesh, dtype)
    lift = build_adf_lifting_from_mesh_faces(
        mesh, ("clamped",), displacement_scale_mm=u_scale_mm
    )
    q_t = torch.tensor(xyz_mm, dtype=dtype).unsqueeze(0)
    n_sc, n_off = lift.scale_and_offset(q_t)
    u_t = torch.tensor(u_mm, dtype=dtype)
    umag = torch.linalg.vector_norm(u_t, dim=-1)
    top_hot = torch.topk(umag, k=min(Q_HOT, len(umag))).indices

    return MechanicsSample3D(
        sample_id=comp_dir.name,
        family=family,
        dataset_source="HaomingLuo/AgentFEM-Pipe-Components",
        split=split,
        mesh=mesh,
        descriptor=desc,
        volume_mm3=desc.volume_mm3,
        e_mpa=e_mpa,
        nu=nu,
        u_scale_mm=u_scale_mm,
        p_vec=p_vec,
        query_xyz_mm=xyz_mm,
        u_true_mm=u_mm,
        clamped_node_ids=clamped_ids,
        loaded_node_ids=loaded_ids,
        surf_tensors=surf,
        lifting=lift,
        n_sc=n_sc,
        n_off=n_off,
        q_xyz_t=q_t,
        u_true_t=u_t,
        top_hot_idx=top_hot,
        loaded_idx_fixed=_fixed_size_index(loaded_ids, Q_LOAD),
        euler_chi=int(desc.surface_euler_characteristic),
        dofs=int(meta["metrics"]["dofs"]),
    )


def _norm_p(p_vec: list[float], p_lo: np.ndarray, p_hi: np.ndarray) -> np.ndarray:
    raw = (
        2.0
        * (np.asarray(p_vec, dtype=np.float64) - p_lo)
        / np.maximum(p_hi - p_lo, 1.0e-6)
        - 1.0
    )
    return np.clip(raw, -2.5, 2.5)


def _eval_sample(
    mt_op: MultiTopologyOperator3D,
    sample: MechanicsSample3D,
    p_lo: np.ndarray,
    p_hi: np.ndarray,
    dtype: torch.dtype,
) -> dict[str, Any]:
    op = mt_op.operator
    op.eval()
    p_t = torch.tensor(_norm_p(sample.p_vec, p_lo, p_hi)[None, :], dtype=dtype)

    t0 = time.perf_counter()
    with torch.no_grad():
        enc = op.encode_surface(*sample.surf_tensors)
        u_pred_t = op.displacement(
            encoding=enc,
            query_xyz=sample.q_xyz_t,
            parameters=p_t,
            lifting_offset=sample.n_off,
            lifting_scale=sample.n_sc,
        )
        latent_vec = enc.slice_tokens.mean(dim=1)[0].cpu().numpy().astype(np.float64)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    u_pred = u_pred_t[0].cpu().numpy().astype(np.float64)
    u_true = sample.u_true_mm

    diff = u_pred - u_true
    rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(u_true), 1.0e-12))
    mae_mm = float(np.mean(np.abs(diff)))
    max_u_true = float(np.max(np.linalg.norm(u_true, axis=1)))
    max_u_pred = float(np.max(np.linalg.norm(u_pred, axis=1)))
    peak_rel_err = abs(max_u_pred - max_u_true) / max(max_u_true, 1.0e-12)
    clamp_err = (
        float(np.max(np.abs(u_pred[sample.clamped_node_ids])))
        if len(sample.clamped_node_ids)
        else 0.0
    )

    w_true_mj = float(
        0.5 * (-LOAD_FORCE_N) * np.mean(u_true[sample.loaded_node_ids, 2])
    )
    w_pred_mj = float(
        0.5 * (-LOAD_FORCE_N) * np.mean(u_pred[sample.loaded_node_ids, 2])
    )
    comp_rel_err = abs(w_pred_mj - w_true_mj) / max(abs(w_true_mj), 1.0e-12)

    return {
        "sample_id": sample.sample_id,
        "family": sample.family,
        "dataset_source": sample.dataset_source,
        "euler_chi": sample.euler_chi,
        "n_nodes": int(len(sample.query_xyz_mm)),
        "n_surface_tris": int(len(sample.mesh.surface_triangles)),
        "dofs": sample.dofs,
        "u_scale_mm": sample.u_scale_mm,
        "u_rel_l2": rel_l2,
        "u_mae_mm": mae_mm,
        "max_u_true_mm": max_u_true,
        "max_u_pred_mm": max_u_pred,
        "peak_u_rel_err": peak_rel_err,
        "compliance_true_mj": w_true_mj,
        "compliance_pred_mj": w_pred_mj,
        "compliance_rel_err": comp_rel_err,
        "clamped_max_err_mm": clamp_err,
        "infer_ms": infer_ms,
        "u_pred": u_pred,
        "latent_vec": latent_vec,
    }


def _train_vectorized(
    mt_op: MultiTopologyOperator3D,
    train_samples: list[MechanicsSample3D],
    p_lo: np.ndarray,
    p_hi: np.ndarray,
    *,
    epochs: int,
    lr: float,
    dtype: torch.dtype,
    dev_samples: list[MechanicsSample3D] | None = None,
    seed: int = 42,
    trainable_only: bool = False,
    warmup_epochs: int = 80,
    log_prefix: str = "      ",
) -> tuple[list[float], dict[str, list[float]], float]:
    """Fully vectorized batched 3D PI-GINOT training with dense 1,024-node queries & DEV checkpoint selection."""
    op = mt_op.operator
    params_to_opt = (
        [p for p in op.parameters() if p.requires_grad]
        if trainable_only
        else list(op.parameters())
    )
    opt = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=5.0e-5)

    def _lr_lambda(ep_idx: int) -> float:
        if warmup_epochs > 0 and ep_idx < warmup_epochs:
            return 0.15 + 0.85 * float(ep_idx + 1) / float(warmup_epochs)
        prog = float(ep_idx - warmup_epochs) / float(max(epochs - warmup_epochs, 1))
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, prog))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    p_mat = torch.tensor(
        np.asarray(
            [_norm_p(s.p_vec, p_lo, p_hi) for s in train_samples], dtype=np.float32
        ),
        dtype=dtype,
    )

    by_family: dict[str, list[int]] = {}
    for idx, s in enumerate(train_samples):
        by_family.setdefault(s.family, []).append(idx)
    families = sorted(by_family.keys())
    fam_weights = (
        {f: (0.44 if f == "bracket" else 0.28) for f in families}
        if len(families) == 3
        else {f: 1.0 / len(families) for f in families}
    )

    loss_hist: list[float] = []
    fam_hist: dict[str, list[float]] = {f: [] for f in families}
    best_dev_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    t0 = time.perf_counter()

    for ep in range(epochs):
        op.train()
        opt.zero_grad(set_to_none=True)

        chosen: list[int] = []
        fam_slices: dict[str, list[int]] = {f: [] for f in families}
        for fam in families:
            pool = by_family[fam]
            k_pick = min(6 if fam == "bracket" else 3, len(pool))
            perm = torch.randperm(len(pool), generator=gen)[:k_pick].tolist()
            for pi in perm:
                fam_slices[fam].append(len(chosen))
                chosen.append(pool[pi])

        s_cent, s_norm, s_area, s_feat, s_mask = pad_and_stack_surface_3d_batch(
            [train_samples[i].surf_tensors for i in chosen]
        )
        enc_all = op.encode_surface(s_cent, s_norm, s_area, s_feat, surface_mask=s_mask)

        # Build batched dense 3D query tensors (B, Q_TOTAL=1024, 3) for single vectorized decoder pass!
        q_list, off_list, sc_list, u_true_list = [], [], [], []
        for s_idx in chosen:
            samp = train_samples[s_idx]
            n_q = int(samp.q_xyz_t.shape[1])
            q_uni = torch.randperm(n_q, generator=gen)[:Q_UNI]
            sel = torch.cat([q_uni, samp.top_hot_idx, samp.loaded_idx_fixed], dim=0)
            q_list.append(samp.q_xyz_t[0, sel, :])
            off_list.append(samp.n_off[0, sel, :])
            sc_list.append(samp.n_sc[0, sel, :])
            u_true_list.append(samp.u_true_t[sel, :])

        q_batch = torch.stack(q_list, dim=0)  # (B, 1024, 3)
        off_batch = torch.stack(off_list, dim=0)
        sc_batch = torch.stack(sc_list, dim=0)
        u_true_b = torch.stack(u_true_list, dim=0)
        p_batch = p_mat[chosen]

        u_pred_b = op.displacement(
            encoding=enc_all,
            query_xyz=q_batch,
            parameters=p_batch,
            lifting_offset=off_batch,
            lifting_scale=sc_batch,
        )  # (B, 1024, 3)

        # Vectorized scale-invariant 3D Rel-L2 (uniform + hotspot + Clapeyron compliance)
        bs = len(chosen)
        u_p_uni, u_t_uni = u_pred_b[:, :Q_UNI, :], u_true_b[:, :Q_UNI, :]
        ref_n = torch.linalg.vector_norm(u_t_uni.reshape(bs, -1), dim=-1).clamp_min(
            1.0e-7
        )
        diff_n = torch.linalg.vector_norm((u_p_uni - u_t_uni).reshape(bs, -1), dim=-1)
        rel_uni = torch.sqrt((diff_n / ref_n).square() + 1.0e-8)

        u_p_hot, u_t_hot = (
            u_pred_b[:, Q_UNI : Q_UNI + Q_HOT, :],
            u_true_b[:, Q_UNI : Q_UNI + Q_HOT, :],
        )
        h_ref = torch.linalg.vector_norm(u_t_hot.reshape(bs, -1), dim=-1).clamp_min(
            1.0e-7
        )
        h_diff = torch.linalg.vector_norm((u_p_hot - u_t_hot).reshape(bs, -1), dim=-1)
        rel_hot = torch.sqrt((h_diff / h_ref).square() + 1.0e-8)

        uz_p = u_pred_b[:, Q_UNI + Q_HOT :, 2].mean(dim=-1)
        uz_t = u_true_b[:, Q_UNI + Q_HOT :, 2].mean(dim=-1)
        rel_comp = torch.abs(uz_p - uz_t) / uz_t.abs().clamp_min(1.0e-7)

        per_sample_loss = (
            0.55 * (rel_uni + 0.5 * rel_uni.square())
            + 0.30 * (rel_hot + 0.5 * rel_hot.square())
            + 0.15 * (rel_comp + 0.25 * rel_comp.square())
        )

        total_loss = torch.zeros((), dtype=dtype)
        for fam in families:
            f_idx = fam_slices[fam]
            total_loss = total_loss + fam_weights[fam] * per_sample_loss[f_idx].mean()
            fam_hist[fam].append(float(rel_uni[f_idx].detach().mean().item()))

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_opt, max_norm=3.0)
        opt.step()
        sched.step()
        loss_hist.append(float(total_loss.detach().item()))

        # Periodic DEV evaluation (strictly DEV-only, zero HOLDOUT leakage) to retain best generalizer
        if dev_samples and ((ep + 1) % 250 == 0 or ep == epochs - 1) and ep >= 499:
            dev_by_f: dict[str, list[float]] = {f: [] for f in families}
            for ds in dev_samples:
                dr = _eval_sample(mt_op, ds, p_lo, p_hi, dtype)
                dev_by_f[ds.family].append(dr["u_rel_l2"])
            dev_score = float(np.mean([float(np.mean(dev_by_f[f])) for f in families]))
            if dev_score < best_dev_score:
                best_dev_score = dev_score
                best_state = {
                    k: v.detach().cpu().clone() for k, v in op.state_dict().items()
                }
            d_str = " | ".join(
                f"{f}={np.mean(dev_by_f[f]) * 100:5.2f}%" for f in families
            )
            print(
                f"{log_prefix}epoch {ep + 1:4d}/{epochs} ({time.perf_counter() - t0:5.1f}s): "
                f"loss={loss_hist[-1]:.4f} | DEV u_L2: {d_str} (best={best_dev_score * 100:.2f}%)",
                flush=True,
            )
        elif ep in (0, 99, 249, epochs - 1) or (
            not dev_samples
            and ep in (epochs // 4 - 1, epochs // 2 - 1, (3 * epochs) // 4 - 1)
        ):
            f_str = " | ".join(f"{f}={fam_hist[f][-1] * 100:5.2f}%" for f in families)
            print(
                f"{log_prefix}epoch {ep + 1:4d}/{epochs} ({time.perf_counter() - t0:5.1f}s): "
                f"loss={loss_hist[-1]:.4f} | train u_L2: {f_str}",
                flush=True,
            )

    if best_state is not None:
        op.load_state_dict(best_state)
    return loss_hist, fam_hist, time.perf_counter() - t0


def main() -> None:
    t_global = time.perf_counter()
    dtype = torch.float32
    torch.manual_seed(42)
    np.random.seed(42)

    if (
        not (BRACKET_DIR / "fields-train-03.zip").is_file()
        or not (PIPE_DIR / "cache_manifest.json").is_file()
    ):
        from fetch_open_source_3d_datasets import fetch_brackets, fetch_pipes

        fetch_brackets()
        fetch_pipes()

    print(
        "[1/5] Loading FULL open-source 3D mechanics corpus (128 Brackets + 2x denser Q2-subdivided Elbows & Pipes + Power OOD)...",
        flush=True,
    )
    train_samples: list[MechanicsSample3D] = []
    dev_samples: list[MechanicsSample3D] = []
    holdout_samples: list[MechanicsSample3D] = []
    ood_power_samples: list[MechanicsSample3D] = []

    # Family 1: ALL 128 TRAIN Brackets (fields-train-00..03.zip) / 12 DEV / 16 BLIND HOLDOUT
    for zname in (
        "fields-train-00.zip",
        "fields-train-01.zip",
        "fields-train-02.zip",
        "fields-train-03.zip",
    ):
        with zipfile.ZipFile(BRACKET_DIR / zname) as zf:
            for d in sorted(
                {n.rsplit("/", 1)[0] for n in zf.namelist() if n.endswith("fields.npz")}
            ):
                train_samples.append(_load_bracket_sample(zf, d, "TRAIN", dtype))
    with zipfile.ZipFile(BRACKET_DIR / "fields-validation-00.zip") as zf:
        for d in sorted(
            {n.rsplit("/", 1)[0] for n in zf.namelist() if n.endswith("fields.npz")}
        )[:12]:
            dev_samples.append(_load_bracket_sample(zf, d, "DEV", dtype))
    with zipfile.ZipFile(BRACKET_DIR / "fields-test-00.zip") as zf:
        for d in sorted(
            {n.rsplit("/", 1)[0] for n in zf.namelist() if n.endswith("fields.npz")}
        )[:16]:
            holdout_samples.append(_load_bracket_sample(zf, d, "HOLDOUT", dtype))
    print(
        "      Loaded Brackets (Genus 4, chi=-6, ~5,574 tris): 128 TRAIN / 12 DEV / 16 UNSEEN HOLDOUT",
        flush=True,
    )

    # Family 2 & 3: AgentFEM-Pipe-Components with 2x denser Q2-subdivided curved surfaces (4,800 tris)
    manifest_pipe = json.loads(
        (PIPE_DIR / "cache_manifest.json").read_text(encoding="utf-8")
    )
    elbows = sorted(manifest_pipe["elbows"])
    straights = sorted(manifest_pipe["straights"])
    power_ood = sorted(manifest_pipe["power_elbows_ood"])

    def _split_24(items: list[str]) -> tuple[list[str], list[str], list[str]]:
        ho = [items[i] for i in (2, 6, 10, 14, 18, 22)]
        dv = [items[i] for i in (1, 9, 13, 21)]
        tr = [x for x in items if x not in ho and x not in dv]
        return tr, dv, ho

    for fam, items in (("pipe-elbow", elbows), ("pipe-straight", straights)):
        tr_n, dv_n, ho_n = _split_24(items)
        for name in tr_n:
            train_samples.append(
                _load_pipe_sample(PIPE_DIR / name, fam, "TRAIN", dtype)
            )
        for name in dv_n:
            dev_samples.append(_load_pipe_sample(PIPE_DIR / name, fam, "DEV", dtype))
        for name in ho_n:
            holdout_samples.append(
                _load_pipe_sample(PIPE_DIR / name, fam, "HOLDOUT", dtype)
            )
        print(
            f"      Loaded {fam:<14s} (Genus 1, chi= 0, 4,800 Q2 tris): {len(tr_n)} TRAIN / {len(dv_n)} DEV / {len(ho_n)} UNSEEN HOLDOUT",
            flush=True,
        )

    for idx, name in enumerate(power_ood):
        sp = "OOD_ADAPT" if idx in (0, 4) else "OOD_TEST"
        ood_power_samples.append(
            _load_pipe_sample(PIPE_DIR / name, "power-elbow-ood", sp, dtype)
        )
    print(
        f"      Loaded Industrial Power-Piping OOD Elbows (35,280 DOFs, 4,992 Q2 tris): 2 ADAPT / {len(ood_power_samples) - 2} BLIND TEST",
        flush=True,
    )

    # Fit normalization bounds strictly on TRAIN
    p_tr_mat = np.asarray([s.p_vec for s in train_samples], dtype=np.float64)
    p_lo = p_tr_mat.min(axis=0)
    p_hi = p_tr_mat.max(axis=0)

    # ------------------------------------------------------------------
    # Stage 2: Train ONE Higher-Capacity 20-Ch MultiTopologyOperator3D Jointly Across ALL 3 Families
    # ------------------------------------------------------------------
    mt_op = MultiTopologyOperator3D(
        parameter_dim=PARAM_DIM,
        hidden_dim=HIDDEN_DIM,
        slice_count=SLICE_COUNT,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        positional_frequencies=POS_FREQS,
        decoder_blocks=DECODER_BLOCKS,
    )
    mt_op.operator = mt_op.operator.to(dtype=dtype)
    n_params = sum(p.numel() for p in mt_op.operator.parameters())
    epochs_main = 2000
    print(
        f"[2/5] Training ONE 20-ch MultiTopologyOperator3D ({n_params:,} params, H={HIDDEN_DIM}, S={SLICE_COUNT}, "
        f"B={NUM_BLOCKS}, F={POS_FREQS}, dec={DECODER_BLOCKS}, Q={Q_TOTAL}/geom, ZERO family IDs) "
        f"for {epochs_main} steps across {len(train_samples)} TRAIN geometries...",
        flush=True,
    )

    loss_hist, fam_hist, train_wall_s = _train_vectorized(
        mt_op,
        train_samples,
        p_lo,
        p_hi,
        epochs=epochs_main,
        lr=2.0e-3,
        dtype=dtype,
        dev_samples=dev_samples,
        seed=42,
        warmup_epochs=80,
    )

    # Fit MultiTopologyOODDetector3D across ALL trained TRAIN geometries (chi = -6 and chi = 0)
    ood_det = MultiTopologyOODDetector3D.fit(
        [s.descriptor for s in train_samples], threshold_multiplier=2.2
    )
    mt_op.ood_detector = ood_det

    ckpt_path = OUT_DIR / "cross_geometry_3d_piginot.safetensors"
    save_safetensors_file(
        {
            k: v.detach().cpu().contiguous()
            for k, v in mt_op.operator.state_dict().items()
        },
        str(ckpt_path),
    )
    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    # ------------------------------------------------------------------
    # Stage 3: Level 1 Blind Unseen Holdout (28 Unseen 3D CAD Parts) + DEV Conformal/AS6983
    # ------------------------------------------------------------------
    print(
        f"[3/5] Level 1: Evaluating ONE unified operator on {len(holdout_samples)} strictly UNSEEN 3D geometries...",
        flush=True,
    )
    dev_conf: list[Conformal3DDevSample] = []
    for s in dev_samples:
        r = _eval_sample(mt_op, s, p_lo, p_hi, dtype)
        dev_conf.append(
            Conformal3DDevSample(
                case_id=s.sample_id,
                predicted_qois={"max_u_norm": r["max_u_pred_mm"] / s.u_scale_mm},
                true_qois={"max_u_norm": r["max_u_true_mm"] / s.u_scale_mm},
                latent_vector=tuple(float(v) for v in r["latent_vec"][:16]),
            )
        )

    ho_records: list[dict[str, Any]] = []
    ho_conf: list[Conformal3DDevSample] = []
    rep_fields: dict[str, Any] = {}
    for s in holdout_samples:
        ood_r = ood_det.score(s.descriptor)
        r = _eval_sample(mt_op, s, p_lo, p_hi, dtype)
        r["ood_status"] = ood_r.status.value
        r["ood_accepted"] = bool(ood_r.accepted)
        r["ood_distance"] = float(ood_r.mahalanobis_distance)
        r["ood_threshold"] = float(ood_r.threshold)
        u_pred_arr = r.pop("u_pred")
        lat_arr = r.pop("latent_vec")
        ho_records.append(r)
        ho_conf.append(
            Conformal3DDevSample(
                case_id=s.sample_id,
                predicted_qois={"max_u_norm": r["max_u_pred_mm"] / s.u_scale_mm},
                true_qois={"max_u_norm": r["max_u_true_mm"] / s.u_scale_mm},
                latent_vector=tuple(float(v) for v in lat_arr[:16]),
            )
        )
        rep_fields[s.sample_id] = (s, u_pred_arr, r)

    cal_rec = Conformal3DCalibrator.fit(
        bundle_hash=ckpt_sha,
        dev_samples=dev_conf,
        qoi_units={"max_u_norm": "u/u_scale"},
        alpha=0.10,
        latent_slope=0.10,
    )
    cov_ho = verify_as6983_holdout_coverage(cal_rec, ho_conf, test_level=0.05)

    fam_summary: dict[str, dict[str, float]] = {}
    print(
        f"\n      === LEVEL 1: BLIND UNSEEN 3D CROSS-GEOMETRY HOLDOUT ({len(ho_records)} UNSEEN CAD PARTS) ===",
        flush=True,
    )
    for fam in ("bracket", "pipe-elbow", "pipe-straight"):
        sub = [r for r in ho_records if r["family"] == fam]
        mean_l2 = float(np.mean([r["u_rel_l2"] for r in sub]))
        med_l2 = float(np.median([r["u_rel_l2"] for r in sub]))
        max_l2 = float(np.max([r["u_rel_l2"] for r in sub]))
        mean_pk = float(np.mean([r["peak_u_rel_err"] for r in sub]))
        mean_comp = float(np.mean([r["compliance_rel_err"] for r in sub]))
        max_clamp = float(np.max([r["clamped_max_err_mm"] for r in sub]))
        mean_ms = float(np.mean([r["infer_ms"] for r in sub]))
        id_acc = float(np.mean([1.0 if r["ood_accepted"] else 0.0 for r in sub]))
        fam_summary[fam] = {
            "count": len(sub),
            "mean_u_rel_l2": mean_l2,
            "median_u_rel_l2": med_l2,
            "worst_u_rel_l2": max_l2,
            "mean_peak_u_rel_err": mean_pk,
            "mean_compliance_rel_err": mean_comp,
            "max_clamped_err_mm": max_clamp,
            "mean_infer_ms": mean_ms,
            "ood_accept_rate": id_acc,
        }
        print(
            f"      {fam:<14s} (n={len(sub):2d}): mean u_L2={mean_l2 * 100:5.2f}% (med={med_l2 * 100:5.2f}%, max={max_l2 * 100:5.2f}%) | "
            f"peak_u={mean_pk * 100:5.2f}% | compliance={mean_comp * 100:5.2f}% | "
            f"clamp={max_clamp:.1e} mm | OOD_acc={id_acc * 100:5.1f}% | {mean_ms:5.1f} ms",
            flush=True,
        )
    cov_u = cov_ho["max_u_norm"]
    print(
        f"      DEV-calibrated AS6983 coverage on {len(ho_records)} unseen geometries: "
        f"{cov_u.empirical_non_underprediction_rate * 100:.1f}% ({cov_u.non_underpredicted_count}/{cov_u.holdout_m}, pass={cov_u.as6983_passed})",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Stage 4: Level 2 OOD Audit + Level 3 Zero-Shot vs Budgeted 3D LoRA on 3.2m Power Elbows
    # ------------------------------------------------------------------
    print(
        "[4/5] Level 2 (OOD Audit) & Level 3 (Zero-Shot vs Budgeted <=5% 3D LoRA on 3.2m Power-Piping Elbows)...",
        flush=True,
    )
    brick_ood = make_brick_tet4_mesh(120.0, 60.0, 40.0, nx=3, ny=2, nz=2)
    brick_desc = compute_geometry_descriptor_3d(
        brick_ood, clamped_face_tags=("x_min",), loaded_face_tags=("x_max",)
    )
    ood_brick = ood_det.score(brick_desc)
    ood_power_reports = [ood_det.score(s.descriptor) for s in ood_power_samples]
    ood_probes = [
        {
            "name": "Unseen Genus-0 Solid Brick (chi=+2)",
            "euler_chi": brick_desc.surface_euler_characteristic,
            "status": ood_brick.status.value,
            "accepted": bool(ood_brick.accepted),
            "distance": None
            if not math.isfinite(ood_brick.mahalanobis_distance)
            else float(ood_brick.mahalanobis_distance),
        }
    ] + [
        {
            "name": f"Out-of-Family Industrial Power Elbow ({s.sample_id})",
            "euler_chi": s.euler_chi,
            "status": rep.status.value,
            "accepted": bool(rep.accepted),
            "distance": float(rep.mahalanobis_distance),
        }
        for s, rep in zip(ood_power_samples, ood_power_reports, strict=True)
    ]
    n_refused = sum(1 for p in ood_probes if not p["accepted"])
    print(
        f"      Adversarial / Out-of-Family OOD Refusal: {n_refused}/{len(ood_probes)} refused "
        f"({ood_brick.status.value} for Genus-0 brick; {ood_power_reports[0].status.value} "
        f"dist={ood_power_reports[0].mahalanobis_distance:.1f} for 3.2m Power Elbow)",
        flush=True,
    )

    adapt_set = [s for s in ood_power_samples if s.split == "OOD_ADAPT"]
    test_ood_set = [s for s in ood_power_samples if s.split == "OOD_TEST"]

    # 3A) Zero-Shot
    zs_recs = []
    for s in test_ood_set:
        r = _eval_sample(mt_op, s, p_lo, p_hi, dtype)
        r.pop("u_pred")
        r.pop("latent_vec")
        zs_recs.append(r)
    zs_mean_l2 = float(np.mean([r["u_rel_l2"] for r in zs_recs]))

    # 3B) Budgeted 3D LoRA (rank=6, 80 steps = 4.0% of 2,000-step budget, <= 3% trainable params!)
    lora_steps = 80
    lora_cfg = LowRankAdapter3DConfig(
        rank=6,
        alpha=12.0,
        target_substrings=("out_proj", "decoder_head", "query_in"),
        max_steps=lora_steps,
        from_scratch_reference_steps=epochs_main,
        max_step_budget_fraction=0.10,
        learning_rate=1.8e-3,
        seed=42,
    )
    adapted_inner, replaced_mods = inject_lora_3d(mt_op.operator, lora_cfg)
    lora_mt = MultiTopologyOperator3D(
        parameter_dim=PARAM_DIM,
        hidden_dim=HIDDEN_DIM,
        slice_count=SLICE_COUNT,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        positional_frequencies=POS_FREQS,
        decoder_blocks=DECODER_BLOCKS,
    )
    lora_mt.operator = adapted_inner
    lora_trainable = sum(
        p.numel() for p in lora_mt.operator.parameters() if p.requires_grad
    )
    lora_total = sum(p.numel() for p in lora_mt.operator.parameters())
    lora_frac = lora_trainable / max(lora_total, 1)

    _, _, lora_wall_s = _train_vectorized(
        lora_mt,
        adapt_set,
        p_lo,
        p_hi,
        epochs=lora_steps,
        lr=1.8e-3,
        dtype=dtype,
        seed=42,
        trainable_only=True,
        warmup_epochs=8,
        log_prefix="      [LoRA] ",
    )
    lora_recs = []
    lora_rep_u = None
    for idx_t, s in enumerate(test_ood_set):
        r = _eval_sample(lora_mt, s, p_lo, p_hi, dtype)
        u_p = r.pop("u_pred")
        r.pop("latent_vec")
        if idx_t == 0:
            lora_rep_u = u_p
        lora_recs.append(r)
    lora_mean_l2 = float(np.mean([r["u_rel_l2"] for r in lora_recs]))

    lora_path = OUT_DIR / "power_elbow_lora_3d.safetensors"
    save_safetensors_file(extract_lora_3d_state_dict(lora_mt.operator), str(lora_path))

    # 3C) From-Scratch on same 2 examples & 80 steps
    scratch_mt = MultiTopologyOperator3D(
        parameter_dim=PARAM_DIM,
        hidden_dim=HIDDEN_DIM,
        slice_count=SLICE_COUNT,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        positional_frequencies=POS_FREQS,
        decoder_blocks=DECODER_BLOCKS,
    )
    scratch_mt.operator = scratch_mt.operator.to(dtype=dtype)
    _, _, scratch_wall_s = _train_vectorized(
        scratch_mt,
        adapt_set,
        p_lo,
        p_hi,
        epochs=lora_steps,
        lr=2.0e-3,
        dtype=dtype,
        seed=42,
        trainable_only=False,
        warmup_epochs=8,
        log_prefix="      [Scratch] ",
    )
    scratch_recs = []
    for s in test_ood_set:
        r = _eval_sample(scratch_mt, s, p_lo, p_hi, dtype)
        r.pop("u_pred")
        r.pop("latent_vec")
        scratch_recs.append(r)
    scratch_mean_l2 = float(np.mean([r["u_rel_l2"] for r in scratch_recs]))

    print(
        f"\n      === LEVEL 3: BLIND 3.2m INDUSTRIAL POWER-PIPING ELBOW TRANSFER (n={len(test_ood_set)}) ===\n"
        f"      Zero-Shot (0 adapt steps, Buckingham-Pi transfer):     mean u_L2 = {zs_mean_l2 * 100:5.2f}%\n"
        f"      Budgeted 3D LoRA ({lora_trainable:,} params = {lora_frac * 100:.2f}%, {lora_steps} steps, {lora_wall_s:.1f}s): mean u_L2 = {lora_mean_l2 * 100:5.2f}%\n"
        f"      From-Scratch ({n_params:,} params = 100%, {lora_steps} steps, {scratch_wall_s:.1f}s):       mean u_L2 = {scratch_mean_l2 * 100:5.2f}%",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Stage 5: Archive Representative 3D Meshes & Canonical JSON Summary
    # ------------------------------------------------------------------
    print(
        "[5/5] Archiving representative unseen 3D geometries & canonical JSON report...",
        flush=True,
    )
    plot_arrays: dict[str, Any] = {
        "loss_history": np.asarray(loss_hist, dtype=np.float64),
        "hist_bracket": np.asarray(fam_hist["bracket"], dtype=np.float64),
        "hist_pipe_elbow": np.asarray(fam_hist["pipe-elbow"], dtype=np.float64),
        "hist_pipe_straight": np.asarray(fam_hist["pipe-straight"], dtype=np.float64),
    }
    rep_ids: dict[str, str] = {}
    for fam, tag in (
        ("bracket", "bracket"),
        ("pipe-elbow", "elbow"),
        ("pipe-straight", "straight"),
    ):
        sub = sorted(
            [r for r in ho_records if r["family"] == fam], key=lambda x: x["u_rel_l2"]
        )
        chosen_id = sub[len(sub) // 2]["sample_id"]
        rep_ids[fam] = chosen_id
        samp, u_pred_arr, _ = rep_fields[chosen_id]
        plot_arrays[f"{tag}_nodes"] = samp.query_xyz_mm
        plot_arrays[f"{tag}_u_true"] = samp.u_true_mm
        plot_arrays[f"{tag}_u_pred"] = u_pred_arr
        plot_arrays[f"{tag}_surf_nodes"] = np.asarray(samp.mesh.nodes, dtype=np.float64)
        plot_arrays[f"{tag}_surf_tris"] = np.asarray(
            samp.mesh.surface_triangles, dtype=np.int64
        )

    s_ood0 = test_ood_set[0]
    plot_arrays["power_nodes"] = s_ood0.query_xyz_mm
    plot_arrays["power_u_true"] = s_ood0.u_true_mm
    plot_arrays["power_u_pred_lora"] = lora_rep_u
    np.savez_compressed(OUT_DIR / "study2_plot_fields.npz", **plot_arrays)

    summary = {
        "schema": "formamech.study2-cross-geometry-3d-v2",
        "datasets": [
            {
                "id": "HaomingLuo/AgentFEM-Bracket-Elasticity",
                "license": "CC-BY-4.0",
                "family": "bracket",
                "euler_chi": -6,
                "genus": 4,
                "surface_tris_approx": 5574,
                "material": {"E_mpa": 70000.0, "nu": 0.33, "alloy": "Al 6061"},
                "splits": {"train": 128, "dev": 12, "holdout": 16},
            },
            {
                "id": "HaomingLuo/AgentFEM-Pipe-Components",
                "license": "CC-BY-4.0",
                "families": [
                    "pipe-elbow (Genus 1, chi=0, 4,800 Q2-subdivided curved tris)",
                    "pipe-straight (Genus 1, chi=0, 4,800 Q2-subdivided curved tris)",
                    "power-elbow-ood (3.2m industrial OOD, 4,992 Q2-subdivided curved tris)",
                ],
                "material": {
                    "E_mpa": 200000.0,
                    "nu": 0.30,
                    "alloy": "Carbon Steel (Power OOD: E=190 GPa)",
                },
                "splits": {
                    "pipe-elbow": {"train": 14, "dev": 4, "holdout": 6},
                    "pipe-straight": {"train": 14, "dev": 4, "holdout": 6},
                    "power-elbow-ood": {"adapt": 2, "blind_test": len(test_ood_set)},
                },
            },
        ],
        "operator": {
            "class": "piginot.operators.multi_topology_3d.MultiTopologyOperator3D",
            "parameter_count": n_params,
            "hidden_dim": HIDDEN_DIM,
            "slice_count": SLICE_COUNT,
            "num_blocks": NUM_BLOCKS,
            "positional_frequencies": POS_FREQS,
            "decoder_blocks": DECODER_BLOCKS,
            "surface_feature_dim": 20,
            "universal_param_dim": PARAM_DIM,
            "query_points_per_step": Q_TOTAL,
            "zero_family_ids_passed": True,
            "buckingham_pi_scaling": "u_scale = F * L_diag / (E * h_eq^2)",
            "exact_3d_adf_lifting": True,
            "checkpoint_file": str(ckpt_path),
            "checkpoint_sha256": ckpt_sha,
            "training_epochs": epochs_main,
            "train_wall_time_s": train_wall_s,
        },
        "epistemic_scope": {
            "framework": "ASME V&V 10",
            "claim_level": "HOLDOUT_VERIFIED_EXTERNAL_FEM_DATASETS",
            "experimental_validation": False,
            "note": "Verified strictly against blind held-out external 3D FEM geometries; not physical coupon validation.",
        },
        "level1_unseen_cross_geometry_holdout": {
            "total_unseen_geometries": len(ho_records),
            "overall_mean_u_rel_l2": float(
                np.mean([r["u_rel_l2"] for r in ho_records])
            ),
            "overall_mean_peak_u_rel_err": float(
                np.mean([r["peak_u_rel_err"] for r in ho_records])
            ),
            "overall_mean_compliance_rel_err": float(
                np.mean([r["compliance_rel_err"] for r in ho_records])
            ),
            "overall_max_clamped_err_mm": float(
                np.max([r["clamped_max_err_mm"] for r in ho_records])
            ),
            "overall_mean_infer_ms": float(
                np.mean([r["infer_ms"] for r in ho_records])
            ),
            "by_family": fam_summary,
            "representative_ids": rep_ids,
            "conformal_as6983_holdout": {k: v.document() for k, v in cov_ho.items()},
            "cases": ho_records,
        },
        "level2_ood_audit": {
            "detector": ood_det.document(),
            "probes": ood_probes,
            "refusal_rate": n_refused / max(len(ood_probes), 1),
        },
        "level3_industrial_power_elbow_transfer": {
            "adapt_sample_ids": [s.sample_id for s in adapt_set],
            "blind_test_sample_ids": [s.sample_id for s in test_ood_set],
            "lora_steps": lora_steps,
            "zero_shot_mean_u_rel_l2": zs_mean_l2,
            "lora_3d_mean_u_rel_l2": lora_mean_l2,
            "from_scratch_mean_u_rel_l2": scratch_mean_l2,
            "lora_trainable_params": lora_trainable,
            "lora_trainable_fraction": lora_frac,
            "lora_replaced_modules": replaced_mods,
            "lora_wall_time_s": lora_wall_s,
            "scratch_wall_time_s": scratch_wall_s,
            "zero_shot_cases": zs_recs,
            "lora_cases": lora_recs,
            "scratch_cases": scratch_recs,
        },
        "total_wall_time_s": time.perf_counter() - t_global,
    }
    summary_path = OUT_DIR / "study2_summary.json"
    summary_path.write_bytes(canonical_json_bytes(summary))
    print(
        f"Study 2 (Upgraded) complete in {time.perf_counter() - t_global:.1f}s -> {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
