"""Study R4 — Hyperelastic Human Brain Surgical Soft-Tissue FEM & AB-UPT/Transolver++ 3D Surrogate (V6-R5).

Validates FormaMech's ultra-soft kPa-scale 3D TET10 hyperelastic FEM (via exact continuum
similitude), ASME V&V 10 experimental grounding on real human brain specimens, V6-R5
AB-UPT + Transolver++ adaptive-temperature Surface3DEnergyOperator, 3D ONNX/TorchScript
runtime export, and AS6983 surgical tissue-tearing peak-strain bounds against:
  - Hyperelastic Human Brain 1-7 experimental tissue specimens (Zenodo 8095559,
    CC-BY-4.0, Tier A — 182 real human brain tissue cylinders in compression & tension)
  - Published regional Neo-Hookean shear moduli for 9 anatomical brain regions
    (Zenodo 21793225, CC-BY-4.0, Tier A — 85.85 Pa corpus callosum to 434.00 Pa cortex)

Outputs:
  - Git-tracked canonical scrubbed summary: docs/assets/studies_robotics/study_r4_summary.json
  - Gitignored D3/Playwright viz payload:   .formamech-runs/robotics_studies/study_r4_viz.json
"""

from __future__ import annotations

import json
import math
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from piginot.bounds.conformal_3d import (
    Conformal3DCalibrator,
    Conformal3DDevSample,
    verify_as6983_holdout_coverage,
)
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.export.onnx_export_3d import export_surface_3d_runtime
from piginot.fem.backend_3d import DirichletBC3D
from piginot.fem.diff_contact_3d import (
    RigidSdfIndenter3D,
    SdfPrimitiveKind,
    solve_3d_sdf_contact,
)
from piginot.geometry.mesh3d import make_brick_tet10_mesh
from piginot.operators.surface_operator_3d import (
    Surface3DConfig,
    Surface3DEnergyOperator,
)
from piginot.validation.component_test_contracts import (
    ComponentTestRecord,
    SensorObservationKind,
)
from piginot.validation.contracts import CouponRole
from piginot.validation.hybrid_grounding import (
    HybridGroundingConfig,
    ground_simulation_with_experiments,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (
    REPO_ROOT / ".formamech-runs" / "datasets" / "robotics" / "r4_surgical_tissue"
)
VIZ_DIR = REPO_ROOT / ".formamech-runs" / "robotics_studies"
ASSETS_DIR = REPO_ROOT / "docs" / "assets" / "studies_robotics"


def _parse_yaml_simple(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def _brain_material(mu_pa: float, nu: float = 0.48) -> MaterialSpec:
    # E = 2 * mu * (1 + nu) in Pa -> MPa (* 1e-6)
    e_mpa = 2.0 * float(mu_pa) * (1.0 + nu) * 1.0e-6
    return MaterialSpec(
        name=f"brain-mu-{mu_pa:.0f}pa",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(e_mpa, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(nu, Unit.ONE).to_canonical(),
        },
    )


def main() -> None:
    t_start = time.perf_counter()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    hbe_zip = DATA_DIR / "HBE_Data.zip"
    nh_csv = DATA_DIR / "HBE_1-7_avg_NeoHooke_params.csv"
    if not hbe_zip.exists() or not nh_csv.exists():
        raise FileNotFoundError(
            "Missing R4 datasets in .formamech-runs/datasets/robotics/r4_surgical_tissue/."
        )

    # 1. Load published regional Neo-Hookean shear moduli across all 9 human brain regions
    regions_df = pd.read_csv(nh_csv)
    regions = [
        {"region": str(r["region"]), "shear_modulus_pa": float(r["shear modulus [Pa]"])}
        for _, r in regions_df.iterrows()
    ]
    mu_min_pa = min(r["shear_modulus_pa"] for r in regions)
    mu_max_pa = max(r["shear_modulus_pa"] for r in regions)

    # 2. Parse real human brain compression & tension curves from HBE_Data.zip (182 specimens)
    specimens: list[dict[str, Any]] = []
    with zipfile.ZipFile(hbe_zip) as zf:
        names_set = set(zf.namelist())
        yamls = sorted(n for n in names_set if n.endswith("geometry.yaml"))
        for ym in yamls:
            c1 = ym.replace("geometry.yaml", "compression_c1.csv")
            if c1 not in names_set:
                continue
            geom = _parse_yaml_simple(zf.read(ym).decode("utf-8", "ignore"))
            if "height" not in geom or "radius" not in geom:
                continue
            h_mm = geom["height"] * 1000.0
            r_mm = geom["radius"] * 1000.0
            area_mm2 = math.pi * r_mm * r_mm
            df_c = pd.read_csv(zf.open(c1))
            if len(df_c) < 10:
                continue
            disp_mm = -df_c["displacement"].to_numpy(dtype=np.float64) * 1000.0
            force_mn = -df_c["force"].to_numpy(dtype=np.float64) * 1000.0  # mN
            strain = disp_mm / max(h_mm, 1e-6)
            # Extract peak stress at 10% nominal compressive strain
            f_10pct_mn = float(np.interp(0.10, strain, force_mn))
            stress_10pct_pa = (f_10pct_mn * 1e-3) / (area_mm2 * 1e-6)  # Pa
            if 10.0 <= stress_10pct_pa <= 4000.0:
                specimens.append(
                    {
                        "id": ym.split("/")[1],
                        "donor": ym.split("/")[0],
                        "height_mm": h_mm,
                        "radius_mm": r_mm,
                        "area_mm2": area_mm2,
                        "strain": strain,
                        "force_mn": force_mn,
                        "stress_10pct_pa": stress_10pct_pa,
                    }
                )

    # 3. Run full 3D TET10 ultra-soft hyperelastic FEM (kPa scale via exact continuum similitude!)
    #    Match average HBE specimen height H = 4.0 mm, square equivalent cross-section (7.1 x 7.1 mm -> A = 50.4 mm^2 ~ pi*4^2)
    tissue_mesh = make_brick_tet10_mesh(8.0, 8.0, 4.0, nx=2, ny=2, nz=1)
    base_bc = (DirichletBC3D("base", ("z_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),)
    # Run FEM flat-platen compression sweep at cortex modulus (434 Pa) and spherical surgical probe indentation
    cortex_mu = 434.0
    mat_cortex = _brain_material(cortex_mu, nu=0.48)
    comp_strains = [0.02, 0.05, 0.08, 0.10, 0.14]
    fem_comp_curve: list[dict[str, Any]] = []
    for eps_c in comp_strains:
        d_mm = eps_c * 4.0
        # Flat platen covering full top face -> uniaxial confined compression
        ind_platen = RigidSdfIndenter3D(
            name=f"platen-{eps_c:.2f}",
            kind=SdfPrimitiveKind.BOX,
            contact_face_tags=("z_max",),
            initial_center_mm=(4.0, 4.0, 4.0 + 3.0 + 0.01),
            center_mm=(4.0, 4.0, 4.0 + 3.0 - d_mm),
            half_extents_mm=(6.0, 6.0, 3.0),
            corner_rounding_mm=0.20,
            penalty_mpa_per_mm=0.05,
        )
        res = solve_3d_sdf_contact(
            tissue_mesh,
            mat_cortex,
            dirichlet_bcs=base_bc,
            indenter=ind_platen,
            load_steps=2,
            max_al_iterations=4,
            penetration_tol_mm=5.0e-3,
        )
        assert res.converged, (
            f"Brain FEM compression failed at strain {eps_c}: {res.reason}"
        )
        fz_mn = float(-res.total_contact_force_n[2]) * 1000.0
        stress_pa = (fz_mn * 1e-3) / (64.0 * 1e-6)  # 8x8 mm^2 -> m^2
        fem_comp_curve.append(
            {
                "strain": eps_c,
                "disp_mm": round(d_mm, 3),
                "fem_cortex_force_mn": round(fz_mn, 3),
                "fem_cortex_stress_pa": round(stress_pa, 2),
                "peak_vm_pa": round(float(res.quadrature_peak_von_mises_mpa) * 1e6, 2),
                "min_det_f": round(float(res.min_det_f), 4),
                "wall_time_s": round(float(res.wall_time_s), 3),
            }
        )

    # Compare 3D FEM compression against median + IQR of all real HBE human brain specimens
    strain_grid = np.array(comp_strains, dtype=np.float64)
    exp_stress_mat = np.array(
        [
            (np.interp(strain_grid, sp["strain"], sp["force_mn"]) * 1e-3)
            / (sp["area_mm2"] * 1e-6)
            for sp in specimens
        ],
        dtype=np.float64,
    )
    exp_median_pa = np.median(exp_stress_mat, axis=0)
    exp_q25_pa = np.percentile(exp_stress_mat, 25, axis=0)
    exp_q75_pa = np.percentile(exp_stress_mat, 75, axis=0)
    exp_std_pa = np.maximum(np.std(exp_stress_mat, axis=0), 15.0)

    # ASME V&V 10 hybrid grounding across real donors: sample up to 6 specimens per donor
    # (HBE_01..HBE_04 -> CALIBRATION, HBE_05..HBE_07 -> strictly blind VALIDATION_HOLDOUT!)
    mu_mean_pa = float(np.mean([r["shear_modulus_pa"] for r in regions]))
    fem_stress_at_10pct_unit_mu = (
        float(fem_comp_curve[3]["fem_cortex_stress_pa"]) / cortex_mu
    )
    by_donor: dict[str, list[dict[str, Any]]] = {}
    for sp in specimens:
        by_donor.setdefault(sp["donor"], []).append(sp)
    vv_recs: list[ComponentTestRecord] = []
    for donor in sorted(by_donor):
        donor_num = int(donor.split("_")[1])
        role = (
            CouponRole.CALIBRATION if donor_num <= 4 else CouponRole.VALIDATION_HOLDOUT
        )
        for sp in by_donor[donor][:6]:
            prior_pa = (
                fem_stress_at_10pct_unit_mu
                * mu_mean_pa
                * (4.0 / sp["height_mm"]) ** 0.25
            )
            vv_recs.append(
                ComponentTestRecord(
                    specimen_id=sp["id"],
                    role=role,
                    sensor_kind=SensorObservationKind.LOAD_CELL_REACTION,
                    condition_vector=(
                        round(sp["height_mm"], 3),
                        round(sp["radius_mm"], 3),
                    ),
                    observed_values=(float(sp["stress_10pct_pa"]),),
                    simulated_prior_values=(float(prior_pa),),
                    measurement_std=float(exp_std_pa[3] * 0.5),
                    unit="Pa",
                    metadata={
                        "donor": sp["donor"],
                        "dataset": "human-brain-hyperelastic",
                    },
                )
            )
    _, hbe_vv = ground_simulation_with_experiments(
        vv_recs,
        config=HybridGroundingConfig(fit_prior_scale_rho=True, length_scale=1.5),
    )

    # 4. Full 3D Surgical Probe Indentation DOE + V6-R5 AB-UPT & Transolver++ Surface3DEnergyOperator
    #    Run 3D TET10 spherical surgical probe indentation on a 16x16x6 mm tissue block
    probe_mesh = make_brick_tet10_mesh(16.0, 16.0, 6.0, nx=3, ny=3, nz=1)
    probe_nodes = np.asarray(probe_mesh.nodes, dtype=np.float64)
    probe_depths = [0.15, 0.30, 0.50, 0.70, 0.90, 1.10]
    probe_fem_cache: list[tuple[float, Any]] = []
    probe_fem_times: list[float] = []
    for pd_mm in probe_depths:
        ind_sp = RigidSdfIndenter3D(
            name=f"surg-probe-{pd_mm:.2f}",
            kind=SdfPrimitiveKind.SPHERE,
            contact_face_tags=("z_max",),
            initial_center_mm=(8.0, 8.0, 6.0 + 2.5 + 0.02),
            center_mm=(8.0, 8.0, 6.0 + 2.5 - pd_mm),
            radius_mm=2.5,
            penalty_mpa_per_mm=0.08,
        )
        res_p = solve_3d_sdf_contact(
            probe_mesh,
            mat_cortex,
            dirichlet_bcs=base_bc,
            indenter=ind_sp,
            load_steps=2,
            max_al_iterations=4,
            penetration_tol_mm=5.0e-3,
        )
        assert res_p.converged, (
            f"Surgical probe FEM failed at d={pd_mm}: {res_p.reason}"
        )
        probe_fem_cache.append((pd_mm, res_p))
        probe_fem_times.append(res_p.wall_time_s)

    # Build 36-case DOE over all 9 real brain regions (85.85..434 Pa) x 4 indentation depths
    # Using exact continuum similitude (u is invariant to mu; Cauchy stress scales linearly with mu!)
    doe_cases: list[dict[str, Any]] = []
    for reg in regions:
        mu_pa = reg["shear_modulus_pa"]
        for d_mm, res_p in probe_fem_cache:
            u_field = res_p.nodal_displacement.copy()
            vm_pa = res_p.nodal_von_mises_mpa * 1.0e6 * (mu_pa / cortex_mu)
            peak_vm_pa = (
                float(res_p.quadrature_peak_von_mises_mpa) * 1.0e6 * (mu_pa / cortex_mu)
            )
            # Peak Green-Lagrange / nominal indentation strain indicator
            peak_strain_pct = (
                float(np.max(np.linalg.norm(u_field, axis=1)) / 6.0) * 100.0
            )
            doe_cases.append(
                {
                    "id": f"{reg['region'].replace(' ', '_')}-d{d_mm:.2f}",
                    "region": reg["region"],
                    "mu_pa": mu_pa,
                    "depth_mm": d_mm,
                    "params": np.array([mu_pa / 500.0, d_mm / 1.2], dtype=np.float64),
                    "u_field": u_field,
                    "vm_pa": vm_pa,
                    "peak_vm_pa": peak_vm_pa,
                    "peak_strain_pct": peak_strain_pct,
                }
            )

    # Split 54 cases deterministically: 24 train / 15 dev / 15 blind holdout (n_dev >= 10 for alpha=0.10 AS6983)
    rng = np.random.default_rng(77)
    order = rng.permutation(len(doe_cases))
    tr_cases = [doe_cases[i] for i in order[:24]]
    dev_cases = [doe_cases[i] for i in order[24:39]]
    hold_cases = [doe_cases[i] for i in order[39:]]

    # Instantiate V6-R5 AB-UPT + Transolver++ Surface3DEnergyOperator
    op_cfg = Surface3DConfig(
        parameter_dim=2,
        surface_feature_dim=6,
        hidden_dim=40,
        slice_count=8,
        num_heads=4,
        num_blocks=2,
        adaptive_temperature=True,
        anchor_count=32,
        low_rank_query_dim=16,
        decode_chunk_size=128,
    )
    torch.manual_seed(17)
    op = Surface3DEnergyOperator(op_cfg).double()
    coords_t = torch.tensor(probe_nodes, dtype=torch.float64)
    # Build surface geometry descriptors (top + bottom + side nodes)
    surf_idx = [
        i
        for i, p in enumerate(probe_nodes)
        if abs(p[2] - 6.0) < 1e-6
        or abs(p[2]) < 1e-6
        or abs(p[0]) < 1e-6
        or abs(p[0] - 16.0) < 1e-6
    ]
    surf_xyz = torch.tensor(probe_nodes[surf_idx], dtype=torch.float64).unsqueeze(0)
    surf_normals = torch.zeros_like(surf_xyz)
    surf_normals[:, :, 2] = torch.where(surf_xyz[:, :, 2] > 3.0, 1.0, -1.0)
    surf_areas = torch.ones((1, len(surf_idx)), dtype=torch.float64)
    surf_feats = torch.cat([surf_xyz / 16.0, surf_normals], dim=-1)

    opt = torch.optim.AdamW(op.parameters(), lr=6e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=220)
    p_tr = torch.tensor(np.stack([c["params"] for c in tr_cases]), dtype=torch.float64)
    u_tr = torch.tensor(np.stack([c["u_field"] for c in tr_cases]), dtype=torch.float64)
    coords_b = coords_t.unsqueeze(0)

    for _ep in range(220):
        opt.zero_grad()
        enc = op.encode_surface(surf_xyz, surf_normals, surf_areas, surf_feats)
        loss = torch.zeros((), dtype=torch.float64)
        for bi in range(len(tr_cases)):
            u_pred = op.displacement(
                encoding=enc, query_xyz=coords_b, parameters=p_tr[bi : bi + 1]
            )[0]
            loss = loss + torch.mean((u_pred - u_tr[bi]) ** 2)
        loss = loss / float(len(tr_cases))
        loss.backward()
        opt.step()
        sched.step()

    op.eval()
    with torch.no_grad():
        enc_frozen = op.encode_surface(surf_xyz, surf_normals, surf_areas, surf_feats)
    export_dir = VIZ_DIR / "study_r4_runtime_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_rep = export_surface_3d_runtime(
        op,
        enc_frozen,
        coords_b,
        export_dir / "brain_surface3d_runtime.onnx",
        sample_parameters=p_tr[:1],
    )

    # Calibrate AS6983 surgical peak-stress safety bounds on DEV cases, verify blind on HOLDOUT
    ref_peak_vm_per_disp_unit_mu = float(
        (probe_fem_cache[-1][1].quadrature_peak_von_mises_mpa * 1e6 / cortex_mu)
        / max(
            float(
                np.max(
                    np.linalg.norm(probe_fem_cache[-1][1].nodal_displacement, axis=1)
                )
            ),
            1e-6,
        )
    )

    def eval_op_case(c: dict[str, Any]) -> tuple[float, float, float]:
        p_t = torch.tensor(c["params"][None, :], dtype=torch.float64)
        t0 = time.perf_counter()
        with torch.no_grad():
            u_pred_t = op.displacement(
                encoding=enc_frozen, query_xyz=coords_b, parameters=p_t
            )[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        u_pred = u_pred_t.numpy()
        max_u = float(np.max(np.linalg.norm(u_pred, axis=1)))
        pred_peak_vm_pa = max_u * ref_peak_vm_per_disp_unit_mu * float(c["mu_pa"])
        u_rel = float(
            np.linalg.norm(u_pred - c["u_field"])
            / max(np.linalg.norm(c["u_field"]), 1e-6)
        )
        return pred_peak_vm_pa, u_rel, dt_ms

    dev_samps: list[Conformal3DDevSample] = []
    for c in dev_cases:
        p_vm, _, _ = eval_op_case(c)
        dev_samps.append(
            Conformal3DDevSample(
                case_id=c["id"],
                predicted_qois={"peak_von_mises_pa": p_vm},
                true_qois={"peak_von_mises_pa": float(c["peak_vm_pa"])},
                latent_vector=tuple(float(v) for v in c["params"]),
            )
        )
    conf_rec = Conformal3DCalibrator.fit(
        bundle_hash=export_rep.artifact_sha256,
        dev_samples=dev_samps,
        qoi_units={"peak_von_mises_pa": "Pa"},
        alpha=0.10,
    )

    hold_samps: list[Conformal3DDevSample] = []
    hold_u_rels: list[float] = []
    hold_lat_ms: list[float] = []
    hold_table: list[dict[str, Any]] = []
    tearing_limit_pa = (
        650.0  # Illustrative conservative surgical cortex stress threshold
    )
    for c in hold_cases:
        p_vm, u_rel, dt_ms = eval_op_case(c)
        hold_u_rels.append(u_rel)
        hold_lat_ms.append(dt_ms)
        assess = conf_rec.evaluate(
            {"peak_von_mises_pa": p_vm},
            latent_vector=tuple(float(v) for v in c["params"]),
        )
        bnd = assess.bounds["peak_von_mises_pa"]
        hold_samps.append(
            Conformal3DDevSample(
                case_id=c["id"],
                predicted_qois={"peak_von_mises_pa": p_vm},
                true_qois={"peak_von_mises_pa": float(c["peak_vm_pa"])},
                latent_vector=tuple(float(v) for v in c["params"]),
            )
        )
        hold_table.append(
            {
                "region": c["region"],
                "mu_pa": round(c["mu_pa"], 1),
                "depth_mm": c["depth_mm"],
                "true_vm_pa": round(float(c["peak_vm_pa"]), 2),
                "pred_vm_pa": round(p_vm, 2),
                "as6983_upper_pa": round(float(bnd.upper_conformal_bound), 2),
                "safe_below_tearing": bool(
                    bnd.upper_conformal_bound < tearing_limit_pa
                ),
            }
        )
    cov_vm = verify_as6983_holdout_coverage(conf_rec, hold_samps)["peak_von_mises_pa"]

    # 5. Build 24 animation frames comparing cortex (434 Pa) vs corpus callosum (85.85 Pa) surgical probe indentation
    top_idx = [i for i, p in enumerate(probe_nodes) if abs(p[2] - 6.0) < 1e-6]
    anim_frames: list[dict[str, Any]] = []
    n_frames = 24
    for fi in range(n_frames):
        # Smooth indent-and-retract surgical palpation cycle (0.05 -> 1.00 -> 0.15 mm)
        t_norm = fi / float(n_frames - 1)
        d_cur = 0.08 + 0.92 * math.sin(math.pi * t_norm)
        idx_hi = min(
            range(len(probe_fem_cache)),
            key=lambda i: abs(probe_fem_cache[i][0] - d_cur),
        )
        d_ref, r_ref = probe_fem_cache[idx_hi]
        u_sc = d_cur / max(d_ref, 1e-6)
        u_top = r_ref.nodal_displacement[top_idx] * u_sc
        pts_top = probe_nodes[top_idx] + u_top
        vm_cortex = r_ref.nodal_von_mises_mpa[top_idx] * 1e6 * u_sc
        vm_cc = vm_cortex * (85.85 / cortex_mu)
        anim_frames.append(
            {
                "frame": fi,
                "depth_mm": round(d_cur, 3),
                "surface_pts": np.round(pts_top, 3).tolist(),
                "vm_cortex_pa": np.round(vm_cortex, 1).tolist(),
                "vm_cc_pa": np.round(vm_cc, 1).tolist(),
                "peak_cortex_pa": round(float(np.max(vm_cortex)), 1),
                "peak_cc_pa": round(float(np.max(vm_cc)), 1),
                "tearing_limit_pa": tearing_limit_pa,
            }
        )

    summary: dict[str, Any] = {
        "study_id": "study_r4_surgical_tissue",
        "title": "Human Brain Ultra-Soft Hyperelastic 3D FEM & AB-UPT/Transolver++ Runtime Surrogate",
        "datasets": [
            {
                "id": "human-brain-hyperelastic",
                "license": "CC-BY-4.0",
                "tier": "A",
                "specimens_parsed": len(specimens),
            },
            {
                "id": "hbe-regional-neohooke",
                "license": "CC-BY-4.0",
                "tier": "A",
                "regions": len(regions),
            },
        ],
        "regional_moduli_pa": {
            "min_corpus_callosum_pa": mu_min_pa,
            "max_cortex_pa": mu_max_pa,
            "regions": regions,
        },
        "asme_vv10_donor_holdout": {
            "state": hbe_vv.state.value,
            "experimental_validation_tier": hbe_vv.experimental_validation_tier.value,
            "raw_fem_holdout_rmse_pa": round(float(hbe_vv.raw_sim_holdout_rmse), 2),
            "grounded_hybrid_holdout_rmse_pa": round(
                float(hbe_vv.grounded_holdout_rmse), 2
            ),
            "exp_10pct_median_stress_pa": round(float(exp_median_pa[3]), 2),
            "fem_cortex_10pct_stress_pa": round(
                float(fem_comp_curve[3]["fem_cortex_stress_pa"]), 2
            ),
        },
        "abupt_transolver_surrogate": {
            "architecture": "Surface3DEnergyOperator (V6-R5 AB-UPT anchors=32, adaptive_temperature=True, low_rank=16)",
            "export_format": export_rep.format_kind,
            "export_parity_max_err": round(float(export_rep.max_abs_diff), 6),
            "export_sha256": export_rep.artifact_sha256,
            "holdout_mean_disp_rel_l2_pct": round(
                float(np.mean(hold_u_rels)) * 100.0, 2
            ),
            "as6983_upper_coverage": round(
                float(cov_vm.non_underpredicted_count) / max(cov_vm.holdout_m, 1), 4
            ),
            "as6983_certified": bool(cov_vm.as6983_passed),
            "median_inference_ms": round(float(np.median(hold_lat_ms)), 3),
            "mean_fem_solve_s": round(float(np.mean(probe_fem_times)), 3),
        },
        "wall_time_s": round(time.perf_counter() - t_start, 2),
    }

    (ASSETS_DIR / "study_r4_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    viz_payload = {
        "summary": summary,
        "regions": regions,
        "compression_validation": [
            {
                "strain_pct": round(float(eps_c) * 100.0, 1),
                "exp_median_pa": round(float(exp_median_pa[i]), 2),
                "exp_q25_pa": round(float(exp_q25_pa[i]), 2),
                "exp_q75_pa": round(float(exp_q75_pa[i]), 2),
                "fem_cortex_pa": fem_comp_curve[i]["fem_cortex_stress_pa"],
                "fem_cc_pa": round(
                    fem_comp_curve[i]["fem_cortex_stress_pa"] * (85.85 / cortex_mu), 2
                ),
            }
            for i, eps_c in enumerate(comp_strains)
        ],
        "holdout_table": hold_table,
        "anim_frames": anim_frames,
    }
    (VIZ_DIR / "study_r4_viz.json").write_text(
        json.dumps(viz_payload), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
