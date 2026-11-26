"""Reproducible real-world ASME V&V 10 validation #2: NIST AM-Bench 2022 (IN718).

Downloads (if absent) and verifies official NIST US Public Domain files
(``10.18434/mds2-2607`` CAD STEP and ``10.18434/mds2-2711`` Cornell CHESS
synchrotron X-ray diffraction strains + CMM wire-EDM deflection) strictly into
gitignored ``.formamech-runs/datasets/nist-ambench/``, runs FormaMech's 3D CAD
mesher, 3D finite-strain thermo-mechanical FEM solver (V5-5), 3D sensor-to-mesh
alignment, ASME V&V 10 hybrid grounding, and 10-seed active X-ray scan planner.
"""

from __future__ import annotations

import hashlib
import math
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from piginot.active.test_planner import (
    PhysicalTestPlanConfig,
    run_retrospective_test_reduction_benchmark,
)
from piginot.benchmarks.external_data import (
    EXTERNAL_DATASET_MANIFEST_SCHEMA,
    resolve_dataset_spec,
)
from piginot.cad.importing_3d import ImportedCad3D, import_cad_3d
from piginot.contracts import (
    CanonicalQuantity,
    MaterialSpec,
    Unit,
    canonical_json_bytes,
)
from piginot.fem.backend_3d import (
    DirichletBC3D,
    TorchFemBackend3D,
    _project_cell_scalar_to_nodes_3d,
    _tet_quadrature,
)
from piginot.fem.electro_thermal_3d import (
    ElectroThermalMaterialSpec,
    solve_3d_electro_thermal_mechanical,
)
from piginot.fem.thermo_mechanical_3d import tet_shape_values_ref
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D
from piginot.validation.component_test_contracts import (
    ComponentTestRecord,
    SensorObservationKind,
)
from piginot.validation.contracts import CouponRole
from piginot.validation.hybrid_grounding import (
    HybridGroundingConfig,
    align_sensor_observations_to_mesh,
    ground_simulation_with_experiments,
)

CACHE_DIR = Path(".formamech-runs/datasets/nist-ambench").resolve()
REPORT_DIR = Path(".formamech-runs/validation").resolve()
FloatArray = NDArray[np.float64]

NIST_URLS: dict[str, str] = {
    "AMB2022_EDD_results_V2.txt": (
        "https://data.nist.gov/od/ds/ark:/88434/mds2-2711/"
        "Solution%20data%20for%20CHAL-AMB2022-01-RS_v1.0/AMB2022_EDD_results_V2.txt"
    ),
    "AMB2022_EDD_results_V2.txt.sha256": (
        "https://data.nist.gov/od/ds/ark:/88434/mds2-2711/"
        "Solution%20data%20for%20CHAL-AMB2022-01-RS_v1.0/AMB2022_EDD_results_V2.txt.sha256"
    ),
    "AMB2022-01-AMMT-PartCAD.STEP": (
        "https://data.nist.gov/od/ds/ark:/88434/mds2-2607/"
        "CAD_Geometry/AMB2022-01-AMMT-PartCAD.STEP"
    ),
    "AMB2022-01-AMMT-PartCAD.STEP.sha256": (
        "https://data.nist.gov/od/ds/ark:/88434/mds2-2607/"
        "CAD_Geometry/AMB2022-01-AMMT-PartCAD.STEP.sha256"
    ),
}
NIST_CMM_RIDGES: tuple[dict[str, float], ...] = (
    {
        "ridge": 1,
        "x_mm": 4.5,
        "y_mm": 2.5,
        "z_mm": 12.25,
        "deflection_mm": -0.003,
        "u_exp_mm": 0.002,
    },
    {
        "ridge": 2,
        "x_mm": 18.5,
        "y_mm": 2.5,
        "z_mm": 12.25,
        "deflection_mm": 0.016,
        "u_exp_mm": 0.002,
    },
    {
        "ridge": 3,
        "x_mm": 32.5,
        "y_mm": 2.5,
        "z_mm": 12.25,
        "deflection_mm": 0.183,
        "u_exp_mm": 0.002,
    },
    {
        "ridge": 4,
        "x_mm": 46.5,
        "y_mm": 2.5,
        "z_mm": 12.25,
        "deflection_mm": 0.523,
        "u_exp_mm": 0.002,
    },
    {
        "ridge": 5,
        "x_mm": 60.5,
        "y_mm": 2.5,
        "z_mm": 12.25,
        "deflection_mm": 1.058,
        "u_exp_mm": 0.002,
    },
    {
        "ridge": 6,
        "x_mm": 74.5,
        "y_mm": 2.5,
        "z_mm": 12.25,
        "deflection_mm": 1.809,
        "u_exp_mm": 0.002,
    },
)
NIST_EDD_U_EXP_UE = 100.0
CAD_UNIT_CELL_PITCH_MM = 14.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_nist_cache() -> dict[str, Any]:
    """Download (if absent) and verify official NIST AM-Bench 2022 SHA-256 digests."""
    spec = resolve_dataset_spec("nist-ambench")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in NIST_URLS.items():
        dest = CACHE_DIR / fname
        if not dest.is_file():
            req = urllib.request.Request(url, headers={"User-Agent": "FormaMech-VV10"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest.write_bytes(resp.read())

    hashes = {fname: _sha256_file(CACHE_DIR / fname) for fname in NIST_URLS}
    for base_f in ("AMB2022_EDD_results_V2.txt", "AMB2022-01-AMMT-PartCAD.STEP"):
        expected = (
            (CACHE_DIR / f"{base_f}.sha256").read_text().strip().split()[0].lower()
        )
        if hashes[base_f] != expected:
            raise ValueError(f"Official NIST SHA-256 verification failed for {base_f}")

    manifest_path = CACHE_DIR / "cache_manifest.json"
    if not manifest_path.is_file():
        manifest_path.write_bytes(
            canonical_json_bytes(
                {
                    "schema": EXTERNAL_DATASET_MANIFEST_SCHEMA,
                    "dataset": spec.document(),
                    "source_mode": "verified_local",
                    "doi": ["10.18434/mds2-2711", "10.18434/mds2-2607"],
                    "file_sha256": hashes,
                }
            )
        )
    return hashes


def _in718_material() -> ElectroThermalMaterialSpec:
    # Zero-traction thermo-elastic strain & displacement are invariant to E scaling;
    # solve in 100-MPa stress units (2000 -> 200,000 MPa) for the Neo-Hookean contract.
    mech = MaterialSpec(
        name="in718-datasheet",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": CanonicalQuantity(2000.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.29, Unit.ONE),
        },
    )
    return ElectroThermalMaterialSpec(
        part_id="in718_bridge",
        conductivity_xyz_w_per_mm_k=(0.0114, 0.0114, 0.0114),
        volumetric_heat_capacity_j_per_mm3_k=3.57e-3,
        thermal_expansion_per_k=13.0e-6,
        reference_temp_k=300.0,
        mechanical=mech,
    )


def _nodal_elastic_strains_ue(
    mesh: CanonicalMesh3D,
    u_nodal: FloatArray,
    t_nodal: FloatArray,
    mat: ElectroThermalMaterialSpec,
) -> tuple[FloatArray, FloatArray]:
    helper = TorchFemBackend3D()
    cells, d_phys, _, cell_vols, _, _ = helper._precompute_geometry(mesh)
    xi_m, _ = _tet_quadrature(mesh.cell_type)
    n_m = tet_shape_values_ref(mesh.cell_type, xi_m)
    t_q = np.einsum("qn,en->eq", n_m, t_nodal[cells])
    lam_th = 1.0 + mat.thermal_expansion_per_k * (t_q - mat.reference_temp_k)
    grad_u = np.einsum("eni,eqnj->eqij", u_nodal[cells], d_phys)
    f_mech = (np.eye(3)[None, None, :, :] + grad_u) / lam_th[:, :, None, None]
    e_m = 0.5 * (
        np.einsum("eqki,eqkj->eqij", f_mech, f_mech) - np.eye(3)[None, None, :, :]
    )
    return (
        _project_cell_scalar_to_nodes_3d(
            mesh, np.mean(e_m[:, :, 0, 0], axis=1) * 1.0e6, cell_vols
        ),
        _project_cell_scalar_to_nodes_3d(
            mesh, np.mean(e_m[:, :, 2, 2], axis=1) * 1.0e6, cell_vols
        ),
    )


def solve_bridge_prior(
    mesh_size_mm: float, *, delta_t_k: float = 500.0
) -> dict[str, Any]:
    """Import official NIST STEP bridge and solve as-built & cut-off 3D FEM."""
    cad: ImportedCad3D = import_cad_3d(
        CACHE_DIR / "AMB2022-01-AMMT-PartCAD.STEP",
        unit=Unit.MM,
        mesh_size_mm=mesh_size_mm,
        element_type=CellType3D.TET4,
        poisson_ratio=0.29,
    )
    mesh = cad.mesh
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    mat = _in718_material()
    anchor_tags = [
        f.name for f in cad.faces if f.centroid_mm[2] < 0.05 and f.normal_mm[2] < -0.9
    ]
    leg_tags = [
        f.name
        for f in cad.faces
        if 0.55 < f.centroid_mm[2] < 0.65
        and f.centroid_mm[0] > 19.5
        and f.normal_mm[2] < -0.9
    ]
    t_field = mat.reference_temp_k - delta_t_k * np.clip(
        nodes[:, 2] / float(np.max(nodes[:, 2])), 0.0, 1.0
    )
    res_ab = solve_3d_electro_thermal_mechanical(
        mesh,
        [mat],
        dirichlet_bcs=[
            DirichletBC3D(
                "build_plate",
                tuple([*anchor_tags, *leg_tags]),
                ("x", "y", "z"),
                (0.0, 0.0, 0.0),
            )
        ],
        prescribed_temperature_field_k=t_field,
        load_steps=2,
    )
    res_cut = solve_3d_electro_thermal_mechanical(
        mesh,
        [mat],
        dirichlet_bcs=[
            DirichletBC3D(
                "anchor_only", tuple(anchor_tags), ("x", "y", "z"), (0.0, 0.0, 0.0)
            )
        ],
        prescribed_temperature_field_k=t_field,
        load_steps=2,
    )
    exx, ezz = _nodal_elastic_strains_ue(mesh, res_ab.nodal_displacement, t_field, mat)
    return {
        "mesh": mesh,
        "nodes": nodes,
        "exx": exx,
        "ezz": ezz,
        "uz": res_cut.nodal_displacement[:, 2],
    }


def main() -> None:
    hashes = ensure_nist_cache()
    rows = []
    for ln in (
        (CACHE_DIR / "AMB2022_EDD_results_V2.txt")
        .read_text(encoding="utf-8", errors="replace")
        .splitlines()[1:]
    ):
        parts = ln.strip().split()
        if len(parts) >= 5:
            rows.append(
                [
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]) * 1e6,
                    float(parts[4]) * 1e6,
                ]
            )
    edd = np.asarray(rows, dtype=np.float64)
    ridge_xyz = np.asarray([[r["x_mm"], r["y_mm"], r["z_mm"]] for r in NIST_CMM_RIDGES])

    coarse = solve_bridge_prior(2.5)
    fine = solve_bridge_prior(1.8)
    sim_exx = align_sensor_observations_to_mesh(
        fine["nodes"], fine["exx"], edd[:, :3]
    ).matched_values
    sim_ezz = align_sensor_observations_to_mesh(
        fine["nodes"], fine["ezz"], edd[:, :3]
    ).matched_values
    sim_uz = align_sensor_observations_to_mesh(
        fine["nodes"], fine["uz"], ridge_xyz
    ).matched_values
    c_exx = align_sensor_observations_to_mesh(
        coarse["nodes"], coarse["exx"], edd[:, :3]
    ).matched_values
    c_ezz = align_sensor_observations_to_mesh(
        coarse["nodes"], coarse["ezz"], edd[:, :3]
    ).matched_values
    c_uz = align_sensor_observations_to_mesh(
        coarse["nodes"], coarse["uz"], ridge_xyz
    ).matched_values
    u_num_exx = float(np.sqrt(np.mean(np.square(sim_exx - c_exx))))
    u_num_ezz = float(np.sqrt(np.mean(np.square(sim_ezz - c_ezz))))
    u_num_uz = float(np.sqrt(np.mean(np.square(sim_uz - c_uz))))

    # Pure-CAD surface wall distances from STEP mesh
    nodes = np.asarray(fine["mesh"].nodes, dtype=np.float64)
    tri_pts = nodes[np.asarray(fine["mesh"].surface_triangles, dtype=np.int64)]
    tri_cent = tri_pts.mean(axis=1)
    nrm = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1.0e-12)
    d_xz, _ = cKDTree(tri_cent[np.abs(nrm[:, 1]) < 0.5][:, [0, 2]]).query(
        edd[:, [0, 2]], k=1
    )
    d_x, _ = cKDTree(tri_cent[np.abs(nrm[:, 0]) > 0.7][:, [0, 2]]).query(
        edd[:, [0, 2]], k=1
    )
    cvecs = [
        (
            float(edd[i, 0]),
            float(edd[i, 2]),
            float(d_x[i]),
            float(d_xz[i]),
            math.sin(2.0 * math.pi * float(edd[i, 0]) / CAD_UNIT_CELL_PITCH_MM),
            math.cos(2.0 * math.pi * float(edd[i, 0]) / CAD_UNIT_CELL_PITCH_MM),
        )
        for i in range(len(edd))
    ]

    # Study 3: Wire-EDM bridge warpage (3 cal / 3 blind ridges)
    pd_cal = {0, 2, 5}
    pd_recs = [
        ComponentTestRecord(
            specimen_id=f"nist-ridge-{int(r['ridge'])}",
            role=CouponRole.CALIBRATION
            if i in pd_cal
            else CouponRole.VALIDATION_HOLDOUT,
            sensor_kind=SensorObservationKind.DEFLECTION_PROFILE,
            condition_vector=(float(r["x_mm"]),),
            observed_values=(float(r["deflection_mm"]),),
            simulated_prior_values=(float(sim_uz[i]),),
            measurement_std=float(r["u_exp_mm"]),
            unit="mm",
        )
        for i, r in enumerate(NIST_CMM_RIDGES)
    ]
    _, d_pd = ground_simulation_with_experiments(
        pd_recs,
        config=HybridGroundingConfig(
            rho_prior_strength=1.0e-3,
            length_scale=1.5,
            u_num=u_num_uz,
            numerical_claim_level="NOT_VERIFIED",
        ),
    )
    print(f"[Study 3 - EDM Bridge Warp] {d_pd.reason} | state={d_pd.state.value}")

    # Study 1A / 1B: 20% synchrotron scan (450 cal / 250 blind)
    perm = np.random.default_rng(2022).permutation(len(edd))
    cal_idx, val_idx = perm[:450], perm[450:700]

    def _recs(
        obs: FloatArray, sim: FloatArray, tag: str
    ) -> tuple[list[ComponentTestRecord], list[ComponentTestRecord]]:
        return (
            [
                ComponentTestRecord(
                    f"{tag}-cal-{int(i):04d}",
                    CouponRole.CALIBRATION,
                    SensorObservationKind.DIC_STRAIN_FIELD,
                    cvecs[int(i)],
                    (float(obs[int(i)]),),
                    (float(sim[int(i)]),),
                    NIST_EDD_U_EXP_UE,
                    "microstrain",
                )
                for i in cal_idx
            ],
            [
                ComponentTestRecord(
                    f"{tag}-val-{int(i):04d}",
                    CouponRole.VALIDATION_HOLDOUT,
                    SensorObservationKind.DIC_STRAIN_FIELD,
                    cvecs[int(i)],
                    (float(obs[int(i)]),),
                    (float(sim[int(i)]),),
                    NIST_EDD_U_EXP_UE,
                    "microstrain",
                )
                for i in val_idx
            ],
        )

    exx_cal, exx_val = _recs(edd[:, 3], sim_exx, "exx")
    ezz_cal, ezz_val = _recs(edd[:, 4], sim_ezz, "ezz")
    cfg_exx = HybridGroundingConfig(
        length_scale=1.0, u_num=u_num_exx, numerical_claim_level="NOT_VERIFIED"
    )
    cfg_ezz = HybridGroundingConfig(
        length_scale=0.85, u_num=u_num_ezz, numerical_claim_level="NOT_VERIFIED"
    )
    _, d_exx = ground_simulation_with_experiments([*exx_cal, *exx_val], config=cfg_exx)
    _, d_ezz = ground_simulation_with_experiments([*ezz_cal, *ezz_val], config=cfg_ezz)
    print(f"[Study 1A - X-Ray e_xx]     {d_exx.reason} | state={d_exx.state.value}")
    print(f"[Study 1B - X-Ray e_zz]     {d_ezz.reason} | state={d_ezz.state.value}")

    # Study 4: Active X-ray scan reduction on e_zz (10 seeds, target <= 1000 ue)
    bench_ezz = run_retrospective_test_reduction_benchmark(
        ezz_cal[:180],
        ezz_val,
        initial_seed_count=10,
        max_rounds=120,
        target_holdout_rmse=1000.0,
        random_seeds=tuple(range(10)),
        grounding_config=cfg_ezz,
        plan_config=PhysicalTestPlanConfig(
            batch_size=1, sim_uncertainty_weight=0.0, length_scale=0.85, noise_std=0.15
        ),
    )
    print(
        f"[Active X-Ray e_zz <=1000]  active={bench_ezz.active_tests_needed} vs "
        f"random_mean={bench_ezz.random_mean_tests_needed:.1f} -> "
        f"saved={100.0 * bench_ezz.tests_saved_fraction:+.1f}%"
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "nist_ambench_2022_cli_run.json"
    out.write_bytes(
        canonical_json_bytes(
            {
                "nist_file_sha256": hashes,
                "study_3_edm_warp": d_pd.document(),
                "study_1a_exx": d_exx.document(),
                "study_1b_ezz": d_ezz.document(),
                "active_ezz_1000": bench_ezz.document(),
            }
        )
    )
    print(f"Wrote run receipt: {out}")


if __name__ == "__main__":
    main()
