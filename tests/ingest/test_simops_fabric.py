"""Gate G-V5-3a tests: Multi-solver SimOps fabric, failure separation & zero-leakage curation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from piginot.contracts import Unit
from piginot.geometry.mesh3d import make_brick_tet4_mesh
from piginot.ingest import (
    CaeSourceFormat,
    FailureDisposition,
    curate_archive_to_sharded_dataset,
    parse_abaqus_convergence_log,
    read_abaqus_simops_entry,
    read_ansys_cdb_and_fields,
    read_lsdyna_k_and_fields,
    read_nastran_simops_entry,
    read_simjeb_sample,
    require_optional_binary_reader,
)
from piginot.store.dataset_3d import ShardedMeshDataset3D


def _write_fields_json(path: Path, disp: np.ndarray, vm: np.ndarray) -> None:
    path.write_text(
        json.dumps({"displacement": disp.tolist(), "von_mises_mpa": vm.tolist()}),
        encoding="utf-8",
    )


def _write_abaqus_inp(path: Path, nodes: np.ndarray, cells: np.ndarray) -> None:
    lines = ["*NODE"]
    for i, (x, y, z) in enumerate(nodes, start=1):
        lines.append(f"{i}, {x:.6f}, {y:.6f}, {z:.6f}")
    lines.append("*ELEMENT, TYPE=C3D4")
    for e, row in enumerate(cells, start=1):
        ns = ", ".join(str(int(v) + 1) for v in row)
        lines.append(f"{e}, {ns}")
    lines.extend(
        [
            "*MATERIAL, NAME=AL6061",
            "*ELASTIC",
            "69000.0, 0.33",
            "*BOUNDARY",
            "1, 1, 3, 0.0",
            "*CLOAD",
            f"{len(nodes)}, 2, -250.0",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_ansys_cdb(path: Path, nodes: np.ndarray, cells: np.ndarray) -> None:
    lines = ["NBLOCK,6,SOLID", "(3i8,6e20.13)"]
    for i, (x, y, z) in enumerate(nodes, start=1):
        lines.append(f"{i} 0 0 {x:.6f} {y:.6f} {z:.6f}")
    lines.append("-1")
    lines.append("EBLOCK,19,SOLID")
    lines.append("(19i8)")
    for e, row in enumerate(cells, start=1):
        ns = " ".join(str(int(v) + 1) for v in row)
        lines.append(f"1 1 1 1 0 0 0 0 4 0 {e} {ns}")
    lines.append("-1")
    lines.extend(["MP,EX,1,210000.0", "MP,PRXY,1,0.30", "D,1,ALL,0.0", "F,2,FY,-100.0"])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_lsdyna_k(path: Path, nodes: np.ndarray, cells: np.ndarray) -> None:
    lines = ["*KEYWORD", "*NODE"]
    for i, (x, y, z) in enumerate(nodes, start=1):
        lines.append(f"{i},{x:.6f},{y:.6f},{z:.6f}")
    lines.append("*ELEMENT_SOLID")
    for e, row in enumerate(cells, start=1):
        n0, n1, n2, n3 = (int(v) + 1 for v in row)
        lines.append(f"{e},1,{n0},{n1},{n2},{n3},{n3},{n3},{n3},{n3}")
    lines.extend(
        [
            "*MAT_ELASTIC",
            "1, 7.85e-9, 205000.0, 0.29",
            "*BOUNDARY_SPC_NODE",
            "1,0,1,1,1,0,0,0",
            "*LOAD_NODE_POINT",
            f"{len(nodes)},2,1,-150.0",
            "*END",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_nastran_bdf(path: Path, nodes: np.ndarray, cells: np.ndarray) -> None:
    lines = ["BEGIN BULK"]
    for i, (x, y, z) in enumerate(nodes, start=1):
        lines.append(f"GRID,{i},0,{x:.6f},{y:.6f},{z:.6f}")
    for e, row in enumerate(cells, start=1):
        ns = ",".join(str(int(v) + 1) for v in row)
        lines.append(f"CTETRA,{e},1,{ns}")
    lines.extend(
        [
            "MAT1,1,71000.0,,0.33",
            "SPC1,1,123,1,2",
            f"FORCE,2,{len(nodes)},0,500.0,0.0,-1.0,0.0",
            "ENDDATA",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def test_simops_fabric_parses_multi_solver_separates_failures_and_locks_splits(
    tmp_path: Path,
) -> None:
    """Gate G-V5-3a: 100% defects quarantined, 100% physical failures preserved, 0 geometry leakage."""
    entries = []

    # 1) Abaqus converged run + 2nd load case on SAME geometry_group_id (must stay in same split!)
    m0 = make_brick_tet4_mesh(12.0, 6.0, 6.0, nx=2, ny=1, nz=1)
    n0, c0 = np.asarray(m0.nodes), np.asarray(m0.cells)
    u0 = 0.01 * n0
    vm0 = np.full(len(n0), 45.0)
    inp0 = tmp_path / "abq_g0_lc1.inp"
    fld0 = tmp_path / "abq_g0_lc1.json"
    sta0 = tmp_path / "abq_g0_lc1.sta"
    _write_abaqus_inp(inp0, n0, c0)
    _write_fields_json(fld0, u0, vm0)
    sta0.write_text(
        "1 1 1 0 1.0 1.0\nTHE ANALYSIS HAS COMPLETED SUCCESSFULLY\n", encoding="utf-8"
    )
    entries.append(
        read_abaqus_simops_entry(
            inp0,
            fld0,
            length_unit=Unit.MM,
            log_path=sta0,
            geometry_group_id="bracket-A",
            load_case_id="lc1",
            parameters={"load": 1.0},
        )
    )

    fld0_lc2 = tmp_path / "abq_g0_lc2.json"
    _write_fields_json(fld0_lc2, 1.5 * u0, 1.5 * vm0)
    entries.append(
        read_abaqus_simops_entry(
            inp0,
            fld0_lc2,
            length_unit=Unit.MM,
            log_path=sta0,
            geometry_group_id="bracket-A",
            load_case_id="lc2",
            sample_id="abq_g0_lc2",
            parameters={"load": 1.5},
        )
    )

    # 2) Abaqus VALID_PHYSICAL_FAILURE (buckling limit point reached) — MUST BE PRESERVED!
    m1 = make_brick_tet4_mesh(14.0, 6.0, 6.0, nx=2, ny=1, nz=1)
    n1, c1 = np.asarray(m1.nodes), np.asarray(m1.cells)
    inp1 = tmp_path / "abq_buckle.inp"
    fld1 = tmp_path / "abq_buckle.json"
    sta1 = tmp_path / "abq_buckle.sta"
    _write_abaqus_inp(inp1, n1, c1)
    _write_fields_json(fld1, 0.02 * n1, np.full(len(n1), 180.0))
    sta1.write_text(
        "1 1 1 0 0.5\n1 2 1U 1 0.8\nLIMIT POINT BUCKLING DETECTED AT LOAD FACTOR 0.82\n",
        encoding="utf-8",
    )
    prov1, label1 = parse_abaqus_convergence_log(sta1)
    assert prov1.convergence_status == "physical_limit_reached"
    assert label1 is not None and "buckling" in label1
    entries.append(
        read_abaqus_simops_entry(
            inp1,
            fld1,
            length_unit=Unit.MM,
            log_path=sta1,
            geometry_group_id="bracket-B",
        )
    )

    # 3) Abaqus QUARANTINED_SOLVER_DEFECT (diverged / too many attempts)
    inp_div = tmp_path / "abq_diverged.inp"
    fld_div = tmp_path / "abq_diverged.json"
    sta_div = tmp_path / "abq_diverged.sta"
    _write_abaqus_inp(inp_div, n1, c1)
    _write_fields_json(fld_div, 0.015 * n1, np.full(len(n1), 99.0))
    sta_div.write_text(
        "1 1 5U 5 0.01\nTOO MANY ATTEMPTS MADE FOR THIS INCREMENT\nTHE ANALYSIS HAS NOT BEEN COMPLETED\n",
        encoding="utf-8",
    )
    entries.append(
        read_abaqus_simops_entry(
            inp_div,
            fld_div,
            length_unit=Unit.MM,
            log_path=sta_div,
            geometry_group_id="bad-div",
        )
    )

    # 4) Ansys .cdb converged run
    m2 = make_brick_tet4_mesh(16.0, 7.0, 6.0, nx=2, ny=1, nz=1)
    n2, c2 = np.asarray(m2.nodes), np.asarray(m2.cells)
    cdb = tmp_path / "ansys_part.cdb"
    cdb_fld = tmp_path / "ansys_part.json"
    _write_ansys_cdb(cdb, n2, c2)
    _write_fields_json(cdb_fld, 0.008 * n2, np.full(len(n2), 60.0))
    ansys_entry = read_ansys_cdb_and_fields(
        cdb, cdb_fld, length_unit=Unit.MM, geometry_group_id="bracket-C"
    )
    assert ansys_entry.sample.source_format is CaeSourceFormat.ANSYS_CDB
    assert ansys_entry.material_summary["elastic_modulus_mpa"] == 210000.0
    entries.append(ansys_entry)

    # 5) LS-DYNA .k converged run
    m3 = make_brick_tet4_mesh(18.0, 7.0, 5.0, nx=2, ny=1, nz=1)
    n3, c3 = np.asarray(m3.nodes), np.asarray(m3.cells)
    kp = tmp_path / "dyna_part.k"
    kp_fld = tmp_path / "dyna_part.json"
    _write_lsdyna_k(kp, n3, c3)
    _write_fields_json(kp_fld, 0.006 * n3, np.full(len(n3), 75.0))
    dyna_entry = read_lsdyna_k_and_fields(
        kp, kp_fld, length_unit=Unit.MM, geometry_group_id="bracket-D"
    )
    assert dyna_entry.sample.source_format is CaeSourceFormat.LSDYNA_K
    entries.append(dyna_entry)

    # 6) Nastran .bdf semantic converged run
    m4 = make_brick_tet4_mesh(20.0, 8.0, 5.0, nx=2, ny=1, nz=1)
    n4, c4 = np.asarray(m4.nodes), np.asarray(m4.cells)
    bdf = tmp_path / "nastran_part.bdf"
    bdf_fld = tmp_path / "nastran_part.json"
    _write_nastran_bdf(bdf, n4, c4)
    _write_fields_json(bdf_fld, 0.007 * n4, np.full(len(n4), 52.0))
    nas_entry = read_nastran_simops_entry(
        bdf, bdf_fld, length_unit=Unit.MM, geometry_group_id="bracket-E"
    )
    assert nas_entry.material_summary["elastic_modulus_mpa"] == 71000.0
    assert nas_entry.bc_summary.net_force_n[1] == -500.0
    entries.append(nas_entry)

    # 7) SimJEB .npz with INVERTED element (det F < 0) -> MUST BE QUARANTINED
    npz_inv = tmp_path / "simjeb_inverted.npz"
    u_inv = -2.5 * n4  # flips all x/y/z coordinates -> det(F) < 0
    np.savez_compressed(npz_inv, nodes=n4, cells=c4, displacement=u_inv)
    entries.append(
        read_simjeb_sample(npz_inv, length_unit=Unit.MM, geometry_group_id="bad-inv")
    )

    # 8) SimJEB .npz with NaN field -> MUST BE QUARANTINED
    npz_nan = tmp_path / "simjeb_nan.npz"
    u_nan = 0.01 * n4.copy()
    u_nan[0, 0] = float("nan")
    np.savez_compressed(npz_nan, nodes=n4, cells=c4, displacement=u_nan)
    entries.append(
        read_simjeb_sample(npz_nan, length_unit=Unit.MM, geometry_group_id="bad-nan")
    )

    ds_dir = tmp_path / "curated_ds"
    report = curate_archive_to_sharded_dataset(entries, ds_dir)

    assert report.total_samples == 9
    assert report.valid_converged_count == 5
    assert report.valid_physical_failure_count == 1
    assert report.quarantined_solver_defect_count == 3
    assert report.dataset_manifest_sha256 is not None
    assert len(report.report_hash) == 64

    # Both load cases of bracket-A stay in the exact same split (zero geometry leakage)
    by_id = {r.sample_id: r for r in report.records}
    assert by_id["abq_g0_lc1"].split == by_id["abq_g0_lc2"].split
    assert by_id["abq_buckle"].disposition is FailureDisposition.VALID_PHYSICAL_FAILURE
    assert by_id["abq_buckle"].physical_failure_label is not None
    assert (
        by_id["abq_diverged"].disposition
        is FailureDisposition.QUARANTINED_SOLVER_DEFECT
    )
    assert (
        by_id["simjeb_inverted_vertical"].disposition
        is FailureDisposition.QUARANTINED_SOLVER_DEFECT
    )
    assert (
        by_id["simjeb_nan_vertical"].disposition
        is FailureDisposition.QUARANTINED_SOLVER_DEFECT
    )

    # Sharded dataset integrity verifies cleanly with 6 accepted 3D records
    ds = ShardedMeshDataset3D(ds_dir)
    integrity = ds.verify_integrity()
    assert integrity["ok"] is True
    assert integrity["total_records"] == 6


def test_optional_binary_readers_fail_closed_with_actionable_hint() -> None:
    """Unknown binary reader raises LookupError; missing optional packages fail closed."""
    with pytest.raises(LookupError, match="unknown binary reader"):
        require_optional_binary_reader("unknown-format")
    try:
        require_optional_binary_reader("ansys-rst")
    except RuntimeError as exc:
        assert "pip install" in str(exc)
