"""Verification tests for Phase 3 (V3-3) CAE archive ingestion, screening & ParaView/GLB export."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from piginot.cad.importing_3d import (
    classify_faces_by_bbox,
    generate_3d_box_step,
    import_cad_3d,
)
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.export.vtu_export import (
    export_deformed_glb,
    export_fields_to_vtu,
    export_trajectory_to_pvd,
)
from piginot.fem.backend_3d import (
    DirichletBC3D,
    FemRequest3D,
    TorchFemBackend3D,
    TractionBC3D,
)
from piginot.geometry.mesh3d import CellType3D
from piginot.ingest import (
    INGEST_PIPELINE_MACHINE,
    CaeSourceFormat,
    IngestPipelineState,
    OutlierReasonCode,
    ParsedCaeSample,
    read_abaqus_inp_and_fields,
    read_calculix_inp_frd,
    read_nastran_bdf_and_fields,
    read_vtu_or_xdmf,
    screen_cae_archive,
)


def test_vtu_pvd_glb_roundtrip_and_cae_readers(tmp_path: Path) -> None:
    step = generate_3d_box_step(tmp_path / "b.step", lx=12.0, ly=6.0, lz=6.0)
    imported = import_cad_3d(step, mesh_size_mm=7.0, element_type=CellType3D.TET10)
    faces = classify_faces_by_bbox(imported)
    mat = MaterialSpec(
        name="nh-vtu",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(10.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.35, Unit.ONE).to_canonical(),
        },
    )
    res = TorchFemBackend3D().solve(
        FemRequest3D(
            request_id="vtu-box",
            mesh=imported.mesh,
            material=mat,
            dirichlet_bcs=(
                DirichletBC3D(
                    "clamp", faces["x_min"], ("x", "y", "z"), (0.0, 0.0, 0.0)
                ),
            ),
            traction_bcs=(TractionBC3D("pull", faces["x_max"], (0.4, 0.0, 0.0)),),
            load_steps=1,
        )
    )
    assert res.converged

    # 1. ParaView .vtu export & read_vtu_or_xdmf round-trip
    vtu_path = tmp_path / "result.vtu"
    vtu_rec = export_fields_to_vtu(
        imported.mesh,
        vtu_path,
        point_fields={
            "displacement_mm": res.nodal_displacement,
            "von_mises_mpa": res.nodal_von_mises_mpa,
        },
        write_manifest=True,
    )
    assert vtu_rec.spatial_dim == 3
    assert vtu_rec.cell_type == "tetra10"
    assert (tmp_path / "result.vtu.manifest.json").is_file()

    parsed_vtu = read_vtu_or_xdmf(vtu_path, sample_id="run-01", parameters={"tx": 0.4})
    assert parsed_vtu.source_format is CaeSourceFormat.VTU
    assert parsed_vtu.cell_type == "tet10"
    assert np.allclose(
        parsed_vtu.point_fields["displacement_mm"], res.nodal_displacement, atol=1e-12
    )

    # 2. ParaView .pvd time-series export
    pvd_info = export_trajectory_to_pvd(
        imported.mesh,
        tmp_path / "pvd_series",
        times=[0.0, 0.5, 1.0],
        point_fields_per_step=[
            {"displacement_mm": f * res.nodal_displacement} for f in (0.0, 0.5, 1.0)
        ],
    )
    assert pvd_info["step_count"] == 3
    assert Path(pvd_info["pvd_path"]).is_file()

    # 3. Binary glTF 2.0 (.glb) export
    glb_path = tmp_path / "deformed.glb"
    glb_info = export_deformed_glb(
        imported.mesh,
        glb_path,
        displacement=res.nodal_displacement,
        scalar_field=res.nodal_von_mises_mpa,
        displacement_scale=2.0,
    )
    raw_glb = glb_path.read_bytes()
    magic, version, total_len = struct.unpack("<III", raw_glb[:12])
    assert magic == 0x46546C67 and version == 2 and total_len == len(raw_glb)
    assert glb_info["triangle_count"] == len(imported.mesh.surface_triangles)

    # 4. Synthetic Abaqus (.inp + .csv), CalculiX (.inp + .frd), and Nastran (.bdf + .csv)
    inp_text = "\n".join(
        [
            "*NODE",
            "1, 0.0, 0.0, 0.0",
            "2, 1.0, 0.0, 0.0",
            "3, 0.0, 1.0, 0.0",
            "4, 0.0, 0.0, 1.0",
            "*ELEMENT, TYPE=C3D4",
            "1, 1, 2, 3, 4",
        ]
    )
    inp_p = tmp_path / "tet.inp"
    inp_p.write_text(inp_text, encoding="utf-8")

    csv_p = tmp_path / "fields.csv"
    csv_p.write_text(
        "node_id,ux,uy,uz,von_mises\n"
        "1,0.0,0.0,0.0,1.2\n"
        "2,0.1,0.0,0.0,1.2\n"
        "3,0.0,-0.03,0.0,1.2\n"
        "4,0.0,0.0,-0.03,1.2\n",
        encoding="utf-8",
    )
    abq = read_abaqus_inp_and_fields(inp_p, csv_p)
    assert abq.source_format is CaeSourceFormat.ABAQUS_INP
    assert abq.cell_type == "tet4"
    assert abq.point_fields["displacement"].shape == (4, 3)

    frd_p = tmp_path / "tet.frd"
    frd_p.write_text(
        " -4  DISP        4    1\n"
        " -1         1 0.00000E+00 0.00000E+00 0.00000E+00\n"
        " -1         2 1.00000E-01 0.00000E+00 0.00000E+00\n"
        " -1         3 0.00000E+00-3.00000E-02 0.00000E+00\n"
        " -1         4 0.00000E+00 0.00000E+00-3.00000E-02\n"
        " -3\n"
        " -4  STRESS      6    1\n"
        " -1         1 1.0 0.0 0.0 0.0 0.0 0.0\n"
        " -1         2 1.0 0.0 0.0 0.0 0.0 0.0\n"
        " -1         3 1.0 0.0 0.0 0.0 0.0 0.0\n"
        " -1         4 1.0 0.0 0.0 0.0 0.0 0.0\n"
        " -3\n",
        encoding="utf-8",
    )
    ccx = read_calculix_inp_frd(inp_p, frd_p)
    assert ccx.source_format is CaeSourceFormat.CALCULIX_FRD
    assert np.allclose(ccx.point_fields["von_mises_mpa"], 1.0, atol=1e-12)

    bdf_p = tmp_path / "tet.bdf"
    bdf_p.write_text(
        "$ Nastran tet\n"
        "GRID,1,,0.0,0.0,0.0\n"
        "GRID,2,,1.0,0.0,0.0\n"
        "GRID,3,,0.0,1.0,0.0\n"
        "GRID,4,,0.0,0.0,1.0\n"
        "CTETRA,1,1,1,2,3,4\n",
        encoding="utf-8",
    )
    nas = read_nastran_bdf_and_fields(bdf_p, csv_p)
    assert nas.source_format is CaeSourceFormat.NASTRAN_BDF
    assert nas.cells.shape == (1, 4)


def test_cae_outlier_screening_quarantines_folded_duplicate_and_zscore_outliers(
    tmp_path: Path,
) -> None:
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)

    def _mk(sid: str, disp_x: float, h_tag: str) -> ParsedCaeSample:
        disp = np.array([[0.0, 0.0], [disp_x, 0.0], [0.0, 0.0]], dtype=np.float64)
        return ParsedCaeSample(
            sample_id=sid,
            source_format=CaeSourceFormat.VTU,
            source_sha256=h_tag,
            nodes=nodes,
            cells=cells,
            cell_type="triangle",
            point_fields={
                "displacement": disp,
                "von_mises_mpa": np.full(3, abs(disp_x) * 10.0),
            },
            cell_fields={},
            parameters={"load": disp_x},
            metadata={},
        )

    # 8 clean runs (disp_x ~ 0.10..0.17) + 1 folded (disp_x=-1.5 -> det F = -0.5)
    # + 1 duplicate hash + 1 50-sigma statistical outlier (disp_x=5.0)
    samples = [_mk(f"ok-{i}", 0.10 + 0.01 * i, f"h-ok-{i}") for i in range(8)]
    samples.append(_mk("bad-folded", -1.5, "h-folded"))
    samples.append(_mk("bad-dup", 0.12, "h-ok-2"))
    samples.append(_mk("bad-zscore", 5.0, "h-zscore"))

    rep = screen_cae_archive(
        samples, zscore_threshold=3.0, output_dir=tmp_path / "screened"
    )
    assert INGEST_PIPELINE_MACHINE.machine_id == "formamech.ingest.cae_pipeline"
    assert rep.state is IngestPipelineState.PARTITIONED
    assert rep.snapshot().state == "PARTITIONED"
    assert rep.accepted_count == 8
    assert rep.quarantined_count == 3
    reasons = {f.sample_id: f.reason_code for f in rep.flags}
    assert reasons["bad-folded"] is OutlierReasonCode.NEGATIVE_JACOBIAN
    assert reasons["bad-dup"] is OutlierReasonCode.DUPLICATE_HASH
    assert reasons["bad-zscore"] is OutlierReasonCode.STATISTICAL_ZSCORE
    assert (tmp_path / "screened" / "archive-manifest.json").is_file()
    assert (tmp_path / "screened" / "outlier-screen.json").is_file()
