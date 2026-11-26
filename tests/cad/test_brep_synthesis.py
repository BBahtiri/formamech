"""Gates G-V5-4a & G-V5-4b tests: Watertight B-Rep STEP synthesis & GD&T tolerance checks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from piginot.cad.brep_synthesis import (
    GdtToleranceBandSpec,
    KeepOutDatumFeature,
    synthesize_watertight_brep_step,
    verify_gdt_tolerance_robustness,
)
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import (
    DirichletBC3D,
    FemRequest3D,
    TorchFemBackend3D,
    TractionBC3D,
)
from piginot.geometry.mesh3d import make_brick_tet4_mesh


def test_watertight_brep_step_synthesis_preserves_bolt_bore_and_remeshed_fem(
    tmp_path: Path,
) -> None:
    """Gate G-V5-4a: OpenCASCADE STEP round-trips with 0 open edges, <=0.05 mm bore, and exact FEM."""
    mesh = make_brick_tet4_mesh(24.0, 12.0, 8.0, nx=4, ny=2, nz=2)
    solid_mask = np.ones(len(mesh.cells), dtype=np.bool_)

    bore = KeepOutDatumFeature(
        name="mounting-bolt-m4",
        kind="cylindrical_bore",
        origin_mm=(12.0, 6.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        radius_mm=2.0,
        length_mm=8.0,
    )
    step_path = tmp_path / "bracket_synthesized.step"
    result = synthesize_watertight_brep_step(
        mesh,
        solid_mask,
        step_path,
        keep_out_features=(bore,),
        n_sections=4,
        verify_roundtrip=True,
        roundtrip_mesh_size_mm=5.0,
        run_remeshed_fem_check=True,
        bore_tolerance_mm=0.05,
    )
    assert step_path.is_file()
    assert result.passed is True, result.reason
    assert result.watertight is True
    assert result.open_boundary_edge_count == 0
    assert result.nonmanifold_edge_count == 0
    assert result.volume_mm3 > 0.0
    assert result.max_bore_center_error_mm <= 0.05
    assert result.remeshed_fem_verified is True
    assert (
        result.remeshed_fem_max_disp_mm is not None
        and result.remeshed_fem_max_disp_mm > 0.0
    )
    assert len(result.receipt_sha256) == 64


def test_gdt_tolerance_robustness_certifies_worst_case_and_fails_closed_on_violation() -> (
    None
):
    """Gate G-V5-4b: Screens +/- tol corners via fast eval, certifies worst-case with exact 3D FEM."""
    mat = MaterialSpec(
        name="gdt-al",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(1000.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.30, Unit.ONE).to_canonical(),
        },
    )

    def _exact_fem(params: Mapping[str, float]) -> tuple[float, float]:
        mesh = make_brick_tet4_mesh(
            float(params["length_mm"]),
            float(params["height_mm"]),
            float(params["width_mm"]),
            nx=3,
            ny=2,
            nz=2,
        )
        req = FemRequest3D(
            request_id="gdt-corner",
            mesh=mesh,
            material=mat,
            dirichlet_bcs=(
                DirichletBC3D("fix", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
            ),
            traction_bcs=(TractionBC3D("load", ("x_max",), (0.0, -0.5, 0.0)),),
            load_steps=1,
        )
        res = TorchFemBackend3D().solve(req)
        assert res.converged
        peak_vm = float(
            res.quadrature_peak_von_mises_mpa
            if res.quadrature_peak_von_mises_mpa is not None
            else res.peak_von_mises_mpa
        )
        return peak_vm, float(res.max_displacement_mm)

    def _fast_surrogate(params: Mapping[str, float]) -> tuple[float, float]:
        # Monotonic Euler-Bernoulli scaling proportional to L/h^2 and L^3/h^3
        l_v = float(params["length_mm"])
        h_v = float(params["height_mm"])
        w_v = float(params["width_mm"])
        stress = 3.0 * (l_v / 20.0) * ((8.0 / h_v) ** 2) * (8.0 / w_v)
        disp = 0.04 * ((l_v / 20.0) ** 3) * ((8.0 / h_v) ** 3) * (8.0 / w_v)
        return stress, disp

    # 1) Generous allowable -> passes all 8 ISO-2768-m corners
    spec_ok = GdtToleranceBandSpec(
        nominal_parameters={"length_mm": 20.0, "height_mm": 8.0, "width_mm": 8.0},
        tolerances_mm={"length_mm": 0.2, "height_mm": 0.15, "width_mm": 0.15},
        max_allowable_stress_mpa=50.0,
        max_allowable_displacement_mm=1.0,
    )
    rep_ok = verify_gdt_tolerance_robustness(
        spec_ok, fast_evaluator=_fast_surrogate, exact_fem_evaluator=_exact_fem
    )
    assert rep_ok.corners_screened == 8
    assert rep_ok.passed is True
    # Worst corner has longest length and thinnest height/width
    assert rep_ok.worst_stress_corner["length_mm"] == 20.2
    assert rep_ok.worst_stress_corner["height_mm"] == 7.85
    assert len(rep_ok.receipt_sha256) == 64

    # 2) Tight limit violated at worst-case in-tolerance corner -> fails closed
    spec_tight = GdtToleranceBandSpec(
        nominal_parameters={"length_mm": 20.0, "height_mm": 8.0, "width_mm": 8.0},
        tolerances_mm={"length_mm": 0.2, "height_mm": 0.15, "width_mm": 0.15},
        max_allowable_stress_mpa=0.5,
        max_allowable_displacement_mm=1.0,
    )
    rep_fail = verify_gdt_tolerance_robustness(
        spec_tight, fast_evaluator=_fast_surrogate, exact_fem_evaluator=_exact_fem
    )
    assert rep_fail.passed is False
    assert "violation" in rep_fail.reason.lower()
