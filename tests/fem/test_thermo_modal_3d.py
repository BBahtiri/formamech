"""Verification tests for Phase 4C (V3-4C) 3D thermo-mechanical & modal eigenfrequency analysis."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from piginot.cad.importing_3d import (
    classify_faces_by_bbox,
    generate_3d_box_step,
    import_cad_3d,
)
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import (
    DirichletBC3D,
    FemRequest3D,
    TorchFemBackend3D,
    TractionBC3D,
)
from piginot.fem.modal_3d import (
    analytical_eigenfrequency_sensitivity,
    assemble_3d_mass_matrix,
    solve_3d_modal_analysis,
)
from piginot.fem.thermo_mechanical_3d import (
    ConvectionBC3D,
    TemperatureBC3D,
    ThermalMaterialSpec,
    solve_3d_steady_heat_conduction,
    solve_3d_thermo_mechanical,
    solve_3d_transient_heat_conduction,
)
from piginot.geometry.mesh3d import CellType3D


def _polymer_composite() -> MaterialSpec:
    # Stays within verified S4 compressible Neo-Hookean envelope 1 <= E <= 10000 MPa
    return MaterialSpec(
        name="composite-v34",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(5_000.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.30, Unit.ONE).to_canonical(),
        },
    )


def test_3d_steady_transient_heat_and_constrained_thermal_stress_analytical(
    tmp_path: Path,
) -> None:
    step = generate_3d_box_step(tmp_path / "tbar.step", lx=20.0, ly=6.0, lz=6.0)
    imp = import_cad_3d(step, mesh_size_mm=7.0, element_type=CellType3D.TET10)
    faces = classify_faces_by_bbox(imp)
    thermal = ThermalMaterialSpec(
        conductivity_w_per_mm_k=0.05,
        density_t_per_mm3=7.85e-9,
        specific_heat_mj_per_t_k=4.6e8,
        thermal_expansion_per_k=1.2e-5,
        reference_temp_k=300.0,
    )

    # 1. Steady 1D-in-3D linear temperature gradient T(x) = 300 + 100 * (x / 20)
    t_steady = solve_3d_steady_heat_conduction(
        imp.mesh,
        thermal,
        temperature_bcs=(
            TemperatureBC3D("cold", faces["x_min"], 300.0),
            TemperatureBC3D("hot", faces["x_max"], 400.0),
        ),
    )
    xs = np.asarray([p[0] for p in imp.mesh.nodes], dtype=np.float64)
    t_exact = 300.0 + 100.0 * (xs / 20.0)
    assert float(np.max(np.abs(t_steady - t_exact))) < 1.0e-10

    # 1b. Consistent P2 Robin convection on TET10 vs exact 1D Biot-number solution
    h_conv = 0.0025  # Bi = h * L / k = 0.0025 * 20 / 0.05 = 1.0
    bi = h_conv * 20.0 / thermal.conductivity_w_per_mm_k
    t_conv = solve_3d_steady_heat_conduction(
        imp.mesh,
        thermal,
        temperature_bcs=(TemperatureBC3D("hot", faces["x_min"], 400.0),),
        convection_bcs=(
            ConvectionBC3D(
                "cool_end", faces["x_max"], h_w_per_mm2_k=h_conv, t_ambient_k=300.0
            ),
        ),
    )
    t_conv_exact = 400.0 - (bi / (1.0 + bi)) * 100.0 * (xs / 20.0)
    assert float(np.max(np.abs(t_conv - t_conv_exact))) < 1.0e-10

    # 2. Transient heating moves smoothly toward steady state
    hist = solve_3d_transient_heat_conduction(
        imp.mesh,
        thermal,
        initial_temperature_k=300.0,
        dt_s=1.0,
        num_steps=3,
        temperature_bcs=(
            TemperatureBC3D("cold", faces["x_min"], 300.0),
            TemperatureBC3D("hot", faces["x_max"], 400.0),
        ),
    )
    assert len(hist) == 4
    assert float(np.mean(hist[-1])) > float(np.mean(hist[0]))

    # 2b. Transient 1D-in-3D sine eigenmode decay verifies exact W <-> mJ diffusivity alpha = k/(rho*cp)
    t_init_sin = 300.0 + 100.0 * np.sin(math.pi * xs / 20.0)
    hist_sin = solve_3d_transient_heat_conduction(
        imp.mesh,
        thermal,
        initial_temperature_k=t_init_sin,
        dt_s=0.1,
        num_steps=10,
        temperature_bcs=(
            TemperatureBC3D("end0", faces["x_min"], 300.0),
            TemperatureBC3D("end1", faces["x_max"], 300.0),
        ),
    )
    alpha_mm2_s = thermal.conductivity_w_per_mm_k / (
        1.0e-3 * thermal.density_t_per_mm3 * thermal.specific_heat_mj_per_t_k
    )
    decay_exact = math.exp(-alpha_mm2_s * ((math.pi / 20.0) ** 2) * 1.0)
    t_sin_exact = 300.0 + 100.0 * decay_exact * np.sin(math.pi * xs / 20.0)
    assert float(np.max(np.abs(hist_sin[-1] - t_sin_exact))) < 1.5

    # 3. Free uniform heating dT = +50 K -> exact zero stress & exact uniform expansion alpha*dT*L
    t_unif = np.full(len(imp.mesh.nodes), 350.0, dtype=np.float64)
    res_free = solve_3d_thermo_mechanical(
        imp.mesh,
        _polymer_composite(),
        thermal,
        dirichlet_bcs=(
            DirichletBC3D("x0", faces["x_min"], ("x",), (0.0,)),
            DirichletBC3D("y0", faces["y_min"], ("y",), (0.0,)),
            DirichletBC3D("z0", faces["z_min"], ("z",), (0.0,)),
        ),
        prescribed_temperature_field_k=t_unif,
        load_steps=1,
    )
    assert res_free.converged
    expected_ux = thermal.thermal_expansion_per_k * 50.0 * 20.0
    end_nodes = np.where(np.abs(xs - 20.0) < 1.0e-3)[0]
    assert (
        abs(float(np.mean(res_free.nodal_displacement[end_nodes, 0])) - expected_ux)
        < 1.0e-9
    )
    assert res_free.peak_von_mises_mpa < 1.0e-6

    # 4. Axially clamped bar under uniform dT = +50 K (nu = 0) -> exact compressive stress sigma = E * alpha * dT
    e_val = 5_000.0
    mat_nu0 = MaterialSpec(
        name="bar-nu0",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(e_val, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.0, Unit.ONE).to_canonical(),
        },
    )
    res_clamped = solve_3d_thermo_mechanical(
        imp.mesh,
        mat_nu0,
        thermal,
        dirichlet_bcs=(
            DirichletBC3D("x0", faces["x_min"], ("x",), (0.0,)),
            DirichletBC3D("x1", faces["x_max"], ("x",), (0.0,)),
            DirichletBC3D("y0", faces["y_min"], ("y",), (0.0,)),
            DirichletBC3D("z0", faces["z_min"], ("z",), (0.0,)),
        ),
        prescribed_temperature_field_k=t_unif,
        load_steps=1,
    )
    assert res_clamped.converged
    sigma_exact = e_val * thermal.thermal_expansion_per_k * 50.0
    assert abs(res_clamped.peak_von_mises_mpa - sigma_exact) / sigma_exact < 0.01


def test_3d_cantilever_modal_euler_bernoulli_prestress_and_sensitivity(
    tmp_path: Path,
) -> None:
    # Slender cantilevered beam: L=100 mm, b=10 mm, h=10 mm
    lx, ly, lz = 100.0, 10.0, 10.0
    step = generate_3d_box_step(tmp_path / "beam.step", lx=lx, ly=ly, lz=lz)
    imp = import_cad_3d(step, mesh_size_mm=10.0, element_type=CellType3D.TET10)
    faces = classify_faces_by_bbox(imp)
    rho = 1.6e-9  # composite tonne/mm^3
    e_mpa = 5_000.0

    # Exact mass conservation check via Duffy degree-5 quadrature
    m_mat = assemble_3d_mass_matrix(imp.mesh, rho)
    e_x = np.zeros(3 * len(imp.mesh.nodes), dtype=np.float64)
    e_x[0::3] = 1.0
    mass_calc = float(e_x @ (m_mat @ e_x))
    mass_exact = rho * lx * ly * lz
    assert abs(mass_calc - mass_exact) / mass_exact < 1.0e-12

    # Unloaded modal solve vs Euler-Bernoulli 1st bending frequency
    modal0 = solve_3d_modal_analysis(
        imp.mesh,
        _polymer_composite(),
        density_t_per_mm3=rho,
        dirichlet_bcs=(
            DirichletBC3D("clamp", faces["x_min"], ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        num_modes=4,
    )
    i_area = ly * (lz**3) / 12.0
    area = ly * lz
    f1_eb = (
        (1.87510407**2)
        / (2.0 * math.pi * lx * lx)
        * math.sqrt(e_mpa * i_area / (rho * area))
    )
    # 3D Timoshenko/TET10 beam frequency closely tracks Euler-Bernoulli (< 6% shear/3D effect)
    assert abs(modal0.frequencies_hz[0] - f1_eb) / f1_eb < 0.06
    assert modal0.document()["schema"] == "formamech.modal-3d-result-v1"

    # Prestressed tensile load increases 1st bending frequency (geometric stress stiffening)
    fem_pre = TorchFemBackend3D().solve(
        FemRequest3D(
            request_id="prestress-beam",
            mesh=imp.mesh,
            material=_polymer_composite(),
            dirichlet_bcs=(
                DirichletBC3D(
                    "clamp", faces["x_min"], ("x", "y", "z"), (0.0, 0.0, 0.0)
                ),
            ),
            traction_bcs=(TractionBC3D("tension", faces["x_max"], (50.0, 0.0, 0.0)),),
            load_steps=1,
        )
    )
    assert fem_pre.converged
    modal_pre = solve_3d_modal_analysis(
        imp.mesh,
        _polymer_composite(),
        density_t_per_mm3=rho,
        dirichlet_bcs=(
            DirichletBC3D("clamp", faces["x_min"], ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        num_modes=4,
        prestress_displacement=fem_pre.nodal_displacement,
    )
    assert modal_pre.prestressed is True
    assert modal_pre.frequencies_hz[0] > modal0.frequencies_hz[0]

    # Exact analytical sensitivity d(omega)/d(rho) = -omega / (2 * rho)
    dm_drho = assemble_3d_mass_matrix(imp.mesh, 1.0)
    from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

    zero_k = csr_matrix(m_mat.shape, dtype=np.float64)
    domega_drho = analytical_eigenfrequency_sensitivity(
        modal0.angular_frequencies_rad_s[0],
        modal0.mode_shapes_nodal[0],
        zero_k,
        dm_drho,
    )
    expected_sens = -modal0.angular_frequencies_rad_s[0] / (2.0 * rho)
    assert abs(domega_drho - expected_sens) / abs(expected_sens) < 1.0e-10
