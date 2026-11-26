"""Tests for V5-5 parametric 3D electronics & thermal management templates and stackup importer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from piginot.cad.electronics_templates_3d import (
    generate_busway_3d,
    generate_chiplet_package_3d,
    generate_cold_plate_3d,
    import_electronics_stackup_json,
)
from piginot.fem.backend_3d import DirichletBC3D
from piginot.fem.electro_thermal_3d import (
    SpatialConvectionBC3D,
    VoltageBC3D,
    solve_3d_electro_thermal_mechanical,
)
from piginot.fem.thermo_mechanical_3d import ConvectionBC3D, TemperatureBC3D


def test_chiplet_package_3d_coupled_warpage_and_step(tmp_path: Path) -> None:
    """Multi-material chiplet package (FR4 + Si + Cu) solves coupled ETM warpage & writes STEP."""
    step_p = tmp_path / "chiplet.step"
    bundle = generate_chiplet_package_3d(chiplet_power_w=15.0, emit_step_path=step_p)
    assert step_p.is_file()
    assert len(bundle.assembly.parts) == 3

    res = solve_3d_electro_thermal_mechanical(
        bundle.assembly,
        bundle.etm_materials,
        dirichlet_bcs=(
            DirichletBC3D("mount", ("z_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        convection_bcs=(
            ConvectionBC3D(
                "spreader_cool", ("z_max",), h_w_per_mm2_k=0.005, t_ambient_k=298.15
            ),
        ),
        volumetric_sources=bundle.power_sources,
        load_steps=1,
    )
    assert res.converged
    assert res.max_temperature_k > 298.15
    assert res.max_displacement_mm > 0.0
    assert res.quadrature_peak_von_mises_mpa > 0.0


def test_cold_plate_3d_and_busway_3d_and_stackup_json(tmp_path: Path) -> None:
    """Cold plate STEP + spatially varying h(x), copper busway Joule heating, and JSON stackup."""
    cp = generate_cold_plate_3d(
        emit_step_path=tmp_path / "cold_plate.step", mesh_size_mm=6.0
    )
    assert cp.step_path is not None and cp.step_path.is_file()
    assert len(cp.assembly.mesh.cells) > 0

    bus = generate_busway_3d(
        length_mm=40.0, width_mm=8.0, thickness_mm=4.0, nx=4, ny=2, nz=2
    )
    res_bus = solve_3d_electro_thermal_mechanical(
        bus.assembly,
        bus.etm_materials,
        dirichlet_bcs=(
            DirichletBC3D("fix", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        voltage_bcs=(
            VoltageBC3D("v0", ("x_min",), 0.0),
            VoltageBC3D("v1", ("x_max",), 0.02),
        ),
        temperature_bcs=(TemperatureBC3D("t0", ("x_min",), 293.15),),
        spatial_convection_bcs=(
            SpatialConvectionBC3D(
                "air",
                ("z_max",),
                h_fn=lambda pts: 1.0e-4 * (1.0 + 0.02 * pts[:, 0]),
                t_amb_fn=lambda pts: np.full(len(pts), 293.15),
            ),
        ),
        load_steps=1,
    )
    assert res_bus.converged
    assert res_bus.total_joule_power_w > 0.0
    assert res_bus.max_temperature_k > 293.15

    stackup_path = tmp_path / "stackup.json"
    stackup_path.write_text(
        json.dumps(
            {
                "name": "ai-accelerator-interposer",
                "width_x_mm": 15.0,
                "width_y_mm": 15.0,
                "layers": [
                    {
                        "name": "pcb",
                        "thickness_mm": 1.0,
                        "conductivity_xyz_w_per_mm_k": [0.02, 0.02, 0.001],
                        "cte_per_k": 1.6e-5,
                        "youngs_modulus_mpa": 2500.0,
                    },
                    {
                        "name": "die",
                        "thickness_mm": 0.8,
                        "conductivity_xyz_w_per_mm_k": [0.148, 0.148, 0.148],
                        "cte_per_k": 2.6e-6,
                        "youngs_modulus_mpa": 9000.0,
                        "power_w": 12.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    imported = import_electronics_stackup_json(stackup_path)
    assert len(imported.assembly.parts) == 2
    assert len(imported.power_sources) == 1
