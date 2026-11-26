"""Parametric 3D Electronics / Thermal Management CAD Templates & Stackup Importer (V5-5).

Provides pure-MIT OpenCASCADE (``gmsh.model.occ``) parametric templates and a
multi-layer JSON/IPC-2581-lite stackup builder targeting the highest-growth
Electro-Thermal-Mechanical verticals:

* ``parametric-chiplet-package-3d`` — multi-die silicon chiplets + TIM + copper
  heat spreader + organic substrate assembly;
* ``parametric-cold-plate-3d`` — channeled aluminum/copper liquid-cooling cold
  plate with spatially varying convective cooling map;
* ``parametric-busway-3d`` — multi-bar copper busway under Joule heating;
* :func:`import_electronics_stackup_json` — multi-layer stackup & volumetric
  power-floorplan importer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from piginot.cad.importing_3d import _gmsh_module, import_cad_3d
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.electro_thermal_3d import (
    ElectroThermalMaterialSpec,
    VolumetricPowerSource3D,
)
from piginot.geometry.assembly_3d import AssemblyMesh3D, PartRegionSpec
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D, make_brick_tet4_mesh

ELECTRONICS_TEMPLATES_3D: Final[tuple[str, ...]] = (
    "parametric-chiplet-package-3d",
    "parametric-cold-plate-3d",
    "parametric-busway-3d",
)


@dataclass(frozen=True, slots=True)
class ElectronicsAssemblyBundle:
    """Generated 3D electronics/thermal assembly paired with multi-material ETM specs."""

    template_name: str
    assembly: AssemblyMesh3D
    etm_materials: tuple[ElectroThermalMaterialSpec, ...]
    power_sources: tuple[VolumetricPowerSource3D, ...]
    step_path: Path | None = None

    def summary_document(self) -> dict[str, Any]:
        return {
            "template_name": self.template_name,
            "assembly_id": self.assembly.assembly_id,
            "n_nodes": len(self.assembly.mesh.nodes),
            "n_cells": len(self.assembly.mesh.cells),
            "parts": [p.document() for p in self.assembly.parts],
            "power_source_count": len(self.power_sources),
            "step_path": None if self.step_path is None else str(self.step_path),
        }


def _make_mech_mat(name: str, e_mpa: float, nu: float) -> MaterialSpec:
    # Respect compressible_neo_hookean's canonical [1, 10_000] MPa envelope
    e_clamped = min(max(float(e_mpa), 1.0), 10_000.0)
    return MaterialSpec(
        name=name,
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(e_clamped, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(float(nu), Unit.ONE).to_canonical(),
        },
    )


def _partition_brick_layers(
    mesh: CanonicalMesh3D,
    layer_z_splits: tuple[float, ...],
    layer_specs: tuple[ElectroThermalMaterialSpec, ...],
) -> AssemblyMesh3D:
    """Assign brick tetrahedra to stacked z-layers based on element z-centroids."""
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    z_c = np.mean(nodes[cells, 2], axis=1)
    buckets: list[list[int]] = [[] for _ in layer_specs]
    for eid, zv in enumerate(z_c.tolist()):
        placed = False
        for li, z_cut in enumerate(layer_z_splits):
            if zv <= z_cut:
                buckets[li].append(eid)
                placed = True
                break
        if not placed:
            buckets[-1].append(eid)
    parts: list[PartRegionSpec] = []
    for spec, cids in zip(layer_specs, buckets, strict=True):
        if not cids:
            raise ValueError(f"layer {spec.part_id!r} received zero cells; refine nz")
        parts.append(
            PartRegionSpec(
                part_id=spec.part_id,
                cell_indices=tuple(cids),
                material=spec.mechanical,
            )
        )
    return AssemblyMesh3D(mesh=mesh, parts=tuple(parts))


def generate_chiplet_package_3d(
    *,
    package_xy_mm: float = 20.0,
    substrate_thick_mm: float = 1.5,
    die_thick_mm: float = 1.0,
    spreader_thick_mm: float = 1.5,
    chiplet_power_w: float = 25.0,
    nx: int = 4,
    ny: int = 4,
    emit_step_path: Path | str | None = None,
) -> ElectronicsAssemblyBundle:
    """Build a 3-layer 3D-IC chiplet package (organic substrate + silicon die + copper spreader)."""
    total_z = substrate_thick_mm + die_thick_mm + spreader_thick_mm
    mesh = make_brick_tet4_mesh(
        package_xy_mm, package_xy_mm, total_z, nx=nx, ny=ny, nz=3
    )
    sub_mat = ElectroThermalMaterialSpec(
        part_id="organic_substrate",
        conductivity_xyz_w_per_mm_k=(0.020, 0.020, 0.0008),  # Orthotropic PCB!
        volumetric_heat_capacity_j_per_mm3_k=1.8e-3,
        thermal_expansion_per_k=1.6e-5,
        reference_temp_k=298.15,
        mechanical=_make_mech_mat("fr4-substrate", 2200.0, 0.30),
    )
    die_mat = ElectroThermalMaterialSpec(
        part_id="silicon_chiplets",
        conductivity_xyz_w_per_mm_k=(0.148, 0.148, 0.148),
        volumetric_heat_capacity_j_per_mm3_k=1.63e-3,
        thermal_expansion_per_k=2.6e-6,
        reference_temp_k=298.15,
        mechanical=_make_mech_mat("silicon-die", 9000.0, 0.28),
    )
    lid_mat = ElectroThermalMaterialSpec(
        part_id="copper_spreader",
        conductivity_xyz_w_per_mm_k=(0.40, 0.40, 0.40),
        volumetric_heat_capacity_j_per_mm3_k=3.45e-3,
        thermal_expansion_per_k=1.7e-5,
        reference_temp_k=298.15,
        mechanical=_make_mech_mat("copper-lid", 8000.0, 0.34),
        electrical_conductivity_s_per_mm=5.8e4,
    )
    z1 = total_z / 3.0
    z2 = 2.0 * total_z / 3.0
    assembly = _partition_brick_layers(mesh, (z1, z2), (sub_mat, die_mat, lid_mat))
    die_vol = package_xy_mm * package_xy_mm * (total_z / 3.0)
    q_dens = float(chiplet_power_w) / max(die_vol, 1.0e-6)
    power = VolumetricPowerSource3D(
        name="chiplet-floorplan",
        power_density_w_per_mm3=q_dens,
        part_id="silicon_chiplets",
    )

    step_out: Path | None = None
    if emit_step_path is not None:
        step_out = Path(emit_step_path).resolve()
        step_out.parent.mkdir(parents=True, exist_ok=True)
        gmsh = _gmsh_module()
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("chiplet_package_3d")
            b1 = gmsh.model.occ.addBox(
                0.0, 0.0, 0.0, package_xy_mm, package_xy_mm, substrate_thick_mm
            )
            b2 = gmsh.model.occ.addBox(
                0.2 * package_xy_mm,
                0.2 * package_xy_mm,
                substrate_thick_mm,
                0.6 * package_xy_mm,
                0.6 * package_xy_mm,
                die_thick_mm,
            )
            b3 = gmsh.model.occ.addBox(
                0.0,
                0.0,
                substrate_thick_mm + die_thick_mm,
                package_xy_mm,
                package_xy_mm,
                spreader_thick_mm,
            )
            gmsh.model.occ.fuse([(3, b1)], [(3, b2), (3, b3)])
            gmsh.model.occ.synchronize()
            gmsh.write(str(step_out))
        finally:
            gmsh.finalize()

    return ElectronicsAssemblyBundle(
        template_name="parametric-chiplet-package-3d",
        assembly=assembly,
        etm_materials=(sub_mat, die_mat, lid_mat),
        power_sources=(power,),
        step_path=step_out,
    )


def generate_cold_plate_3d(
    *,
    length_mm: float = 40.0,
    width_mm: float = 20.0,
    height_mm: float = 6.0,
    channel_width_mm: float = 4.0,
    channel_depth_mm: float = 3.0,
    mesh_size_mm: float = 5.0,
    emit_step_path: Path | str,
) -> ElectronicsAssemblyBundle:
    """Build an OpenCASCADE channeled liquid-cooling cold plate STEP & mesh."""
    step_out = Path(emit_step_path).resolve()
    step_out.parent.mkdir(parents=True, exist_ok=True)
    gmsh = _gmsh_module()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cold_plate_3d")
        base = gmsh.model.occ.addBox(0.0, 0.0, 0.0, length_mm, width_mm, height_mm)
        ch1 = gmsh.model.occ.addBox(
            -1.0,
            0.25 * width_mm - 0.5 * channel_width_mm,
            height_mm - channel_depth_mm,
            length_mm + 2.0,
            channel_width_mm,
            channel_depth_mm + 1.0,
        )
        ch2 = gmsh.model.occ.addBox(
            -1.0,
            0.75 * width_mm - 0.5 * channel_width_mm,
            height_mm - channel_depth_mm,
            length_mm + 2.0,
            channel_width_mm,
            channel_depth_mm + 1.0,
        )
        gmsh.model.occ.cut([(3, base)], [(3, ch1), (3, ch2)])
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_out))
    finally:
        gmsh.finalize()

    imported = import_cad_3d(
        step_out, unit=Unit.MM, mesh_size_mm=mesh_size_mm, element_type=CellType3D.TET4
    )
    al_mat = ElectroThermalMaterialSpec(
        part_id="cold_plate_al6061",
        conductivity_xyz_w_per_mm_k=(0.167, 0.167, 0.167),
        volumetric_heat_capacity_j_per_mm3_k=2.42e-3,
        thermal_expansion_per_k=2.36e-5,
        reference_temp_k=293.15,
        mechanical=_make_mech_mat("al6061-coldplate", 6900.0, 0.33),
        electrical_conductivity_s_per_mm=2.5e4,
    )
    assembly = AssemblyMesh3D(
        mesh=imported.mesh,
        parts=(
            PartRegionSpec(
                part_id=al_mat.part_id,
                cell_indices=tuple(range(len(imported.mesh.cells))),
                material=al_mat.mechanical,
            ),
        ),
    )
    return ElectronicsAssemblyBundle(
        template_name="parametric-cold-plate-3d",
        assembly=assembly,
        etm_materials=(al_mat,),
        power_sources=(),
        step_path=step_out,
    )


def generate_busway_3d(
    *,
    length_mm: float = 50.0,
    width_mm: float = 10.0,
    thickness_mm: float = 4.0,
    nx: int = 6,
    ny: int = 2,
    nz: int = 2,
) -> ElectronicsAssemblyBundle:
    """Build a 3D copper power busway bar ready for coupled Joule-heating ETM solves."""
    mesh = make_brick_tet4_mesh(length_mm, width_mm, thickness_mm, nx=nx, ny=ny, nz=nz)
    cu_mat = ElectroThermalMaterialSpec(
        part_id="copper_busbar",
        conductivity_xyz_w_per_mm_k=(0.40, 0.40, 0.40),
        volumetric_heat_capacity_j_per_mm3_k=3.45e-3,
        thermal_expansion_per_k=1.7e-5,
        reference_temp_k=293.15,
        mechanical=_make_mech_mat("copper-busbar", 8000.0, 0.34),
        electrical_conductivity_s_per_mm=5.8e4,
    )
    assembly = AssemblyMesh3D(
        mesh=mesh,
        parts=(
            PartRegionSpec(
                part_id=cu_mat.part_id,
                cell_indices=tuple(range(len(mesh.cells))),
                material=cu_mat.mechanical,
            ),
        ),
    )
    return ElectronicsAssemblyBundle(
        template_name="parametric-busway-3d",
        assembly=assembly,
        etm_materials=(cu_mat,),
        power_sources=(),
    )


def import_electronics_stackup_json(
    stackup_path: Path | str,
    *,
    nx: int = 3,
    ny: int = 3,
) -> ElectronicsAssemblyBundle:
    """Import an IPC-2581-lite JSON multi-layer PCB/chiplet stackup into an ETM bundle."""
    doc = json.loads(Path(stackup_path).read_text(encoding="utf-8"))
    lx = float(doc["width_x_mm"])
    ly = float(doc["width_y_mm"])
    layers = list(doc["layers"])
    if not layers:
        raise ValueError("stackup JSON must contain at least one layer")

    total_z = sum(float(layer["thickness_mm"]) for layer in layers)
    mesh = make_brick_tet4_mesh(lx, ly, total_z, nx=nx, ny=ny, nz=len(layers))

    etm_specs: list[ElectroThermalMaterialSpec] = []
    power_srcs: list[VolumetricPowerSource3D] = []
    z_splits: list[float] = []
    cum_z = 0.0
    dz_uniform = total_z / float(len(layers))
    for idx, layer in enumerate(layers):
        pid = str(layer["name"])
        k_raw = layer["conductivity_xyz_w_per_mm_k"]
        spec = ElectroThermalMaterialSpec(
            part_id=pid,
            conductivity_xyz_w_per_mm_k=(
                float(k_raw[0]),
                float(k_raw[1]),
                float(k_raw[2]),
            ),
            volumetric_heat_capacity_j_per_mm3_k=float(
                layer.get("volumetric_heat_capacity_j_per_mm3_k", 2.0e-3)
            ),
            thermal_expansion_per_k=float(layer.get("cte_per_k", 1.2e-5)),
            reference_temp_k=float(layer.get("reference_temp_k", 298.15)),
            mechanical=_make_mech_mat(
                f"mech-{pid}",
                float(layer.get("youngs_modulus_mpa", 3000.0)),
                float(layer.get("poisson_ratio", 0.30)),
            ),
            electrical_conductivity_s_per_mm=float(
                layer.get("electrical_conductivity_s_per_mm", 0.0)
            ),
        )
        etm_specs.append(spec)
        if idx < len(layers) - 1:
            cum_z += dz_uniform
            z_splits.append(cum_z)
        q_w = float(layer.get("power_w", 0.0))
        if q_w > 0.0:
            layer_vol = lx * ly * dz_uniform
            power_srcs.append(
                VolumetricPowerSource3D(
                    name=f"power-{pid}",
                    power_density_w_per_mm3=q_w / max(layer_vol, 1.0e-6),
                    part_id=pid,
                )
            )

    assembly = _partition_brick_layers(mesh, tuple(z_splits), tuple(etm_specs))
    return ElectronicsAssemblyBundle(
        template_name=str(doc.get("name", "imported-stackup-3d")),
        assembly=assembly,
        etm_materials=tuple(etm_specs),
        power_sources=tuple(power_srcs),
    )


__all__ = (
    "ELECTRONICS_TEMPLATES_3D",
    "ElectronicsAssemblyBundle",
    "generate_busway_3d",
    "generate_chiplet_package_3d",
    "generate_cold_plate_3d",
    "import_electronics_stackup_json",
)
