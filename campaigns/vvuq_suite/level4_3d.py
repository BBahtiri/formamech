"""Level 3c (2.5D Axisymmetric) & Level 4 (3D NAFEMS LE10, 3D Cook, OOD Shift) VVUQ suite."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    ExecutionProfile,
    LoadSpec,
    MaterialSpec,
    QoISpec,
    Quantity,
    Unit,
)
from piginot.fem.models import FemRequest, FemSolverConfig
from piginot.fem.torch_backend import TorchFemBackend
from piginot.geometry import (
    CellType,
    CellType3D,
    MeshPayload,
    canonicalize_mesh,
    make_brick_tet10_mesh,
)

LEVEL4_BENCHMARK_SCHEMA: Final = "formamech.vvuq-level4-spec-v1"


@dataclass(frozen=True, slots=True)
class Level4BenchmarkSpec:
    """Declarative ASME V&V 10 / NAFEMS Level 4 3D benchmark contract."""

    benchmark_id: str
    title: str
    reference_source: str
    physics_mode: str
    required_cell_types: tuple[CellType3D, ...]
    poisson_ratio: float
    target_qoi_name: str
    target_qoi_value: float | None
    target_qoi_unit: str
    tolerance_relative: float | None
    notes: str

    def document(self) -> dict[str, Any]:
        return {
            "schema": LEVEL4_BENCHMARK_SCHEMA,
            "benchmark_id": self.benchmark_id,
            "title": self.title,
            "reference_source": self.reference_source,
            "physics_mode": self.physics_mode,
            "required_cell_types": [c.value for c in self.required_cell_types],
            "poisson_ratio": self.poisson_ratio,
            "target_qoi_name": self.target_qoi_name,
            "target_qoi_value": self.target_qoi_value,
            "target_qoi_unit": self.target_qoi_unit,
            "tolerance_relative": self.tolerance_relative,
            "notes": self.notes,
        }


def nafems_le10_spec() -> Level4BenchmarkSpec:
    """NAFEMS LE10 thick elliptic plate with hole 3D benchmark contract."""
    return Level4BenchmarkSpec(
        benchmark_id="nafems-le10-3d",
        title="NAFEMS LE10 Thick Elliptic Plate with Central Elliptic Hole",
        reference_source="NAFEMS Standard Benchmarks, TNSB Rev 3 (LE10)",
        physics_mode="3d-finite-strain",
        required_cell_types=(CellType3D.TET10, CellType3D.HEX8),
        poisson_ratio=0.30,
        target_qoi_name="sigma_yy_point_D",
        target_qoi_value=-5.38,
        target_qoi_unit="MPa",
        tolerance_relative=0.02,
        notes=(
            "Quarter-symmetry thick elliptic plate (E=210 GPa, nu=0.3, p=1 MPa "
            "normal pressure on outer edge); target direct stress sigma_yy=-5.38 MPa "
            "at point D (2000, 0, 300) mm."
        ),
    )


def cooks_membrane_3d_spec() -> Level4BenchmarkSpec:
    """3D nearly incompressible Cook's Membrane benchmark contract (refuses TET4)."""
    return Level4BenchmarkSpec(
        benchmark_id="cooks-membrane-3d-incompressible",
        title="3D Tapered Cook's Membrane (Nearly Incompressible Neo-Hookean)",
        reference_source="Simo & Armero (1992) / Reese et al. (2000)",
        physics_mode="3d-finite-strain",
        required_cell_types=(CellType3D.TET10, CellType3D.HEX8),
        poisson_ratio=0.49,
        target_qoi_name="tip_vertical_displacement",
        target_qoi_value=None,
        target_qoi_unit="mm",
        tolerance_relative=0.02,
        notes=(
            "Clamped tapered 3D block under dead end shear; nu=0.49 explicitly "
            "triggers TET4 volumetric-locking refusal, requiring TET10 or HEX8."
        ),
    )


@dataclass(frozen=True, slots=True)
class OodShiftGateReport:
    """Conformal coverage shift and OOD false-acceptance verification gate."""

    nominal_coverage: float
    empirical_id_coverage: float
    empirical_ood_coverage: float
    ood_false_acceptance_rate: float
    total_ood_samples: int
    false_acceptances: int
    gate_passed: bool

    def document(self) -> dict[str, Any]:
        return {
            "nominal_coverage": self.nominal_coverage,
            "empirical_id_coverage": self.empirical_id_coverage,
            "empirical_ood_coverage": self.empirical_ood_coverage,
            "ood_false_acceptance_rate": self.ood_false_acceptance_rate,
            "total_ood_samples": self.total_ood_samples,
            "false_acceptances": self.false_acceptances,
            "gate_passed": self.gate_passed,
        }


def evaluate_ood_conformal_shift_gate(
    *,
    nominal_coverage: float,
    id_errors: Sequence[float],
    id_bounds: Sequence[float],
    ood_errors: Sequence[float],
    ood_bounds: Sequence[float],
    ood_accepted_flags: Sequence[bool],
) -> OodShiftGateReport:
    """Audit empirical ID vs OOD coverage and require zero OOD false acceptances."""
    if len(id_errors) != len(id_bounds) or not id_errors:
        raise ValueError("id_errors and id_bounds must be non-empty and same length")
    if (
        not (len(ood_errors) == len(ood_bounds) == len(ood_accepted_flags))
        or not ood_errors
    ):
        raise ValueError("ood sequences must be non-empty and share the same length")
    id_cov = sum(
        1 for err, bnd in zip(id_errors, id_bounds, strict=True) if err <= bnd
    ) / len(id_errors)
    ood_cov = sum(
        1 for err, bnd in zip(ood_errors, ood_bounds, strict=True) if err <= bnd
    ) / len(ood_errors)
    false_acc = sum(
        1
        for err, bnd, acc in zip(
            ood_errors, ood_bounds, ood_accepted_flags, strict=True
        )
        if acc and err > bnd
    )
    far = false_acc / len(ood_errors)
    return OodShiftGateReport(
        nominal_coverage=nominal_coverage,
        empirical_id_coverage=id_cov,
        empirical_ood_coverage=ood_cov,
        ood_false_acceptance_rate=far,
        total_ood_samples=len(ood_errors),
        false_acceptances=false_acc,
        gate_passed=(false_acc == 0),
    )


def axisymmetric_cylinder_mesh(
    *,
    r_inner: float = 10.0,
    r_outer: float = 20.0,
    height: float = 5.0,
    nr: int = 8,
    nz: int = 2,
) -> CanonicalMesh:
    """Structured P1 triangle mesh of a thick-walled cylinder meridional section."""
    nodes: list[tuple[float, float]] = []
    for j in range(nz + 1):
        z = height * j / nz
        for i in range(nr + 1):
            r = r_inner + (r_outer - r_inner) * i / nr
            nodes.append((r, z))

    def _nid(i: int, j: int) -> int:
        return j * (nr + 1) + i

    cells: list[tuple[int, int, int]] = []
    for j in range(nz):
        for i in range(nr):
            n00, n10 = _nid(i, j), _nid(i + 1, j)
            n01, n11 = _nid(i, j + 1), _nid(i + 1, j + 1)
            cells.append((n00, n10, n11))
            cells.append((n00, n11, n01))

    facets: list[tuple[int, int]] = []
    tags: dict[str, list[int]] = {"inner": [], "outer": [], "bottom": [], "top": []}
    for i in range(nr):
        tags["bottom"].append(len(facets))
        facets.append((_nid(i, 0), _nid(i + 1, 0)))
    for j in range(nz):
        tags["outer"].append(len(facets))
        facets.append((_nid(nr, j), _nid(nr, j + 1)))
    for i in range(nr, 0, -1):
        tags["top"].append(len(facets))
        facets.append((_nid(i, nz), _nid(i - 1, nz)))
    for j in range(nz, 0, -1):
        tags["inner"].append(len(facets))
        facets.append((_nid(0, j), _nid(0, j - 1)))

    return canonicalize_mesh(
        MeshPayload(
            tuple(nodes),
            tuple(cells),
            CellType.TRIANGLE,
            tuple(facets),
            {k: tuple(v) for k, v in tags.items()},
            (1,) * len(cells),
            Unit.MM,
            {"kind": "axisymmetric-thick-cylinder"},
        )
    )


def run_level3c_pressurized_cylinder_axisymmetric(
    *,
    r_inner: float = 10.0,
    r_outer: float = 20.0,
    height: float = 4.0,
    pressure_mpa: float = 0.05,
    elastic_modulus_mpa: float = 100.0,
    poisson_ratio: float = 0.3,
    nr: int = 12,
    nz: int = 2,
    element_order: int = 2,
) -> dict[str, Any]:
    """Axisymmetric pressurized thick-walled cylinder verified against Lamé's solution.

    Plane-strain Lamé closed-form radial displacement (small-strain limit ``p << E``):
    ``u_r(R) = (1 + nu) * p * a^2 / (E * (b^2 - a^2)) * ((1 - 2*nu)*R + b^2 / R)``.
    """
    mesh = axisymmetric_cylinder_mesh(
        r_inner=r_inner, r_outer=r_outer, height=height, nr=nr, nz=nz
    )
    mat = MaterialSpec(
        name="lame-neo-hookean",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(elastic_modulus_mpa, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(poisson_ratio, Unit.ONE).to_canonical(),
        },
    )
    bcs = (
        BoundaryConditionSpec(
            name="fix_bottom_z",
            boundary_tag="bottom",
            kind="dirichlet",
            components=("y",),
            values=(Quantity(0.0, Unit.MM).to_canonical(),),
        ),
        BoundaryConditionSpec(
            name="fix_top_z",
            boundary_tag="top",
            kind="dirichlet",
            components=("y",),
            values=(Quantity(0.0, Unit.MM).to_canonical(),),
        ),
    )
    # Internal radial pressure pushes outward (+R) on the inner wall
    loads = (
        LoadSpec(
            name="internal_pressure",
            kind="traction",
            boundary_tag="inner",
            components=("x", "y"),
            values=(
                Quantity(pressure_mpa, Unit.MPA).to_canonical(),
                Quantity(0.0, Unit.MPA).to_canonical(),
            ),
        ),
    )
    profile = ExecutionProfile(
        python_version="3.13",
        library_versions={"torch": "runtime"},
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )
    req = FemRequest(
        request_id="vvuq-level3c-axi-cylinder",
        mesh=mesh,
        material=mat,
        boundary_conditions=bcs,
        loads=loads,
        qois=(
            QoISpec(
                name="max_u",
                field="u",
                reduction="max_norm",
                region_tag="inner",
                unit=Unit.MM,
            ),
        ),
        config=FemSolverConfig(
            element_order=element_order,
            solver_tolerance=1.0e-11,
            max_newton_iterations=16,
            load_steps=1,
        ),
        execution_profile=profile,
    )
    backend = TorchFemBackend(physics_mode="2d-axisymmetric-finite-strain")
    attempt = backend.solve_many((req,))[0]
    if not attempt.success or attempt.result is None:
        raise RuntimeError(f"Axisymmetric cylinder failed: {attempt.message}")

    coords = np.asarray(attempt.result.displacement_dof_coordinates, dtype=np.float64)
    u = np.asarray(attempt.result.displacement, dtype=np.float64)
    inner_mask = np.isclose(coords[:, 0], r_inner, atol=1.0e-9)
    outer_mask = np.isclose(coords[:, 0], r_outer, atol=1.0e-9)
    ur_inner_fem = float(u[inner_mask, 0].mean())
    ur_outer_fem = float(u[outer_mask, 0].mean())

    # Exact plane-strain Lamé formula
    a2, b2 = r_inner * r_inner, r_outer * r_outer
    pref = (1.0 + poisson_ratio) * pressure_mpa * a2 / (elastic_modulus_mpa * (b2 - a2))
    ur_inner_lame = pref * ((1.0 - 2.0 * poisson_ratio) * r_inner + b2 / r_inner)
    ur_outer_lame = pref * ((1.0 - 2.0 * poisson_ratio) * r_outer + b2 / r_outer)
    rel_err_inner = abs(ur_inner_fem - ur_inner_lame) / abs(ur_inner_lame)
    rel_err_outer = abs(ur_outer_fem - ur_outer_lame) / abs(ur_outer_lame)

    return {
        "benchmark": "level3c-axisymmetric-pressurized-cylinder",
        "backend_id": backend.backend_id,
        "ur_inner_fem_mm": ur_inner_fem,
        "ur_inner_lame_mm": ur_inner_lame,
        "ur_outer_fem_mm": ur_outer_fem,
        "ur_outer_lame_mm": ur_outer_lame,
        "rel_error_inner": rel_err_inner,
        "rel_error_outer": rel_err_outer,
        "passed": bool(rel_err_inner < 5.0e-3 and rel_err_outer < 5.0e-3),
    }


def build_level4_smoke_tet10_mesh() -> dict[str, Any]:
    """Construct a valid TET10 3D mesh and validate it against both Level 4 specs."""
    mesh = make_brick_tet10_mesh(48.0, 44.0, 10.0, nx=2, ny=2, nz=1)
    cooks = cooks_membrane_3d_spec()
    mesh.validate_incompressibility_locking(cooks.poisson_ratio)
    return {
        "mesh_id": mesh.mesh_id,
        "cell_type": mesh.cell_type.value,
        "node_count": len(mesh.nodes),
        "cell_count": len(mesh.cells),
        "surface_triangle_count": len(mesh.surface_triangles),
        "specs": [nafems_le10_spec().document(), cooks.document()],
    }


__all__ = (
    "LEVEL4_BENCHMARK_SCHEMA",
    "Level4BenchmarkSpec",
    "OodShiftGateReport",
    "axisymmetric_cylinder_mesh",
    "build_level4_smoke_tet10_mesh",
    "cooks_membrane_3d_spec",
    "evaluate_ood_conformal_shift_gate",
    "nafems_le10_spec",
    "run_level3c_pressurized_cylinder_axisymmetric",
)
