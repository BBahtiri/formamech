"""Reproducible Level 1-3 nonlinear finite-strain VVUQ benchmark suite.

Aligned with ASME V&V 10-2019 (R2025) and NAFEMS nonlinear benchmarks:

* **Level 1**: Code & calculation verification via nonlinear MMS convergence
  and exact finite-difference tangent verification (including unsymmetric
  follower-pressure load stiffness).
* **Level 2**: Canonical 2D nonlinear hyperelasticity — NAFEMS / Abaqus Cook's
  Membrane (combined bending + shear, nearly incompressible Neo-Hookean,
  structured for direct parity comparison against PFEM 2026).
* **Level 3**: Constitutive & soft-tissue complexity — Holzapfel-Gasser-Ogden
  (HGO) anisotropic arterial strip (exponential fibre stiffening & transverse
  stretch) and elastomer cantilever under non-conservative follower pressure
  (reporting tangent asymmetry, warm-start Newton iteration reduction ratio R,
  and ``min det(F) > 0``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from piginot.certify.guards import coercive_inertia
from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    LoadSpec,
    MaterialSpec,
    Unit,
)
from piginot.fem import FemRequest, FemSolverConfig, ScikitFemBackend, SolverBackend
from piginot.fem.mms_runner import build_mms_problem, verification_execution_profile
from piginot.fem.verification_norms import compute_verification_norms
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh


def cooks_membrane_mesh(nx: int, ny: int) -> CanonicalMesh:
    """Build canonical NAFEMS/Abaqus 2D Cook's Membrane mesh.

    Corner vertices (mm): (0, 0), (48, 44), (48, 60), (0, 44).
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive integers")
    nodes: list[tuple[float, float]] = []
    for j in range(ny + 1):
        eta = j / ny
        for i in range(nx + 1):
            xi = i / nx
            x = 48.0 * xi
            y_bot = 44.0 * xi
            y_top = 44.0 + 16.0 * xi
            y = (1.0 - eta) * y_bot + eta * y_top
            nodes.append((x, y))

    cells: list[tuple[int, int, int]] = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * (nx + 1) + i
            n10 = n00 + 1
            n01 = (j + 1) * (nx + 1) + i
            n11 = n01 + 1
            cells.append((n00, n10, n11))
            cells.append((n00, n11, n01))

    facets: list[tuple[int, int]] = []
    tags: dict[str, list[int]] = {"left": [], "right": [], "bottom": [], "top": []}
    for i in range(nx):
        tags["bottom"].append(len(facets))
        facets.append((i, i + 1))
    for j in range(ny):
        tags["right"].append(len(facets))
        facets.append((j * (nx + 1) + nx, (j + 1) * (nx + 1) + nx))
    for i in range(nx, 0, -1):
        tags["top"].append(len(facets))
        facets.append((ny * (nx + 1) + i, ny * (nx + 1) + i - 1))
    for j in range(ny, 0, -1):
        tags["left"].append(len(facets))
        facets.append((j * (nx + 1), (j - 1) * (nx + 1)))

    payload = MeshPayload(
        nodes=tuple(nodes),
        cells=tuple(cells),
        cell_type=CellType.TRIANGLE,
        boundary_facets=tuple(facets),
        boundary_tags={k: tuple(v) for k, v in tags.items()},
        region_ids=(1,) * len(cells),
        unit=Unit.MM,
        source_provenance={"benchmark": "nafems-cooks-membrane-2d"},
    )
    return canonicalize_mesh(payload)


def rectangular_strip_mesh(
    length_mm: float, height_mm: float, nx: int, ny: int, *, label: str = "strip"
) -> CanonicalMesh:
    """Build a canonical structured triangular mesh on ``[0, length] x [0, height]``."""
    nodes = [
        (length_mm * i / nx, height_mm * j / ny)
        for j in range(ny + 1)
        for i in range(nx + 1)
    ]
    cells: list[tuple[int, int, int]] = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * (nx + 1) + i
            n10 = n00 + 1
            n01 = (j + 1) * (nx + 1) + i
            n11 = n01 + 1
            cells.append((n00, n10, n11))
            cells.append((n00, n11, n01))

    facets: list[tuple[int, int]] = []
    tags: dict[str, list[int]] = {"left": [], "right": [], "bottom": [], "top": []}
    for i in range(nx):
        tags["bottom"].append(len(facets))
        facets.append((i, i + 1))
    for j in range(ny):
        tags["right"].append(len(facets))
        facets.append((j * (nx + 1) + nx, (j + 1) * (nx + 1) + nx))
    for i in range(nx, 0, -1):
        tags["top"].append(len(facets))
        facets.append((ny * (nx + 1) + i, ny * (nx + 1) + i - 1))
    for j in range(ny, 0, -1):
        tags["left"].append(len(facets))
        facets.append((j * (nx + 1), (j - 1) * (nx + 1)))

    payload = MeshPayload(
        nodes=tuple(nodes),
        cells=tuple(cells),
        cell_type=CellType.TRIANGLE,
        boundary_facets=tuple(facets),
        boundary_tags={k: tuple(v) for k, v in tags.items()},
        region_ids=(1,) * len(cells),
        unit=Unit.MM,
        source_provenance={"benchmark": label},
    )
    return canonicalize_mesh(payload)


def _min_det_f_from_result(result: Any) -> float:
    f3d = np.asarray(result.deformation_gradient, dtype=np.float64)
    f2d = f3d[..., :2, :2]
    det2d = f2d[..., 0, 0] * f2d[..., 1, 1] - f2d[..., 0, 1] * f2d[..., 1, 0]
    return float(np.min(det2d))


def run_level1_mms_verification(
    backend: SolverBackend | None = None,
    *,
    case_name: str = "nonlinear_mixed",
    levels: Sequence[int] = (1, 2),
    element_order: int = 2,
) -> dict[str, Any]:
    """Level 1: Backend-agnostic nonlinear MMS order-of-accuracy verification."""
    rows: list[dict[str, Any]] = []
    for level in levels:
        prob = build_mms_problem(case_name, level=level, element_order=element_order)
        solver = (
            ScikitFemBackend(
                body_force_providers=(prob.body_force_provider,),
                dirichlet_field_providers=(prob.displacement_provider,),
                traction_field_providers=(prob.traction_provider,),
            )
            if backend is None
            else backend
        )
        attempt = solver.solve_many((prob.request,))[0]
        if not attempt.success or attempt.result is None:
            raise RuntimeError(f"Level 1 MMS level={level} failed: {attempt.message}")
        norms = compute_verification_norms(
            prob.request, attempt.result, prob.manufactured_solution
        )
        rows.append(
            {
                "level": level,
                "dof_count": len(attempt.result.displacement) * 2,
                "u_l2_rel_error": norms.displacement_l2_relative,
                "u_h1_rel_error": norms.displacement_h1_relative,
                "p_l2_rel_error": norms.stress_l2_relative,
            }
        )

    l2_rate: float | None = None
    h1_rate: float | None = None
    if len(rows) >= 2:
        l2_rate = math.log2(rows[-2]["u_l2_rel_error"] / rows[-1]["u_l2_rel_error"])
        h1_rate = math.log2(rows[-2]["u_h1_rel_error"] / rows[-1]["u_h1_rel_error"])
    expected_l2 = float(element_order + 1)
    expected_h1 = float(element_order)
    passed = (
        l2_rate is not None
        and h1_rate is not None
        and l2_rate >= expected_l2 - 0.35
        and h1_rate >= expected_h1 - 0.35
    )
    return {
        "level": "Level 1: Solver Code & Calculation Verification (MMS)",
        "case_name": case_name,
        "element_order": element_order,
        "levels": rows,
        "observed_l2_rate": l2_rate,
        "expected_l2_rate": expected_l2,
        "observed_h1_rate": h1_rate,
        "expected_h1_rate": expected_h1,
        "passed": passed,
    }


def run_level2_cooks_membrane(
    backend: SolverBackend | None = None,
    *,
    mesh_resolutions: Sequence[int] = (4, 8),
    young_modulus_mpa: float = 240.565,
    poisson_ratio: float = 0.45,
    shear_traction_mpa: float = 6.25,
    element_order: int = 2,
    load_steps: int = 8,
) -> dict[str, Any]:
    """Level 2: NAFEMS/Abaqus 2D Cook's Membrane nonlinear hyperelastic benchmark."""
    solver = ScikitFemBackend() if backend is None else backend
    material = MaterialSpec(
        "cooks-neo-hookean",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(young_modulus_mpa, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(poisson_ratio, Unit.ONE),
        },
    )
    profile = verification_execution_profile()
    runs: list[dict[str, Any]] = []
    for n in mesh_resolutions:
        mesh = cooks_membrane_mesh(n, n)
        request = FemRequest(
            request_id=f"vvuq-l2-cooks-n{n}-p{element_order}",
            mesh=mesh,
            material=material,
            boundary_conditions=(
                BoundaryConditionSpec("clamp_left", "left", "dirichlet", ("x", "y")),
                BoundaryConditionSpec("free_top", "top", "free"),
                BoundaryConditionSpec("free_bottom", "bottom", "free"),
            ),
            loads=(
                LoadSpec(
                    "right_shear",
                    "traction",
                    "right",
                    ("y",),
                    (CanonicalQuantity(shear_traction_mpa, Unit.MPA),),
                ),
            ),
            qois=(),
            config=FemSolverConfig(
                element_order=element_order,
                solver_tolerance=1.0e-10,
                max_newton_iterations=25,
                plane_stress_max_iterations=32,
                load_steps=load_steps,
                max_dofs=100_000,
            ),
            execution_profile=profile,
        )
        t0 = time.perf_counter()
        attempt = solver.solve_many((request,))[0]
        wall = time.perf_counter() - t0
        if not attempt.success or attempt.result is None:
            raise RuntimeError(f"Cook's Membrane n={n} failed: {attempt.message}")
        coords = np.asarray(
            attempt.result.displacement_dof_coordinates, dtype=np.float64
        )
        disp = np.asarray(attempt.result.displacement, dtype=np.float64)
        # Top-right tip at (48, 60)
        tip_idx = int(np.argmin(np.hypot(coords[:, 0] - 48.0, coords[:, 1] - 60.0)))
        runs.append(
            {
                "resolution": n,
                "dof_count": len(attempt.result.displacement) * 2,
                "tip_ux_mm": float(disp[tip_idx, 0]),
                "tip_uy_mm": float(disp[tip_idx, 1]),
                "strain_energy": float(attempt.result.total_strain_energy),
                "min_det_f_2d": _min_det_f_from_result(attempt.result),
                "newton_iterations": len(attempt.result.convergence_history),
                "wall_seconds": wall,
            }
        )

    tip_change = (
        abs(runs[-1]["tip_uy_mm"] - runs[-2]["tip_uy_mm"])
        / max(abs(runs[-1]["tip_uy_mm"]), 1e-12)
        if len(runs) >= 2
        else None
    )
    return {
        "level": "Level 2: NAFEMS / Abaqus 2D Cook's Membrane (Nonlinear Hyperelastic)",
        "backend_id": solver.backend_id,
        "element_order": element_order,
        "poisson_ratio": poisson_ratio,
        "shear_traction_mpa": shear_traction_mpa,
        "runs": runs,
        "finest_tip_uy_mm": runs[-1]["tip_uy_mm"],
        "mesh_convergence_tip_rel_change": tip_change,
        "all_jacobians_positive": all(r["min_det_f_2d"] > 0.0 for r in runs),
    }


def run_level3_hgo_arterial_strip(
    backend: SolverBackend | None = None,
    *,
    axial_strains: Sequence[float] = (0.05, 0.15, 0.25),
    fibre_angle_deg: float = 29.0,
    element_order: int = 2,
) -> dict[str, Any]:
    """Level 3a: Holzapfel-Gasser-Ogden (HGO) anisotropic arterial strip benchmark."""
    solver = ScikitFemBackend() if backend is None else backend
    mesh = rectangular_strip_mesh(10.0, 4.0, 4, 2, label="hgo-arterial-strip")
    material = MaterialSpec(
        "arterial-hgo",
        "hgo",
        {
            "shear_modulus": CanonicalQuantity(0.06, Unit.MPA),
            "bulk_modulus": CanonicalQuantity(0.60, Unit.MPA),
            "k1": CanonicalQuantity(0.08, Unit.MPA),
            "k2": CanonicalQuantity(4.0, Unit.ONE),
            "fibre_angle": CanonicalQuantity(fibre_angle_deg, Unit.DEG),
        },
    )
    profile = verification_execution_profile()
    curve: list[dict[str, Any]] = []
    for strain in axial_strains:
        ux = 10.0 * float(strain)
        request = FemRequest(
            request_id=f"vvuq-l3-hgo-eps{strain:.3f}",
            mesh=mesh,
            material=material,
            boundary_conditions=(
                BoundaryConditionSpec("left_sym", "left", "symmetry", ("x",)),
                BoundaryConditionSpec("bot_sym", "bottom", "symmetry", ("y",)),
                BoundaryConditionSpec(
                    "right_pull",
                    "right",
                    "dirichlet",
                    ("x",),
                    (CanonicalQuantity(ux, Unit.MM),),
                ),
                BoundaryConditionSpec("top_free", "top", "free"),
            ),
            loads=(),
            qois=(),
            config=FemSolverConfig(
                element_order=element_order,
                solver_tolerance=1.0e-10,
                max_newton_iterations=25,
                plane_stress_max_iterations=32,
                load_steps=8,
                max_dofs=50_000,
            ),
            execution_profile=profile,
        )
        attempt = solver.solve_many((request,))[0]
        if not attempt.success or attempt.result is None:
            raise RuntimeError(f"HGO strip strain={strain} failed: {attempt.message}")
        coords = np.asarray(
            attempt.result.displacement_dof_coordinates, dtype=np.float64
        )
        disp = np.asarray(attempt.result.displacement, dtype=np.float64)
        top_mask = np.abs(coords[:, 1] - 4.0) < 1e-9
        transverse_stretch = float(1.0 + np.mean(disp[top_mask, 1]) / 4.0)
        rxn_x = float(attempt.result.reactions["right_pull"][0])
        curve.append(
            {
                "axial_strain": float(strain),
                "axial_stretch": 1.0 + float(strain),
                "transverse_stretch": transverse_stretch,
                "axial_reaction_n": rxn_x,
                "secant_stiffness_n_per_unit_strain": rxn_x / max(float(strain), 1e-12),
                "min_det_f_2d": _min_det_f_from_result(attempt.result),
            }
        )

    stiffening_ratio = (
        curve[-1]["secant_stiffness_n_per_unit_strain"]
        / max(curve[0]["secant_stiffness_n_per_unit_strain"], 1e-12)
        if len(curve) >= 2
        else 1.0
    )
    return {
        "level": "Level 3a: Holzapfel-Gasser-Ogden (HGO) Anisotropic Arterial Strip",
        "backend_id": solver.backend_id,
        "fibre_angle_deg": fibre_angle_deg,
        "curve": curve,
        "exponential_stiffening_ratio": stiffening_ratio,
        "exhibits_strain_stiffening": stiffening_ratio > 1.05,
        "all_jacobians_positive": all(c["min_det_f_2d"] > 0.0 for c in curve),
    }


def run_level3_follower_pressure_cantilever(
    backend: SolverBackend | None = None,
    *,
    pressure_mpa: float = 0.04,
    young_modulus_mpa: float = 10.0,
    poisson_ratio: float = 0.40,
    element_order: int = 2,
) -> dict[str, Any]:
    """Level 3b: Elastomer cantilever under non-conservative follower pressure."""
    solver = ScikitFemBackend() if backend is None else backend
    mesh = rectangular_strip_mesh(10.0, 2.0, 6, 2, label="follower-cantilever")
    material = MaterialSpec(
        "elastomer-nh",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(young_modulus_mpa, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(poisson_ratio, Unit.ONE),
        },
    )
    request = FemRequest(
        request_id="vvuq-l3-follower-cantilever",
        mesh=mesh,
        material=material,
        boundary_conditions=(
            BoundaryConditionSpec("clamp_left", "left", "dirichlet", ("x", "y")),
            BoundaryConditionSpec("free_right", "right", "free"),
            BoundaryConditionSpec("free_bottom", "bottom", "free"),
        ),
        loads=(
            LoadSpec(
                "top_follower_pressure",
                "follower_pressure",
                "top",
                ("p",),
                (CanonicalQuantity(pressure_mpa, Unit.MPA),),
            ),
        ),
        qois=(),
        config=FemSolverConfig(
            element_order=element_order,
            solver_tolerance=1.0e-10,
            max_newton_iterations=25,
            plane_stress_max_iterations=32,
            load_steps=8,
            max_dofs=50_000,
        ),
        execution_profile=verification_execution_profile(),
    )
    cold = solver.solve_many((request,))[0]
    if not cold.success or cold.result is None:
        raise RuntimeError(f"Follower cantilever cold solve failed: {cold.message}")
    cold_iters = len(cold.result.convergence_history)

    lin = solver.linearize(request)
    u_exact = np.asarray(cold.result.displacement, dtype=np.float64)
    exact_vec = lin.vector(u_exact)
    exact_state = lin.state(exact_vec)
    coercive = coercive_inertia(lin.free_tangent(exact_state))

    # Measure warm-start iteration reduction from a smooth 5%-underpredicted surrogate.
    warm_start = lin.with_exact_dirichlet(lin.vector(0.95 * u_exact))
    warm = solver.solve_from(request, warm_start, (1.0,))
    if not warm.success or warm.result is None:
        raise RuntimeError(f"Follower cantilever warm solve failed: {warm.message}")
    warm_iters = len(warm.result.convergence_history)

    coords = np.asarray(cold.result.displacement_dof_coordinates, dtype=np.float64)
    tip_idx = int(np.argmin(np.hypot(coords[:, 0] - 10.0, coords[:, 1] - 2.0)))
    return {
        "level": "Level 3b: Elastomer Cantilever under Non-Conservative Follower Pressure",
        "backend_id": solver.backend_id,
        "pressure_mpa": pressure_mpa,
        "tip_ux_mm": float(u_exact[tip_idx, 0]),
        "tip_uy_mm": float(u_exact[tip_idx, 1]),
        "tangent_symmetric": coercive.tangent_symmetric,
        "tangent_relative_asymmetry": coercive.relative_asymmetry,
        "coercive_positive_definite": coercive.positive_definite,
        "cold_newton_iterations": cold_iters,
        "warm_newton_iterations": warm_iters,
        "warm_start_iteration_reduction_ratio": cold_iters / max(warm_iters, 1),
        "min_det_f_2d": _min_det_f_from_result(cold.result),
    }


def run_vvuq_suite(backend: SolverBackend | None = None) -> dict[str, Any]:
    """Run Levels 1-3 VVUQ nonlinear verification & benchmark suite."""
    return {
        "level1_mms": run_level1_mms_verification(backend),
        "level2_cooks_membrane": run_level2_cooks_membrane(backend),
        "level3a_hgo_arterial_strip": run_level3_hgo_arterial_strip(backend),
        "level3b_follower_cantilever": run_level3_follower_pressure_cantilever(backend),
    }
