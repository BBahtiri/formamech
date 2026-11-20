"""CPU-sized end-to-end training, FEM comparison, and diagnostic gate pilots.

Runs five real, reproducible CPU pilots covering all six recommended features:
  * Pilot 1 (P1 + P4): Hierarchical vs. baseline energy on a 2D plate-with-hole
    + CAD remesh/reference convergence + DWR adjoint trust report & D-GATE stats.
  * Pilot 2 (P2): Amortized multi-geometry BoundaryEnergyOperator + GeometryOOD
    + <=10% cosine-scheduled low-rank adapter vs. zero-shot vs. from-scratch.
  * Pilot 3 (P6 / F1): Point-local history-aware finite-strain viscoelastic
    operator trained via incremental variational potential + 2D incremental FEM
    reference + GuardedStatefulStepper fallback & replay checks.
  * Pilot 4 (P3): Certificate-driven active FEM acquisition (3-arm x 3-seed
    comparison: random_doe vs. physics_indicator vs. physics_diversity) with
    TRAIN-only guardrails, FEMCostLedger, and multi-fidelity energy distillation.
  * Pilot 5 (P5): FEM-guarded differentiable CAD design optimization (plate
    area minimization under P2-FEM-verified stiffness constraint) vs. Pure-FEM.

All outputs are written strictly under ``.formamech-runs/pilots/`` (gitignored).
These are diagnostic CPU pilots, not sealed holdout qualifications.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from piginot.active import (
    AcquisitionCandidate,
    AcquisitionPlan,
    AcquisitionWeights,
    FEMCostLedger,
    FEMCostLedgerEntry,
    MultiFidelityConfig,
    multifidelity_energy_loss,
    select_acquisition_batch,
)
from piginot.cad.hierarchy import (
    GeometryConvergenceSample,
    build_mesh_hierarchy,
    build_reference_convergence_report,
    evaluate_geometry_convergence,
)
from piginot.cad.refinement import refine_fixed_mesh
from piginot.certify import (
    DiscretizationErrorRecord,
    GeometryErrorRecord,
    ReactionForceQoi,
    RegionalAverageStressQoi,
    estimate_qoi_dwr,
    evaluate_cad_trust,
)
from piginot.compiler.reference_convergence import (
    compare_fem_pair,
    compare_remeshed_cad_fem_pair,
)
from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    ExecutionProfile,
    MaterialSpec,
    Quantity,
    Unit,
)
from piginot.contracts.expressions import DerivedExpression
from piginot.contracts.problem_spec import ProblemSpec
from piginot.contracts.stateful import (
    BoundTrajectoryState,
    LoadSegment,
    StatefulVerificationPolicy,
    TimeGrid,
    evaluate_load_path,
)
from piginot.fem import FemRequest, FemSolverConfig, ScikitFemBackend
from piginot.fem.linearization import FemLinearization
from piginot.fem.models import FemSolveFailure
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh
from piginot.lifting import AffineDirichletLifting
from piginot.objectives import (
    HierarchicalEnergyConfig,
    PotentialEnergyConfig,
    TriangleDomain,
    check_quadrature_energy_convergence,
    deterministic_triangle_quadrature,
    hierarchical_plane_stress_energy,
    plane_stress_potential_energy,
    stratified_triangle_quadrature,
    subdivide_triangle_domain,
)
from piginot.operators import (
    GEOMETRY_FEATURE_NAMES,
    AdaptationPlan,
    BoundaryEnergyConfig,
    BoundaryEnergyOperator,
    BoundaryRoleMap,
    EnergyFormOperator,
    EnergyOperatorConfig,
    GeometryOODDetector,
    GeometryOODStatus,
    GeometryTokenizer,
    LowRankAdapterConfig,
    QueryFrame,
    adapt_boundary_operator_energy,
    boundary_energy_module_digest,
    compute_geometry_descriptor,
    constitutive_in_plane_stress,
)
from piginot.optimization import (
    DesignConstraint,
    DesignOptimizationStatus,
    DesignParameter,
    DesignProblem,
    FemDesignEvaluation,
    SurrogateDesignEvaluation,
    run_fem_guarded_design_optimization,
)
from piginot.stateful import (
    STEP_CONTROLLER_VERSION,
    CandidateStepOutput,
    DirichletHistoryBC,
    GeneralizedMaxwellConfig,
    GeneralizedMaxwellViscoelastic,
    GuardedStatefulStepper,
    HistoryAwareViscoelasticOperator,
    MaxwellBranch,
    ViscoelasticOperatorConfig,
    hash_state_tensors,
    load_trajectory_checkpoint,
    project_cell_state_to_nodes,
    save_trajectory_checkpoint,
    solve_incremental_fem_trajectory,
    summarize_viscous_state,
    train_viscoelastic_operator_energy,
    verify_algorithmic_tangent,
)
from piginot.stateful.fem_stepper import (
    _build_mesh_and_basis,
    _constrained_dofs,
    _solve_single_increment,
)
from piginot.verification.status import ClaimLevel

OUTPUT_DIR = Path(".formamech-runs/pilots")


# ---------------------------------------------------------------------------
# Shared geometry, FEM, and metric helpers
# ---------------------------------------------------------------------------


def make_quarter_hole_mesh(
    length: float,
    height: float,
    radius: float,
    *,
    n_half_arc: int = 4,
    n_radial: int = 4,
    label: str = "quarter-hole",
) -> CanonicalMesh:
    """Structured graded triangulation of [0, L] x [0, H] minus quarter circle R."""
    n_theta = 2 * n_half_arc
    outer: list[tuple[float, float]] = []
    angles: list[float] = []
    for k in range(n_theta + 1):
        if k <= n_half_arc:
            ox = length
            oy = height * (k / n_half_arc)
        else:
            ox = length * (n_theta - k) / n_half_arc
            oy = height
        outer.append((ox, oy))
        angles.append(math.atan2(oy, ox))

    nodes: list[tuple[float, float]] = []
    idx_grid: list[list[int]] = []
    for i in range(n_radial + 1):
        s = (i / n_radial) ** 1.15
        row: list[int] = []
        for k in range(n_theta + 1):
            ix = radius * math.cos(angles[k])
            iy = radius * math.sin(angles[k])
            ox, oy = outer[k]
            row.append(len(nodes))
            nodes.append(((1.0 - s) * ix + s * ox, (1.0 - s) * iy + s * oy))
        idx_grid.append(row)

    cells: list[tuple[int, int, int]] = []
    for i in range(n_radial):
        for k in range(n_theta):
            a = idx_grid[i][k]
            b = idx_grid[i + 1][k]
            c = idx_grid[i + 1][k + 1]
            d = idx_grid[i][k + 1]
            cells.append((a, b, c))
            cells.append((a, c, d))

    facets: list[tuple[int, int]] = []
    tags: dict[str, list[int]] = {
        "hole": [],
        "right": [],
        "top": [],
        "bottom": [],
        "left": [],
    }
    for k in range(n_theta):
        tags["hole"].append(len(facets))
        facets.append((idx_grid[0][k + 1], idx_grid[0][k]))
    for k in range(n_half_arc):
        tags["right"].append(len(facets))
        facets.append((idx_grid[n_radial][k], idx_grid[n_radial][k + 1]))
    for k in range(n_half_arc, n_theta):
        tags["top"].append(len(facets))
        facets.append((idx_grid[n_radial][k], idx_grid[n_radial][k + 1]))
    for i in range(n_radial):
        tags["bottom"].append(len(facets))
        facets.append((idx_grid[i][0], idx_grid[i + 1][0]))
    for i in range(n_radial):
        tags["left"].append(len(facets))
        facets.append((idx_grid[i + 1][n_theta], idx_grid[i][n_theta]))

    return canonicalize_mesh(
        MeshPayload(
            tuple(nodes),
            tuple(cells),
            CellType.TRIANGLE,
            tuple(facets),
            {k: tuple(v) for k, v in tags.items()},
            (1,) * len(cells),
            Unit.MM,
            {"label": label, "L": str(length), "H": str(height), "R": str(radius)},
        )
    )


def hole_roles() -> BoundaryRoleMap:
    return BoundaryRoleMap(
        {
            "left": "symmetry",
            "bottom": "symmetry",
            "right": "dirichlet_x",
            "top": "traction_free",
            "hole": "traction_free",
        }
    )


def neo_hookean_spec(e_mpa: float = 100.0, nu: float = 0.3) -> MaterialSpec:
    return MaterialSpec(
        name="nh-pilot",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(e_mpa, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(nu, Unit.ONE).to_canonical(),
        },
    )


def quarter_hole_fem_request(
    mesh: CanonicalMesh,
    length: float,
    strain: float,
    *,
    request_id: str,
    element_order: int = 2,
    load_steps: int = 8,
    material: MaterialSpec | None = None,
) -> FemRequest:
    bcs = (
        BoundaryConditionSpec(
            name="left_sym",
            boundary_tag="left",
            kind="symmetry",
            components=("x",),
            values=(Quantity(0.0, Unit.MM).to_canonical(),),
        ),
        BoundaryConditionSpec(
            name="bottom_sym",
            boundary_tag="bottom",
            kind="symmetry",
            components=("y",),
            values=(Quantity(0.0, Unit.MM).to_canonical(),),
        ),
        BoundaryConditionSpec(
            name="right_disp",
            boundary_tag="right",
            kind="dirichlet",
            components=("x",),
            values=(Quantity(strain * length, Unit.MM).to_canonical(),),
        ),
    )
    profile = ExecutionProfile(
        python_version="3.13.11",
        library_versions={
            "numpy": str(np.__version__),
            "torch": str(torch.__version__),
            "scikit-fem": "installed",
            "scipy": "installed",
        },
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )
    return FemRequest(
        request_id=request_id,
        mesh=mesh,
        material=neo_hookean_spec() if material is None else material,
        boundary_conditions=bcs,
        loads=(),
        qois=(),
        config=FemSolverConfig(element_order=element_order, load_steps=load_steps),
        execution_profile=profile,
    )


def quarter_hole_lifting(length: float, height: float, strain: float):
    def lift(query_xy: torch.Tensor) -> AffineDirichletLifting:
        x = query_xy[..., 0]
        y = query_xy[..., 1]
        offset = torch.stack((strain * x, torch.zeros_like(y)), dim=-1)
        scale_x = (x / length) * (1.0 - x / length) * 4.0
        scale_y = y / height
        scale = torch.stack((scale_x, scale_y), dim=-1)
        return AffineDirichletLifting(offset=offset, scale=scale)

    return lift


def make_problem_spec_for_peaks(mesh: CanonicalMesh) -> ProblemSpec:
    """Valid v1 ProblemSpec for compare_fem_pair / compare_remeshed_cad_fem_pair."""
    doc = {
        "schema": "piginot.problem-spec-v1",
        "name": "quarter-hole-pilot",
        "physics": "2d-plane-stress-finite-strain",
        "parameters": [
            {
                "name": "L_plate",
                "role": "geometry",
                "lo": 4,
                "hi": 4,
                "unit": "mm",
                "distribution": "fixed",
            },
            {
                "name": "H_plate",
                "role": "geometry",
                "lo": 3,
                "hi": 3,
                "unit": "mm",
                "distribution": "fixed",
            },
            {
                "name": "strain",
                "role": "load",
                "lo": 0.05,
                "hi": 0.2,
                "unit": "1",
                "distribution": "uniform",
            },
            {
                "name": "elastic_modulus",
                "role": "material",
                "lo": 100,
                "hi": 100,
                "unit": "MPa",
                "distribution": "fixed",
            },
            {
                "name": "poisson_ratio",
                "role": "material",
                "lo": 0.3,
                "hi": 0.3,
                "unit": "1",
                "distribution": "fixed",
            },
        ],
        "geometry": {
            "reference_mesh_id": mesh.mesh_id,
            "morph_map_sha256": "0" * 64,
            "regions": ["bottom", "hole", "left", "right", "top"],
            "frame": {"origin": ["0", "0"], "extent": ["L_plate", "H_plate"]},
        },
        "material": {
            "model": "compressible_neo_hookean",
            "parameters": {
                "elastic_modulus": "elastic_modulus",
                "poisson_ratio": "poisson_ratio",
            },
        },
        "boundary_conditions": [
            {
                "name": "left_sym",
                "region": "left",
                "kind": "symmetry",
                "components": ["x"],
            },
            {
                "name": "bottom_sym",
                "region": "bottom",
                "kind": "symmetry",
                "components": ["y"],
            },
            {
                "name": "right_disp",
                "region": "right",
                "kind": "dirichlet",
                "components": ["x"],
                "values": [{"nominal_strain": "strain"}],
            },
            {"name": "top_free", "region": "top", "kind": "traction_free"},
            {"name": "hole_free", "region": "hole", "kind": "traction_free"},
        ],
        "lifting": {"kind": "quarter", "displacement_scale": "L_plate * strain"},
        "qois": [
            {
                "name": "hole_peak_p",
                "field": "P_served",
                "reduction": "max-frobenius",
                "region": "hole",
                "exclude_regions": [],
            }
        ],
        "verification": {
            "quantile": 0.9,
            "tolerances": {
                "u_relative_l2": 0.05,
                "p_relative_l2": 0.1,
                "absolute_peak_error": 0.1,
            },
        },
        "split": {"pilot": 2, "train": 4, "development": 4, "holdout": 4, "reserve": 2},
        "fem": {
            "material_name": "nh-pilot",
            "element_order": 2,
            "load_steps": 8,
            "max_dofs": 100000,
            "max_newton_iterations": 20,
            "plane_stress_max_iterations": 32,
            "solver_tolerance": 1e-10,
        },
    }
    return ProblemSpec.from_document(doc)


def eval_field_errors_against_fem(
    predict_u_and_grad_fn,
    fem_request: FemRequest,
    fem_result,
    material: MaterialSpec,
) -> dict[str, float]:
    """Compute exact quadrature L2 displacement & first-Piola stress errors vs. FEM."""
    linearization = FemLinearization(fem_request)
    qpts = np.asarray(fem_result.quadrature_points, dtype=np.float64).reshape(-1, 2)
    qwts = np.asarray(fem_result.quadrature_weights, dtype=np.float64).reshape(-1)
    u_fem_nodal = np.asarray(fem_result.displacement, dtype=np.float64)
    u_fem_q = np.stack(
        [linearization.scalar_basis.interpolate(u_fem_nodal[:, ax]) for ax in (0, 1)],
        axis=-1,
    ).reshape(-1, 2)
    p_fem_q = np.asarray(fem_result.first_piola_stress, dtype=np.float64)[
        ..., :2, :2
    ].reshape(-1, 2, 2)

    q_tensor = torch.tensor(qpts, dtype=torch.float64)
    u_pred_q_t, grad_pred_q_t = predict_u_and_grad_fn(q_tensor)
    p_pred_q_t = constitutive_in_plane_stress(grad_pred_q_t, material)

    u_pred_q = u_pred_q_t.detach().cpu().numpy()
    p_pred_q = p_pred_q_t.detach().cpu().numpy()

    u_diff = np.sum((u_pred_q - u_fem_q) ** 2, axis=-1)
    u_ref = np.sum(u_fem_q**2, axis=-1)
    u_rel_l2 = math.sqrt(float(np.sum(u_diff * qwts)) / float(np.sum(u_ref * qwts)))

    p_diff = np.sum((p_pred_q - p_fem_q) ** 2, axis=(-2, -1))
    p_ref = np.sum(p_fem_q**2, axis=(-2, -1))
    p_rel_l2 = math.sqrt(float(np.sum(p_diff * qwts)) / float(np.sum(p_ref * qwts)))

    # Evaluate nodal prediction at FEM DOFs for exact discrete QoI comparison.
    dof_xy = torch.tensor(linearization.dof_coordinates, dtype=torch.float64)
    u_pred_nodal_t, _ = predict_u_and_grad_fn(dof_xy)
    u_pred_nodal = u_pred_nodal_t.detach().cpu().numpy()

    fem_state = linearization.state(linearization.vector(u_fem_nodal))
    rxn_qoi = ReactionForceQoi("right_reaction", "right_disp")
    avg_qoi = RegionalAverageStressQoi("avg_p11", component=(0, 0))
    rxn_fem = rxn_qoi.evaluate(linearization, fem_state)
    avg_fem = avg_qoi.evaluate(linearization, fem_state)

    try:
        pred_state = linearization.state(
            linearization.with_exact_dirichlet(linearization.vector(u_pred_nodal))
        )
        rxn_pred = rxn_qoi.evaluate(linearization, pred_state)
        avg_pred = avg_qoi.evaluate(linearization, pred_state)
        admissible = True
        rxn_rel_err = abs(rxn_pred - rxn_fem) / abs(rxn_fem)
        avg_rel_err = abs(avg_pred - avg_fem) / abs(avg_fem)
    except FemSolveFailure:
        admissible = False
        rxn_pred = float("nan")
        avg_pred = float("nan")
        rxn_rel_err = float("nan")
        avg_rel_err = float("nan")

    return {
        "admissible": admissible,
        "u_rel_l2": u_rel_l2,
        "p_rel_l2": p_rel_l2,
        "rxn_fem_N": rxn_fem,
        "rxn_pred_N": rxn_pred,
        "rxn_rel_err": rxn_rel_err,
        "avg_p11_fem_MPa": avg_fem,
        "avg_p11_pred_MPa": avg_pred,
        "avg_p11_rel_err": avg_rel_err,
        "u_pred_nodal": u_pred_nodal,  # stripped before JSON dump
    }


# ---------------------------------------------------------------------------
# PILOT 1: P1 (Hierarchical Energy vs Baseline) + P4 (DWR Trust Report)
# ---------------------------------------------------------------------------


def run_pilot_1_p1_p4() -> dict[str, Any]:
    print("\n=== PILOT 1: P1 (Hierarchical Energy + Convergence) + P4 (DWR Trust) ===")
    t_start = time.perf_counter()
    L, H, R = 4.0, 3.0, 1.0
    backend = ScikitFemBackend()
    mat = neo_hookean_spec()
    source_hash = "1" * 64

    # 1A. Distinct curved-arc CAD remeshes -> GeometryConvergenceReport + common-interior FEM.
    mesh_arc6 = make_quarter_hole_mesh(L, H, R, n_half_arc=3, n_radial=3, label="arc6")
    mesh_arc12 = make_quarter_hole_mesh(
        L, H, R, n_half_arc=6, n_radial=6, label="arc12"
    )
    req_arc6 = quarter_hole_fem_request(mesh_arc6, L, 0.15, request_id="arc6")
    req_arc12 = quarter_hole_fem_request(mesh_arc12, L, 0.15, request_id="arc12")
    res_arc6 = backend.solve_many((req_arc6,))[0]
    res_arc12 = backend.solve_many((req_arc12,))[0]
    assert res_arc6.success and res_arc6.result is not None
    assert res_arc12.success and res_arc12.result is not None
    spec_arc = make_problem_spec_for_peaks(mesh_arc6)
    remesh_fem = compare_remeshed_cad_fem_pair(
        spec_arc, req_arc6, res_arc6.result, req_arc12, res_arc12.result
    )
    geom_samples = [
        GeometryConvergenceSample.from_mesh(
            "n_arc_6",
            mesh_arc6,
            cad_source_hash=source_hash,
            declared_curve_tolerance_mm=0.05,
            achieved_max_chord_error_mm=R * (1.0 - math.cos(math.pi / 24.0)),
        ),
        GeometryConvergenceSample.from_mesh(
            "n_arc_12",
            mesh_arc12,
            cad_source_hash=source_hash,
            declared_curve_tolerance_mm=0.05,
            achieved_max_chord_error_mm=R * (1.0 - math.cos(math.pi / 48.0)),
        ),
    ]
    geom_conv = evaluate_geometry_convergence(
        geom_samples,
        max_relative_area_change=0.01,
        max_relative_perimeter_change=0.01,
        max_chord_error_mm=0.01,
        fem_remesh_comparisons=(remesh_fem,),
    )

    # 1B. Nested fixed-polygon M0 -> M1 -> M2 ReferenceConvergenceReport.
    base_mesh = make_quarter_hole_mesh(L, H, R, n_half_arc=4, n_radial=4, label="m0")
    hierarchy = build_mesh_hierarchy(
        base_mesh,
        cad_source_hash=source_hash,
        geometry_tolerance_mm=0.02,
        quadrature_rule="dunavant-deg2-v1",
    )
    spec_m0 = make_problem_spec_for_peaks(hierarchy.m0_coarse)
    req_m0 = quarter_hole_fem_request(hierarchy.m0_coarse, L, 0.15, request_id="m0")
    req_m1 = quarter_hole_fem_request(hierarchy.m1_training, L, 0.15, request_id="m1")
    req_m2 = quarter_hole_fem_request(hierarchy.m2_reference, L, 0.15, request_id="m2")
    sol_m0, sol_m1, sol_m2 = backend.solve_many((req_m0, req_m1, req_m2))
    assert sol_m0.result and sol_m1.result and sol_m2.result
    p01 = compare_fem_pair(
        spec_m0, req_m0, sol_m0.result, req_m1, sol_m1.result, tolerances=None
    )
    p12 = compare_fem_pair(
        spec_m0, req_m1, sol_m1.result, req_m2, sol_m2.result, tolerances=None
    )
    ref_conv = build_reference_convergence_report(
        hierarchy,
        p01,
        p12,
        reference_floor={
            "u_relative_l2": 0.01,
            "p_relative_l2": 0.02,
            "absolute_peak_error": 0.05,
        },
    )
    print(
        f"  Geometry conv: {geom_conv.status.value} | "
        f"Ref M1->M2 conv: {ref_conv.status.value} "
        f"(u_l2={p12['quantities']['u_relative_l2']:.4e}, "
        f"p_l2={p12['quantities']['p_relative_l2']:.4e}, "
        f"peak={p12['quantities']['absolute_peak_error']:.4e})"
    )

    # 1C. Train Arm A (Baseline stratified energy) vs Arm B (Hierarchical M0->M1 energy).
    domain_m0 = TriangleDomain(
        vertices=torch.tensor(hierarchy.m0_coarse.nodes, dtype=torch.float64),
        cells=torch.tensor(hierarchy.m0_coarse.cells, dtype=torch.int64),
    )
    domain_m1 = subdivide_triangle_domain(domain_m0)
    frame = QueryFrame(
        origin=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        extent=torch.tensor([[L, H]], dtype=torch.float64),
    )
    train_strains = (0.05, 0.10, 0.15, 0.20)
    test_strains = (0.06, 0.08, 0.11, 0.13, 0.14, 0.16, 0.18, 0.19)
    mean_s, std_s = 0.125, 0.0559016994

    def norm_param(s: float) -> torch.Tensor:
        return torch.tensor([[(s - mean_s) / std_s]], dtype=torch.float64)

    op_cfg = EnergyOperatorConfig(
        parameter_names=("strain",),
        hidden_dim=24,
        latent_token_count=8,
        parameter_hidden_dim=32,
        num_attention_heads=4,
        decoder_layers=2,
        positional_frequencies=4,
    )

    def train_arm(hierarchical: bool) -> tuple[EnergyFormOperator, dict[str, Any]]:
        torch.manual_seed(7)
        model = EnergyFormOperator(op_cfg).to(torch.float64)
        base_cfg = PotentialEnergyConfig(
            points_per_cell=3, barrier_weight=1e3, barrier_threshold=0.20
        )
        hier_cfg = HierarchicalEnergyConfig(
            base=base_cfg,
            cross_mesh_weight=0.10,
            local_physics_weight=0.05,
            quadrature_degree=2,
            use_deterministic_quadrature=False,
        )
        t0 = time.perf_counter()
        n_steps = 220
        opt = torch.optim.Adam(model.parameters(), lr=3e-3, foreach=False, fused=False)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=3e-4
        )
        quad_gen = torch.Generator().manual_seed(101)
        loss_start, loss_end = 0.0, 0.0
        min_det_seen = float("inf")
        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), dtype=torch.float64)
            for eps in train_strains:
                p_in = norm_param(eps)
                lift = quarter_hole_lifting(L, H, eps)
                if hierarchical:

                    def _grad(pts: torch.Tensor, p=p_in, lf=lift) -> torch.Tensor:
                        _, g = model.displacement_and_gradient(
                            query_xy=pts.unsqueeze(0),
                            parameters=p,
                            frame=frame,
                            lifting=lf,
                            create_graph=True,
                        )
                        return g[0]

                    terms = hierarchical_plane_stress_energy(
                        _grad,
                        domain_m0,
                        mat,
                        config=hier_cfg,
                        fine_domain=domain_m1,
                        generator=quad_gen,
                    )
                    total_loss = total_loss + terms.total / len(train_strains)
                    min_det_seen = min(min_det_seen, terms.primary.minimum_determinant)
                else:
                    pts, wts = stratified_triangle_quadrature(
                        domain_m0, base_cfg.points_per_cell, generator=quad_gen
                    )
                    _, g = model.displacement_and_gradient(
                        query_xy=pts.unsqueeze(0),
                        parameters=p_in,
                        frame=frame,
                        lifting=lift,
                        create_graph=True,
                    )
                    terms_b = plane_stress_potential_energy(
                        g[0], wts, mat, config=base_cfg, reference_area=domain_m0.area
                    )
                    total_loss = total_loss + terms_b.total / len(train_strains)
                    min_det_seen = min(min_det_seen, terms_b.minimum_determinant)
            val = float(total_loss.detach().item())
            if step == 0:
                loss_start = val
            loss_end = val
            total_loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()

        return model, {
            "wall_seconds": round(time.perf_counter() - t0, 2),
            "initial_loss": loss_start,
            "final_loss": loss_end,
            "adam_steps": n_steps,
            "min_det_f_train": min_det_seen,
        }

    arm_a, arm_a_train = train_arm(hierarchical=False)
    arm_b, arm_b_train = train_arm(hierarchical=True)

    # Check quadrature energy convergence on Arm B at strain=0.15.
    q_conv = check_quadrature_energy_convergence(
        lambda pts: arm_b.displacement_and_gradient(
            query_xy=pts.unsqueeze(0),
            parameters=norm_param(0.15),
            frame=frame,
            lifting=quarter_hole_lifting(L, H, 0.15),
            create_graph=False,
        )[1][0],
        domain_m0,
        mat,
        relative_tolerance=5e-3,
    )

    # 1D. Evaluate both arms against P2 FEM on 8 unseen test strains + P4 DWR trust report.
    test_reqs = tuple(
        quarter_hole_fem_request(
            hierarchy.m1_training, L, eps, request_id=f"test-{i}", element_order=2
        )
        for i, eps in enumerate(test_strains)
    )
    test_sols = backend.solve_many(test_reqs)

    arm_a_cases: list[dict[str, float]] = []
    arm_b_cases: list[dict[str, float]] = []
    dwr_rxn_est: list[float] = []
    dwr_rxn_true: list[float] = []
    dwr_rxn_eff: list[float] = []
    dwr_avg_est: list[float] = []
    dwr_avg_true: list[float] = []
    dwr_avg_eff: list[float] = []
    sample_trust_doc: dict[str, Any] = {}

    rxn_qoi = ReactionForceQoi("right_reaction", "right_disp")
    avg_qoi = RegionalAverageStressQoi("avg_p11", component=(0, 0))
    geom_rec = GeometryErrorRecord(
        cad_source_hash=source_hash,
        declared_curve_tolerance_mm=0.05,
        achieved_max_chord_error_mm=geom_conv.final_achieved_chord_error_mm,
        remesh_status=geom_conv.status.value,
        excluded_weight_fraction=remesh_fem["excluded_weight_fraction"],
    )
    disc_rec = DiscretizationErrorRecord(
        reference_status=ref_conv.status.value,
        element_order=2,
        dof_count=len(hierarchy.m1_training.nodes) * 2,
        m1_to_m2_u_rel_l2=p12["quantities"]["u_relative_l2"],
        m1_to_m2_p_rel_l2=p12["quantities"]["p_relative_l2"],
    )

    for idx, (eps, req, attempt) in enumerate(
        zip(test_strains, test_reqs, test_sols, strict=True)
    ):
        assert attempt.success and attempt.result is not None
        p_in = norm_param(eps)
        lift = quarter_hole_lifting(L, H, eps)

        def _pred(model: EnergyFormOperator, p=p_in, lf=lift):
            def fn(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                u, g = model.displacement_and_gradient(
                    query_xy=pts.unsqueeze(0),
                    parameters=p,
                    frame=frame,
                    lifting=lf,
                    create_graph=False,
                )
                return u[0], g[0]

            return fn

        res_a = eval_field_errors_against_fem(_pred(arm_a), req, attempt.result, mat)
        res_b = eval_field_errors_against_fem(_pred(arm_b), req, attempt.result, mat)
        u_nodal_b = res_b.pop("u_pred_nodal")
        res_a.pop("u_pred_nodal")
        arm_a_cases.append({"strain": eps, **res_a})
        arm_b_cases.append({"strain": eps, **res_b})

        # P4 DWR adjoint trust evaluation on Arm B.
        lin = FemLinearization(req)
        dwr_r = estimate_qoi_dwr(lin, u_nodal_b, rxn_qoi)
        dwr_s = estimate_qoi_dwr(lin, u_nodal_b, avg_qoi)
        true_r_err = res_b["rxn_fem_N"] - res_b["rxn_pred_N"]
        true_s_err = res_b["avg_p11_fem_MPa"] - res_b["avg_p11_pred_MPa"]
        if dwr_r.dwr_estimate is not None and abs(true_r_err) > 1e-9:
            dwr_rxn_est.append(abs(dwr_r.dwr_estimate))
            dwr_rxn_true.append(abs(true_r_err))
            dwr_rxn_eff.append(dwr_r.dwr_estimate / true_r_err)
        if dwr_s.dwr_estimate is not None and abs(true_s_err) > 1e-9:
            dwr_avg_est.append(abs(dwr_s.dwr_estimate))
            dwr_avg_true.append(abs(true_s_err))
            dwr_avg_eff.append(dwr_s.dwr_estimate / true_s_err)

        if idx == 3:
            trust = evaluate_cad_trust(
                req,
                u_nodal_b,
                (rxn_qoi, avg_qoi),
                problem_hash=spec_m0.problem_hash,
                geometry_error=geom_rec,
                discretization_error=disc_rec,
                claim_level=ClaimLevel.NOT_VERIFIED,
            )
            sample_trust_doc = trust.document()

    def _summarize(cases: list[dict[str, float]]) -> dict[str, float]:
        return {
            "p90_u_rel_l2": float(np.percentile([c["u_rel_l2"] for c in cases], 90)),
            "p90_p_rel_l2": float(np.percentile([c["p_rel_l2"] for c in cases], 90)),
            "p90_rxn_rel_err": float(
                np.percentile([c["rxn_rel_err"] for c in cases], 90)
            ),
            "p90_avg_p11_rel_err": float(
                np.percentile([c["avg_p11_rel_err"] for c in cases], 90)
            ),
            "mean_u_rel_l2": float(np.mean([c["u_rel_l2"] for c in cases])),
            "mean_p_rel_l2": float(np.mean([c["p_rel_l2"] for c in cases])),
        }

    sum_a = _summarize(arm_a_cases)
    sum_b = _summarize(arm_b_cases)
    rxn_spearman = float(spearmanr(dwr_rxn_est, dwr_rxn_true).statistic)
    avg_spearman = float(spearmanr(dwr_avg_est, dwr_avg_true).statistic)
    rxn_eff_in_band = sum(0.5 <= e <= 2.0 for e in dwr_rxn_eff) / max(
        len(dwr_rxn_eff), 1
    )
    avg_eff_in_band = sum(0.5 <= e <= 2.0 for e in dwr_avg_eff) / max(
        len(dwr_avg_eff), 1
    )

    print(
        f"  Arm A (Baseline):     p90 u_L2={sum_a['p90_u_rel_l2'] * 100:.2f}%, "
        f"p90 P_L2={sum_a['p90_p_rel_l2'] * 100:.2f}%, "
        f"p90 rxn={sum_a['p90_rxn_rel_err'] * 100:.2f}% ({arm_a_train['wall_seconds']}s)"
    )
    print(
        f"  Arm B (Hierarchical): p90 u_L2={sum_b['p90_u_rel_l2'] * 100:.2f}%, "
        f"p90 P_L2={sum_b['p90_p_rel_l2'] * 100:.2f}%, "
        f"p90 rxn={sum_b['p90_rxn_rel_err'] * 100:.2f}% ({arm_b_train['wall_seconds']}s)"
    )
    print(
        f"  P4 DWR Reaction:  Spearman={rxn_spearman:.3f}, "
        f"eff in [0.5,2.0]={rxn_eff_in_band * 100:.0f}% (median eff={float(np.median(dwr_rxn_eff)):.3f})"
    )
    print(
        f"  P4 DWR Avg P11:   Spearman={avg_spearman:.3f}, "
        f"eff in [0.5,2.0]={avg_eff_in_band * 100:.0f}% (median eff={float(np.median(dwr_avg_eff)):.3f})"
    )

    return {
        "status": "DIAGNOSTIC_PILOT",
        "claim_level": "NOT_VERIFIED",
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "geometry_convergence": geom_conv.document(),
        "reference_convergence": ref_conv.document(),
        "quadrature_convergence": {
            "converged": q_conv.converged,
            "degree_2_to_5_relative_diff": q_conv.degree_2_to_5_relative_diff,
            "mesh_subdivision_relative_diff": q_conv.mesh_subdivision_relative_diff,
        },
        "arm_a_baseline": {
            "training": arm_a_train,
            "summary": sum_a,
            "cases": arm_a_cases,
        },
        "arm_b_hierarchical": {
            "training": arm_b_train,
            "summary": sum_b,
            "cases": arm_b_cases,
        },
        "p4_dwr_diagnostics": {
            "reaction_force": {
                "spearman": rxn_spearman,
                "effectivity_in_0p5_2p0_fraction": rxn_eff_in_band,
                "effectivities": dwr_rxn_eff,
            },
            "regional_average_p11": {
                "spearman": avg_spearman,
                "effectivity_in_0p5_2p0_fraction": avg_eff_in_band,
                "effectivities": dwr_avg_eff,
            },
            "sample_cad_trust_report": sample_trust_doc,
        },
    }


# ---------------------------------------------------------------------------
# PILOT 2: P2 (Amortized Multi-Geometry BoundaryEnergyOperator + <=10% LoRA)
# ---------------------------------------------------------------------------


def run_pilot_2_p2() -> dict[str, Any]:
    print(
        "\n=== PILOT 2: P2 (Multi-Geometry BoundaryEnergyOperator + 10% Low-Rank Adapter) ==="
    )
    t_start = time.perf_counter()
    backend = ScikitFemBackend()
    mat = neo_hookean_spec()
    roles = hole_roles()
    tok = GeometryTokenizer(subdivisions_per_facet=1)

    # 6 diverse TRAIN geometries bracketing the unseen test geometry + 1 OOD geometry.
    train_specs = [
        (3.5, 2.5, 0.70, "train-g1"),
        (4.0, 3.0, 0.75, "train-g2"),
        (4.0, 3.0, 1.15, "train-g3"),
        (4.5, 2.6, 0.85, "train-g4"),
        (5.0, 3.0, 0.95, "train-g5"),
        (5.0, 3.4, 1.15, "train-g6"),
    ]
    unseen_spec = (4.5, 3.0, 1.0, "unseen-test")
    ood_spec = (12.0, 2.0, 0.4, "ood-slender")

    train_meshes = [
        make_quarter_hole_mesh(l_, h_, r_, n_half_arc=4, n_radial=4, label=lbl)
        for l_, h_, r_, lbl in train_specs
    ]
    unseen_mesh = make_quarter_hole_mesh(
        unseen_spec[0],
        unseen_spec[1],
        unseen_spec[2],
        n_half_arc=4,
        n_radial=4,
        label=unseen_spec[3],
    )
    ood_mesh = make_quarter_hole_mesh(
        ood_spec[0],
        ood_spec[1],
        ood_spec[2],
        n_half_arc=4,
        n_radial=4,
        label=ood_spec[3],
    )

    # Fit GeometryOODDetector on TRAIN geometries only.
    train_descs = [
        compute_geometry_descriptor(m, roles, tokenizer=tok) for m in train_meshes
    ]
    ood_detector = GeometryOODDetector.fit(train_descs, descriptor_threshold=3.5)
    ood_unseen = ood_detector.score(unseen_mesh, roles, tokenizer=tok)
    ood_slender = ood_detector.score(ood_mesh, roles, tokenizer=tok)
    print(
        f"  OOD check -> unseen: {ood_unseen.status.value} (d={ood_unseen.descriptor_distance:.2f}) | "
        f"slender: {ood_slender.status.value} (d={ood_slender.descriptor_distance:.2f})"
    )

    # Build shared BoundaryEnergyOperator.
    b_cfg = BoundaryEnergyConfig(
        energy=EnergyOperatorConfig(
            parameter_names=("strain",),
            hidden_dim=24,
            latent_token_count=8,
            parameter_hidden_dim=32,
            num_attention_heads=4,
            decoder_layers=2,
            positional_frequencies=4,
        ),
        boundary_feature_names=GEOMETRY_FEATURE_NAMES,
    )
    torch.manual_seed(11)
    shared_op = BoundaryEnergyOperator(b_cfg).to(torch.float64)
    base_energy_cfg = PotentialEnergyConfig(
        points_per_cell=2, barrier_weight=1e3, barrier_threshold=0.20
    )

    train_batches = [
        tok.tokenize_batch([(m, roles)], dtype=torch.float64) for m in train_meshes
    ]
    train_domains = [
        TriangleDomain(
            torch.tensor(m.nodes, dtype=torch.float64),
            torch.tensor(m.cells, dtype=torch.int64),
        )
        for m in train_meshes
    ]

    # Pretrain shared operator across 6 geometries x 2 strains (0.08, 0.16), label-free,
    # redrawing fresh unbiased stratified quadrature points every step.
    t_pre = time.perf_counter()
    pre_steps = 180
    pre_gen = torch.Generator().manual_seed(303)
    opt = torch.optim.Adam(shared_op.parameters(), lr=3e-3, foreach=False, fused=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=pre_steps, eta_min=3e-4
    )
    for _step in range(pre_steps):
        opt.zero_grad(set_to_none=True)
        loss_acc = torch.zeros((), dtype=torch.float64)
        for (l_, h_, _r, _lbl), batch, dom in zip(
            train_specs, train_batches, train_domains, strict=True
        ):
            enc = shared_op.encode_boundary(
                boundary_xy=batch.boundary_xy,
                boundary_normals=batch.boundary_normals,
                boundary_features=batch.boundary_features,
                weights=batch.weights,
                mask=batch.mask,
                frame=batch.frame,
            )
            pts, wts = stratified_triangle_quadrature(dom, 3, generator=pre_gen)
            for eps in (0.08, 0.16):
                p_in = torch.tensor([[(eps - 0.12) / 0.04]], dtype=torch.float64)
                lift = quarter_hole_lifting(l_, h_, eps)
                _, g = shared_op.displacement_and_gradient(
                    encoding=enc,
                    query_xy=pts.unsqueeze(0),
                    parameters=p_in,
                    lifting=lift,
                    create_graph=True,
                )
                terms = plane_stress_potential_energy(
                    g[0], wts, mat, config=base_energy_cfg, reference_area=dom.area
                )
                loss_acc = loss_acc + terms.total / (len(train_specs) * 2)
        loss_acc.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(shared_op.parameters(), max_norm=1.0)
        opt.step()
        sched.step()
    pretrain_wall = round(time.perf_counter() - t_pre, 2)

    # Evaluate on UNSEEN geometry at strain = 0.12 against converged P2 FEM.
    eps_test = 0.12
    p_test = torch.tensor([[(eps_test - 0.12) / 0.04]], dtype=torch.float64)
    lift_unseen = quarter_hole_lifting(unseen_spec[0], unseen_spec[1], eps_test)
    unseen_req = quarter_hole_fem_request(
        refine_fixed_mesh(unseen_mesh, 2),
        unseen_spec[0],
        eps_test,
        request_id="unseen-fem",
        element_order=2,
    )
    unseen_fem = backend.solve_many((unseen_req,))[0]
    assert unseen_fem.success and unseen_fem.result is not None

    unseen_batch = tok.tokenize_batch([(unseen_mesh, roles)], dtype=torch.float64)
    unseen_dom = TriangleDomain(
        torch.tensor(unseen_mesh.nodes, dtype=torch.float64),
        torch.tensor(unseen_mesh.cells, dtype=torch.int64),
    )

    def _eval_bop(op_any) -> dict[str, float]:
        with torch.no_grad():
            enc = op_any.encode_boundary(
                boundary_xy=unseen_batch.boundary_xy,
                boundary_normals=unseen_batch.boundary_normals,
                boundary_features=unseen_batch.boundary_features,
                weights=unseen_batch.weights,
                mask=unseen_batch.mask,
                frame=unseen_batch.frame,
            )

        def fn(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            u, g = op_any.displacement_and_gradient(
                encoding=enc,
                query_xy=pts.unsqueeze(0),
                parameters=p_test,
                lifting=lift_unseen,
                create_graph=False,
            )
            return u[0], g[0]

        res = eval_field_errors_against_fem(fn, unseen_req, unseen_fem.result, mat)
        res.pop("u_pred_nodal")
        return res

    # (1) Zero-shot error.
    zero_shot = _eval_bop(shared_op)

    # (2) 10% Low-Rank Adapter (20 steps vs 200-step from-scratch budget) with cosine schedule & grad clipping.
    from_scratch_steps = 200
    adapt_steps = 20
    t_ad = time.perf_counter()
    adapter_cfg = LowRankAdapterConfig(
        rank=4,
        alpha=8.0,
        target_substrings=("query_projection", "out_proj", "mlp", "films", "output"),
        max_steps=adapt_steps,
        from_scratch_reference_steps=from_scratch_steps,
        max_step_budget_fraction=0.10,
        learning_rate=5e-3,
        use_cosine_schedule=True,
        grad_clip_norm=1.0,
        seed=5,
    )
    plan = AdaptationPlan(
        base_model_digest=boundary_energy_module_digest(shared_op),
        target_mesh_id=unseen_mesh.mesh_id,
        adapter_config=adapter_cfg,
        energy_config=PotentialEnergyConfig(
            points_per_cell=4, barrier_weight=1e3, barrier_threshold=0.20
        ),
        ood_report=ood_unseen,
    )
    adapted_op, adapt_res = adapt_boundary_operator_energy(
        shared_op,
        plan=plan,
        domain=unseen_dom,
        boundary_xy=unseen_batch.boundary_xy,
        boundary_normals=unseen_batch.boundary_normals,
        boundary_features=unseen_batch.boundary_features,
        boundary_weights=unseen_batch.weights,
        boundary_mask=unseen_batch.mask,
        frame=unseen_batch.frame,
        parameters=p_test,
        lifting=lift_unseen,
        material=mat,
    )
    adapt_wall = round(time.perf_counter() - t_ad, 2)
    assert adapted_op is not None
    adapted_err = _eval_bop(adapted_op)

    # (3) 100% From-Scratch baseline (200 steps) on the unseen geometry.
    t_fs = time.perf_counter()
    torch.manual_seed(11)
    scratch_op = BoundaryEnergyOperator(b_cfg).to(torch.float64)
    opt_fs = torch.optim.Adam(
        scratch_op.parameters(), lr=3e-3, foreach=False, fused=False
    )
    sched_fs = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_fs, T_max=from_scratch_steps, eta_min=3e-4
    )
    fs_gen = torch.Generator().manual_seed(22)
    for _ in range(from_scratch_steps):
        opt_fs.zero_grad(set_to_none=True)
        enc_fs = scratch_op.encode_boundary(
            boundary_xy=unseen_batch.boundary_xy,
            boundary_normals=unseen_batch.boundary_normals,
            boundary_features=unseen_batch.boundary_features,
            weights=unseen_batch.weights,
            mask=unseen_batch.mask,
            frame=unseen_batch.frame,
        )
        pts_s, wts_s = stratified_triangle_quadrature(unseen_dom, 3, generator=fs_gen)
        _, g_fs = scratch_op.displacement_and_gradient(
            encoding=enc_fs,
            query_xy=pts_s.unsqueeze(0),
            parameters=p_test,
            lifting=lift_unseen,
            create_graph=True,
        )
        loss_fs = plane_stress_potential_energy(
            g_fs[0],
            wts_s,
            mat,
            config=PotentialEnergyConfig(
                points_per_cell=3, barrier_weight=1e3, barrier_threshold=0.20
            ),
            reference_area=unseen_dom.area,
        ).total
        loss_fs.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(scratch_op.parameters(), max_norm=1.0)
        opt_fs.step()
        sched_fs.step()
    scratch_wall = round(time.perf_counter() - t_fs, 2)
    scratch_err = _eval_bop(scratch_op)

    print(
        f"  Zero-shot:            u_L2={zero_shot['u_rel_l2'] * 100:.2f}%, "
        f"P_L2={zero_shot['p_rel_l2'] * 100:.2f}%, rxn={zero_shot['rxn_rel_err'] * 100:.2f}%"
    )
    print(
        f"  10% LoRA Adapter:     u_L2={adapted_err['u_rel_l2'] * 100:.2f}%, "
        f"P_L2={adapted_err['p_rel_l2'] * 100:.2f}%, rxn={adapted_err['rxn_rel_err'] * 100:.2f}% "
        f"({adapt_steps} steps, {adapt_wall}s, {adapt_res.adapter_parameter_count}/{adapt_res.base_parameter_count} params)"
    )
    print(
        f"  100% From-Scratch:    u_L2={scratch_err['u_rel_l2'] * 100:.2f}%, "
        f"P_L2={scratch_err['p_rel_l2'] * 100:.2f}%, rxn={scratch_err['rxn_rel_err'] * 100:.2f}% "
        f"({from_scratch_steps} steps, {scratch_wall}s)"
    )

    return {
        "status": "DIAGNOSTIC_PILOT",
        "claim_level": "NOT_VERIFIED",
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "pretrain": {
            "train_geometries": len(train_specs),
            "steps": pre_steps,
            "wall_seconds": pretrain_wall,
        },
        "ood_reports": {
            "unseen_in_family": ood_unseen.document(),
            "slender_ood": ood_slender.document(),
        },
        "unseen_geometry_comparison": {
            "zero_shot": zero_shot,
            "low_rank_adapter_10pct": {
                **adapted_err,
                "wall_seconds": adapt_wall,
                "steps": adapt_steps,
                "adapter_params": adapt_res.adapter_parameter_count,
                "base_params": adapt_res.base_parameter_count,
                "ratio_vs_from_scratch_u_l2": adapted_err["u_rel_l2"]
                / max(scratch_err["u_rel_l2"], 1e-12),
                "ratio_vs_from_scratch_p_l2": adapted_err["p_rel_l2"]
                / max(scratch_err["p_rel_l2"], 1e-12),
                "ratio_vs_from_scratch_rxn": adapted_err["rxn_rel_err"]
                / max(scratch_err["rxn_rel_err"], 1e-12),
            },
            "from_scratch_100pct": {
                **scratch_err,
                "wall_seconds": scratch_wall,
                "steps": from_scratch_steps,
            },
        },
    }


# ---------------------------------------------------------------------------
# PILOT 3: P6 / F1 (Viscoelastic Neural Operator + 2D FEM + Guarded Stepper)
# ---------------------------------------------------------------------------


def run_pilot_3_p6_f1(ckpt_root: Path) -> dict[str, Any]:
    print(
        "\n=== PILOT 3: P6 / F1 (Viscoelastic Operator + 2D FEM + Guarded Stepper) ==="
    )
    t_start = time.perf_counter()
    L, H, R = 4.0, 3.0, 1.0
    mesh = make_quarter_hole_mesh(L, H, R, n_half_arc=3, n_radial=3, label="visco-hole")

    visco = GeneralizedMaxwellViscoelastic(
        GeneralizedMaxwellConfig(
            equilibrium_modulus_mpa=100.0,
            poisson_ratio=0.3,
            branches=(
                MaxwellBranch(stiffness_ratio=0.6, relaxation_time=0.5),
                MaxwellBranch(stiffness_ratio=0.4, relaxation_time=2.5),
            ),
        )
    )

    # 3A. F1-GATE material tangent check.
    f_probe = torch.tensor(
        [[[1.12, 0.04, 0.0], [0.02, 0.96, 0.0], [0.0, 0.0, 0.95]]], dtype=torch.float64
    )
    t_check = verify_algorithmic_tangent(
        visco,
        torch.eye(3, dtype=torch.float64).unsqueeze(0),
        f_probe,
        visco.initial_state((1,)),
        dt=0.25,
    )

    # 3B. Converged 2D incremental viscoelastic FEM reference on unseen load amp u_max = 0.44 mm.
    bcs = (
        DirichletHistoryBC("left_sym", "left", "x", load_key=None),
        DirichletHistoryBC("bottom_sym", "bottom", "y", load_key=None),
        DirichletHistoryBC("right_disp", "right", "x", load_key="ux"),
    )
    segments = (
        LoadSegment(0.0, 0.5, "ramp", {"ux": DerivedExpression.parse("u_max * s")}),
        LoadSegment(0.5, 1.5, "hold", {"ux": DerivedExpression.parse("u_max")}),
        LoadSegment(
            1.5, 2.0, "ramp", {"ux": DerivedExpression.parse("u_max * (1 - s)")}
        ),
    )
    grid_coarse = TimeGrid(0.0, 2.0, dt_initial=0.25, dt_min=0.05, dt_max=0.25)
    grid_fine = TimeGrid(0.0, 2.0, dt_initial=0.125, dt_min=0.025, dt_max=0.125)
    unseen_u_max = 0.44

    t_fem = time.perf_counter()
    fem_ref = solve_incremental_fem_trajectory(
        mesh,
        visco,
        bcs,
        grid_coarse,
        segments,
        problem_spec_hash="f" * 64,
        load_path_hash="l" * 64,
        parameters={"u_max": unseen_u_max},
        element_order=2,
    )
    fem_wall = round(time.perf_counter() - t_fem, 2)
    fem_ref_replay = solve_incremental_fem_trajectory(
        mesh,
        visco,
        bcs,
        grid_coarse,
        segments,
        problem_spec_hash="f" * 64,
        load_path_hash="l" * 64,
        parameters={"u_max": unseen_u_max},
        element_order=2,
    )
    fem_fine = solve_incremental_fem_trajectory(
        mesh,
        visco,
        bcs,
        grid_fine,
        segments,
        problem_spec_hash="f" * 64,
        load_path_hash="l" * 64,
        parameters={"u_max": unseen_u_max},
        element_order=2,
    )
    assert fem_ref.converged and fem_fine.converged
    replay_exact = [s.bound_state.state_hash for s in fem_ref.steps] == [
        s.bound_state.state_hash for s in fem_ref_replay.steps
    ]
    peak_c = max(s.reactions["right_disp"] for s in fem_ref.steps)
    peak_f = max(s.reactions["right_disp"] for s in fem_fine.steps)
    dt_halving_rel = abs(peak_c - peak_f) / abs(peak_f)

    # 3C. Train HistoryAwareViscoelasticOperator (pure incremental variational energy).
    domain = TriangleDomain(
        torch.tensor(mesh.nodes, dtype=torch.float64),
        torch.tensor(mesh.cells, dtype=torch.int64),
    )
    frame = QueryFrame(
        origin=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        extent=torch.tensor([[L, H]], dtype=torch.float64),
    )
    times = grid_coarse.fixed_times()
    train_u_max_list = (0.20, 0.36, 0.52)
    train_trajs = []
    for u_m in train_u_max_list:
        seq = []
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            ux = evaluate_load_path(segments, times[i], {"u_max": u_m})["ux"]
            seq.append((torch.tensor([ux / L], dtype=torch.float64), dt))
        train_trajs.append(seq)

    op = HistoryAwareViscoelasticOperator(
        ViscoelasticOperatorConfig(
            load_dim=1,
            n_branches=len(visco.config.branches),
            hidden_dim=64,
            num_layers=3,
            positional_frequencies=4,
            include_local_state=True,
        )
    ).to(torch.float64)

    t_tr = time.perf_counter()
    tr_summary = train_viscoelastic_operator_energy(
        op,
        material=visco,
        domain=domain,
        frame=frame,
        load_trajectories=train_trajs,
        lifting_factory=lambda load_vec: quarter_hole_lifting(
            L, H, float(load_vec[0].item())
        ),
        epochs=80,
        learning_rate=3e-3,
        points_per_cell=4,
        seed=3,
    )
    train_wall = round(time.perf_counter() - t_tr, 2)

    # 3D. Evaluate unseen trajectory (u_max=0.44) against converged P2 FEM reference.
    skmesh, basis, scalar_basis = _build_mesh_and_basis(mesh, element_order=2)
    by_name, _, free_dofs = _constrained_dofs(basis, skmesh, mesh, bcs)
    dof_xy = torch.tensor(
        np.asarray(scalar_basis.doflocs, dtype=np.float64).T, dtype=torch.float64
    )
    n_verts = int(domain.vertices.shape[0])
    assert (
        float(torch.max(torch.abs(dof_xy[:n_verts] - domain.vertices)).item()) < 1e-13
    )
    facet_v0 = torch.tensor(skmesh.facets[0], dtype=torch.int64)
    facet_v1 = torch.tensor(skmesh.facets[1], dtype=torch.int64)
    q_shape = tuple(int(d) for d in basis.dx.shape)
    dx_t = torch.tensor(np.asarray(basis.dx, dtype=np.float64), dtype=torch.float64)
    x_dofs, y_dofs = (np.asarray(v) for v in basis.split_indices())

    def _local_node_feats(state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        cell_ev = state_dict["viscous_strain"].mean(dim=1)  # (C, M, 3, 3)
        vert_feats = project_cell_state_to_nodes(domain, cell_ev)  # (V, 4*M)
        mid_feats = 0.5 * (vert_feats[facet_v0] + vert_feats[facet_v1])  # (F, 4*M)
        return torch.cat((vert_feats, mid_feats), dim=0).unsqueeze(0)

    def _to_full_vec(nodal_u: np.ndarray) -> np.ndarray:
        vec = np.zeros(basis.N, dtype=np.float64)
        vec[x_dofs] = nodal_u[:, 0]
        vec[y_dofs] = nodal_u[:, 1]
        return vec

    def _eval_fem_step_residual(
        u_old_vec: np.ndarray,
        u_new_vec: np.ndarray,
        state_old: dict[str, torch.Tensor],
        dt: float,
    ) -> tuple[float, dict[str, torch.Tensor], float, float, float, float]:
        from skfem import asm  # type: ignore[import-untyped]

        from piginot.stateful.fem_stepper import _internal_form

        g_old = np.asarray(basis.interpolate(u_old_vec).grad, dtype=np.float64)
        g_new = np.asarray(basis.interpolate(u_new_vec).grad, dtype=np.float64)
        f2d_old = torch.from_numpy(
            np.ascontiguousarray(np.transpose(g_old, (2, 3, 0, 1)) + np.eye(2))
        )
        f2d_new = torch.from_numpy(
            np.ascontiguousarray(np.transpose(g_new, (2, 3, 0, 1)) + np.eye(2))
        )
        ps = visco.plane_stress_update(f2d_old, f2d_new, state_old, dt)
        stress_2d = ps.update_3d.stress[..., :2, :2].detach().cpu().numpy()
        internal = np.asarray(
            asm(_internal_form, basis, stress=np.transpose(stress_2d, (2, 3, 0, 1))),
            dtype=np.float64,
        )
        res_norm = float(np.linalg.norm(internal[free_dofs]))
        rxn = float(np.sum(internal[by_name["right_disp"]]))
        min_det = float(ps.update_3d.det_f.min().item())
        min_diss = float(ps.update_3d.dissipation.min().item())
        tot_diss = float(
            np.sum(ps.update_3d.dissipation.detach().cpu().numpy() * basis.dx)
        )
        s_new = {k: v.detach().clone() for k, v in ps.update_3d.state_new.items()}
        return res_norm, s_new, rxn, min_det, min_diss, tot_diss

    # Pure operator rollout on unseen trajectory.
    state_op = visco.initial_state(q_shape, dtype=torch.float64)
    u_old_vec = np.zeros(basis.N, dtype=np.float64)
    op_rxns: list[float] = []
    fem_rxns = [s.reactions["right_disp"] for s in fem_ref.steps]
    op_residuals: list[float] = []
    op_min_diss = float("inf")
    op_min_det = float("inf")

    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        ux = evaluate_load_path(segments, times[i], {"u_max": unseen_u_max})["ux"]
        load_vec = torch.tensor([ux / L], dtype=torch.float64)
        hist_feat = summarize_viscous_state(state_op, weights=dx_t)
        local_feat = _local_node_feats(state_op)
        with torch.no_grad():
            u_nodal_t = op.displacement(
                query_xy=dof_xy.unsqueeze(0),
                frame=frame,
                load_vector=load_vec,
                dt=dt,
                history_features=hist_feat,
                lifting=quarter_hole_lifting(L, H, ux / L),
                local_state_features=local_feat,
            )[0]
        u_new_vec = _to_full_vec(u_nodal_t.cpu().numpy())
        res_n, state_op, rxn, mdet, mdiss, _ = _eval_fem_step_residual(
            u_old_vec, u_new_vec, state_op, dt
        )
        op_rxns.append(rxn)
        op_residuals.append(res_n)
        op_min_det = min(op_min_det, mdet)
        op_min_diss = min(op_min_diss, mdiss)
        u_old_vec = u_new_vec

    rxn_abs_errs = [abs(a - b) for a, b in zip(op_rxns, fem_rxns, strict=True)]
    rxn_scale = max(abs(r) for r in fem_rxns)
    rxn_rel_errs = [e / rxn_scale for e in rxn_abs_errs]
    ev_op = state_op["viscous_strain"]
    ev_fem = fem_ref.final_state["viscous_strain"]
    ev_end_rel_err = float(
        torch.linalg.vector_norm(ev_op - ev_fem) / torch.linalg.vector_norm(ev_fem)
    )

    # GuardedStatefulStepper rollout with real single-step incremental FEM fallback.
    from piginot.contracts.stateful import IntegratorSpec

    integ = IntegratorSpec()

    def _op_predictor(u_old_t, s_old, load_next, dt_s):
        ux = load_next["ux"]
        load_vec = torch.tensor([ux / L], dtype=torch.float64)
        hist_feat = summarize_viscous_state(s_old, weights=dx_t)
        local_feat = _local_node_feats(s_old)
        with torch.no_grad():
            u_nodal_t = op.displacement(
                query_xy=dof_xy.unsqueeze(0),
                frame=frame,
                load_vector=load_vec,
                dt=dt_s,
                history_features=hist_feat,
                lifting=quarter_hole_lifting(L, H, ux / L),
                local_state_features=local_feat,
            )[0]
        u_new_v = _to_full_vec(u_nodal_t.cpu().numpy())
        res_n, s_new, _, mdet, mdiss, tdiss = _eval_fem_step_residual(
            u_old_t.numpy(), u_new_v, dict(s_old), dt_s
        )
        return CandidateStepOutput(
            displacement=torch.from_numpy(u_new_v),
            state_new=s_new,
            min_det_f=mdet,
            min_dissipation=mdiss,
            total_dissipation=tdiss,
            residual_norm=res_n,
            admissible=mdet > 0.0 and mdiss >= -1e-12,
        )

    def _fem_fallback(u_old_t, s_old, load_next, dt_s):
        ux = load_next["ux"]
        target_d: dict[int, float] = {}
        for bc in bcs:
            val = 0.0 if bc.load_key is None else float(ux)
            for dof in by_name[bc.name]:
                target_d[int(dof)] = val
        ok, u_n, s_n, _, _, res_n, _, tdiss, mdet, _, reason = _solve_single_increment(
            basis=basis,
            model=visco,
            u_old=u_old_t.numpy(),
            state_old=s_old,
            f33_old=torch.ones(q_shape, dtype=torch.float64),
            target_dirichlet=target_d,
            free_dofs=free_dofs,
            dt=dt_s,
            integrator=integ,
        )
        return CandidateStepOutput(
            displacement=torch.from_numpy(u_n),
            state_new=s_n,
            min_det_f=mdet,
            min_dissipation=0.0 if ok else -1.0,
            total_dissipation=tdiss,
            residual_norm=res_n,
            admissible=ok,
            reason=reason,
        )

    def _run_guarded(
        res_tol: float,
    ) -> tuple[list[str], BoundTrajectoryState, torch.Tensor, dict[str, torch.Tensor]]:
        stp = GuardedStatefulStepper(
            problem_spec_hash="f" * 64,
            mesh_hash=mesh.mesh_id,
            material_schema_hash=visco.state_schema.schema_hash,
            load_path_hash="l" * 64,
            predictor=_op_predictor,
            fem_fallback=_fem_fallback,
            verification=StatefulVerificationPolicy(allow_fem_fallback=True),
            residual_tolerance=res_tol,
            max_substep_halvings=0,
        )
        gs = visco.initial_state(q_shape, dtype=torch.float64)
        gu = torch.zeros(basis.N, dtype=torch.float64)
        gb = BoundTrajectoryState(
            problem_spec_hash="f" * 64,
            mesh_hash=mesh.mesh_id,
            material_schema_hash=visco.state_schema.schema_hash,
            load_path_hash="l" * 64,
            time_index=0,
            time_value=0.0,
            step_controller_version=STEP_CONTROLLER_VERSION,
            state_hash=hash_state_tensors(gs),
        )
        statuses: list[str] = []
        for idx_t in range(1, len(times)):
            dt_t = times[idx_t] - times[idx_t - 1]
            ux_t = evaluate_load_path(segments, times[idx_t], {"u_max": unseen_u_max})[
                "ux"
            ]
            sr = stp.advance(
                bound_state=gb,
                displacement_old=gu,
                state_old=gs,
                load_next={"ux": ux_t},
                dt=dt_t,
            )
            statuses.append(sr.status.value)
            assert (
                sr.accepted
                and sr.displacement is not None
                and sr.state_new is not None
                and sr.bound_state is not None
            )
            gu = sr.displacement
            gs = dict(sr.state_new)
            gb = sr.bound_state
        return statuses, gb, gu, gs

    step_statuses, g_bound, g_u, g_state = _run_guarded(res_tol=0.5)
    hybrid_tol = float(np.median(op_residuals)) * 1.15
    hybrid_statuses, _, _, _ = _run_guarded(res_tol=hybrid_tol)

    # Verify checkpoint save & reload roundtrip on final guarded state.
    save_trajectory_checkpoint(
        ckpt_root / "visco-final",
        bound_state=g_bound,
        displacement=g_u,
        state_tensors=g_state,
    )
    reloaded_bound, _, _ = load_trajectory_checkpoint(ckpt_root / "visco-final")
    ckpt_verified = reloaded_bound == g_bound

    p90_rxn_err = float(np.percentile(rxn_rel_errs, 90))
    max_rxn_err = float(max(rxn_rel_errs))
    print(
        f"  Tangent FD rel err: {t_check.relative_error:.2e} | "
        f"FEM dt-halving diff: {dt_halving_rel * 100:.2f}% | CPU replay exact: {replay_exact}"
    )
    print(
        f"  Pure F1 Operator:   p90 rxn err={p90_rxn_err * 100:.2f}%, "
        f"max rxn err={max_rxn_err * 100:.2f}%, end E_v err={ev_end_rel_err * 100:.2f}%, "
        f"min_diss={op_min_diss:.2e}, min_det={op_min_det:.3f} (train={train_wall}s)"
    )
    print(
        f"  Guarded (strict 0.5N): {sum(s == 'ACCEPTED_OPERATOR' for s in step_statuses)} op / "
        f"{sum(s == 'ACCEPTED_FEM_FALLBACK' for s in step_statuses)} FEM | "
        f"Hybrid ({hybrid_tol:.2f}N): {sum(s == 'ACCEPTED_OPERATOR' for s in hybrid_statuses)} op / "
        f"{sum(s == 'ACCEPTED_FEM_FALLBACK' for s in hybrid_statuses)} FEM | ckpt={ckpt_verified}"
    )

    return {
        "status": "DIAGNOSTIC_PILOT",
        "claim_level": "NOT_VERIFIED",
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "f1_material_tangent_fd_rel_err": t_check.relative_error,
        "f1_fem_reference": {
            "wall_seconds": fem_wall,
            "dt_halving_peak_reaction_rel_diff": dt_halving_rel,
            "cpu_replay_exact": replay_exact,
            "reaction_history_N": fem_rxns,
        },
        "f1_operator_training": {
            "include_local_state": True,
            "epochs": tr_summary.epochs,
            "total_steps": tr_summary.total_steps,
            "initial_loss": tr_summary.initial_loss,
            "final_loss": tr_summary.final_loss,
            "wall_seconds": train_wall,
        },
        "f1_unseen_trajectory": {
            "unseen_u_max_mm": unseen_u_max,
            "operator_reaction_history_N": op_rxns,
            "p90_reaction_rel_err": p90_rxn_err,
            "max_reaction_rel_err": max_rxn_err,
            "end_state_viscous_strain_rel_err": ev_end_rel_err,
            "min_det_f": op_min_det,
            "min_dissipation": op_min_diss,
            "operator_residual_norms_N": op_residuals,
        },
        "guarded_stepper": {
            "strict_residual_tol_N": 0.5,
            "step_statuses": step_statuses,
            "operator_accepted_count": sum(
                s == "ACCEPTED_OPERATOR" for s in step_statuses
            ),
            "fem_fallback_count": sum(
                s == "ACCEPTED_FEM_FALLBACK" for s in step_statuses
            ),
            "hybrid_residual_tol_N": hybrid_tol,
            "hybrid_step_statuses": hybrid_statuses,
            "hybrid_operator_accepted_count": sum(
                s == "ACCEPTED_OPERATOR" for s in hybrid_statuses
            ),
            "hybrid_fem_fallback_count": sum(
                s == "ACCEPTED_FEM_FALLBACK" for s in hybrid_statuses
            ),
            "checkpoint_roundtrip_verified": ckpt_verified,
        },
    }


# ---------------------------------------------------------------------------
# PILOT 4: P3 (Certificate-Driven Active FEM + Multi-Fidelity Learning)
# ---------------------------------------------------------------------------


def run_pilot_4_p3_active_fem() -> dict[str, Any]:
    print(
        "\n=== PILOT 4: P3 (Certificate-Driven Active FEM: 3-Arm x 3-Seed Multi-Fidelity) ==="
    )
    t_start = time.perf_counter()
    L, H, R = 4.0, 3.0, 1.0
    backend = ScikitFemBackend()
    mesh_m0 = make_quarter_hole_mesh(L, H, R, n_half_arc=3, n_radial=3, label="p3-m0")
    mesh_m1 = refine_fixed_mesh(mesh_m0, 2)
    domain_m0 = TriangleDomain(
        vertices=torch.tensor(mesh_m0.nodes, dtype=torch.float64),
        cells=torch.tensor(mesh_m0.cells, dtype=torch.int64),
    )
    frame = QueryFrame(
        origin=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        extent=torch.tensor([[L, H]], dtype=torch.float64),
    )

    anchors = [
        (0.06, 0.24),
        (0.08, 0.28),
        (0.10, 0.25),
        (0.11, 0.30),
    ]
    # Realistic imbalanced 2D CAD parameter pool: 10 routine nominal TRAIN queries
    # plus 6 extreme high-strain / low- and high-Poisson corners across 2D space.
    train_pool_specs: list[tuple[str, float, float]] = [
        ("train-01", 0.055, 0.23),
        ("train-02", 0.065, 0.27),
        ("train-03", 0.075, 0.25),
        ("train-04", 0.085, 0.29),
        ("train-05", 0.095, 0.24),
        ("train-06", 0.105, 0.28),
        ("train-07", 0.110, 0.26),
        ("train-08", 0.115, 0.31),
        ("train-09", 0.120, 0.27),
        ("train-10", 0.125, 0.30),
        ("train-11", 0.195, 0.16),
        ("train-12", 0.235, 0.17),
        ("train-13", 0.210, 0.31),
        ("train-14", 0.230, 0.40),
        ("train-15", 0.235, 0.41),
        ("train-16", 0.240, 0.42),
    ]
    # Strictly disjoint 6-case DEVELOPMENT set spanning nominal + both 2D extreme corners.
    dev_specs: list[tuple[str, float, float]] = [
        ("dev-01", 0.080, 0.26),
        ("dev-02", 0.165, 0.20),
        ("dev-03", 0.170, 0.38),
        ("dev-04", 0.220, 0.18),
        ("dev-05", 0.225, 0.30),
        ("dev-06", 0.230, 0.41),
    ]
    s_mean, s_std = 0.145, 0.06
    nu_mean, nu_std = 0.29, 0.08

    def norm_p(strain: float, nu: float) -> torch.Tensor:
        return torch.tensor(
            [[(strain - s_mean) / s_std, (nu - nu_mean) / nu_std]], dtype=torch.float64
        )

    op_cfg = EnergyOperatorConfig(
        parameter_names=("strain", "poisson_ratio"),
        hidden_dim=24,
        latent_token_count=8,
        parameter_hidden_dim=32,
        num_attention_heads=4,
        decoder_layers=2,
        positional_frequencies=4,
    )
    base_energy_cfg = PotentialEnergyConfig(
        points_per_cell=3, barrier_weight=1e3, barrier_threshold=0.20
    )

    torch.manual_seed(19)
    base_model = EnergyFormOperator(op_cfg).to(torch.float64)
    opt_b = torch.optim.Adam(
        base_model.parameters(), lr=3e-3, foreach=False, fused=False
    )
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=80, eta_min=3e-4)
    gen_b = torch.Generator().manual_seed(202)
    for _ in range(80):
        opt_b.zero_grad(set_to_none=True)
        loss_b = torch.zeros((), dtype=torch.float64)
        for eps, nu in anchors:
            mat_a = neo_hookean_spec(100.0, nu)
            pts, wts = stratified_triangle_quadrature(domain_m0, 3, generator=gen_b)
            _, g = base_model.displacement_and_gradient(
                query_xy=pts.unsqueeze(0),
                parameters=norm_p(eps, nu),
                frame=frame,
                lifting=quarter_hole_lifting(L, H, eps),
                create_graph=True,
            )
            loss_b = loss_b + plane_stress_potential_energy(
                g[0], wts, mat_a, config=base_energy_cfg, reference_area=domain_m0.area
            ).total / len(anchors)
        loss_b.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
        opt_b.step()
        sched_b.step()

    # Solve strictly disjoint DEVELOPMENT P2 FEM references (never read by acquisition).
    dev_fems: list[tuple[str, float, float, MaterialSpec, FemRequest, Any]] = []
    for cid, eps, nu in dev_specs:
        mat_d = neo_hookean_spec(100.0, nu)
        req_d = quarter_hole_fem_request(
            mesh_m1, L, eps, request_id=cid, element_order=2, material=mat_d
        )
        att_d = backend.solve_many((req_d,))[0]
        assert att_d.success and att_d.result is not None
        dev_fems.append((cid, eps, nu, mat_d, req_d, att_d.result))

    def _eval_on_dev(model: EnergyFormOperator) -> dict[str, float]:
        u_errs: list[float] = []
        p_errs: list[float] = []
        r_errs: list[float] = []
        for _cid, eps, nu, mat_d, req_d, res_d in dev_fems:
            p_in = norm_p(eps, nu)
            lf = quarter_hole_lifting(L, H, eps)

            def _fn(
                pts: torch.Tensor, p=p_in, lift=lf
            ) -> tuple[torch.Tensor, torch.Tensor]:
                u, g = model.displacement_and_gradient(
                    query_xy=pts.unsqueeze(0),
                    parameters=p,
                    frame=frame,
                    lifting=lift,
                    create_graph=False,
                )
                return u[0], g[0]

            ev = eval_field_errors_against_fem(_fn, req_d, res_d, mat_d)
            u_errs.append(ev["u_rel_l2"])
            p_errs.append(ev["p_rel_l2"])
            r_errs.append(ev["rxn_rel_err"])
        return {
            "p90_u_rel_l2": float(np.percentile(u_errs, 90)),
            "p90_p_rel_l2": float(np.percentile(p_errs, 90)),
            "p90_rxn_rel_err": float(np.percentile(r_errs, 90)),
            "mean_p_rel_l2": float(np.mean(p_errs)),
            "mean_u_rel_l2": float(np.mean(u_errs)),
        }

    base_dev = _eval_on_dev(base_model)

    centroids = domain_m0.vertices[domain_m0.cells].mean(dim=1)
    anchor_norm = [((a - s_mean) / s_std, (b - nu_mean) / nu_std) for a, b in anchors]

    def _compute_physics_indicator(
        model: EnergyFormOperator, eps: float, nu: float
    ) -> float:
        mat_c = neo_hookean_spec(100.0, nu)
        pts = centroids.detach().clone().requires_grad_(True)
        _, grad = model.displacement_and_gradient(
            query_xy=pts.unsqueeze(0),
            parameters=norm_p(eps, nu),
            frame=frame,
            lifting=quarter_hole_lifting(L, H, eps),
            create_graph=True,
        )
        stress = constitutive_in_plane_stress(grad[0], mat_c)
        div_components = []
        for row in range(2):
            dp_dx = torch.autograd.grad(
                stress[:, row, 0].sum(), pts, retain_graph=True
            )[0][:, 0]
            dp_dy = torch.autograd.grad(
                stress[:, row, 1].sum(), pts, retain_graph=(row == 0)
            )[0][:, 1]
            div_components.append(dp_dx + dp_dy)
        div_p = torch.stack(div_components, dim=-1)
        area_w = domain_m0.areas
        # Normalize force-imbalance residual ||div P||_L2 by constant reference
        # stiffness E_0 = 100 MPa (not case-dependent ||P||, which would inflate
        # low-strain cases where ||P|| is near zero).
        num = torch.sqrt(torch.sum(torch.sum(div_p.square(), dim=-1) * area_w))
        return float((num / (100.0 * math.sqrt(domain_m0.area))).detach().item())

    pool_candidates: list[AcquisitionCandidate] = []
    pool_lookup: dict[str, tuple[float, float]] = {}
    for cid, eps, nu in train_pool_specs:
        pool_lookup[cid] = (eps, nu)
        eta_val = _compute_physics_indicator(base_model, eps, nu)
        nf = ((eps - s_mean) / s_std, (nu - nu_mean) / nu_std)
        ood_d = min(math.hypot(nf[0] - ax, nf[1] - ay) for ax, ay in anchor_norm)
        pool_candidates.append(
            AcquisitionCandidate(
                case_id=cid,
                partition="TRAIN",
                feature_vector=(eps, nu),
                eta_energy=eta_val,
                ood_distance=ood_d,
                ensemble_disagreement=0.0,
                parameters={"strain": eps, "poisson_ratio": nu},
            )
        )

    replay_plan = AcquisitionPlan(
        plan_id="replay-check",
        arm_kind="physics_diversity",
        budget_solves=4,
        batch_size=2,
        seed=0,
    )
    ev_fwd = select_acquisition_batch(
        pool_candidates, already_acquired_ids=(), plan=replay_plan
    )
    ev_rev = select_acquisition_batch(
        list(reversed(pool_candidates)), already_acquired_ids=(), plan=replay_plan
    )
    deterministic_replay_ok = (
        ev_fwd.selection_hash == ev_rev.selection_hash
        and ev_fwd.selected_case_ids == ev_rev.selected_case_ids
    )

    mf_cfg = MultiFidelityConfig(
        base_energy=base_energy_cfg,
        fem_displacement_weight=1.5,
        fem_stress_weight=0.35,
    )
    seeds = (0, 1, 2)
    arm_names = ("random_doe", "physics_indicator", "physics_diversity")
    arm_results: dict[str, dict[str, Any]] = {}

    for arm in arm_names:
        seed_runs: list[dict[str, Any]] = []
        for seed in seeds:
            plan = AcquisitionPlan(
                plan_id=f"p3-{arm}-s{seed}",
                arm_kind=arm,  # type: ignore[arg-type]
                budget_solves=4,
                batch_size=2,
                seed=seed,
                weights=AcquisitionWeights(
                    eta_weight=0.40,
                    ood_weight=0.15,
                    ensemble_weight=0.0,
                    diversity_weight=0.45,
                ),
            )
            ledger = FEMCostLedger(plan_hash=plan.plan_hash, budget_solves=4)
            acquired_ids: list[str] = []
            acquired_feats: list[tuple[float, ...]] = []
            acquired_targets: list[
                tuple[
                    float,
                    float,
                    MaterialSpec,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ]
            ] = []
            selection_hashes: list[str] = []

            model_arm = copy.deepcopy(base_model)
            round_steps = (35, 45)
            for rnd in range(2):
                # Closed-loop re-scoring of remaining TRAIN candidates using current model.
                cur_pool: list[AcquisitionCandidate] = []
                for cid, eps, nu in train_pool_specs:
                    eta_cur = (
                        _compute_physics_indicator(model_arm, eps, nu)
                        if rnd > 0
                        else next(
                            c.eta_energy for c in pool_candidates if c.case_id == cid
                        )
                    )
                    nf = ((eps - s_mean) / s_std, (nu - nu_mean) / nu_std)
                    ood_d = min(
                        math.hypot(nf[0] - ax, nf[1] - ay) for ax, ay in anchor_norm
                    )
                    cur_pool.append(
                        AcquisitionCandidate(
                            case_id=cid,
                            partition="TRAIN",
                            feature_vector=(eps, nu),
                            eta_energy=eta_cur,
                            ood_distance=ood_d,
                            ensemble_disagreement=0.0,
                            parameters={"strain": eps, "poisson_ratio": nu},
                        )
                    )
                ev = select_acquisition_batch(
                    cur_pool,
                    already_acquired_ids=acquired_ids,
                    plan=plan,
                    round_index=rnd,
                    already_acquired_features=acquired_feats,
                )
                selection_hashes.append(ev.selection_hash)
                for cid in ev.selected_case_ids:
                    eps, nu = pool_lookup[cid]
                    mat_c = neo_hookean_spec(100.0, nu)
                    req_c = quarter_hole_fem_request(
                        mesh_m1,
                        L,
                        eps,
                        request_id=f"{arm}-s{seed}-{cid}",
                        element_order=2,
                        material=mat_c,
                    )
                    t_sol = time.perf_counter()
                    att = backend.solve_many((req_c,))[0]
                    w_sec = time.perf_counter() - t_sol
                    lin_c = FemLinearization(req_c)
                    ledger = ledger.append(
                        FEMCostLedgerEntry(
                            case_id=cid,
                            partition="TRAIN",
                            attempt_index=ledger.attempted_count + 1,
                            succeeded=bool(att.success and att.result is not None),
                            wall_seconds=w_sec,
                            dof_count=lin_c.dof_coordinates.shape[0] * 2,
                            result_hash=mesh_m1.mesh_id if att.success else None,
                        )
                    )
                    assert att.success and att.result is not None
                    qpts = np.asarray(
                        att.result.quadrature_points, dtype=np.float64
                    ).reshape(-1, 2)
                    qwts = np.asarray(
                        att.result.quadrature_weights, dtype=np.float64
                    ).reshape(-1)
                    u_nodal = np.asarray(att.result.displacement, dtype=np.float64)
                    u_q = np.stack(
                        [
                            lin_c.scalar_basis.interpolate(u_nodal[:, ax])
                            for ax in (0, 1)
                        ],
                        axis=-1,
                    ).reshape(-1, 2)
                    p_q = np.asarray(att.result.first_piola_stress, dtype=np.float64)[
                        ..., :2, :2
                    ].reshape(-1, 2, 2)
                    acquired_targets.append(
                        (
                            eps,
                            nu,
                            mat_c,
                            torch.tensor(qpts, dtype=torch.float64),
                            torch.tensor(qwts, dtype=torch.float64),
                            torch.tensor(u_q, dtype=torch.float64),
                            torch.tensor(p_q, dtype=torch.float64),
                        )
                    )
                    acquired_ids.append(cid)
                    acquired_feats.append((eps, nu))

                n_steps_rnd = round_steps[rnd]
                opt_a = torch.optim.Adam(
                    model_arm.parameters(),
                    lr=2.5e-3 if rnd == 0 else 1.5e-3,
                    foreach=False,
                    fused=False,
                )
                sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt_a, T_max=n_steps_rnd, eta_min=3e-4
                )
                gen_a = torch.Generator().manual_seed(500 + seed * 10 + rnd)
                for _ in range(n_steps_rnd):
                    opt_a.zero_grad(set_to_none=True)
                    tot_loss = torch.zeros((), dtype=torch.float64)
                    denom = len(acquired_targets) + len(anchors)
                    for (
                        eps,
                        nu,
                        mat_c,
                        qpts_t,
                        qwts_t,
                        u_ref_t,
                        p_ref_t,
                    ) in acquired_targets:
                        u_pr, g_pr = model_arm.displacement_and_gradient(
                            query_xy=qpts_t.unsqueeze(0),
                            parameters=norm_p(eps, nu),
                            frame=frame,
                            lifting=quarter_hole_lifting(L, H, eps),
                            create_graph=True,
                        )
                        mf_terms = multifidelity_energy_loss(
                            u_pred=u_pr[0],
                            grad_pred=g_pr[0],
                            weights=qwts_t,
                            material=mat_c,
                            reference_area=domain_m0.area,
                            config=mf_cfg,
                            u_fem_target=u_ref_t,
                            p_fem_target=p_ref_t,
                            partition="TRAIN",
                        )
                        tot_loss = tot_loss + mf_terms.total / denom
                    for eps, nu in anchors:
                        mat_a = neo_hookean_spec(100.0, nu)
                        pts, wts = stratified_triangle_quadrature(
                            domain_m0, 3, generator=gen_a
                        )
                        _, g_a = model_arm.displacement_and_gradient(
                            query_xy=pts.unsqueeze(0),
                            parameters=norm_p(eps, nu),
                            frame=frame,
                            lifting=quarter_hole_lifting(L, H, eps),
                            create_graph=True,
                        )
                        tot_loss = (
                            tot_loss
                            + plane_stress_potential_energy(
                                g_a[0],
                                wts,
                                mat_a,
                                config=base_energy_cfg,
                                reference_area=domain_m0.area,
                            ).total
                            / denom
                        )
                    tot_loss.backward()  # type: ignore[no-untyped-call]
                    torch.nn.utils.clip_grad_norm_(model_arm.parameters(), max_norm=1.0)
                    opt_a.step()
                    sched_a.step()

            dev_metrics = _eval_on_dev(model_arm)
            seed_runs.append(
                {
                    "seed": seed,
                    "selected_case_ids": acquired_ids,
                    "selection_hashes": selection_hashes,
                    "ledger": {
                        "attempted": ledger.attempted_count,
                        "succeeded": ledger.succeeded_count,
                        "failed": ledger.failed_count,
                        "wall_seconds": round(ledger.total_wall_seconds, 3),
                        "ledger_hash": ledger.ledger_hash,
                    },
                    "dev_metrics": dev_metrics,
                }
            )

        arm_results[arm] = {
            "mean_p90_u_rel_l2": float(
                np.mean([r["dev_metrics"]["p90_u_rel_l2"] for r in seed_runs])
            ),
            "std_p90_u_rel_l2": float(
                np.std([r["dev_metrics"]["p90_u_rel_l2"] for r in seed_runs])
            ),
            "mean_p90_p_rel_l2": float(
                np.mean([r["dev_metrics"]["p90_p_rel_l2"] for r in seed_runs])
            ),
            "std_p90_p_rel_l2": float(
                np.std([r["dev_metrics"]["p90_p_rel_l2"] for r in seed_runs])
            ),
            "mean_p90_rxn_rel_err": float(
                np.mean([r["dev_metrics"]["p90_rxn_rel_err"] for r in seed_runs])
            ),
            "std_p90_rxn_rel_err": float(
                np.std([r["dev_metrics"]["p90_rxn_rel_err"] for r in seed_runs])
            ),
            "runs": seed_runs,
        }

    print(
        f"  Base Physics-Only (0 FEM):   DEV p90 u_L2={base_dev['p90_u_rel_l2'] * 100:.2f}%, "
        f"P_L2={base_dev['p90_p_rel_l2'] * 100:.2f}%, rxn={base_dev['p90_rxn_rel_err'] * 100:.2f}%"
    )
    for arm in arm_names:
        ar = arm_results[arm]
        print(
            f"  {arm:20s} (4 FEM): DEV p90 u_L2={ar['mean_p90_u_rel_l2'] * 100:.2f}% "
            f"(+-{ar['std_p90_u_rel_l2'] * 100:.2f}%), "
            f"P_L2={ar['mean_p90_p_rel_l2'] * 100:.2f}% (+-{ar['std_p90_p_rel_l2'] * 100:.2f}%), "
            f"rxn={ar['mean_p90_rxn_rel_err'] * 100:.2f}%"
        )

    return {
        "status": "DIAGNOSTIC_PILOT",
        "claim_level": "NOT_VERIFIED",
        "training_mode_disclosure": mf_cfg.disclosure,
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "deterministic_replay_verified": deterministic_replay_ok,
        "train_pool_scores": [
            {
                "case_id": c.case_id,
                "strain": c.parameters["strain"],
                "poisson_ratio": c.parameters["poisson_ratio"],
                "eta_energy": c.eta_energy,
                "ood_distance": c.ood_distance,
            }
            for c in pool_candidates
        ],
        "base_physics_only_dev": base_dev,
        "arms": arm_results,
    }


# ---------------------------------------------------------------------------
# PILOT 5: P5 (FEM-Guarded Differentiable CAD Design Optimization vs Pure-FEM)
# ---------------------------------------------------------------------------


def run_pilot_5_p5_design_opt() -> dict[str, Any]:
    print(
        "\n=== PILOT 5: P5 (FEM-Guarded Differentiable CAD Design Optimization vs. Pure-FEM) ==="
    )
    t_start = time.perf_counter()
    L = 4.0
    eps = 0.12
    f_min_n = 26.0
    backend = ScikitFemBackend()
    mat = neo_hookean_spec()
    roles = hole_roles()
    tok = GeometryTokenizer(subdivisions_per_facet=1)
    rxn_qoi = ReactionForceQoi("right_reaction", "right_disp")

    train_rh = [
        (0.60, 2.40, "opt-g1"),
        (0.60, 3.60, "opt-g2"),
        (0.95, 2.40, "opt-g3"),
        (0.95, 3.00, "opt-g4"),
        (1.30, 2.80, "opt-g5"),
        (1.30, 3.60, "opt-g6"),
    ]
    train_meshes = [
        make_quarter_hole_mesh(L, h_, r_, n_half_arc=4, n_radial=4, label=lbl)
        for r_, h_, lbl in train_rh
    ]
    train_descs = [
        compute_geometry_descriptor(m, roles, tokenizer=tok) for m in train_meshes
    ]
    ood_detector = GeometryOODDetector.fit(train_descs, descriptor_threshold=4.5)

    b_cfg = BoundaryEnergyConfig(
        energy=EnergyOperatorConfig(
            parameter_names=("strain",),
            hidden_dim=24,
            latent_token_count=8,
            parameter_hidden_dim=32,
            num_attention_heads=4,
            decoder_layers=2,
            positional_frequencies=4,
        ),
        boundary_feature_names=GEOMETRY_FEATURE_NAMES,
    )
    torch.manual_seed(29)
    design_op = BoundaryEnergyOperator(b_cfg).to(torch.float64)
    energy_cfg = PotentialEnergyConfig(
        points_per_cell=3, barrier_weight=1e3, barrier_threshold=0.20
    )
    train_batches = [
        tok.tokenize_batch([(m, roles)], dtype=torch.float64) for m in train_meshes
    ]
    train_domains = [
        TriangleDomain(
            torch.tensor(m.nodes, dtype=torch.float64),
            torch.tensor(m.cells, dtype=torch.int64),
        )
        for m in train_meshes
    ]
    p_fixed = torch.tensor([[0.0]], dtype=torch.float64)
    opt = torch.optim.Adam(design_op.parameters(), lr=3e-3, foreach=False, fused=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=180, eta_min=3e-4)
    pre_gen = torch.Generator().manual_seed(707)
    t_pre = time.perf_counter()
    for _ in range(180):
        opt.zero_grad(set_to_none=True)
        loss_acc = torch.zeros((), dtype=torch.float64)
        for (_r, h_, _lbl), batch, dom in zip(
            train_rh, train_batches, train_domains, strict=True
        ):
            enc = design_op.encode_boundary(
                boundary_xy=batch.boundary_xy,
                boundary_normals=batch.boundary_normals,
                boundary_features=batch.boundary_features,
                weights=batch.weights,
                mask=batch.mask,
                frame=batch.frame,
            )
            pts, wts = stratified_triangle_quadrature(dom, 3, generator=pre_gen)
            _, g = design_op.displacement_and_gradient(
                encoding=enc,
                query_xy=pts.unsqueeze(0),
                parameters=p_fixed,
                lifting=quarter_hole_lifting(L, h_, eps),
                create_graph=True,
            )
            loss_acc = loss_acc + plane_stress_potential_energy(
                g[0], wts, mat, config=energy_cfg, reference_area=dom.area
            ).total / len(train_rh)
        loss_acc.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(design_op.parameters(), max_norm=1.0)
        opt.step()
        sched.step()
    surrogate_pretrain_wall = round(time.perf_counter() - t_pre, 2)

    problem = DesignProblem(
        name="quarter-hole-lightweighting",
        geometry_version="quarter-hole-cad-v1",
        problem_spec_hash="e" * 64,
        parameters=(
            DesignParameter(
                "hole_radius", lower=0.60, upper=1.30, initial=0.70, unit="mm"
            ),
            DesignParameter(
                "plate_height", lower=2.40, upper=3.60, initial=3.40, unit="mm"
            ),
        ),
        objective_name="material_area",
        objective_unit="mm^2",
        minimize=True,
        constraints=(
            DesignConstraint(
                name="reaction_force",
                kind="inequality_ge",
                limit=f_min_n,
                tolerance=0.05,
                unit="N",
            ),
        ),
        max_iterations=25,
        max_fem_evaluations=10,
        initial_trust_radius=0.22,
        min_trust_radius=0.01,
        gradient_check_interval=6,
        gradient_cos_min=0.25,
        accept_eta_tolerance=0.05,
        constraint_penalty_weight=5.0,
    )

    def _area(r_val: float, h_val: float) -> float:
        return L * h_val - 0.25 * math.pi * (r_val * r_val)

    def _surrogate_eval(p: Mapping[str, float]) -> SurrogateDesignEvaluation:
        r_val = float(p["hole_radius"])
        h_val = float(p["plate_height"])
        if r_val >= h_val - 0.35:
            return SurrogateDesignEvaluation(
                objective=float("inf"),
                gradient={"hole_radius": 0.0, "plate_height": 0.0},
                constraints={"reaction_force": 0.0},
                in_envelope=False,
                admissible=False,
                eta_indicator=1.0,
                reason="hole violates minimum ligament clearance",
            )
        mesh_p = make_quarter_hole_mesh(L, h_val, r_val, n_half_arc=4, n_radial=4)
        ood = ood_detector.score(mesh_p, roles, tokenizer=tok)
        batch_p = tok.tokenize_batch([(mesh_p, roles)], dtype=torch.float64)
        dom_p = TriangleDomain(
            torch.tensor(mesh_p.nodes, dtype=torch.float64),
            torch.tensor(mesh_p.cells, dtype=torch.int64),
        )
        with torch.no_grad():
            enc_p = design_op.encode_boundary(
                boundary_xy=batch_p.boundary_xy,
                boundary_normals=batch_p.boundary_normals,
                boundary_features=batch_p.boundary_features,
                weights=batch_p.weights,
                mask=batch_p.mask,
                frame=batch_p.frame,
            )
        lift_p = quarter_hole_lifting(L, h_val, eps)
        q2_pts, q2_wts = deterministic_triangle_quadrature(dom_p, degree=2)
        _, g2 = design_op.displacement_and_gradient(
            encoding=enc_p,
            query_xy=q2_pts.unsqueeze(0),
            parameters=p_fixed,
            lifting=lift_p,
            create_graph=False,
        )
        p_stress = constitutive_in_plane_stress(g2[0], mat)
        rxn_surr = float(torch.sum(p_stress[:, 0, 0] * q2_wts).item()) / L
        e2 = float(
            plane_stress_potential_energy(
                g2[0], q2_wts, mat, config=energy_cfg, reference_area=dom_p.area
            )
            .energy.detach()
            .item()
        )
        q5_pts, q5_wts = deterministic_triangle_quadrature(dom_p, degree=5)
        _, g5 = design_op.displacement_and_gradient(
            encoding=enc_p,
            query_xy=q5_pts.unsqueeze(0),
            parameters=p_fixed,
            lifting=lift_p,
            create_graph=False,
        )
        e5 = float(
            plane_stress_potential_energy(
                g5[0], q5_wts, mat, config=energy_cfg, reference_area=dom_p.area
            )
            .energy.detach()
            .item()
        )
        eta_q = abs(e5 - e2) / max(abs(e5), 1e-12)
        return SurrogateDesignEvaluation(
            objective=_area(r_val, h_val),
            gradient={"hole_radius": -0.5 * math.pi * r_val, "plate_height": L},
            constraints={"reaction_force": rxn_surr},
            in_envelope=ood.status is GeometryOODStatus.IN_DISTRIBUTION,
            admissible=True,
            eta_indicator=eta_q,
        )

    fem_call_counter = 0

    def _fem_eval(p: Mapping[str, float]) -> FemDesignEvaluation:
        nonlocal fem_call_counter
        fem_call_counter += 1
        r_val = float(p["hole_radius"])
        h_val = float(p["plate_height"])
        mesh_p = make_quarter_hole_mesh(L, h_val, r_val, n_half_arc=4, n_radial=4)
        req_p = quarter_hole_fem_request(
            mesh_p, L, eps, request_id=f"opt-fem-{fem_call_counter}", element_order=2
        )
        att = backend.solve_many((req_p,))[0]
        if not att.success or att.result is None:
            return FemDesignEvaluation(
                succeeded=False,
                objective=None,
                constraints={},
                reason="FEM solve failed",
            )
        lin_p = FemLinearization(req_p)
        u_nodal = np.asarray(att.result.displacement, dtype=np.float64)
        st = lin_p.state(lin_p.vector(u_nodal))
        rxn_val = rxn_qoi.evaluate(lin_p, st)
        return FemDesignEvaluation(
            succeeded=True,
            objective=_area(r_val, h_val),
            constraints={"reaction_force": rxn_val},
            reference_hash=mesh_p.mesh_id,
        )

    t_gd = time.perf_counter()
    guarded_report = run_fem_guarded_design_optimization(
        problem,
        surrogate_evaluator=_surrogate_eval,
        fem_evaluator=_fem_eval,
        fd_step_fraction=2e-2,
    )
    guarded_wall = round(time.perf_counter() - t_gd, 2)

    refusal_report = run_fem_guarded_design_optimization(
        problem,
        surrogate_evaluator=_surrogate_eval,
        fem_evaluator=_fem_eval,
        geometry_supported=False,
        unsupported_reason="topology change adds unparameterized second cutout",
    )
    refusal_verified = (
        refusal_report.status is DesignOptimizationStatus.REFUSED_UNSUPPORTED_GEOMETRY
        and refusal_report.fem_evaluations == 0
    )

    t_pf = time.perf_counter()
    pure_fem_calls = 0
    pure_trace: list[dict[str, Any]] = []

    def _pure_fem_merit(p: Mapping[str, float]) -> tuple[float, float, float, bool]:
        nonlocal pure_fem_calls
        pure_fem_calls += 1
        ev = _fem_eval(p)
        assert ev.succeeded and ev.objective is not None
        feas, viols = problem.check_constraints(ev.constraints)
        merit = ev.objective + problem.constraint_penalty_weight * sum(
            v * v for v in viols.values()
        )
        return merit, ev.objective, ev.constraints["reaction_force"], feas

    pf_p = problem.initial_point()
    pf_merit, pf_obj, pf_rxn, pf_feas = _pure_fem_merit(pf_p)
    best_pf = (pf_p.copy(), pf_obj, pf_rxn, pf_feas)
    pf_rad = problem.initial_trust_radius
    pure_trace.append(
        {
            "iteration": 0,
            "parameters": dict(pf_p),
            "fem_objective": pf_obj,
            "fem_reaction_N": pf_rxn,
            "fem_feasible": pf_feas,
        }
    )
    for it in range(1, 16):
        if pf_rad < problem.min_trust_radius:
            break
        g_vec: list[float] = []
        for p_spec in problem.parameters:
            h = 2e-2 * p_spec.span
            p_pert = dict(pf_p)
            sign = 1.0 if pf_p[p_spec.name] + h <= p_spec.upper else -1.0
            p_pert[p_spec.name] = p_spec.clamp(pf_p[p_spec.name] + sign * h)
            dh = p_pert[p_spec.name] - pf_p[p_spec.name]
            m_pert, _, _, _ = _pure_fem_merit(p_pert)
            g_vec.append(((m_pert - pf_merit) / dh) * p_spec.span)
        gn = math.sqrt(sum(v * v for v in g_vec))
        if gn <= 1e-10:
            break
        trial = {
            p_spec.name: p_spec.clamp(
                pf_p[p_spec.name] - pf_rad * (g_s / gn) * p_spec.span
            )
            for p_spec, g_s in zip(problem.parameters, g_vec, strict=True)
        }
        t_merit, t_obj, t_rxn, t_feas = _pure_fem_merit(trial)
        if t_merit < pf_merit - 1e-9:
            pf_p, pf_merit, pf_obj, pf_rxn, pf_feas = (
                trial,
                t_merit,
                t_obj,
                t_rxn,
                t_feas,
            )
            pf_rad = min(problem.initial_trust_radius * 1.5, pf_rad * 1.15)
            if t_feas and (not best_pf[3] or t_obj < best_pf[1]):
                best_pf = (pf_p.copy(), t_obj, t_rxn, t_feas)
        else:
            pf_rad *= 0.5
        pure_trace.append(
            {
                "iteration": it,
                "parameters": dict(pf_p),
                "fem_objective": pf_obj,
                "fem_reaction_N": pf_rxn,
                "fem_feasible": pf_feas,
            }
        )
    pure_fem_wall = round(time.perf_counter() - t_pf, 2)

    assert guarded_report.best_design is not None
    guarded_obj = guarded_report.best_design.fem_objective
    pure_obj = best_pf[1]
    obj_gap_rel = abs(guarded_obj - pure_obj) / max(abs(pure_obj), 1e-12)
    fem_reduction_ratio = pure_fem_calls / max(guarded_report.fem_evaluations, 1)
    init_area = _area(0.70, 3.40)

    print(
        f"  Initial design:     R=0.70, H=3.40 -> Area={init_area:.3f} mm^2 (min F_x >= {f_min_n:.1f} N)"
    )
    print(
        f"  FEM-Guarded AI:     status={guarded_report.status.value}, "
        f"R={guarded_report.best_design.parameters['hole_radius']:.3f}, "
        f"H={guarded_report.best_design.parameters['plate_height']:.3f} -> "
        f"FEM Area={guarded_obj:.3f} mm^2, FEM F_x={guarded_report.best_design.fem_constraints['reaction_force']:.2f} N "
        f"({guarded_report.fem_evaluations} FEM calls, {guarded_report.surrogate_evaluations} surr calls, {guarded_wall}s)"
    )
    print(
        f"  Pure-FEM Baseline:  R={best_pf[0]['hole_radius']:.3f}, "
        f"H={best_pf[0]['plate_height']:.3f} -> "
        f"FEM Area={pure_obj:.3f} mm^2, FEM F_x={best_pf[2]:.2f} N "
        f"({pure_fem_calls} FEM calls, {pure_fem_wall}s)"
    )
    print(
        f"  Comparison:         objective gap={obj_gap_rel * 100:.2f}%, "
        f"FEM call reduction={fem_reduction_ratio:.2f}x, refusal_verified={refusal_verified}"
    )

    return {
        "status": "DIAGNOSTIC_PILOT",
        "claim_level": "NOT_VERIFIED",
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "surrogate_pretrain_wall_seconds": surrogate_pretrain_wall,
        "initial_area_mm2": init_area,
        "reaction_force_limit_N": f_min_n,
        "refusal_verified": refusal_verified,
        "guarded_optimization": {
            "wall_seconds": guarded_wall,
            "report": guarded_report.document(),
        },
        "pure_fem_baseline": {
            "wall_seconds": pure_fem_wall,
            "fem_evaluations": pure_fem_calls,
            "best_parameters": best_pf[0],
            "best_fem_objective_mm2": pure_obj,
            "best_fem_reaction_N": best_pf[2],
            "best_fem_feasible": best_pf[3],
            "trace": pure_trace,
        },
        "comparison": {
            "fem_verified_objective_rel_gap": obj_gap_rel,
            "fem_evaluation_reduction_ratio": fem_reduction_ratio,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FormaMech CPU diagnostic pilots.")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=("pilot1", "pilot2", "pilot3", "pilot4", "pilot5"),
        default=None,
        help="Optional subset of pilots to run; merges into existing report JSON.",
    )
    args = parser.parse_args()
    selected = (
        set(args.only)
        if args.only
        else {"pilot1", "pilot2", "pilot3", "pilot4", "pilot5"}
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "cpu_pilots_report.json"
    if args.only and out_path.exists():
        report: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        report = {
            "note": "CPU-sized diagnostic pilots covering P1-P6; NOT a sealed holdout qualification."
        }

    if "pilot1" in selected:
        report["pilot_1_p1_p4"] = run_pilot_1_p1_p4()
    if "pilot2" in selected:
        report["pilot_2_p2"] = run_pilot_2_p2()
    if "pilot3" in selected:
        report["pilot_3_p6_f1"] = run_pilot_3_p6_f1(OUTPUT_DIR / "checkpoints")
    if "pilot4" in selected:
        report["pilot_4_p3_active_fem"] = run_pilot_4_p3_active_fem()
    if "pilot5" in selected:
        report["pilot_5_p5_design_opt"] = run_pilot_5_p5_design_opt()

    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nSaved complete pilot evidence to {out_path}")


if __name__ == "__main__":
    main()
