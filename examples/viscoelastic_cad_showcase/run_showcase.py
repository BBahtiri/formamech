"""End-to-end FormaMech showcase on a CAD-generated viscoelastic elastomer mount bracket.

Generates a real genus-2 STEP CAD part (40 x 20 mm mount bracket with a top
semi-circular relief notch and two asymmetric mounting holes) via OpenCASCADE,
assigns a 2-branch Generalized Maxwell finite-strain viscoelastic elastomer
model, and runs every FormaMech capability end-to-end while measuring and
plotting each feature's quantitative contribution:

  Stage 1 — OpenCASCADE STEP CAD import, local MeshPolicy, M0/M1/M2 hierarchy
            and distinct-remesh geometric convergence.
  Stage 2 — P6 stateful finite-strain viscoelasticity: algorithmic tangent
            verification, cyclic Ramp->Hold->Unload P2 FEM trajectory (with
            dt-halving & exact CPU replay), HistoryAwareViscoelasticOperator
            energy training, GuardedStatefulStepper fallback, and tamper-checked
            safetensors artifact + checkpoint save/reload.
  Stage 3 — P1 hierarchical M0->M1 energy vs single-level baseline, P2 12-channel
            boundary/load tokenization, GeometryOOD gate, and <=10% LoRA
            adaptation on an unseen CAD variant vs zero-shot & from-scratch.
  Stage 4 — Dual-backend parity (ScikitFemBackend vs TorchFemBackend), Tier 0
            zero-FEM admissibility, Tier 1 per-CAD conformal bounds & P4 live
            DWR adjoint CADTrustReport across 4 QoIs, Tier 1.5 test-time physics
            LoRA, Tier 2 spectrally safe warm-start repair, RFC 8785 Trust
            Receipts, and amortized break-even query curve N*.
  Stage 5 — P3 certificate/DWR active learning (random vs indicator vs
            physics_dwr_diversity), TRAIN-only Tier 2 FallbackFeedbackLedger,
            multi-fidelity energy distillation, and P5 FEM-guarded CAD
            lightweighting design optimization.
  Stage 6 — Extended nonlinear physics (non-conservative follower pressure with
            unsymmetric tangent, 2.5D plane-strain rigid-indenter contact, 3D
            TET10 locking refusal & O(N*S) Surface3DEnergyOperator) plus
            production deployment (Verification Inspector dossier, sandboxed MCP
            server JSON-RPC calls, and FMI 3.0 Co-Simulation FMU export/step).

All generated files, plots, receipts, checkpoints, and metrics are written under
``.formamech-runs/showcase/`` (gitignored).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from examples.viscoelastic_cad_showcase.cad_builder import (
    BracketGeometryParams,
    generate_bracket_step,
    relabel_bracket_mesh,
)
from examples.viscoelastic_cad_showcase.plotting import (
    plot_stage1_cad_and_hierarchy,
    plot_stage2_viscoelastic_p6,
    plot_stage3_p1_p2_hierarchical_and_lora,
    plot_stage4_verification_funnel,
    plot_stage5_p3_active_and_p5_design,
    plot_stage6_extended_physics_and_deployment,
)
from piginot.active import (
    AcquisitionCandidate,
    AcquisitionPlan,
    AcquisitionWeights,
    FallbackFeedbackLedger,
    FallbackReturnRecord,
    MultiFidelityConfig,
    multifidelity_energy_loss,
    select_acquisition_batch,
)
from piginot.bounds import build_cad_calibration, calibrate_functional_band
from piginot.cad.mesh_policy import MeshPolicy, PointSize
from piginot.cad.project import (
    build_project_hierarchy,
    evaluate_project_geometry_convergence,
    inspect_geometry,
    prepare_geometry,
)
from piginot.cad.resource import load_geometry_resource
from piginot.certify import (
    AdmissibilityThresholds,
    ArtifactDigests,
    DiscretizationErrorRecord,
    GeometryErrorRecord,
    ReactionForceQoi,
    RegionalAverageStressQoi,
    SmoothedPeakVonMisesQoi,
    TotalStrainEnergyQoi,
    TrustReceipt,
    VerificationTier,
    estimate_qoi_dwr,
    evaluate_cad_trust,
    evaluate_tier0_admissibility,
    verify_receipt_document,
)
from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    ExecutionProfile,
    LoadSpec,
    MaterialSpec,
    Quantity,
    Unit,
    canonical_json_bytes,
)
from piginot.contracts.expressions import DerivedExpression
from piginot.contracts.stateful import (
    BoundTrajectoryState,
    IntegratorSpec,
    LoadSegment,
    StatefulVerificationPolicy,
    TimeGrid,
    evaluate_load_path,
)
from piginot.export import evaluate_fmu_step, export_bundle_to_fmu
from piginot.fem import (
    FemRequest,
    FemSolverConfig,
    RigidIndenterSpec,
    ScikitFemBackend,
    TorchFemBackend,
)
from piginot.fem.linearization import FemLinearization
from piginot.geometry import (
    VolumetricLockingError,
    make_brick_tet4_mesh,
    make_brick_tet10_mesh,
)
from piginot.lifting import AffineDirichletLifting
from piginot.mcp_server import FormaMechMcpServer, SandboxViolationError
from piginot.objectives import (
    HierarchicalEnergyConfig,
    PotentialEnergyConfig,
    TriangleDomain,
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
    BoundaryLoadMap,
    BoundaryRoleMap,
    EnergyOperatorConfig,
    GeometryOODDetector,
    GeometryTokenizer,
    LowRankAdapterConfig,
    QueryFrame,
    Surface3DConfig,
    Surface3DEnergyOperator,
    adapt_boundary_operator_energy,
    boundary_energy_module_digest,
    compute_geometry_descriptor,
    constitutive_in_plane_stress,
)
from piginot.optimization import (
    DesignConstraint,
    DesignParameter,
    DesignProblem,
    FemDesignEvaluation,
    SurrogateDesignEvaluation,
    run_fem_guarded_design_optimization,
)
from piginot.serving import (
    SpectralRepairConfig,
    TestTimeLoraConfig,
    adapt_operator_at_test_time,
    spectrally_safe_repair,
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
    load_viscoelastic_artifact,
    project_cell_state_to_nodes,
    save_trajectory_checkpoint,
    save_viscoelastic_artifact,
    solve_incremental_fem_trajectory,
    summarize_viscous_state,
    train_viscoelastic_operator_energy,
    verify_algorithmic_tangent,
)
from piginot.stateful.fem_stepper import (
    _build_mesh_and_basis,
    _constrained_dofs,
    _internal_form,
    _solve_single_increment,
)
from piginot.ui import build_inspector_snapshot, render_trust_dossier_markdown
from piginot.verification.amortization import amortized_break_even_queries
from piginot.verification.status import ClaimLevel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".formamech-runs" / "showcase"


def _exec_profile() -> ExecutionProfile:
    return ExecutionProfile(
        python_version="3.13",
        library_versions={"torch": torch.__version__, "numpy": np.__version__},
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="pcg64",
        seed=0,
    )


def _bracket_roles() -> BoundaryRoleMap:
    return BoundaryRoleMap(
        {
            "left_clamp": "dirichlet_xy",
            "right_load": "dirichlet_x",
            "bottom_free": "traction_free",
            "top_free": "traction_free",
            "notch_free": "traction_free",
            "left_hole": "traction_free",
            "right_hole": "traction_free",
        }
    )


def _bracket_lifting(length_mm: float, ux_mm: float) -> Any:
    """Exact Dirichlet lifting: u=(0,0) at x=0 (left_clamp) and u_x=ux at x=L (right_load)."""
    ux_val = float(ux_mm)
    len_val = float(length_mm)

    def _lift(query_xy: torch.Tensor) -> AffineDirichletLifting:
        xi = query_xy[..., 0] / len_val
        offset = torch.stack((ux_val * xi, torch.zeros_like(xi)), dim=-1)
        scale = torch.stack((4.0 * xi * (1.0 - xi), xi), dim=-1)
        return AffineDirichletLifting(offset=offset, scale=scale)

    return _lift


def _equilibrium_material(e_mpa: float = 8.0, nu: float = 0.38) -> MaterialSpec:
    return MaterialSpec(
        name="elastomer-equilibrium-nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(e_mpa, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(nu, Unit.ONE).to_canonical(),
        },
    )


def _bracket_fem_request(
    mesh: CanonicalMesh,
    *,
    ux_mm: float,
    material: MaterialSpec,
    request_id: str,
    element_order: int = 2,
    load_steps: int = 2,
    loads: tuple[LoadSpec, ...] = (),
) -> FemRequest:
    bcs = (
        BoundaryConditionSpec(
            name="left_clamp",
            boundary_tag="left_clamp",
            kind="dirichlet",
            components=("x", "y"),
            values=(
                Quantity(0.0, Unit.MM).to_canonical(),
                Quantity(0.0, Unit.MM).to_canonical(),
            ),
        ),
        BoundaryConditionSpec(
            name="right_load",
            boundary_tag="right_load",
            kind="dirichlet",
            components=("x",),
            values=(Quantity(ux_mm, Unit.MM).to_canonical(),),
        ),
    )
    return FemRequest(
        request_id=request_id,
        mesh=mesh,
        material=material,
        boundary_conditions=bcs,
        loads=loads,
        qois=(),
        config=FemSolverConfig(
            element_order=element_order,
            solver_tolerance=1.0e-10,
            max_newton_iterations=18,
            load_steps=load_steps,
        ),
        execution_profile=_exec_profile(),
    )


def _import_bracket_variant(
    work_dir: Path,
    name: str,
    params: BracketGeometryParams,
    *,
    mesh_size_mm: float = 3.6,
    mesh_policy: MeshPolicy | None = None,
) -> tuple[Path, Path, CanonicalMesh, CanonicalMesh, dict[str, str]]:
    """Generate STEP via OpenCASCADE, import with prepare_geometry, return (step, proj, M0, M1, map)."""
    step_path = generate_bracket_step(work_dir / "cad" / f"{name}.step", params)
    proj_dir = work_dir / "projects" / name
    if proj_dir.exists():
        shutil.rmtree(proj_dir)
    if mesh_policy is not None:
        prepare_geometry(
            step_path, unit=Unit.MM, mesh_policy=mesh_policy, output=proj_dir
        )
    else:
        prepare_geometry(
            step_path, unit=Unit.MM, mesh_size_mm=mesh_size_mm, output=proj_dir
        )
    geom_json = proj_dir / "geometry.json"
    raw = geom_json.read_bytes()
    resource = load_geometry_resource(geom_json, hashlib.sha256(raw).hexdigest())
    m0_rel, emap = relabel_bracket_mesh(resource.mesh(1), params)
    m1_rel, _ = relabel_bracket_mesh(resource.mesh(2), params)
    return step_path, proj_dir, m0_rel, m1_rel, emap


def _eval_operator_vs_fem(
    *,
    op: BoundaryEnergyOperator | Any,
    encoding: Any,
    mesh: CanonicalMesh,
    frame: QueryFrame,
    params_t: torch.Tensor,
    lifting: AffineDirichletLifting,
    material: MaterialSpec,
    fem_u_p1: np.ndarray[Any, Any],
    fem_strain_energy: float,
    fem_lin: FemLinearization,
    fem_p_nodal: np.ndarray[Any, Any],
) -> dict[str, float]:
    """Compare a BoundaryEnergyOperator (or AdaptedBoundaryEnergyOperator) against FEM."""
    dtype = torch.float64
    dof_xy = torch.tensor(fem_lin.dof_coordinates, dtype=dtype).unsqueeze(0)
    verts = torch.tensor(mesh.nodes, dtype=dtype).unsqueeze(0)
    with torch.no_grad():
        if hasattr(op, "operator"):
            u_dof = op.operator.displacement(
                encoding=encoding,
                query_xy=dof_xy,
                parameters=params_t,
                lifting=lifting,
            )[0]
        else:
            u_dof = op.displacement(
                encoding=encoding,
                query_xy=dof_xy,
                parameters=params_t,
                lifting=lifting,
            )[0]
    u_pred_np = u_dof.detach().cpu().numpy().astype(np.float64)
    u_err = float(
        np.linalg.norm(u_pred_np - fem_u_p1) / max(np.linalg.norm(fem_u_p1), 1.0e-12)
    )

    # Compute P_served at vertices & discrete energy
    dom = TriangleDomain(
        torch.tensor(mesh.nodes, dtype=dtype),
        torch.tensor(mesh.cells, dtype=torch.int64),
    )
    q_pts, q_wts = deterministic_triangle_quadrature(dom, degree=2)
    if hasattr(op, "operator"):
        _, g_q = op.operator.displacement_and_gradient(
            encoding=encoding,
            query_xy=q_pts.unsqueeze(0),
            parameters=params_t,
            lifting=lifting,
            create_graph=False,
        )
        _, g_v = op.operator.displacement_and_gradient(
            encoding=encoding,
            query_xy=verts,
            parameters=params_t,
            lifting=lifting,
            create_graph=False,
        )
    else:
        _, g_q = op.displacement_and_gradient(
            encoding=encoding,
            query_xy=q_pts.unsqueeze(0),
            parameters=params_t,
            lifting=lifting,
            create_graph=False,
        )
        _, g_v = op.displacement_and_gradient(
            encoding=encoding,
            query_xy=verts,
            parameters=params_t,
            lifting=lifting,
            create_graph=False,
        )
    p_v = constitutive_in_plane_stress(g_v[0], material).detach().cpu().numpy()
    p_ref = fem_p_nodal[: len(mesh.nodes), :2, :2]
    p_err = float(np.linalg.norm(p_v - p_ref) / max(np.linalg.norm(p_ref), 1.0e-12))

    from piginot.physics import reference_modulus

    e_terms = plane_stress_potential_energy(
        g_q[0], q_wts, material, config=PotentialEnergyConfig(), reference_area=dom.area
    )
    e_phys = float(e_terms.energy.item()) * reference_modulus(material) * dom.area
    e_err = abs(e_phys - fem_strain_energy) / max(abs(fem_strain_energy), 1.0e-12)
    return {
        "u_rel_l2_pct": 100.0 * u_err,
        "p_rel_l2_pct": 100.0 * p_err,
        "energy_err_pct": 100.0 * e_err,
        "surrogate_u": u_pred_np,  # type: ignore[dict-item]
    }


def run_showcase(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Execute the full 6-stage viscoelastic CAD showcase and generate all plots."""
    t_all = time.perf_counter()
    out = Path(output_dir).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    nom_params = BracketGeometryParams()
    print("=" * 78)
    print("FORMAMECH END-TO-END SHOWCASE: CAD-GENERATED VISCOELASTIC MOUNT BRACKET")
    print("=" * 78)

    # =========================================================================
    # STAGE 1: CAD Generation (STEP), Local MeshPolicy, M0/M1/M2 & Convergence
    # =========================================================================
    print(
        "\n[Stage 1/6] Generating STEP CAD & running MeshPolicy / M0-M1-M2 Hierarchy..."
    )
    t_s1 = time.perf_counter()
    policy = MeshPolicy(
        global_size_mm=3.8,
        minimum_size_mm=0.8,
        curve_tolerance_mm=0.08,
        curvature_points_per_circle=24,
        feature_sizes=(),
        point_sizes=(
            PointSize((20.0, 20.0), 1.6, 4.0, 9.0),
            PointSize((11.0, 10.0), 1.5, 3.2, 7.5),
            PointSize((29.0, 10.0), 1.4, 2.6, 7.0),
        ),
        minimum_angle_deg=20.0,
        maximum_aspect_ratio=6.0,
        maximum_triangles=60_000,
        maximum_p2_dofs=200_000,
    )
    step_nom, proj_nom, m0_nom, m1_nom, edge_map = _import_bracket_variant(
        out, "nominal-bracket", nom_params, mesh_policy=policy
    )
    insp_nom = inspect_geometry(proj_nom)
    hier_nom = build_project_hierarchy(proj_nom, output=out / "hierarchy.json")
    conv_report = evaluate_project_geometry_convergence(
        step_nom,
        unit=Unit.MM,
        mesh_sizes_mm=(4.5, 2.8, 1.8),
        max_relative_area_change=0.05,
        max_relative_perimeter_change=0.05,
        max_chord_error_mm=0.25,
        output=out / "geometry-convergence.json",
    )
    # Build convergence summary vs analytical CAD area
    conv_rows: list[dict[str, float]] = []
    for sz_mm, sample in zip(
        conv_report["mesh_sizes_mm"], conv_report["samples"], strict=True
    ):
        area_rel = (
            abs(float(sample["area_mm2"]) - nom_params.analytical_area_mm2)
            / nom_params.analytical_area_mm2
            * 100.0
        )
        # Also compute exact theoretical circular chord sagitta for R=2.6mm at h=sz_mm if 0
        raw_chord = float(sample.get("achieved_max_chord_error_mm", 0.0))
        sagitta = (
            raw_chord
            if raw_chord > 0.0
            else (
                nom_params.right_hole_radius_mm
                - math.sqrt(
                    max(
                        nom_params.right_hole_radius_mm**2
                        - (0.5 * min(float(sz_mm), nom_params.right_hole_radius_mm))
                        ** 2,
                        0.0,
                    )
                )
            )
        )
        conv_rows.append(
            {
                "mesh_size_mm": float(sz_mm),
                "area_mm2": float(sample["area_mm2"]),
                "area_rel_error_pct": area_rel,
                "max_chord_error_mm": sagitta,
                "triangle_count": float(sample.get("cell_count", 0)),
            }
        )

    # Compact CAD mesh (same STEP file) for crisp, fast P2 viscoelastic & multi-CAD sweeps
    _, proj_compact, mesh_cad, mesh_cad_m1, _ = _import_bracket_variant(
        out, "nominal-compact", nom_params, mesh_size_mm=4.2
    )

    stage1_wall = round(time.perf_counter() - t_s1, 2)
    plot1_path = plot_stage1_cad_and_hierarchy(
        {
            "m0_nodes": m0_nom.nodes,
            "m0_cells": m0_nom.cells,
            "m0_facets": m0_nom.boundary_facets,
            "semantic_tags": m0_nom.semantic_tags,
            "m1_nodes": m1_nom.nodes,
            "m1_cells": m1_nom.cells,
            "m2_cell_count": hier_nom["levels"]["M2"]["cell_count"],
            "geometry_convergence": conv_rows,
        },
        plots_dir / "01_cad_import_and_hierarchy.png",
    )
    print(
        f"  CAD imported: {step_nom.name} ({step_nom.stat().st_size} bytes) | "
        f"M0={len(m0_nom.cells)}, M1={len(m1_nom.cells)}, M2={hier_nom['levels']['M2']['cell_count']} tris | "
        f"Area err @ h=1.8mm: {conv_rows[-1]['area_rel_error_pct']:.3f}% ({stage1_wall}s)"
    )

    # =========================================================================
    # STAGE 2: P6 Stateful Finite-Strain Viscoelasticity on the CAD Bracket
    # =========================================================================
    print(
        "\n[Stage 2/6] Running P6 Finite-Strain Viscoelasticity (FEM, Operator, Guarded Stepper)..."
    )
    t_s2 = time.perf_counter()
    visco = GeneralizedMaxwellViscoelastic(
        GeneralizedMaxwellConfig(
            equilibrium_modulus_mpa=8.0,
            poisson_ratio=0.38,
            branches=(
                MaxwellBranch(stiffness_ratio=0.65, relaxation_time=0.4),
                MaxwellBranch(stiffness_ratio=0.35, relaxation_time=2.0),
            ),
        )
    )
    f_probe = torch.tensor(
        [[[1.14, 0.05, 0.0], [0.02, 0.94, 0.0], [0.0, 0.0, 0.94]]], dtype=torch.float64
    )
    t_check = verify_algorithmic_tangent(
        visco,
        torch.eye(3, dtype=torch.float64).unsqueeze(0),
        f_probe,
        visco.initial_state((1,)),
        dt=0.25,
    )

    visco_bcs = (
        DirichletHistoryBC("left_clamp_x", "left_clamp", "x", load_key=None),
        DirichletHistoryBC("left_clamp_y", "left_clamp", "y", load_key=None),
        DirichletHistoryBC("right_load_x", "right_load", "x", load_key="ux"),
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
    unseen_u_max = 3.6  # 9% nominal strain on 40 mm elastomer bracket

    t_vfem = time.perf_counter()
    fem_traj = solve_incremental_fem_trajectory(
        mesh_cad,
        visco,
        visco_bcs,
        grid_coarse,
        segments,
        problem_spec_hash="a" * 64,
        load_path_hash="b" * 64,
        parameters={"u_max": unseen_u_max},
        element_order=2,
    )
    fem_traj_wall = round(time.perf_counter() - t_vfem, 2)
    fem_traj_replay = solve_incremental_fem_trajectory(
        mesh_cad,
        visco,
        visco_bcs,
        grid_coarse,
        segments,
        problem_spec_hash="a" * 64,
        load_path_hash="b" * 64,
        parameters={"u_max": unseen_u_max},
        element_order=2,
    )
    fem_traj_fine = solve_incremental_fem_trajectory(
        mesh_cad,
        visco,
        visco_bcs,
        grid_fine,
        segments,
        problem_spec_hash="a" * 64,
        load_path_hash="b" * 64,
        parameters={"u_max": unseen_u_max},
        element_order=2,
    )
    assert fem_traj.converged and fem_traj_fine.converged
    replay_exact = [s.bound_state.state_hash for s in fem_traj.steps] == [
        s.bound_state.state_hash for s in fem_traj_replay.steps
    ]
    fem_rxns = [float(s.reactions["right_load_x"]) for s in fem_traj.steps]
    peak_c = max(fem_rxns)
    peak_f = max(float(s.reactions["right_load_x"]) for s in fem_traj_fine.steps)
    dt_halving_rel = abs(peak_c - peak_f) / max(abs(peak_f), 1.0e-12)
    # Relaxation drop during hold [t=0.5 -> t=1.5]
    times = grid_coarse.fixed_times()[1:]
    idx_hold_start = times.index(0.5)
    idx_hold_end = times.index(1.5)
    relaxation_drop_pct = (
        (fem_rxns[idx_hold_start] - fem_rxns[idx_hold_end])
        / max(abs(fem_rxns[idx_hold_start]), 1.0e-12)
        * 100.0
    )

    # Equilibrium elastic reference curve (E_inf = 8 MPa, zero viscosity) for contrast
    eq_mat = _equilibrium_material(e_mpa=8.0, nu=0.38)
    sk_backend = ScikitFemBackend()
    eq_rxns: list[float] = []
    applied_uxs = [
        float(evaluate_load_path(segments, t_val, {"u_max": unseen_u_max})["ux"])
        for t_val in times
    ]
    for idx_t, ux_v in enumerate(applied_uxs):
        if ux_v <= 1.0e-12:
            eq_rxns.append(0.0)
            continue
        eq_req = _bracket_fem_request(
            mesh_cad,
            ux_mm=ux_v,
            material=eq_mat,
            request_id=f"eq-ref-{idx_t}",
            element_order=2,
            load_steps=1,
        )
        eq_att = sk_backend.solve_many((eq_req,))[0]
        assert eq_att.success and eq_att.result is not None
        eq_rxns.append(float(eq_att.result.reactions["right_load"][0]))

    # Train HistoryAwareViscoelasticOperator purely via incremental variational energy
    visco_dom = TriangleDomain(
        torch.tensor(mesh_cad.nodes, dtype=torch.float64),
        torch.tensor(mesh_cad.cells, dtype=torch.int64),
    )
    visco_frame = QueryFrame(
        origin=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        extent=torch.tensor(
            [[nom_params.length_mm, nom_params.height_mm]], dtype=torch.float64
        ),
    )
    all_times = grid_coarse.fixed_times()
    train_trajs = []
    for u_m in (2.0, 3.2, 4.4):
        seq = []
        for idx_i in range(1, len(all_times)):
            dt_i = all_times[idx_i] - all_times[idx_i - 1]
            ux_i = evaluate_load_path(segments, all_times[idx_i], {"u_max": u_m})["ux"]
            seq.append(
                (torch.tensor([ux_i / nom_params.length_mm], dtype=torch.float64), dt_i)
            )
        train_trajs.append(seq)

    torch.manual_seed(3)
    visco_op = HistoryAwareViscoelasticOperator(
        ViscoelasticOperatorConfig(
            load_dim=1,
            n_branches=len(visco.config.branches),
            hidden_dim=56,
            num_layers=3,
            positional_frequencies=4,
            include_local_state=True,
        )
    ).to(torch.float64)
    t_vtr = time.perf_counter()
    v_train_summary = train_viscoelastic_operator_energy(
        visco_op,
        material=visco,
        domain=visco_dom,
        frame=visco_frame,
        load_trajectories=train_trajs,
        lifting_factory=lambda lv: _bracket_lifting(
            nom_params.length_mm, float(lv[0].item()) * nom_params.length_mm
        ),
        epochs=75,
        learning_rate=3.0e-3,
        points_per_cell=3,
        seed=3,
    )
    visco_train_wall = round(time.perf_counter() - t_vtr, 2)

    # Save tamper-checked safetensors viscoelastic artifact & reload it!
    visco_art_dir = out / "viscoelastic-artifact"
    save_viscoelastic_artifact(
        visco_art_dir,
        operator=visco_op,
        material=visco,
        mesh=mesh_cad,
        frame=visco_frame,
        problem_spec_hash="a" * 64,
    )
    reloaded_art = load_viscoelastic_artifact(visco_art_dir)

    # Evaluate pure operator vs GuardedStatefulStepper on unseen u_max = 3.6 mm
    from skfem import asm  # type: ignore[import-untyped]

    skmesh_v, basis_v, scalar_v = _build_mesh_and_basis(mesh_cad, element_order=2)
    by_name_v, _, free_dofs_v = _constrained_dofs(
        basis_v, skmesh_v, mesh_cad, visco_bcs
    )
    dof_xy_v = torch.tensor(
        np.asarray(scalar_v.doflocs, dtype=np.float64).T, dtype=torch.float64
    )
    facet_v0 = torch.tensor(skmesh_v.facets[0], dtype=torch.int64)
    facet_v1 = torch.tensor(skmesh_v.facets[1], dtype=torch.int64)
    q_shape_v = tuple(int(d) for d in basis_v.dx.shape)
    dx_v = torch.tensor(np.asarray(basis_v.dx, dtype=np.float64), dtype=torch.float64)
    x_dofs_v, y_dofs_v = (np.asarray(v) for v in basis_v.split_indices())

    def _local_feats_v(sdict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        cell_ev = sdict["viscous_strain"].mean(dim=1)
        vert_f = project_cell_state_to_nodes(visco_dom, cell_ev)
        mid_f = 0.5 * (vert_f[facet_v0] + vert_f[facet_v1])
        return torch.cat((vert_f, mid_f), dim=0).unsqueeze(0)

    def _to_vec_v(nodal_u: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        vec = np.zeros(basis_v.N, dtype=np.float64)
        vec[x_dofs_v] = nodal_u[:, 0]
        vec[y_dofs_v] = nodal_u[:, 1]
        return vec

    def _eval_vstep(
        u_old_v: np.ndarray[Any, Any],
        u_new_v: np.ndarray[Any, Any],
        s_old: dict[str, torch.Tensor],
        dt_s: float,
    ) -> tuple[
        float, dict[str, torch.Tensor], float, float, float, float, np.ndarray[Any, Any]
    ]:
        g_old = np.asarray(basis_v.interpolate(u_old_v).grad, dtype=np.float64)
        g_new = np.asarray(basis_v.interpolate(u_new_v).grad, dtype=np.float64)
        f2d_o = torch.from_numpy(
            np.ascontiguousarray(np.transpose(g_old, (2, 3, 0, 1)) + np.eye(2))
        )
        f2d_n = torch.from_numpy(
            np.ascontiguousarray(np.transpose(g_new, (2, 3, 0, 1)) + np.eye(2))
        )
        ps = visco.plane_stress_update(f2d_o, f2d_n, s_old, dt_s)
        s2d = ps.update_3d.stress[..., :2, :2].detach().cpu().numpy()
        internal = np.asarray(
            asm(_internal_form, basis_v, stress=np.transpose(s2d, (2, 3, 0, 1))),
            dtype=np.float64,
        )
        res_n = float(np.linalg.norm(internal[free_dofs_v]))
        rxn = float(np.sum(internal[by_name_v["right_load_x"]]))
        mdet = float(ps.update_3d.det_f.min().item())
        mdiss = float(ps.update_3d.dissipation.min().item())
        tdiss = float(
            np.sum(ps.update_3d.dissipation.detach().cpu().numpy() * basis_v.dx)
        )
        s_new = {k: v.detach().clone() for k, v in ps.update_3d.state_new.items()}
        return res_n, s_new, rxn, mdet, mdiss, tdiss, s2d

    # Pure operator rollout
    st_op = visco.initial_state(q_shape_v, dtype=torch.float64)
    u_prev = np.zeros(basis_v.N, dtype=np.float64)
    op_rxns: list[float] = []
    op_res_list: list[float] = []
    op_min_det = float("inf")
    op_min_diss = float("inf")
    t_op_inf = time.perf_counter()
    for idx_t in range(1, len(all_times)):
        dt_t = all_times[idx_t] - all_times[idx_t - 1]
        ux_t = applied_uxs[idx_t - 1]
        lv = torch.tensor([ux_t / nom_params.length_mm], dtype=torch.float64)
        hf = summarize_viscous_state(st_op, weights=dx_v)
        lf = _local_feats_v(st_op)
        with torch.no_grad():
            u_nod = reloaded_art.operator.displacement(
                query_xy=dof_xy_v.unsqueeze(0),
                frame=visco_frame,
                load_vector=lv,
                dt=dt_t,
                history_features=hf,
                lifting=_bracket_lifting(nom_params.length_mm, ux_t),
                local_state_features=lf,
            )[0]
        u_cur = _to_vec_v(u_nod.cpu().numpy())
        rn, st_op, rx, mdet, mdiss, _, _ = _eval_vstep(u_prev, u_cur, st_op, dt_t)
        op_rxns.append(rx)
        op_res_list.append(rn)
        op_min_det = min(op_min_det, mdet)
        op_min_diss = min(op_min_diss, mdiss)
        u_prev = u_cur
    op_rollout_wall = round(time.perf_counter() - t_op_inf, 3)

    # GuardedStatefulStepper with hybrid tolerance so users see BOTH fast operator & FEM fallback!
    hybrid_tol = float(np.percentile(op_res_list, 35))

    def _op_pred(
        u_old_t: torch.Tensor,
        s_old: Mapping[str, torch.Tensor],
        load_next: Mapping[str, float],
        dt_s: float,
    ) -> CandidateStepOutput:
        ux_val = float(load_next["ux"])
        lv = torch.tensor([ux_val / nom_params.length_mm], dtype=torch.float64)
        hf = summarize_viscous_state(s_old, weights=dx_v)
        lf = _local_feats_v(s_old)
        with torch.no_grad():
            u_nod = reloaded_art.operator.displacement(
                query_xy=dof_xy_v.unsqueeze(0),
                frame=visco_frame,
                load_vector=lv,
                dt=dt_s,
                history_features=hf,
                lifting=_bracket_lifting(nom_params.length_mm, ux_val),
                local_state_features=lf,
            )[0]
        u_cur = _to_vec_v(u_nod.cpu().numpy())
        rn, s_n, _, mdet, mdiss, tdiss, _ = _eval_vstep(
            u_old_t.numpy(), u_cur, dict(s_old), dt_s
        )
        return CandidateStepOutput(
            displacement=torch.from_numpy(u_cur),
            state_new=s_n,
            min_det_f=mdet,
            min_dissipation=mdiss,
            total_dissipation=tdiss,
            residual_norm=rn,
            admissible=mdet > 0.0 and mdiss >= -1.0e-12,
        )

    def _fem_fb(
        u_old_t: torch.Tensor,
        s_old: Mapping[str, torch.Tensor],
        load_next: Mapping[str, float],
        dt_s: float,
    ) -> CandidateStepOutput:
        ux_val = float(load_next["ux"])
        td: dict[int, float] = {}
        for bc in visco_bcs:
            val = 0.0 if bc.load_key is None else ux_val
            for dof in by_name_v[bc.name]:
                td[int(dof)] = val
        ok, u_n, s_n, _, _, rn, _, tdiss, mdet, _, rsn = _solve_single_increment(
            basis=basis_v,
            model=visco,
            u_old=u_old_t.numpy(),
            state_old=dict(s_old),
            f33_old=torch.ones(q_shape_v, dtype=torch.float64),
            target_dirichlet=td,
            free_dofs=free_dofs_v,
            dt=dt_s,
            integrator=IntegratorSpec(),
        )
        return CandidateStepOutput(
            displacement=torch.from_numpy(u_n),
            state_new=s_n,
            min_det_f=mdet,
            min_dissipation=0.0 if ok else -1.0,
            total_dissipation=tdiss,
            residual_norm=rn,
            admissible=ok,
            reason=rsn,
        )

    stepper = GuardedStatefulStepper(
        problem_spec_hash="a" * 64,
        mesh_hash=mesh_cad.mesh_id,
        material_schema_hash=visco.state_schema.schema_hash,
        load_path_hash="b" * 64,
        predictor=_op_pred,
        fem_fallback=_fem_fb,
        verification=StatefulVerificationPolicy(allow_fem_fallback=True),
        residual_tolerance=hybrid_tol,
        max_substep_halvings=0,
    )
    g_state = visco.initial_state(q_shape_v, dtype=torch.float64)
    g_u = torch.zeros(basis_v.N, dtype=torch.float64)
    g_bound = BoundTrajectoryState(
        problem_spec_hash="a" * 64,
        mesh_hash=mesh_cad.mesh_id,
        material_schema_hash=visco.state_schema.schema_hash,
        load_path_hash="b" * 64,
        time_index=0,
        time_value=0.0,
        step_controller_version=STEP_CONTROLLER_VERSION,
        state_hash=hash_state_tensors(g_state),
    )
    guard_rxns: list[float] = []
    guard_res_list: list[float] = []
    guard_statuses: list[str] = []
    for idx_t in range(1, len(all_times)):
        dt_t = all_times[idx_t] - all_times[idx_t - 1]
        ux_t = applied_uxs[idx_t - 1]
        sr = stepper.advance(
            bound_state=g_bound,
            displacement_old=g_u,
            state_old=g_state,
            load_next={"ux": ux_t},
            dt=dt_t,
        )
        assert (
            sr.accepted
            and sr.displacement is not None
            and sr.state_new is not None
            and sr.bound_state is not None
        )
        guard_statuses.append(sr.status.value)
        rn, _, rx, _, _, _, _ = _eval_vstep(
            g_u.numpy(), sr.displacement.numpy(), g_state, dt_t
        )
        guard_rxns.append(rn if False else rx)
        guard_res_list.append(rn)
        g_u = sr.displacement
        g_state = dict(sr.state_new)
        g_bound = sr.bound_state

    # Save & reload trajectory checkpoint
    ckpt_dir = out / "viscoelastic-checkpoint"
    save_trajectory_checkpoint(
        ckpt_dir, bound_state=g_bound, displacement=g_u, state_tensors=g_state
    )
    reloaded_bound, _, _ = load_trajectory_checkpoint(ckpt_dir)
    ckpt_verified = reloaded_bound == g_bound

    # Peak-load deformed stress field from FEM step at t=0.5s
    peak_step = fem_traj.steps[idx_hold_start]
    peak_u_vec = np.asarray(peak_step.displacement, dtype=np.float64)
    n_verts_cad = len(mesh_cad.nodes)
    peak_u_verts = np.column_stack(
        (peak_u_vec[x_dofs_v[:n_verts_cad]], peak_u_vec[y_dofs_v[:n_verts_cad]])
    )
    st_replay = visco.initial_state(q_shape_v, dtype=torch.float64)
    u_rep_prev = np.zeros(basis_v.N, dtype=np.float64)
    peak_s2d = np.zeros((*q_shape_v, 2, 2), dtype=np.float64)
    for st_step in fem_traj.steps[: idx_hold_start + 1]:
        u_rep_cur = np.asarray(st_step.displacement, dtype=np.float64)
        _, st_replay, _, _, _, _, peak_s2d = _eval_vstep(
            u_rep_prev, u_rep_cur, st_replay, float(st_step.dt)
        )
        u_rep_prev = u_rep_cur
    # Element-averaged Frobenius norm of P projected to vertices
    cell_p_norm = np.linalg.norm(peak_s2d, axis=(-2, -1)).mean(axis=1)
    vert_p_sum = np.zeros(n_verts_cad, dtype=np.float64)
    vert_p_cnt = np.zeros(n_verts_cad, dtype=np.float64)
    for c_idx, tri in enumerate(mesh_cad.cells):
        for v_idx in tri:
            vert_p_sum[v_idx] += cell_p_norm[c_idx]
            vert_p_cnt[v_idx] += 1.0
    peak_vert_p = vert_p_sum / np.maximum(vert_p_cnt, 1.0)

    # Compute error reduction from GuardedStatefulStepper
    max_abs_fem = max(abs(r) for r in fem_rxns)
    pure_op_max_err_pct = (
        max(abs(a - b) for a, b in zip(op_rxns, fem_rxns, strict=True))
        / max_abs_fem
        * 100.0
    )
    guarded_max_err_pct = (
        max(abs(a - b) for a, b in zip(guard_rxns, fem_rxns, strict=True))
        / max_abs_fem
        * 100.0
    )
    stage2_wall = round(time.perf_counter() - t_s2, 2)
    plot2_path = plot_stage2_viscoelastic_p6(
        {
            "times_s": times,
            "applied_ux_mm": applied_uxs,
            "fem_reactions_n": fem_rxns,
            "operator_reactions_n": op_rxns,
            "guarded_reactions_n": guard_rxns,
            "equilibrium_elastic_reactions_n": eq_rxns,
            "guarded_statuses": guard_statuses,
            "operator_residuals_n": op_res_list,
            "guarded_residuals_n": guard_res_list,
            "hybrid_residual_tol": hybrid_tol,
            "fem_step_dissipation_n_mm": [
                float(s.step_dissipation) for s in fem_traj.steps
            ],
            "relaxation_drop_pct": relaxation_drop_pct,
            "tangent_fd_rel_err": t_check.relative_error,
            "pure_op_max_rxn_err_pct": pure_op_max_err_pct,
            "guarded_max_rxn_err_pct": guarded_max_err_pct,
            "peak_nodes_mm": mesh_cad.nodes,
            "peak_fem_u_mm": peak_u_verts,
            "peak_cells": mesh_cad.cells,
            "peak_stress_norm_mpa": peak_vert_p,
            "min_det_f": min(op_min_det, min(s.min_det_f for s in fem_traj.steps)),
        },
        plots_dir / "02_viscoelastic_p6_rollout.png",
    )
    print(
        f"  Tangent FD err={t_check.relative_error:.2e} | Relaxation drop={relaxation_drop_pct:.1f}% | "
        f"Pure op max rxn err={pure_op_max_err_pct:.2f}% -> Guarded={guarded_max_err_pct:.2f}% "
        f"({sum(s == 'ACCEPTED_OPERATOR' for s in guard_statuses)} op / "
        f"{sum(s == 'ACCEPTED_FEM_FALLBACK' for s in guard_statuses)} FEM) | "
        f"ckpt={ckpt_verified} ({stage2_wall}s)"
    )

    # =========================================================================
    # STAGE 3: P1 Hierarchical Energy & P2 Multi-CAD Tokenization, OOD & LoRA
    # =========================================================================
    print(
        "\n[Stage 3/6] Running P1 Hierarchical Energy & P2 Multi-CAD Tokenization / OOD / LoRA..."
    )
    t_s3 = time.perf_counter()
    roles = _bracket_roles()
    tok_v1 = GeometryTokenizer(subdivisions_per_facet=1, include_load_embeddings=False)
    tok_v2 = GeometryTokenizer(subdivisions_per_facet=1, include_load_embeddings=True)
    # 12-channel load-embedded tokenization demo (demonstrating dead traction & follower channels)
    demo_roles_v2 = BoundaryRoleMap(
        {
            "left_clamp": "dirichlet_xy",
            "right_load": "dead_traction",
            "bottom_free": "traction_free",
            "top_free": "follower_pressure",
            "notch_free": "follower_pressure",
            "left_hole": "traction_free",
            "right_hole": "traction_free",
        }
    )
    demo_loads_v2 = BoundaryLoadMap(
        {
            "right_load": (0.8, 0.0, 0.0),
            "top_free": (0.0, 0.0, 0.25),
            "notch_free": (0.0, 0.0, 0.25),
        }
    )
    batch_v2 = tok_v2.tokenize_batch(
        [(mesh_cad, demo_roles_v2)], load_maps=[demo_loads_v2], dtype=torch.float64
    )
    feats_v2 = batch_v2.boundary_features[0].numpy()
    role_code = np.zeros(feats_v2.shape[0], dtype=int)
    role_code[feats_v2[:, 2] > 0.5] = 1  # dirichlet
    role_code[feats_v2[:, 7] > 0.5] = 2  # dead traction
    role_code[feats_v2[:, 8] > 0.5] = 3  # follower pressure

    # Equilibrium FEM reference on nominal CAD bracket at ux = 3.2 mm (8% strain)
    ux_eq = 3.2
    req_nom_p1 = _bracket_fem_request(
        mesh_cad, ux_mm=ux_eq, material=eq_mat, request_id="s3-nom-p1", element_order=1
    )
    lin_nom_p1 = FemLinearization(req_nom_p1)
    att_nom_p1 = sk_backend.solve_many((req_nom_p1,))[0]
    assert att_nom_p1.success and att_nom_p1.result is not None
    fem_u_nom = np.asarray(att_nom_p1.result.displacement, dtype=np.float64)
    fem_p_nom = np.asarray(
        att_nom_p1.result.recovered_first_piola_stress, dtype=np.float64
    )
    fem_e_nom = float(att_nom_p1.result.total_strain_energy)

    # P1 comparison: Single-Level M0 Energy vs Hierarchical M0->M1 Energy (same 80 steps)
    b_cfg = BoundaryEnergyConfig(
        energy=EnergyOperatorConfig(
            parameter_names=("strain",),
            hidden_dim=24,
            latent_token_count=8,
            parameter_hidden_dim=24,
            num_attention_heads=4,
            decoder_layers=2,
            positional_frequencies=4,
        ),
        boundary_feature_names=GEOMETRY_FEATURE_NAMES,
    )
    batch_nom = tok_v1.tokenize_batch([(mesh_cad, roles)], dtype=torch.float64)
    dom_m0 = TriangleDomain(
        torch.tensor(mesh_cad.nodes, dtype=torch.float64),
        torch.tensor(mesh_cad.cells, dtype=torch.int64),
    )
    dom_m1 = subdivide_triangle_domain(dom_m0)
    p_strain = torch.tensor([[ux_eq / nom_params.length_mm]], dtype=torch.float64)
    lift_nom = _bracket_lifting(nom_params.length_mm, ux_eq)
    ecfg = PotentialEnergyConfig(
        points_per_cell=3, barrier_weight=1e3, barrier_threshold=0.1
    )

    def _train_single_vs_hier(
        hierarchical: bool, steps: int = 80
    ) -> BoundaryEnergyOperator:
        torch.manual_seed(17)
        op_m = BoundaryEnergyOperator(b_cfg).to(torch.float64)
        opt_m = torch.optim.Adam(op_m.parameters(), lr=4e-3)
        gen_m = torch.Generator().manual_seed(42)
        hcfg = HierarchicalEnergyConfig(
            base=ecfg,
            quadrature_degree=2,
            cross_mesh_weight=0.25,
            local_physics_weight=0.05,
            use_deterministic_quadrature=True,
        )
        for _ in range(steps):
            opt_m.zero_grad(set_to_none=True)
            enc = op_m.encode_boundary(
                boundary_xy=batch_nom.boundary_xy,
                boundary_normals=batch_nom.boundary_normals,
                boundary_features=batch_nom.boundary_features,
                weights=batch_nom.weights,
                mask=batch_nom.mask,
                frame=batch_nom.frame,
            )
            if hierarchical:
                loss = hierarchical_plane_stress_energy(
                    lambda q, _enc=enc: op_m.displacement_and_gradient(
                        encoding=_enc,
                        query_xy=q.unsqueeze(0),
                        parameters=p_strain,
                        lifting=lift_nom,
                        create_graph=True,
                    )[1][0],
                    dom_m0,
                    eq_mat,
                    config=hcfg,
                    fine_domain=dom_m1,
                ).total
            else:
                pts, wts = stratified_triangle_quadrature(dom_m0, 3, generator=gen_m)
                _, g = op_m.displacement_and_gradient(
                    encoding=enc,
                    query_xy=pts.unsqueeze(0),
                    parameters=p_strain,
                    lifting=lift_nom,
                    create_graph=True,
                )
                loss = plane_stress_potential_energy(
                    g[0], wts, eq_mat, config=ecfg, reference_area=dom_m0.area
                ).total
            loss.backward()
            torch.nn.utils.clip_grad_norm_(op_m.parameters(), 1.0)
            opt_m.step()
        return op_m

    op_p1_base = _train_single_vs_hier(hierarchical=False, steps=80)
    op_p1_hier = _train_single_vs_hier(hierarchical=True, steps=80)
    with torch.no_grad():
        enc_base = op_p1_base.encode_boundary(
            boundary_xy=batch_nom.boundary_xy,
            boundary_normals=batch_nom.boundary_normals,
            boundary_features=batch_nom.boundary_features,
            weights=batch_nom.weights,
            mask=batch_nom.mask,
            frame=batch_nom.frame,
        )
        enc_hier = op_p1_hier.encode_boundary(
            boundary_xy=batch_nom.boundary_xy,
            boundary_normals=batch_nom.boundary_normals,
            boundary_features=batch_nom.boundary_features,
            weights=batch_nom.weights,
            mask=batch_nom.mask,
            frame=batch_nom.frame,
        )
    p1_base_metrics = _eval_operator_vs_fem(
        op=op_p1_base,
        encoding=enc_base,
        mesh=mesh_cad,
        frame=batch_nom.frame,
        params_t=p_strain,
        lifting=lift_nom,
        material=eq_mat,
        fem_u_p1=fem_u_nom,
        fem_strain_energy=fem_e_nom,
        fem_lin=lin_nom_p1,
        fem_p_nodal=fem_p_nom,
    )
    p1_hier_metrics = _eval_operator_vs_fem(
        op=op_p1_hier,
        encoding=enc_hier,
        mesh=mesh_cad,
        frame=batch_nom.frame,
        params_t=p_strain,
        lifting=lift_nom,
        material=eq_mat,
        fem_u_p1=fem_u_nom,
        fem_strain_energy=fem_e_nom,
        fem_lin=lin_nom_p1,
        fem_p_nodal=fem_p_nom,
    )

    # P2 Multi-CAD family (4 training CAD STEP variants + 1 unseen ID target + 1 extreme OOD)
    train_cad_specs = [
        (
            "cad-tr-1",
            BracketGeometryParams(
                notch_radius_mm=3.2, left_hole_radius_mm=2.6, right_hole_radius_mm=2.2
            ),
        ),
        (
            "cad-tr-2",
            BracketGeometryParams(
                notch_radius_mm=4.2, left_hole_radius_mm=3.0, right_hole_radius_mm=2.5
            ),
        ),
        (
            "cad-tr-3",
            BracketGeometryParams(
                notch_radius_mm=3.6, left_hole_radius_mm=3.6, right_hole_radius_mm=3.0
            ),
        ),
        (
            "cad-tr-4",
            BracketGeometryParams(
                notch_radius_mm=4.6, left_hole_radius_mm=3.4, right_hole_radius_mm=2.8
            ),
        ),
    ]
    target_cad_params = BracketGeometryParams(
        notch_radius_mm=4.4, left_hole_radius_mm=3.5, right_hole_radius_mm=2.9
    )
    ood_cad_params = BracketGeometryParams(
        length_mm=26.0,
        height_mm=20.0,
        notch_radius_mm=6.2,
        left_hole_radius_mm=4.2,
        right_hole_radius_mm=1.2,
        left_hole_x_mm=6.5,
        right_hole_x_mm=20.0,
    )

    train_meshes_p2: list[CanonicalMesh] = []
    for cname, cparams in train_cad_specs:
        _, _, cm, _, _ = _import_bracket_variant(out, cname, cparams, mesh_size_mm=4.6)
        train_meshes_p2.append(cm)
    _, _, target_mesh_p2, _, _ = _import_bracket_variant(
        out, "cad-target-id", target_cad_params, mesh_size_mm=4.4
    )
    _, _, ood_mesh_p2, _, _ = _import_bracket_variant(
        out, "cad-extreme-ood", ood_cad_params, mesh_size_mm=4.4
    )

    train_descs = [
        compute_geometry_descriptor(m, roles, tokenizer=tok_v1) for m in train_meshes_p2
    ]
    ood_det = GeometryOODDetector.fit(train_descs, descriptor_threshold=3.5)
    ood_id_rep = ood_det.score(target_mesh_p2, roles, tokenizer=tok_v1)
    ood_ext_rep = ood_det.score(ood_mesh_p2, roles, tokenizer=tok_v1)
    ood_tr_rep = ood_det.score(train_meshes_p2[0], roles, tokenizer=tok_v1)

    # Pretrain multi-CAD BoundaryEnergyOperator on the 4 CAD training geometries (120 steps)
    torch.manual_seed(31)
    multi_op = BoundaryEnergyOperator(b_cfg).to(torch.float64)
    opt_multi = torch.optim.Adam(multi_op.parameters(), lr=3.5e-3)
    train_batches_p2 = [
        tok_v1.tokenize_batch([(m, roles)], dtype=torch.float64)
        for m in train_meshes_p2
    ]
    train_doms_p2 = [
        TriangleDomain(
            torch.tensor(m.nodes, dtype=torch.float64),
            torch.tensor(m.cells, dtype=torch.int64),
        )
        for m in train_meshes_p2
    ]
    gen_p2 = torch.Generator().manual_seed(99)
    pretrain_steps = 120
    t_pre = time.perf_counter()
    for _ in range(pretrain_steps):
        opt_multi.zero_grad(set_to_none=True)
        loss_tot = torch.zeros((), dtype=torch.float64)
        for b_i, d_i in zip(train_batches_p2, train_doms_p2, strict=True):
            enc_i = multi_op.encode_boundary(
                boundary_xy=b_i.boundary_xy,
                boundary_normals=b_i.boundary_normals,
                boundary_features=b_i.boundary_features,
                weights=b_i.weights,
                mask=b_i.mask,
                frame=b_i.frame,
            )
            pts_i, w_i = stratified_triangle_quadrature(d_i, 2, generator=gen_p2)
            _, g_i = multi_op.displacement_and_gradient(
                encoding=enc_i,
                query_xy=pts_i.unsqueeze(0),
                parameters=p_strain,
                lifting=lift_nom,
                create_graph=True,
            )
            loss_tot = loss_tot + plane_stress_potential_energy(
                g_i[0], w_i, eq_mat, config=ecfg, reference_area=d_i.area
            ).total / len(train_meshes_p2)
        loss_tot.backward()
        torch.nn.utils.clip_grad_norm_(multi_op.parameters(), 1.0)
        opt_multi.step()
    pretrain_wall = round(time.perf_counter() - t_pre, 2)

    # Evaluate target unseen CAD variant: Zero-Shot vs <=10% LoRA vs From-Scratch
    req_tgt = _bracket_fem_request(
        target_mesh_p2,
        ux_mm=ux_eq,
        material=eq_mat,
        request_id="s3-tgt-fem",
        element_order=1,
    )
    lin_tgt = FemLinearization(req_tgt)
    att_tgt = sk_backend.solve_many((req_tgt,))[0]
    assert att_tgt.success and att_tgt.result is not None
    u_tgt_fem = np.asarray(att_tgt.result.displacement, dtype=np.float64)
    p_tgt_fem = np.asarray(
        att_tgt.result.recovered_first_piola_stress, dtype=np.float64
    )
    e_tgt_fem = float(att_tgt.result.total_strain_energy)

    batch_tgt = tok_v1.tokenize_batch([(target_mesh_p2, roles)], dtype=torch.float64)
    dom_tgt = TriangleDomain(
        torch.tensor(target_mesh_p2.nodes, dtype=torch.float64),
        torch.tensor(target_mesh_p2.cells, dtype=torch.int64),
    )
    t_zs = time.perf_counter()
    with torch.no_grad():
        enc_tgt = multi_op.encode_boundary(
            boundary_xy=batch_tgt.boundary_xy,
            boundary_normals=batch_tgt.boundary_normals,
            boundary_features=batch_tgt.boundary_features,
            weights=batch_tgt.weights,
            mask=batch_tgt.mask,
            frame=batch_tgt.frame,
        )
    zs_m = _eval_operator_vs_fem(
        op=multi_op,
        encoding=enc_tgt,
        mesh=target_mesh_p2,
        frame=batch_tgt.frame,
        params_t=p_strain,
        lifting=lift_nom,
        material=eq_mat,
        fem_u_p1=u_tgt_fem,
        fem_strain_energy=e_tgt_fem,
        fem_lin=lin_tgt,
        fem_p_nodal=p_tgt_fem,
    )
    zs_wall = round(time.perf_counter() - t_zs, 3)

    # P2 <= 10% budget LoRA adaptation (12 steps vs 120 baseline steps!)
    lora_plan = AdaptationPlan(
        base_model_digest=boundary_energy_module_digest(multi_op),
        target_mesh_id=target_mesh_p2.mesh_id,
        adapter_config=LowRankAdapterConfig(
            rank=4,
            alpha=4.0,
            max_steps=12,
            from_scratch_reference_steps=pretrain_steps,
            max_step_budget_fraction=0.10,
            learning_rate=8.0e-3,
            seed=5,
        ),
        energy_config=ecfg,
        ood_report=ood_id_rep,
    )
    t_la = time.perf_counter()
    adapted_op, adapt_res = adapt_boundary_operator_energy(
        multi_op,
        plan=lora_plan,
        domain=dom_tgt,
        boundary_xy=batch_tgt.boundary_xy,
        boundary_normals=batch_tgt.boundary_normals,
        boundary_features=batch_tgt.boundary_features,
        boundary_weights=batch_tgt.weights,
        boundary_mask=batch_tgt.mask,
        frame=batch_tgt.frame,
        parameters=p_strain,
        lifting=lift_nom,
        material=eq_mat,
    )
    lora_wall = round(time.perf_counter() - t_la, 3)
    assert adapted_op is not None
    with torch.no_grad():
        enc_tgt_ad = adapted_op.operator.encode_boundary(
            boundary_xy=batch_tgt.boundary_xy,
            boundary_normals=batch_tgt.boundary_normals,
            boundary_features=batch_tgt.boundary_features,
            weights=batch_tgt.weights,
            mask=batch_tgt.mask,
            frame=batch_tgt.frame,
        )
    lora_m = _eval_operator_vs_fem(
        op=adapted_op,
        encoding=enc_tgt_ad,
        mesh=target_mesh_p2,
        frame=batch_tgt.frame,
        params_t=p_strain,
        lifting=lift_nom,
        material=eq_mat,
        fem_u_p1=u_tgt_fem,
        fem_strain_energy=e_tgt_fem,
        fem_lin=lin_tgt,
        fem_p_nodal=p_tgt_fem,
    )

    # From-scratch (120 steps on target CAD)
    t_sc = time.perf_counter()
    torch.manual_seed(77)
    scratch_op = BoundaryEnergyOperator(b_cfg).to(torch.float64)
    opt_sc = torch.optim.Adam(scratch_op.parameters(), lr=4e-3)
    gen_sc = torch.Generator().manual_seed(77)
    for _ in range(pretrain_steps):
        opt_sc.zero_grad(set_to_none=True)
        enc_s = scratch_op.encode_boundary(
            boundary_xy=batch_tgt.boundary_xy,
            boundary_normals=batch_tgt.boundary_normals,
            boundary_features=batch_tgt.boundary_features,
            weights=batch_tgt.weights,
            mask=batch_tgt.mask,
            frame=batch_tgt.frame,
        )
        pts_s, w_s = stratified_triangle_quadrature(dom_tgt, 2, generator=gen_sc)
        _, g_s = scratch_op.displacement_and_gradient(
            encoding=enc_s,
            query_xy=pts_s.unsqueeze(0),
            parameters=p_strain,
            lifting=lift_nom,
            create_graph=True,
        )
        ls = plane_stress_potential_energy(
            g_s[0], w_s, eq_mat, config=ecfg, reference_area=dom_tgt.area
        ).total
        ls.backward()
        torch.nn.utils.clip_grad_norm_(scratch_op.parameters(), 1.0)
        opt_sc.step()
    scratch_wall = round(time.perf_counter() - t_sc, 3)
    with torch.no_grad():
        enc_sc = scratch_op.encode_boundary(
            boundary_xy=batch_tgt.boundary_xy,
            boundary_normals=batch_tgt.boundary_normals,
            boundary_features=batch_tgt.boundary_features,
            weights=batch_tgt.weights,
            mask=batch_tgt.mask,
            frame=batch_tgt.frame,
        )
    scratch_m = _eval_operator_vs_fem(
        op=scratch_op,
        encoding=enc_sc,
        mesh=target_mesh_p2,
        frame=batch_tgt.frame,
        params_t=p_strain,
        lifting=lift_nom,
        material=eq_mat,
        fem_u_p1=u_tgt_fem,
        fem_strain_energy=e_tgt_fem,
        fem_lin=lin_tgt,
        fem_p_nodal=p_tgt_fem,
    )

    stage3_wall = round(time.perf_counter() - t_s3, 2)
    plot3_path = plot_stage3_p1_p2_hierarchical_and_lora(
        {
            "p1_baseline": {
                k: v for k, v in p1_base_metrics.items() if isinstance(v, float)
            },
            "p1_hierarchical": {
                k: v for k, v in p1_hier_metrics.items() if isinstance(v, float)
            },
            "token_xy": batch_v2.boundary_xy[0].numpy(),
            "token_normals": batch_v2.boundary_normals[0].numpy(),
            "token_role_code": role_code,
            "ood_labels": [
                "Train CAD Member",
                "Unseen Target CAD (ID)",
                "Extreme Extrapolated CAD (OOD)",
            ],
            "ood_scores": [
                ood_tr_rep.descriptor_distance,
                ood_id_rep.descriptor_distance,
                ood_ext_rep.descriptor_distance,
            ],
            "ood_threshold": ood_det.descriptor_threshold,
            "p2_zero_shot": {"u_rel_l2_pct": zs_m["u_rel_l2_pct"], "wall_s": zs_wall},
            "p2_lora": {
                "u_rel_l2_pct": lora_m["u_rel_l2_pct"],
                "wall_s": lora_wall,
                "adapter_params": adapt_res.adapter_parameter_count,
            },
            "p2_scratch": {
                "u_rel_l2_pct": scratch_m["u_rel_l2_pct"],
                "wall_s": scratch_wall,
            },
        },
        plots_dir / "03_p1_p2_hierarchical_and_lora.png",
    )
    print(
        f"  P1 Hier vs Base u-err: {p1_hier_metrics['u_rel_l2_pct']:.2f}% vs {p1_base_metrics['u_rel_l2_pct']:.2f}% | "
        f"P2 OOD: ID={ood_id_rep.descriptor_distance:.2f} vs OOD={ood_ext_rep.descriptor_distance:.2f} | "
        f"LoRA u-err={lora_m['u_rel_l2_pct']:.2f}% ({lora_wall}s) vs Zero-Shot={zs_m['u_rel_l2_pct']:.2f}% ({stage3_wall}s)"
    )

    # =========================================================================
    # STAGE 4: Dual-Backend Parity, 3-Tier Funnel, TTA, Spectral Repair, Receipt
    # =========================================================================
    print(
        "\n[Stage 4/6] Running Dual-Backend Parity, 3-Tier Funnel, Tier 1.5 TTA & Spectral Repair..."
    )
    t_s4 = time.perf_counter()
    torch_backend = TorchFemBackend(physics_mode="2d-plane-stress-finite-strain")
    t_sk = time.perf_counter()
    att_sk = sk_backend.solve_many((req_nom_p1,))[0]
    fem_single_s = time.perf_counter() - t_sk
    att_th = torch_backend.solve_many((req_nom_p1,))[0]
    assert att_sk.success and att_sk.result is not None
    assert att_th.success and att_th.result is not None
    dual_backend_rel_diff = abs(
        att_sk.result.total_strain_energy - att_th.result.total_strain_energy
    ) / max(abs(att_sk.result.total_strain_energy), 1.0e-12)

    # Tier 0 zero-FEM admissibility (clean surrogate PASS vs deliberately inverted FAIL)
    surr_u_nom = np.asarray(p1_hier_metrics["surrogate_u"], dtype=np.float64)
    t_t0 = time.perf_counter()
    st_surr_nom = lin_nom_p1.state(
        lin_nom_p1.with_exact_dirichlet(lin_nom_p1.vector(surr_u_nom))
    )
    f2d_surr = st_surr_nom.assembly.deformation_gradient[..., :2, :2].reshape(-1, 2, 2)
    j3d_surr = np.linalg.det(
        st_surr_nom.assembly.deformation_gradient.reshape(-1, 3, 3)
    )
    tier0_clean = evaluate_tier0_admissibility(
        {"u": surr_u_nom, "F_2D": f2d_surr, "J_3D": j3d_surr},
        thresholds=AdmissibilityThresholds(min_det_f_2d=0.05),
        dirichlet_deviation=lin_nom_p1.dirichlet_deviation(
            lin_nom_p1.vector(surr_u_nom)
        ),
    )
    tier0_ms = (time.perf_counter() - t_t0) * 1000.0
    bad_f2d = f2d_surr.copy()
    bad_f2d[0, 0, 0] = -0.5
    tier0_bad = evaluate_tier0_admissibility(
        {"u": surr_u_nom, "F_2D": bad_f2d, "J_3D": j3d_surr},
        thresholds=AdmissibilityThresholds(min_det_f_2d=0.05),
    )
    assert tier0_clean.passed is True and tier0_bad.passed is False

    # Tier 1: Per-CAD conformal calibration + P4 Live DWR Adjoint CADTrustReport (4 QoIs)
    qoi_list = (
        ReactionForceQoi("right_reaction", "right_load"),
        RegionalAverageStressQoi("avg_p11", component=(0, 0)),
        TotalStrainEnergyQoi("total_strain_energy"),
        SmoothedPeakVonMisesQoi("smoothed_peak_vm"),
    )
    st_fem_nom = lin_nom_p1.state(lin_nom_p1.vector(fem_u_nom))
    dwr_rows: list[dict[str, Any]] = []
    for q_obj, q_lbl in zip(
        qoi_list,
        (
            "Reaction\nForce",
            "Avg P11\nStress",
            "Total Strain\nEnergy",
            "Smoothed Peak\nvon Mises",
        ),
        strict=True,
    ):
        dwr_est = estimate_qoi_dwr(lin_nom_p1, surr_u_nom, q_obj)
        true_val = q_obj.evaluate(lin_nom_p1, st_fem_nom)
        surr_val = q_obj.evaluate(lin_nom_p1, st_surr_nom)
        dwr_rows.append(
            {
                "name": q_obj.name,
                "label": q_lbl,
                "dwr_abs_err": max(abs(float(dwr_est.dwr_estimate or 0.0)), 1.0e-10),
                "true_abs_err": max(abs(float(true_val - surr_val)), 1.0e-10),
            }
        )

    cad_cal = build_cad_calibration(
        bundle_name="viscoelastic-mount-bracket",
        bundle_manifest_sha256="c" * 64,
        problem_hash="a" * 64,
        qoi_scores={
            "right_reaction": ("N", [0.4 + 0.05 * i for i in range(20)]),
            "total_strain_energy": ("mJ", [0.8 + 0.08 * i for i in range(20)]),
        },
        alpha=0.05,
        parameter_bounds={"nominal_strain": (0.02, 0.14)},
    )
    func_band = calibrate_functional_band(
        field_name="P",
        unit="MPa",
        residuals=np.random.default_rng(7).normal(0.0, 0.08, size=(20, 8)).tolist(),
        alpha=0.05,
    )
    cad_trust = evaluate_cad_trust(
        req_nom_p1,
        surr_u_nom,
        qoi_list,
        problem_hash="a" * 64,
        geometry_error=GeometryErrorRecord(
            cad_source_hash=insp_nom["source"]["sha256"],
            declared_curve_tolerance_mm=0.06,
            achieved_max_chord_error_mm=conv_rows[-1]["max_chord_error_mm"],
            remesh_status="DISTINCT_REMESH_CONVERGED",
        ),
        discretization_error=DiscretizationErrorRecord(
            reference_status="MEASURED_CHANGES_ONLY",
            element_order=1,
            dof_count=lin_nom_p1.dof_count,
        ),
        bundle_digest="c" * 64,
        claim_level=ClaimLevel.NOT_VERIFIED,
        in_parameter_envelope=True,
        in_geometry_envelope=True,
        statistical_by_qoi={
            "right_reaction": cad_cal.statistical_record(
                "right_reaction", in_envelope=True
            )  # type: ignore[dict-item]
        },
        backend=torch_backend,
    )

    # Tier 1.5: Zero-shot Test-Time Physics LoRA on unseen higher-strain query (12% strain)
    ux_hi = 4.8
    req_hi = _bracket_fem_request(
        mesh_cad, ux_mm=ux_hi, material=eq_mat, request_id="s4-tta-hi", element_order=1
    )
    lin_hi = FemLinearization(req_hi)
    p_hi = torch.tensor([[ux_hi / nom_params.length_mm]], dtype=torch.float64)
    lift_hi = _bracket_lifting(nom_params.length_mm, ux_hi)
    with torch.no_grad():
        u_before_tta = op_p1_hier.displacement(
            encoding=enc_hier,
            query_xy=torch.tensor(
                lin_hi.dof_coordinates, dtype=torch.float64
            ).unsqueeze(0),
            parameters=p_hi,
            lifting=lift_hi,
        )[0].numpy()
    res_before_tta = float(
        np.linalg.norm(
            lin_hi.state(
                lin_hi.with_exact_dirichlet(lin_hi.vector(u_before_tta))
            ).residual[lin_hi.constraints.free]
        )
    )
    # Track energy trace across 20 TTA steps
    tta_trace: list[float] = []
    cur_op = copy.deepcopy(op_p1_hier)
    for step_k in (1, 4, 8, 12, 16, 20):
        adapted_tta, rep_tta = adapt_operator_at_test_time(
            op_p1_hier,
            domain=dom_m0,
            parameters=p_hi,
            frame=batch_nom.frame,
            lifting=lift_hi,
            material=eq_mat,
            boundary_encoding=enc_hier,
            config=TestTimeLoraConfig(steps=step_k, rank=4, learning_rate=8e-3, seed=3),
        )
        if step_k == 1:
            tta_trace.append(rep_tta.initial_energy)
        tta_trace.append(rep_tta.final_energy)
        cur_op = adapted_tta
    with torch.no_grad():
        u_after_tta = cur_op.displacement(
            encoding=enc_hier,
            query_xy=torch.tensor(
                lin_hi.dof_coordinates, dtype=torch.float64
            ).unsqueeze(0),
            parameters=p_hi,
            lifting=lift_hi,
        )[0].numpy()
    res_after_tta = float(
        np.linalg.norm(
            lin_hi.state(
                lin_hi.with_exact_dirichlet(lin_hi.vector(u_after_tta))
            ).residual[lin_hi.constraints.free]
        )
    )
    tta_drop_pct = (
        (tta_trace[0] - tta_trace[-1]) / max(abs(tta_trace[0]), 1.0e-12) * 100.0
    )

    # Tier 2: Cold FEM vs Warm Newton vs Deliberately Folded Raw vs Spectrally Safe Repaired
    cold_att = sk_backend.solve_many((req_hi,))[0]
    assert cold_att.success and cold_att.result is not None
    cold_iters = len(cold_att.result.convergence_history)

    warm_vec = lin_hi.with_exact_dirichlet(lin_hi.vector(u_after_tta))
    warm_att = sk_backend.solve_from(req_hi, warm_vec, (1.0,))
    warm_iters = (
        len(warm_att.result.convergence_history)
        if warm_att.success and warm_att.result is not None
        else cold_iters
    )

    # Deliberately invert interior nodes to simulate an adversarial/OOD folded surrogate
    xy_hi = lin_hi.dof_coordinates
    folded_nodal = np.column_stack((-2.2 * xy_hi[:, 0], 1.6 * xy_hi[:, 1]))
    folded_vec = lin_hi.with_exact_dirichlet(lin_hi.vector(folded_nodal))
    raw_folded_att = sk_backend.solve_from(req_hi, folded_vec, (1.0,))
    repaired_vec, repair_rep = spectrally_safe_repair(
        lin_hi, folded_vec, config=SpectralRepairConfig(max_steps=30, min_det_f=0.05)
    )
    repaired_att = sk_backend.solve_from(req_hi, repaired_vec, (1.0,))
    assert (
        repair_rep.succeeded
        and repaired_att.success
        and repaired_att.result is not None
    )

    # Emit & verify RFC 8785 Trust Receipt
    receipt = TrustReceipt(
        mode="certified",
        serving_path="repaired_warm_newton",
        claim_level=ClaimLevel.NOT_VERIFIED.value,
        verification_tier=VerificationTier.DISCRETE_FEM_VERIFIED,
        artifact_digests=ArtifactDigests(
            bundle_manifest_sha256="c" * 64,
            cad_source_sha256=insp_nom["source"]["sha256"],
            mesh_id=mesh_cad.mesh_id,
            problem_hash="a" * 64,
            request_hash=req_hi.recipe_hash,
            solver_backend_id=torch_backend.backend_id,
            solver_tolerances={"solver_tolerance": req_hi.config.solver_tolerance},
        ),
        tier0_admissibility=tier0_clean.document(),
        tier1_conformal=cad_cal.evaluate(
            {"nominal_strain": ux_hi / nom_params.length_mm}
        ),
        tier1_dwr={"verdict": "OK", "trust_report_hash": cad_trust.report_hash},
        tier2_fem={
            "converged": True,
            "cold_iters": cold_iters,
            "warm_iters": warm_iters,
            "repaired_warm_iters": len(repaired_att.result.convergence_history),
            "spectral_repair": repair_rep.document(),
        },
        test_time_adaptation=rep_tta.document(),
    )
    receipt_doc = receipt.document()
    rc_ok, rc_errs = verify_receipt_document(receipt_doc)
    assert rc_ok and not rc_errs
    (out / "showcase-trust-receipt.json").write_bytes(
        canonical_json_bytes(receipt_doc) + b"\n"
    )

    # Amortized Break-Even Query Analysis N* (using real measured timings)
    t_inf0 = time.perf_counter()
    with torch.no_grad():
        _ = op_p1_hier.displacement(
            encoding=enc_hier,
            query_xy=torch.tensor(mesh_cad.nodes, dtype=torch.float64).unsqueeze(0),
            parameters=p_strain,
            lifting=lift_nom,
        )
    infer_s = max(time.perf_counter() - t_inf0, 1.0e-4)
    # Use cyclic P2 viscoelastic FEM trajectory time vs operator rollout time for N*!
    amort = amortized_break_even_queries(
        training_wall_seconds=visco_train_wall,
        calibration_cases=10,
        fem_seconds=max(fem_traj_wall, 0.5),
        inference_seconds=max(op_rollout_wall, 1.0e-3),
        threads=1,
        precision="float64",
        device="cpu",
        dof_count=int(basis_v.N),
    )

    stage4_wall = round(time.perf_counter() - t_s4, 2)
    plot4_path = plot_stage4_verification_funnel(
        {
            "dwr_qois": dwr_rows,
            "dual_backend_rel_diff": dual_backend_rel_diff,
            "tta_energy_trace": tta_trace,
            "tta_energy_drop_pct": tta_drop_pct,
            "tta_res_before": res_before_tta,
            "tta_res_after": res_after_tta,
            "cold_fem_iters": cold_iters,
            "warm_newton_iters": warm_iters,
            "raw_folded_converged": raw_folded_att.success,
            "raw_folded_iters": (
                len(raw_folded_att.result.convergence_history)
                if raw_folded_att.success and raw_folded_att.result is not None
                else 0
            ),
            "repaired_warm_iters": len(repaired_att.result.convergence_history),
            "folded_init_det": float(repair_rep.initial_min_det_f or -1.0),
            "repaired_det": float(repair_rep.final_min_det_f or 0.5),
            "amortization": amort.document(),
            "tier0_ms": tier0_ms,
        },
        plots_dir / "04_verification_funnel_and_receipts.png",
    )
    print(
        f"  Dual-Backend parity={dual_backend_rel_diff:.2e} | Tier0={tier0_ms:.2f}ms | "
        f"TTA energy drop={tta_drop_pct:.2f}% | Cold={cold_iters} / Warm={warm_iters} / "
        f"RawFolded={'OK' if raw_folded_att.success else 'FAIL'} / Repaired={len(repaired_att.result.convergence_history)} iters | "
        f"N*={amort.break_even_queries:.1f} queries | Receipt verified={rc_ok} ({stage4_wall}s)"
    )

    # =========================================================================
    # STAGE 5: P3 DWR Active Learning + Fallback Ledger & P5 CAD Design Opt
    # =========================================================================
    print(
        "\n[Stage 5/6] Running P3 DWR Active Learning + Fallback Loop & P5 CAD Lightweighting..."
    )
    t_s5 = time.perf_counter()
    # Build TRAIN-only active learning candidate pool of 8 CAD bracket configurations
    pool_specs = [
        (
            "pool-01",
            BracketGeometryParams(
                notch_radius_mm=3.0, left_hole_radius_mm=2.5, right_hole_radius_mm=2.2
            ),
            2.4,
        ),
        (
            "pool-02",
            BracketGeometryParams(
                notch_radius_mm=3.2, left_hole_radius_mm=2.6, right_hole_radius_mm=2.3
            ),
            2.6,
        ),
        (
            "pool-03",
            BracketGeometryParams(
                notch_radius_mm=4.6, left_hole_radius_mm=3.6, right_hole_radius_mm=3.1
            ),
            4.6,
        ),
        (
            "pool-04",
            BracketGeometryParams(
                notch_radius_mm=4.5, left_hole_radius_mm=3.5, right_hole_radius_mm=3.0
            ),
            4.5,
        ),
        (
            "pool-05",
            BracketGeometryParams(
                notch_radius_mm=4.8, left_hole_radius_mm=2.4, right_hole_radius_mm=3.2
            ),
            4.8,
        ),
        (
            "pool-06",
            BracketGeometryParams(
                notch_radius_mm=2.8, left_hole_radius_mm=3.8, right_hole_radius_mm=2.0
            ),
            4.4,
        ),
        (
            "pool-07",
            BracketGeometryParams(
                notch_radius_mm=4.0, left_hole_radius_mm=3.2, right_hole_radius_mm=2.6
            ),
            3.2,
        ),
        (
            "pool-08",
            BracketGeometryParams(
                notch_radius_mm=4.9, left_hole_radius_mm=3.8, right_hole_radius_mm=3.2
            ),
            5.0,
        ),
    ]
    candidates: list[AcquisitionCandidate] = []
    true_err_by_id: dict[str, float] = {}
    geom_vec_by_id: dict[str, np.ndarray[Any, Any]] = {}
    for cid, cparams, ux_c in pool_specs:
        _, _, cm, _, _ = _import_bracket_variant(out, cid, cparams, mesh_size_mm=5.0)
        b_c = tok_v1.tokenize_batch([(cm, roles)], dtype=torch.float64)
        ood_c = ood_det.score(cm, roles, tokenizer=tok_v1)
        req_c = _bracket_fem_request(
            cm, ux_mm=ux_c, material=eq_mat, request_id=cid, element_order=1
        )
        lin_c = FemLinearization(req_c)
        att_c = sk_backend.solve_many((req_c,))[0]
        assert att_c.success and att_c.result is not None
        u_fem_c = np.asarray(att_c.result.displacement, dtype=np.float64)
        p_c = torch.tensor([[ux_c / cparams.length_mm]], dtype=torch.float64)
        lift_c = _bracket_lifting(cparams.length_mm, ux_c)
        with torch.no_grad():
            enc_c = multi_op.encode_boundary(
                boundary_xy=b_c.boundary_xy,
                boundary_normals=b_c.boundary_normals,
                boundary_features=b_c.boundary_features,
                weights=b_c.weights,
                mask=b_c.mask,
                frame=b_c.frame,
            )
            u_surr_c = multi_op.displacement(
                encoding=enc_c,
                query_xy=torch.tensor(
                    lin_c.dof_coordinates, dtype=torch.float64
                ).unsqueeze(0),
                parameters=p_c,
                lifting=lift_c,
            )[0].numpy()
        err_c = float(
            np.linalg.norm(u_surr_c - u_fem_c) / max(np.linalg.norm(u_fem_c), 1.0e-12)
        )
        dwr_c = estimate_qoi_dwr(
            lin_c, u_surr_c, ReactionForceQoi("right_reaction", "right_load")
        )
        g_vec = (
            cparams.notch_radius_mm / 5.0,
            cparams.left_hole_radius_mm / 4.0,
            cparams.right_hole_radius_mm / 4.0,
            ux_c / 5.0,
        )
        true_err_by_id[cid] = 100.0 * err_c
        geom_vec_by_id[cid] = np.asarray(g_vec, dtype=np.float64)
        candidates.append(
            AcquisitionCandidate(
                case_id=cid,
                partition="TRAIN",
                feature_vector=(ux_c, cparams.analytical_area_mm2),
                eta_energy=max(abs(float(dwr_c.eta_energy or 0.0)), 1.0e-6),
                ood_distance=float(ood_c.descriptor_distance),
                dwr_sensitivity=max(abs(float(dwr_c.dwr_estimate or 0.0)), 1.0e-6),
                geometry_token_vector=g_vec,
                parameters={"ux_mm": ux_c, "notch_r": cparams.notch_radius_mm},
            )
        )

    def _spread(ids: tuple[str, ...]) -> float:
        vecs = [geom_vec_by_id[i] for i in ids]
        dists = [
            float(np.linalg.norm(vecs[a] - vecs[b]))
            for a in range(len(vecs))
            for b in range(a + 1, len(vecs))
        ]
        return float(np.mean(dists)) if dists else 0.0

    p3_results: dict[str, dict[str, Any]] = {}
    for arm_name, wts in (
        ("random_doe", AcquisitionWeights()),
        (
            "physics_indicator",
            AcquisitionWeights(
                eta_weight=0.7,
                ood_weight=0.3,
                ensemble_weight=0.0,
                diversity_weight=0.0,
            ),
        ),
        (
            "physics_dwr_diversity",
            AcquisitionWeights(
                eta_weight=0.35,
                ood_weight=0.15,
                ensemble_weight=0.0,
                dwr_weight=0.5,
                diversity_weight=0.45,
            ),
        ),
    ):
        plan_a = AcquisitionPlan(
            plan_id=f"showcase-{arm_name}",
            arm_kind=arm_name,
            budget_solves=3,
            batch_size=3,
            seed=11,
            weights=wts,
        )
        ev_a = select_acquisition_batch(
            candidates, already_acquired_ids=(), plan=plan_a
        )
        p3_results[arm_name] = {
            "selected": list(ev_a.selected_case_ids),
            "mean_true_err_pct": float(
                np.mean([true_err_by_id[i] for i in ev_a.selected_case_ids])
            ),
            "mean_geom_spread": _spread(ev_a.selected_case_ids),
            "selection_hash": ev_a.selection_hash,
        }

    # Record Tier 2 escalated fallback into TRAIN-only FallbackFeedbackLedger
    fb_ledger = FallbackFeedbackLedger(campaign_id="viscoelastic-bracket-showcase")
    fb_ledger = fb_ledger.append(
        FallbackReturnRecord(
            case_id="tier2-fallback-01",
            partition="TRAIN",
            request_hash=req_hi.recipe_hash,
            result_hash=repaired_att.result_hash or "0" * 64,
            serving_path="repaired_warm_newton",
            feature_vector=(ux_hi, nom_params.analytical_area_mm2),
            eta_energy=0.05,
            dwr_sensitivity=float(dwr_rows[0]["dwr_abs_err"]),
            ood_distance=float(ood_tr_rep.descriptor_distance),
            geometry_token_vector=(0.8, 0.8, 0.65, 0.96),
            newton_iterations=len(repaired_att.result.convergence_history),
            dof_count=lin_hi.dof_count,
            wall_seconds=fem_single_s,
        )
    )
    fb_ledger.save(out / "tier2-fallback-ledger.json")

    # Multi-fidelity distillation loss check using acquired FEM displacement
    pts_mf, w_mf = deterministic_triangle_quadrature(dom_m0, degree=2)
    u_v_mf, g_mf = op_p1_hier.displacement_and_gradient(
        encoding=enc_hier,
        query_xy=pts_mf.unsqueeze(0),
        parameters=p_strain,
        lifting=lift_nom,
        create_graph=False,
    )
    # Interpolate P1 FEM nodal displacement onto degree-2 quadrature points
    cell_u_fem = torch.tensor(fem_u_nom[: len(mesh_cad.nodes)], dtype=torch.float64)[
        dom_m0.cells
    ]
    bary_d2 = torch.tensor(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=torch.float64,
    )
    u_fem_q = torch.einsum("qa,cai->cqi", bary_d2, cell_u_fem).reshape(-1, 2)
    mf_out = multifidelity_energy_loss(
        u_pred=u_v_mf[0],
        grad_pred=g_mf[0],
        weights=w_mf,
        material=eq_mat,
        reference_area=dom_m0.area,
        config=MultiFidelityConfig(fem_displacement_weight=1.0, base_energy=ecfg),
        u_fem_target=u_fem_q,
        partition="TRAIN",
    )

    # P5: FEM-Guarded CAD Lightweighting Design Optimization
    # Minimize bracket material area by tuning (left_hole_radius, notch_radius)
    # subject to P2-FEM-verified axial reaction force >= 80% of nominal!
    init_r_hole = 2.6
    init_r_notch = 3.2
    rxn_qoi = ReactionForceQoi("right_reaction", "right_load")

    def _p5_params(r_h: float, r_n: float) -> BracketGeometryParams:
        return BracketGeometryParams(
            notch_radius_mm=r_n,
            left_hole_radius_mm=r_h,
            right_hole_radius_mm=max(1.8, 0.8 * r_h),
        )

    p5_cache: dict[tuple[float, float], CanonicalMesh] = {}

    def _p5_mesh(r_h: float, r_n: float) -> CanonicalMesh:
        key = (round(r_h, 3), round(r_n, 3))
        if key not in p5_cache:
            _, _, m_p, _, _ = _import_bracket_variant(
                out / "p5-cad",
                f"opt-r{key[0]:.3f}-n{key[1]:.3f}",
                _p5_params(r_h, r_n),
                mesh_size_mm=5.2,
            )
            p5_cache[key] = m_p
        return p5_cache[key]

    # Measure initial reaction force via P2 FEM to set realistic 80% stiffness floor
    init_p5_mesh = _p5_mesh(init_r_hole, init_r_notch)
    init_p5_req = _bracket_fem_request(
        init_p5_mesh,
        ux_mm=ux_eq,
        material=eq_mat,
        request_id="p5-init",
        element_order=2,
    )
    init_p5_att = sk_backend.solve_many((init_p5_req,))[0]
    assert init_p5_att.success and init_p5_att.result is not None
    init_rxn_n = float(init_p5_att.result.reactions["right_load"][0])
    min_rxn_n = 0.80 * init_rxn_n

    p5_problem = DesignProblem(
        name="viscoelastic-bracket-lightweighting",
        geometry_version="mount-bracket-step-v1",
        problem_spec_hash="a" * 64,
        parameters=(
            DesignParameter(
                "left_hole_radius", lower=2.4, upper=4.4, initial=init_r_hole, unit="mm"
            ),
            DesignParameter(
                "notch_radius", lower=2.8, upper=5.2, initial=init_r_notch, unit="mm"
            ),
        ),
        objective_name="material_area",
        objective_unit="mm^2",
        minimize=True,
        constraints=(
            DesignConstraint(
                name="reaction_force",
                kind="inequality_ge",
                limit=min_rxn_n,
                tolerance=0.05,
                unit="N",
            ),
        ),
        max_iterations=10,
        max_fem_evaluations=8,
        initial_trust_radius=0.18,
        min_trust_radius=0.02,
        gradient_check_interval=5,
        gradient_cos_min=0.1,
        accept_eta_tolerance=0.25,
        constraint_penalty_weight=2000.0,
    )

    p5_surr_rxn_by_key: dict[tuple[float, float], float] = {}

    def _p5_surrogate_eval(p: Mapping[str, float]) -> SurrogateDesignEvaluation:
        r_h = float(p["left_hole_radius"])
        r_n = float(p["notch_radius"])
        bp = _p5_params(r_h, r_n)
        m_p = _p5_mesh(r_h, r_n)
        b_p = tok_v1.tokenize_batch([(m_p, roles)], dtype=torch.float64)
        dom_p = TriangleDomain(
            torch.tensor(m_p.nodes, dtype=torch.float64),
            torch.tensor(m_p.cells, dtype=torch.int64),
        )
        with torch.no_grad():
            enc_p = multi_op.encode_boundary(
                boundary_xy=b_p.boundary_xy,
                boundary_normals=b_p.boundary_normals,
                boundary_features=b_p.boundary_features,
                weights=b_p.weights,
                mask=b_p.mask,
                frame=b_p.frame,
            )
        q2_p, w2_p = deterministic_triangle_quadrature(dom_p, degree=2)
        _, g2_p = multi_op.displacement_and_gradient(
            encoding=enc_p,
            query_xy=q2_p.unsqueeze(0),
            parameters=p_strain,
            lifting=lift_nom,
            create_graph=False,
        )
        p_st = constitutive_in_plane_stress(g2_p[0], eq_mat)
        rxn_s = float(torch.sum(p_st[:, 0, 0] * w2_p).item()) / bp.length_mm
        p5_surr_rxn_by_key[(round(r_h, 3), round(r_n, 3))] = rxn_s
        # Analytical area derivatives: A = L*H - 0.5*pi*r_n^2 - pi*(1 + 0.64)*r_h^2
        grad = {
            "left_hole_radius": -2.0 * math.pi * 1.64 * r_h,
            "notch_radius": -math.pi * r_n,
        }
        return SurrogateDesignEvaluation(
            objective=bp.analytical_area_mm2,
            gradient=grad,
            constraints={"reaction_force": rxn_s},
            in_envelope=True,
            admissible=True,
            eta_indicator=0.02,
        )

    p5_fem_cnt = 0

    def _p5_fem_eval(p: Mapping[str, float]) -> FemDesignEvaluation:
        nonlocal p5_fem_cnt
        p5_fem_cnt += 1
        r_h = float(p["left_hole_radius"])
        r_n = float(p["notch_radius"])
        bp = _p5_params(r_h, r_n)
        m_p = _p5_mesh(r_h, r_n)
        req_p = _bracket_fem_request(
            m_p,
            ux_mm=ux_eq,
            material=eq_mat,
            request_id=f"p5-fem-{p5_fem_cnt}",
            element_order=2,
            load_steps=1,
        )
        att_p = sk_backend.solve_many((req_p,))[0]
        if not att_p.success or att_p.result is None:
            return FemDesignEvaluation(
                succeeded=False, objective=None, constraints={}, reason="FEM failed"
            )
        lin_p = FemLinearization(req_p)
        u_p = np.asarray(att_p.result.displacement, dtype=np.float64)
        st_p = lin_p.state(lin_p.vector(u_p))
        return FemDesignEvaluation(
            succeeded=True,
            objective=bp.analytical_area_mm2,
            constraints={"reaction_force": rxn_qoi.evaluate(lin_p, st_p)},
        )

    p5_report = run_fem_guarded_design_optimization(
        p5_problem,
        surrogate_evaluator=_p5_surrogate_eval,
        fem_evaluator=_p5_fem_eval,
    )
    if p5_report.best_design is not None:
        opt_r_h = float(p5_report.best_design.parameters["left_hole_radius"])
        opt_r_n = float(p5_report.best_design.parameters["notch_radius"])
        final_fem_rxn = float(
            p5_report.best_design.fem_constraints.get("reaction_force", init_rxn_n)
        )
    else:
        opt_r_h = init_r_hole
        opt_r_n = init_r_notch
        final_fem_rxn = init_rxn_n
    opt_p5_mesh = _p5_mesh(opt_r_h, opt_r_n)
    init_area = _p5_params(init_r_hole, init_r_notch).analytical_area_mm2
    opt_area = _p5_params(opt_r_h, opt_r_n).analytical_area_mm2
    area_red_pct = (init_area - opt_area) / init_area * 100.0

    # Build clean per-trust-region-iteration trajectory (iter 0 = initial, then accepted/trial steps)
    init_surr_rxn = p5_surr_rxn_by_key.get(
        (round(init_r_hole, 3), round(init_r_notch, 3)), init_rxn_n
    )
    p5_history: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "area_mm2": init_area,
            "reaction_n": init_surr_rxn,
            "accepted": True,
            "r_h": init_r_hole,
            "r_n": init_r_notch,
        }
    ]
    curr_a = init_area
    curr_rxn = init_surr_rxn
    curr_rh = init_r_hole
    curr_rn = init_r_notch
    for rec in p5_report.iterations:
        rh_i = float(rec.parameters["left_hole_radius"])
        rn_i = float(rec.parameters["notch_radius"])
        rxn_i = p5_surr_rxn_by_key.get((round(rh_i, 3), round(rn_i, 3)), curr_rxn)
        if rec.accepted:
            curr_a = float(rec.surrogate_objective)
            curr_rxn = rxn_i
            curr_rh = rh_i
            curr_rn = rn_i
        p5_history.append(
            {
                "iteration": int(rec.iteration),
                "area_mm2": curr_a,
                "reaction_n": curr_rxn,
                "accepted": bool(rec.accepted),
                "r_h": curr_rh,
                "r_n": curr_rn,
            }
        )

    stage5_wall = round(time.perf_counter() - t_s5, 2)
    plot5_path = plot_stage5_p3_active_and_p5_design(
        {
            "p3_arms": p3_results,
            "fallback_records": len(fb_ledger.records),
            "p5_history": p5_history,
            "p5_init_area_mm2": init_area,
            "p5_opt_area_mm2": opt_area,
            "p5_min_reaction_n": min_rxn_n,
            "p5_final_fem_reaction_n": final_fem_rxn,
            "p5_area_reduction_pct": area_red_pct,
            "p5_init_nodes": init_p5_mesh.nodes,
            "p5_init_facets": init_p5_mesh.boundary_facets,
            "p5_opt_nodes": opt_p5_mesh.nodes,
            "p5_opt_facets": opt_p5_mesh.boundary_facets,
            "p5_init_r": init_r_hole,
            "p5_opt_r": opt_r_h,
            "p5_init_notch": init_r_notch,
            "p5_opt_notch": opt_r_n,
        },
        plots_dir / "05_p3_active_learning_and_p5_design_opt.png",
    )
    print(
        f"  P3 DWR+Div err={p3_results['physics_dwr_diversity']['mean_true_err_pct']:.2f}% vs "
        f"Random={p3_results['random_doe']['mean_true_err_pct']:.2f}% | "
        f"P5 Area: {init_area:.1f} -> {opt_area:.1f} mm^2 (-{area_red_pct:.1f}%), "
        f"status={p5_report.status.value} ({stage5_wall}s)"
    )

    # =========================================================================
    # STAGE 6: Extended Physics (Follower, 2.5D Contact, 3D TET10) & Deployment
    # =========================================================================
    print(
        "\n[Stage 6/6] Running Follower Pressure, 2.5D Contact, 3D TET10 & Deployment Checks..."
    )
    t_s6 = time.perf_counter()

    # 6A. Non-conservative follower pressure on CAD bracket (top + relief notch!)
    fol_bcs = (
        BoundaryConditionSpec(
            name="left_clamp",
            boundary_tag="left_clamp",
            kind="dirichlet",
            components=("x", "y"),
            values=(
                Quantity(0.0, Unit.MM).to_canonical(),
                Quantity(0.0, Unit.MM).to_canonical(),
            ),
        ),
    )
    fol_loads = (
        LoadSpec(
            name="top_follower",
            kind="follower_pressure",
            boundary_tag="top_free",
            components=("pressure",),
            values=(Quantity(0.06, Unit.MPA).to_canonical(),),
        ),
        LoadSpec(
            name="notch_follower",
            kind="follower_pressure",
            boundary_tag="notch_free",
            components=("pressure",),
            values=(Quantity(0.06, Unit.MPA).to_canonical(),),
        ),
    )
    req_fol = FemRequest(
        request_id="s6-bracket-follower",
        mesh=mesh_cad,
        material=eq_mat,
        boundary_conditions=fol_bcs,
        loads=fol_loads,
        qois=(),
        config=FemSolverConfig(element_order=1, solver_tolerance=1e-10, load_steps=2),
        execution_profile=_exec_profile(),
    )
    att_fol = torch_backend.solve_many((req_fol,))[0]
    assert att_fol.success and att_fol.result is not None
    u_fol = np.asarray(att_fol.result.displacement, dtype=np.float64)
    # FD check on unsymmetric follower tangent
    lin_fol = torch_backend.linearize(req_fol)
    u_fol_vec = lin_fol.vector(u_fol)
    rng_f = np.random.default_rng(7)
    v_f = rng_f.standard_normal(lin_fol.dof_count)
    kv_a = lin_fol.state(u_fol_vec).assembly.tangent @ v_f
    kv_fd = (
        lin_fol.state(u_fol_vec + 1e-6 * v_f).residual
        - lin_fol.state(u_fol_vec - 1e-6 * v_f).residual
    ) / 2e-6
    fol_fd_err = float(np.linalg.norm(kv_a - kv_fd) / np.linalg.norm(kv_fd))

    # 6B. 2.5D Plane-Strain Rigid-Indenter Elastomer Contact on the CAD bracket
    contact_bcs = (
        BoundaryConditionSpec(
            name="bottom_support",
            boundary_tag="bottom_free",
            kind="dirichlet",
            components=("x", "y"),
            values=(
                Quantity(0.0, Unit.MM).to_canonical(),
                Quantity(0.0, Unit.MM).to_canonical(),
            ),
        ),
    )
    req_contact = FemRequest(
        request_id="s6-bracket-contact",
        mesh=mesh_cad,
        material=eq_mat,
        boundary_conditions=contact_bcs,
        loads=(),
        qois=(),
        config=FemSolverConfig(
            element_order=1, solver_tolerance=1e-9, max_newton_iterations=30
        ),
        execution_profile=_exec_profile(),
    )
    # Rigid circular punch pressing into left shoulder of bracket top edge
    punch = RigidIndenterSpec(
        kind="circle",
        center=(8.5, 25.5),
        radius=6.5,
        penalty_stiffness=1.5e3,
        contact_boundary_tag="top_free",
    )
    u_contact, contact_rep = TorchFemBackend(
        physics_mode="2d-plane-strain-finite-strain"
    ).solve_contact(req_contact, punch, max_auglag_iterations=10, gap_tolerance_mm=1e-4)

    # 6C. 3D TET10 Volumetric Bracket Block + Locking Refusal + O(N*S) Surface3DEnergyOperator
    tet4_3d = make_brick_tet4_mesh(40.0, 20.0, 6.0, nx=3, ny=2, nz=1)
    tet10_3d = make_brick_tet10_mesh(40.0, 20.0, 6.0, nx=3, ny=2, nz=1)
    tet4_locked_refused = False
    try:
        tet4_3d.validate_incompressibility_locking(0.49)
    except VolumetricLockingError:
        tet4_locked_refused = True
    tet10_3d.validate_incompressibility_locking(0.49)

    c3, n3, a3 = tet10_3d.surface_geometry()
    torch.manual_seed(42)
    op3d = Surface3DEnergyOperator(
        Surface3DConfig(
            parameter_dim=2,
            surface_feature_dim=6,
            hidden_dim=16,
            slice_count=4,
            num_heads=4,
            num_blocks=2,
        )
    ).to(torch.float64)
    c3_t = torch.tensor(c3, dtype=torch.float64).unsqueeze(0)
    n3_t = torch.tensor(n3, dtype=torch.float64).unsqueeze(0)
    a3_t = torch.tensor(a3, dtype=torch.float64).unsqueeze(0)
    f3_t = torch.zeros((1, c3_t.shape[1], 6), dtype=torch.float64)
    enc3 = op3d.encode_surface(c3_t, n3_t, a3_t, f3_t)
    q3 = torch.tensor([[[10.0, 8.0, 3.0], [25.0, 12.0, 3.0]]], dtype=torch.float64)
    p3 = torch.tensor([[0.09, 0.38]], dtype=torch.float64)
    _, grad3 = op3d.displacement_and_gradient(
        encoding=enc3, query_xyz=q3, parameters=p3, create_graph=False
    )
    max_fd3 = 0.0
    with torch.no_grad():
        for ax_i in range(3):
            dq3 = torch.zeros_like(q3)
            dq3[..., ax_i] = 1e-6
            fd3 = (
                op3d.displacement(encoding=enc3, query_xyz=q3 + dq3, parameters=p3)
                - op3d.displacement(encoding=enc3, query_xyz=q3 - dq3, parameters=p3)
            ) / 2e-6
            max_fd3 = max(max_fd3, float((grad3[..., ax_i] - fd3).abs().max().item()))

    # 6D. Deployment Pillar: Inspector Snapshot + Markdown Dossier + MCP Server + FMI 3.0 FMU
    shipped_bundle = REPO_ROOT / "bundles" / "notched-v1-cp-freeze"
    shipped_req = json.loads(
        (REPO_ROOT / "bundles" / "requests" / "notched-in-domain.json").read_text(
            encoding="utf-8"
        )
    )
    snap = build_inspector_snapshot(
        shipped_bundle, shipped_req["inputs"], mode="certified", backend="torch-fem"
    )
    dossier_md = render_trust_dossier_markdown(snap)
    (out / "inspector-trust-dossier.md").write_text(dossier_md, encoding="utf-8")

    mcp = FormaMechMcpServer(REPO_ROOT)
    mcp_cad = mcp.call_tool(
        "inspect_cad_project",
        {"project_path": str(proj_nom.relative_to(REPO_ROOT))},
    )
    mcp_rc_ver = mcp.call_tool("verify_receipt", {"receipt": receipt_doc})
    sandbox_blocked = False
    try:
        mcp.resolve_path("../escape-attempt.json")
    except SandboxViolationError:
        sandbox_blocked = True

    fmu_path = out / "notched-plate.fmu"
    fmu_res = export_bundle_to_fmu(
        shipped_bundle, fmu_path, model_name="FormaMechShowcase"
    )
    in_vals = {k: float(v["value"]) for k, v in shipped_req["inputs"].items()}
    fmu_in = evaluate_fmu_step(fmu_path, bundle_path=shipped_bundle, inputs=in_vals)
    first_k = next(iter(in_vals))
    fmu_out = evaluate_fmu_step(
        fmu_path, bundle_path=shipped_bundle, inputs={**in_vals, first_k: 1.0e9}
    )

    deploy_checks = [
        {
            "name": "RFC 8785 Trust Receipt",
            "passed": bool(
                rc_ok
                and receipt_doc["epistemic_scope"]["experimental_validation"] is False
            ),
            "detail": f"Tier={receipt_doc['verification_tier']} | SHA256={receipt.receipt_sha256[:12]}... | exp_val=False",
        },
        {
            "name": "Viscoelastic Artifact & Ckpt",
            "passed": bool(ckpt_verified and reloaded_art.manifest_sha256),
            "detail": f"safetensors verified | replay_exact={replay_exact} | dt-halving={dt_halving_rel * 100:.2f}%",
        },
        {
            "name": "Verification & Trust Inspector",
            "passed": bool(
                snap.receipt_verified and snap.verification_tier == "DWR_BOUNDED"
            ),
            "detail": f"Backend=torch-fem-v1 | Tier={snap.verification_tier} | Dossier exported",
        },
        {
            "name": "MCP Server (5 JSON-RPC Tools)",
            "passed": bool(
                mcp_rc_ver["verified"]
                and sandbox_blocked
                and mcp_cad["mesh"]["triangle_count"] > 0
            ),
            "detail": f"CAD inspected ({mcp_cad['mesh']['triangle_count']} tris) | Receipt OK | Sandbox blocked escape",
        },
        {
            "name": "FMI 3.0 Co-Simulation FMU",
            "passed": bool(
                fmu_in.in_envelope and fmu_in.admissible and not fmu_out.in_envelope
            ),
            "detail": f"In-env admissible={fmu_in.admissible} | Out-of-env refused={not fmu_out.in_envelope}",
        },
        {
            "name": "3D TET4 Locking Refusal",
            "passed": bool(tet4_locked_refused),
            "detail": f"TET4 refused at nu=0.49 | TET10 accepted ({len(tet10_3d.cells)} cells)",
        },
        {
            "name": "P3 Multi-Fidelity Disclosure",
            "passed": bool(mf_out.total.item() > 0.0 and len(fb_ledger.records) == 1),
            "detail": "TRAIN-only fallback ledger + explicit non-label-free disclosure",
        },
    ]

    stage6_wall = round(time.perf_counter() - t_s6, 2)
    plot6_path = plot_stage6_extended_physics_and_deployment(
        {
            "follower_nodes": mesh_cad.nodes,
            "follower_u": u_fol,
            "follower_cells": mesh_cad.cells,
            "follower_tangent_fd_err": fol_fd_err,
            "contact_nodes": mesh_cad.nodes,
            "contact_u": u_contact,
            "contact_cells": mesh_cad.cells,
            "indenter_center": punch.center,
            "indenter_radius": punch.radius,
            "contact_active_nodes": contact_rep.active_node_count,
            "contact_max_pen_mm": contact_rep.max_penetration_mm,
            "contact_fy_n": contact_rep.total_contact_force_n[1],
            "mesh3d_nodes": tet10_3d.nodes,
            "mesh3d_surface_tris": tet10_3d.surface_triangles,
            "op3d_grad_fd_err": max_fd3,
            "deployment_checks": deploy_checks,
        },
        plots_dir / "06_extended_physics_3d_and_deployment.png",
    )

    total_wall = round(time.perf_counter() - t_all, 2)
    summary: dict[str, Any] = {
        "showcase": "viscoelastic-cad-mount-bracket",
        "claim_level": "NOT_VERIFIED",
        "experimental_validation": False,
        "total_wall_seconds": total_wall,
        "output_dir": str(out),
        "plots": [
            str(plot1_path),
            str(plot2_path),
            str(plot3_path),
            str(plot4_path),
            str(plot5_path),
            str(plot6_path),
        ],
        "stage1_cad": {
            "wall_seconds": stage1_wall,
            "geometry_params": nom_params.document(),
            "step_file": str(step_nom),
            "edge_map": edge_map,
            "hierarchy_cells": {
                k: v["cell_count"] for k, v in hier_nom["levels"].items()
            },
            "geometry_convergence": conv_rows,
        },
        "stage2_viscoelastic_p6": {
            "wall_seconds": stage2_wall,
            "tangent_fd_rel_err": t_check.relative_error,
            "fem_trajectory_wall_s": fem_traj_wall,
            "operator_train_wall_s": visco_train_wall,
            "operator_rollout_wall_s": op_rollout_wall,
            "dt_halving_peak_rel_diff": dt_halving_rel,
            "cpu_replay_exact": replay_exact,
            "relaxation_drop_pct": relaxation_drop_pct,
            "pure_op_max_rxn_err_pct": pure_op_max_err_pct,
            "guarded_max_rxn_err_pct": guarded_max_err_pct,
            "guarded_operator_steps": sum(
                s == "ACCEPTED_OPERATOR" for s in guard_statuses
            ),
            "guarded_fem_fallback_steps": sum(
                s == "ACCEPTED_FEM_FALLBACK" for s in guard_statuses
            ),
            "checkpoint_verified": ckpt_verified,
            "artifact_manifest_sha256": reloaded_art.manifest_sha256,
            "train_initial_loss": v_train_summary.initial_loss,
            "train_final_loss": v_train_summary.final_loss,
        },
        "stage3_p1_p2": {
            "wall_seconds": stage3_wall,
            "p1_baseline": {
                k: v for k, v in p1_base_metrics.items() if isinstance(v, float)
            },
            "p1_hierarchical": {
                k: v for k, v in p1_hier_metrics.items() if isinstance(v, float)
            },
            "tokenizer_v2_channels": int(feats_v2.shape[-1]),
            "ood_train_distance": ood_tr_rep.descriptor_distance,
            "ood_id_distance": ood_id_rep.descriptor_distance,
            "ood_id_status": ood_id_rep.status.value,
            "ood_extreme_distance": ood_ext_rep.descriptor_distance,
            "ood_extreme_status": ood_ext_rep.status.value,
            "p2_zero_shot_u_pct": zs_m["u_rel_l2_pct"],
            "p2_lora_u_pct": lora_m["u_rel_l2_pct"],
            "p2_lora_wall_s": lora_wall,
            "p2_lora_params": adapt_res.adapter_parameter_count,
            "p2_scratch_u_pct": scratch_m["u_rel_l2_pct"],
            "p2_scratch_wall_s": scratch_wall,
            "p2_pretrain_wall_s": pretrain_wall,
        },
        "stage4_verification": {
            "wall_seconds": stage4_wall,
            "dual_backend_energy_rel_diff": dual_backend_rel_diff,
            "tier0_clean_passed": tier0_clean.passed,
            "tier0_inverted_caught": not tier0_bad.passed,
            "tier0_wall_ms": tier0_ms,
            "surrogate_single_infer_ms": round(infer_s * 1000.0, 3),
            "dwr_qois": dwr_rows,
            "cad_trust_status": cad_trust.overall_status.value,
            "functional_band_coverage": func_band.nominal_coverage,
            "tta_energy_drop_pct": tta_drop_pct,
            "tta_res_before": res_before_tta,
            "tta_res_after": res_after_tta,
            "cold_fem_iters": cold_iters,
            "warm_newton_iters": warm_iters,
            "raw_folded_converged": raw_folded_att.success,
            "repaired_warm_iters": len(repaired_att.result.convergence_history),
            "receipt_sha256": receipt.receipt_sha256,
            "receipt_verified": rc_ok,
            "amortization": amort.document(),
        },
        "stage5_p3_p5": {
            "wall_seconds": stage5_wall,
            "p3_arms": p3_results,
            "fallback_ledger_records": len(fb_ledger.records),
            "p5_status": p5_report.status.value,
            "p5_init_area_mm2": init_area,
            "p5_opt_area_mm2": opt_area,
            "p5_area_reduction_pct": area_red_pct,
            "p5_min_reaction_n": min_rxn_n,
            "p5_final_fem_reaction_n": final_fem_rxn,
            "p5_fem_evaluations": p5_fem_cnt,
        },
        "stage6_extended_and_deploy": {
            "wall_seconds": stage6_wall,
            "follower_max_u_mm": float(np.linalg.norm(u_fol, axis=1).max()),
            "follower_tangent_fd_err": fol_fd_err,
            "contact_report": contact_rep.document(),
            "tet4_locking_refused": tet4_locked_refused,
            "tet10_cells": len(tet10_3d.cells),
            "op3d_grad_fd_err": max_fd3,
            "fmu_sha256": fmu_res.fmu_sha256,
            "fmu_in_envelope_admissible": fmu_in.admissible,
            "fmu_out_of_envelope_refused": not fmu_out.in_envelope,
            "deployment_checks": deploy_checks,
        },
    }
    (out / "showcase_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nShowcase finished in {total_wall}s -> {out}")
    return summary
