"""Phase 6 tests: TorchFemBackend, 2.5D plane strain/axisymmetric, contact, 3D & Level 4."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from campaigns.vvuq_suite import (
    axisymmetric_cylinder_mesh,
    build_level4_smoke_tet10_mesh,
    cooks_membrane_3d_spec,
    cooks_membrane_mesh,
    evaluate_ood_conformal_shift_gate,
    nafems_le10_spec,
    rectangular_strip_mesh,
    run_level3c_pressurized_cylinder_axisymmetric,
)
from piginot.bundle import load_bundle
from piginot.certify import verify_receipt_document
from piginot.cli import parse_request
from piginot.contracts import (
    BoundaryConditionSpec,
    ExecutionProfile,
    LoadSpec,
    MaterialSpec,
    Quantity,
    Unit,
)
from piginot.fem import (
    DiscreteLinearization,
    FemRequest,
    FemSolverConfig,
    RigidIndenterSpec,
    ScikitFemBackend,
    SolverBackend,
    TorchFemBackend,
    assemble_contact_terms,
    available_backends,
    resolve_backend,
)
from piginot.geometry import (
    CellType3D,
    VolumetricLockingError,
    make_brick_hex8_mesh,
    make_brick_tet4_mesh,
    make_brick_tet10_mesh,
)
from piginot.objectives import (
    axisymmetric_potential_energy,
    plane_strain_potential_energy,
)
from piginot.operators import (
    SURFACE_3D_ARCHITECTURE_ID,
    Surface3DConfig,
    Surface3DEnergyOperator,
)
from piginot.physics import reference_modulus
from piginot.serving import ServingMode, load_recipe, serve

ROOT = Path(__file__).resolve().parents[2]
NOTCHED_BUNDLE = ROOT / "bundles" / "notched-v1-cp-freeze"
NOTCHED_REQUEST = ROOT / "bundles" / "requests" / "notched-in-domain.json"


def _profile() -> ExecutionProfile:
    return ExecutionProfile(
        python_version="3.13",
        library_versions={"torch": "runtime"},
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )


def _neo_hookean(e_mpa: float = 10.0, nu: float = 0.3) -> MaterialSpec:
    return MaterialSpec(
        name="nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(e_mpa, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(nu, Unit.ONE).to_canonical(),
        },
    )


def test_torch_fem_registry_protocol_and_cooks_membrane_parity_with_skfem() -> None:
    assert "torch-fem" in available_backends()
    t_backend = resolve_backend("torch-fem")
    assert isinstance(t_backend, TorchFemBackend)
    assert isinstance(t_backend, SolverBackend)
    assert t_backend.capabilities.supports_gpu is True
    assert t_backend.capabilities.supports_contact is True
    assert t_backend.capabilities.license_spdx == "MIT"

    mesh = cooks_membrane_mesh(nx=4, ny=4)
    mat = _neo_hookean(e_mpa=25.0, nu=0.35)
    bcs = (
        BoundaryConditionSpec(
            name="clamp_left",
            boundary_tag="left",
            kind="dirichlet",
            components=("x", "y"),
            values=(
                Quantity(0.0, Unit.MM).to_canonical(),
                Quantity(0.0, Unit.MM).to_canonical(),
            ),
        ),
    )
    loads = (
        LoadSpec(
            name="shear_right",
            kind="traction",
            boundary_tag="right",
            components=("x", "y"),
            values=(
                Quantity(0.0, Unit.MPA).to_canonical(),
                Quantity(0.25, Unit.MPA).to_canonical(),
            ),
        ),
    )
    for order in (1, 2):
        req = FemRequest(
            request_id=f"cooks-parity-p{order}",
            mesh=mesh,
            material=mat,
            boundary_conditions=bcs,
            loads=loads,
            qois=(),
            config=FemSolverConfig(
                element_order=order,
                solver_tolerance=1.0e-10,
                max_newton_iterations=16,
                load_steps=2,
            ),
            execution_profile=_profile(),
        )
        lin = t_backend.linearize(req)
        assert isinstance(lin, DiscreteLinearization)

        sk_att = ScikitFemBackend().solve_many((req,))[0]
        th_att = t_backend.solve_many((req,))[0]
        assert sk_att.success and sk_att.result is not None
        assert th_att.success and th_att.result is not None

        assert th_att.result.total_strain_energy == pytest.approx(
            sk_att.result.total_strain_energy, rel=1.0e-6
        )
        assert th_att.result.total_potential_energy == pytest.approx(
            sk_att.result.total_potential_energy, rel=1.0e-6
        )
        sk_u = np.asarray(sk_att.result.displacement, dtype=np.float64)
        th_u = np.asarray(th_att.result.displacement, dtype=np.float64)
        assert np.max(np.linalg.norm(th_u, axis=1)) == pytest.approx(
            np.max(np.linalg.norm(sk_u, axis=1)), rel=1.0e-6
        )

    # End-to-end certified serving with backend="torch-fem" on live bundle
    bundle = load_bundle(NOTCHED_BUNDLE)
    recipe = load_recipe(bundle, NOTCHED_BUNDLE)
    inputs, queries = parse_request(json.loads(NOTCHED_REQUEST.read_text("utf-8")))
    served = serve(
        bundle,
        inputs,
        queries,
        mode=ServingMode.CERTIFIED,
        recipe=recipe,
        backend="torch-fem",
        emit_receipt=True,
    )
    doc = served.document()
    assert (
        doc["trust_receipt"]["artifact_digests"]["solver_backend_id"] == "torch-fem-v1"
    )
    ok, errs = verify_receipt_document(doc["trust_receipt"])
    assert ok is True and errs == ()


def test_25d_plane_strain_axisymmetric_objectives_tangent_and_lame_cylinder() -> None:
    # 1) Plane strain objective vs TorchFemBackend energy parity
    mesh = rectangular_strip_mesh(length_mm=4.0, height_mm=2.0, nx=3, ny=2)
    mat = _neo_hookean(e_mpa=20.0, nu=0.4)
    req_ps = FemRequest(
        request_id="ps-energy-parity",
        mesh=mesh,
        material=mat,
        boundary_conditions=(
            BoundaryConditionSpec(
                name="clamp_left",
                boundary_tag="left",
                kind="dirichlet",
                components=("x", "y"),
                values=(
                    Quantity(0.0, Unit.MM).to_canonical(),
                    Quantity(0.0, Unit.MM).to_canonical(),
                ),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(element_order=1),
        execution_profile=_profile(),
    )
    ps_backend = TorchFemBackend(physics_mode="2d-plane-strain-finite-strain")
    lin_ps = ps_backend.linearize(req_ps)
    coords = lin_ps.dof_coordinates
    u_nodal = np.column_stack(
        (0.08 * coords[:, 0], -0.02 * coords[:, 1] + 0.01 * coords[:, 0])
    )
    st_ps = lin_ps.state(lin_ps.vector(u_nodal))

    grad_u = st_ps.assembly.deformation_gradient[..., :2, :2].reshape(
        -1, 2, 2
    ) - np.eye(2)
    w_flat = lin_ps.quadrature_weights().reshape(-1)
    terms_ps = plane_strain_potential_energy(
        torch.tensor(grad_u, dtype=torch.float64),
        torch.tensor(w_flat, dtype=torch.float64),
        mat,
    )
    phys_energy_ps = (
        float(terms_ps.energy.item()) * reference_modulus(mat) * float(w_flat.sum())
    )
    assert phys_energy_ps == pytest.approx(
        st_ps.assembly.total_strain_energy, rel=1.0e-10
    )

    # 2) Axisymmetric objective & exact consistent tangent FD check
    cyl_mesh = axisymmetric_cylinder_mesh(
        r_inner=10.0, r_outer=15.0, height=3.0, nr=3, nz=2
    )
    req_axi = FemRequest(
        request_id="axi-tangent-fd",
        mesh=cyl_mesh,
        material=mat,
        boundary_conditions=(
            BoundaryConditionSpec(
                name="fix_bottom",
                boundary_tag="bottom",
                kind="dirichlet",
                components=("y",),
                values=(Quantity(0.0, Unit.MM).to_canonical(),),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(element_order=1),
        execution_profile=_profile(),
    )
    axi_backend = TorchFemBackend(physics_mode="2d-axisymmetric-finite-strain")
    lin_axi = axi_backend.linearize(req_axi)
    rng = np.random.default_rng(123)
    u0 = 0.03 * rng.standard_normal(lin_axi.dof_count)
    st0 = lin_axi.state(u0)

    u_elem = u0.reshape(-1, 2)[lin_axi.disc.conn]
    u_q = np.einsum("qa,eai->eqi", lin_axi.disc.n_ref, u_elem).reshape(-1, 2)
    grad_q = st0.assembly.deformation_gradient[..., :2, :2].reshape(-1, 2, 2) - np.eye(
        2
    )
    r_q = lin_axi.disc.quad_points[..., 0].reshape(-1)
    w_area = lin_axi.disc.quad_area_weights.reshape(-1)
    terms_axi = axisymmetric_potential_energy(
        torch.tensor(u_q, dtype=torch.float64),
        torch.tensor(grad_q, dtype=torch.float64),
        torch.tensor(r_q, dtype=torch.float64),
        torch.tensor(w_area, dtype=torch.float64),
        mat,
    )
    vol_axi = float(np.sum((2.0 * math.pi) * r_q * w_area))
    phys_energy_axi = float(terms_axi.energy.item()) * reference_modulus(mat) * vol_axi
    assert phys_energy_axi == pytest.approx(
        st0.assembly.total_strain_energy, rel=1.0e-10
    )

    eps = 1.0e-6
    v_dir = rng.standard_normal(lin_axi.dof_count)
    kv_analytic = st0.assembly.tangent @ v_dir
    kv_fd = (
        lin_axi.state(u0 + eps * v_dir).residual
        - lin_axi.state(u0 - eps * v_dir).residual
    ) / (2.0 * eps)
    assert np.linalg.norm(kv_analytic - kv_fd) / np.linalg.norm(kv_fd) < 1.0e-6

    # 3) Analytical Lamé pressurized thick-walled cylinder benchmark
    lame = run_level3c_pressurized_cylinder_axisymmetric()
    assert lame["passed"] is True
    assert lame["rel_error_inner"] < 5.0e-3
    assert lame["rel_error_outer"] < 5.0e-3


def test_frictionless_rigid_indenter_contact_and_dolfinx_plugin_isolation() -> None:
    # 1) Exact contact Hessian finite-difference verification
    indenter = RigidIndenterSpec(
        kind="circle",
        center=(5.0, 4.8),
        radius=3.0,
        penalty_stiffness=500.0,
        contact_boundary_tag="top",
    )
    ref_xy = np.array([[4.5, 2.0], [5.0, 2.0], [5.5, 2.0]], dtype=np.float64)
    u0 = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    cand = np.array([0, 1, 2], dtype=np.int64)
    _, f0, h0, _ = assemble_contact_terms(indenter, ref_xy, u0, cand)
    assert np.any(f0[:, 1] > 0.0)
    eps = 1.0e-6
    for idx in range(3):
        for comp in range(2):
            du = np.zeros_like(u0)
            du[idx, comp] = eps
            _, fp, _, _ = assemble_contact_terms(indenter, ref_xy, u0 + du, cand)
            _, fm, _, _ = assemble_contact_terms(indenter, ref_xy, u0 - du, cand)
            col_fd = (fp[idx] - fm[idx]) / (2.0 * eps)
            assert np.allclose(h0[idx, :, comp], col_fd, atol=1.0e-5, rtol=1.0e-5)

    # 2) Live 2D elastomeric indentation of Neo-Hookean block by rigid circular punch
    block = rectangular_strip_mesh(length_mm=10.0, height_mm=4.0, nx=10, ny=4)
    mat = _neo_hookean(e_mpa=10.0, nu=0.35)
    req = FemRequest(
        request_id="elastomer-indentation",
        mesh=block,
        material=mat,
        boundary_conditions=(
            BoundaryConditionSpec(
                name="fix_bottom",
                boundary_tag="bottom",
                kind="dirichlet",
                components=("x", "y"),
                values=(
                    Quantity(0.0, Unit.MM).to_canonical(),
                    Quantity(0.0, Unit.MM).to_canonical(),
                ),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(
            element_order=1,
            solver_tolerance=1.0e-9,
            max_newton_iterations=20,
        ),
        execution_profile=_profile(),
    )
    punch = RigidIndenterSpec(
        kind="circle",
        center=(5.0, 6.75),
        radius=3.0,
        penalty_stiffness=2.0e3,
        contact_boundary_tag="top",
    )
    u_contact, report = TorchFemBackend(
        physics_mode="2d-plane-strain-finite-strain"
    ).solve_contact(req, punch, max_auglag_iterations=10, gap_tolerance_mm=1.0e-4)
    assert report.admissible is True
    assert report.active_node_count >= 1
    assert report.max_penetration_mm < 1.0e-3
    assert float(u_contact[:, 1].min()) < -0.15
    assert report.total_contact_force_n[1] < 0.0

    # 3) Verify isolated LGPL-3.0 DolfinxBackend plugin contract without modifying MIT core
    plugin_src = str(ROOT / "plugins" / "formamech-dolfinx" / "src")
    sys.path.insert(0, plugin_src)
    try:
        from formamech_dolfinx import (  # type: ignore[import-not-found]
            DolfinxBackend,
            ThirdMediumContactConfig,
        )

        tmc = ThirdMediumContactConfig(k_third=1.0e-5, k_reg=1.0e-3)
        assert tmc.document()["regularization"] == "huhu"
        db = DolfinxBackend(tmc_config=tmc)
        assert isinstance(db, SolverBackend)
        assert db.capabilities.license_spdx == "LGPL-3.0-or-later"
        assert db.capabilities.supports_contact is True
        if not db.is_runtime_available():
            att = db.solve_many((req,))[0]
            assert att.success is False
            assert "dolfinx runtime unavailable" in att.message
    finally:
        sys.path.remove(plugin_src)


def test_3d_mesh_locking_refusal_surface_operator_and_level4_vvuq() -> None:
    tet4 = make_brick_tet4_mesh(10.0, 5.0, 4.0, nx=2, ny=1, nz=1)
    tet10 = make_brick_tet10_mesh(10.0, 5.0, 4.0, nx=2, ny=1, nz=1)
    hex8 = make_brick_hex8_mesh(10.0, 5.0, 4.0, nx=2, ny=1, nz=1)
    assert tet4.cell_type is CellType3D.TET4
    assert tet10.cell_type is CellType3D.TET10
    assert hex8.cell_type is CellType3D.HEX8

    tet4.validate_incompressibility_locking(0.30)
    with pytest.raises(VolumetricLockingError, match="TET4"):
        tet4.validate_incompressibility_locking(0.49)
    tet10.validate_incompressibility_locking(0.49)
    hex8.validate_incompressibility_locking(0.49)
    assert len(tet10.mesh_id) == 64

    centroids, normals, areas = tet10.surface_geometry()
    dtype = torch.float64
    c_t = torch.tensor(centroids, dtype=dtype).unsqueeze(0)
    n_t = torch.tensor(normals, dtype=dtype).unsqueeze(0)
    a_t = torch.tensor(areas, dtype=dtype).unsqueeze(0)
    m_faces = c_t.shape[1]
    feat_t = torch.zeros((1, m_faces, 6), dtype=dtype)
    feat_t[0, :4, 0] = 1.0

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
    ).to(dtype)
    assert op3d.architecture_id == SURFACE_3D_ARCHITECTURE_ID

    enc = op3d.encode_surface(c_t, n_t, a_t, feat_t)
    assert enc.slice_tokens.shape == (1, 4, 16)

    q_xyz = torch.tensor(
        [[[2.5, 1.5, 1.0], [5.0, 2.5, 2.0], [7.5, 3.5, 3.0]]], dtype=dtype
    )
    params = torch.tensor([[0.2, -0.1]], dtype=dtype)
    u3d, grad_u3d = op3d.displacement_and_gradient(
        encoding=enc, query_xyz=q_xyz, parameters=params, create_graph=True
    )
    assert u3d.shape == (1, 3, 3)
    assert grad_u3d.shape == (1, 3, 3, 3)

    eps = 1.0e-6
    with torch.no_grad():
        for axis in range(3):
            dq = torch.zeros_like(q_xyz)
            dq[..., axis] = eps
            up = op3d.displacement(
                encoding=enc, query_xyz=q_xyz + dq, parameters=params
            )
            um = op3d.displacement(
                encoding=enc, query_xyz=q_xyz - dq, parameters=params
            )
            fd = (up - um) / (2.0 * eps)
            assert torch.allclose(grad_u3d[..., axis], fd, atol=1.0e-6, rtol=1.0e-5)

    loss = (grad_u3d**2).mean()
    loss.backward()
    assert any(
        p.grad is not None and float(p.grad.abs().max().item()) > 0.0
        for p in op3d.parameters()
    )

    le10 = nafems_le10_spec()
    cooks3d = cooks_membrane_3d_spec()
    assert le10.target_qoi_value == pytest.approx(-5.38)
    assert CellType3D.TET4 not in cooks3d.required_cell_types
    smoke = build_level4_smoke_tet10_mesh()
    assert smoke["cell_type"] == "tet10"
    assert len(smoke["specs"]) == 2

    gate_ok = evaluate_ood_conformal_shift_gate(
        nominal_coverage=0.95,
        id_errors=[0.1, 0.2, 0.3, 0.4],
        id_bounds=[0.5, 0.5, 0.5, 0.5],
        ood_errors=[1.2, 0.9, 1.5],
        ood_bounds=[0.5, 0.5, 0.5],
        ood_accepted_flags=[False, False, False],
    )
    assert gate_ok.gate_passed is True
    assert gate_ok.ood_false_acceptance_rate == 0.0

    gate_fail = evaluate_ood_conformal_shift_gate(
        nominal_coverage=0.95,
        id_errors=[0.1, 0.2],
        id_bounds=[0.5, 0.5],
        ood_errors=[1.2, 0.9],
        ood_bounds=[0.5, 0.5],
        ood_accepted_flags=[True, False],
    )
    assert gate_fail.gate_passed is False
    assert gate_fail.false_acceptances == 1


def test_torch_fem_backend_dwr_all_four_qois_parity_with_skfem() -> None:
    """Verify TorchFemBackend matches ScikitFemBackend on all 4 live DWR QoIs."""
    from piginot.certify import (
        ReactionForceQoi,
        RegionalAverageStressQoi,
        SmoothedPeakVonMisesQoi,
        TotalStrainEnergyQoi,
        estimate_qoi_dwr,
    )

    mesh = rectangular_strip_mesh(length_mm=20.0, height_mm=10.0, nx=6, ny=3)
    req = FemRequest(
        request_id="torch-dwr-parity",
        mesh=mesh,
        material=_neo_hookean(e_mpa=10.0, nu=0.32),
        boundary_conditions=(
            BoundaryConditionSpec(
                name="left_fix",
                boundary_tag="left",
                kind="dirichlet",
                components=("x", "y"),
                values=(
                    Quantity(0.0, Unit.MM).to_canonical(),
                    Quantity(0.0, Unit.MM).to_canonical(),
                ),
            ),
            BoundaryConditionSpec(
                name="right_pull",
                boundary_tag="right",
                kind="dirichlet",
                components=("x",),
                values=(Quantity(1.2, Unit.MM).to_canonical(),),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(element_order=1, solver_tolerance=1.0e-11),
        execution_profile=_profile(),
    )
    sk_backend = ScikitFemBackend()
    th_backend = TorchFemBackend()
    att = sk_backend.solve_many((req,))[0]
    assert att.success and att.result is not None
    u_fem = np.asarray(att.result.displacement, dtype=np.float64)
    # Perturb interior DOFs slightly to create a non-trivial surrogate residual
    rng = np.random.default_rng(7)
    u_surr = u_fem.copy()
    nodes_arr = np.asarray(mesh.nodes, dtype=np.float64)
    interior = (nodes_arr[:, 0] > 1.0e-9) & (nodes_arr[:, 0] < 20.0 - 1.0e-9)
    u_surr[interior] += 0.02 * rng.standard_normal(u_surr[interior].shape)

    lin_sk = sk_backend.linearize(req)
    lin_th = th_backend.linearize(req)
    qois = (
        ReactionForceQoi("rxn", "right_pull"),
        RegionalAverageStressQoi("avg_p11", component=(0, 0)),
        TotalStrainEnergyQoi("energy"),
        SmoothedPeakVonMisesQoi("peak_vm"),
    )
    for qoi in qois:
        est_sk = estimate_qoi_dwr(lin_sk, u_surr, qoi)
        est_th = estimate_qoi_dwr(lin_th, u_surr, qoi)
        assert est_sk.guards.get("admissible") is True
        assert est_th.guards.get("admissible") is True
        assert est_sk.dwr_estimate is not None and est_th.dwr_estimate is not None
        assert est_th.qoi_value == pytest.approx(est_sk.qoi_value, rel=1.0e-9)
        assert est_th.dwr_estimate == pytest.approx(est_sk.dwr_estimate, rel=1.0e-7)
