"""Gate G-V5-5b tests: Vector-BC coupled multiphysics 3D operator & guarded transient rollout."""

from __future__ import annotations

import time

import numpy as np
import torch

from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import DirichletBC3D, TractionBC3D
from piginot.fem.electro_thermal_3d import (
    ElectroThermalMaterialSpec,
    VolumetricPowerSource3D,
    _assemble_multimat_thermal_matrices,
    solve_3d_electro_thermal_mechanical,
    solve_3d_multi_material_heat_conduction,
)
from piginot.fem.thermo_mechanical_3d import TemperatureBC3D
from piginot.geometry.mesh3d import make_brick_tet4_mesh
from piginot.operators.multiphysics_operator_3d import (
    GuardedTransient3DStepper,
    Multiphysics3DConfig,
    Multiphysics3DOperator,
    VectorBcSurfaceSpec3D,
    build_vector_bc_surface_tokens_3d,
)
from piginot.operators.surface_operator_3d import (
    Surface3DConfig,
    Surface3DEnergyOperator,
)


def _make_bc_shift_dataset() -> tuple[
    list[tuple[object, VectorBcSurfaceSpec3D, np.ndarray, np.ndarray, list[float]]],
    list[tuple[object, VectorBcSurfaceSpec3D, np.ndarray, np.ndarray, list[float]]],
]:
    """Generate 3D coupled thermal-mechanical runs across varied traction & power-map locations."""
    mesh = make_brick_tet4_mesh(16.0, 8.0, 6.0, nx=3, ny=2, nz=2)
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    mat = ElectroThermalMaterialSpec(
        part_id="chip",
        conductivity_xyz_w_per_mm_k=(0.15, 0.15, 0.15),
        volumetric_heat_capacity_j_per_mm3_k=2.0e-3,
        thermal_expansion_per_k=1.5e-5,
        reference_temp_k=293.15,
        mechanical=MaterialSpec(
            name="chip-mech",
            model="compressible_neo_hookean",
            properties={
                "elastic_modulus": Quantity(1000.0, Unit.MPA).to_canonical(),
                "poisson_ratio": Quantity(0.30, Unit.ONE).to_canonical(),
            },
        ),
    )

    def _solve_case(
        load_face: str, traction: tuple[float, float, float], hot_x: float, q_mag: float
    ) -> tuple[object, VectorBcSurfaceSpec3D, np.ndarray, np.ndarray, list[float]]:
        def _q_fn(pts: np.ndarray) -> np.ndarray:
            d2 = np.sum((pts - np.array([hot_x, 4.0, 3.0])) ** 2, axis=1)
            return q_mag * np.exp(-d2 / 12.0)

        res = solve_3d_electro_thermal_mechanical(
            mesh,
            [mat],
            dirichlet_bcs=(
                DirichletBC3D("fix", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
            ),
            traction_bcs=(TractionBC3D("load", (load_face,), traction),),
            temperature_bcs=(TemperatureBC3D("sink", ("x_min",), 293.15),),
            volumetric_sources=(VolumetricPowerSource3D("hot", spatial_fn=_q_fn),),
            load_steps=1,
        )
        assert res.converged
        spec = VectorBcSurfaceSpec3D(
            dirichlet_face_masks={"x_min": (1.0, 1.0, 1.0)},
            dirichlet_face_values={"x_min": (0.0, 0.0, 0.0)},
            traction_face_vectors={load_face: traction},
            temp_dirichlet_faces={"x_min": 293.15},
            heat_flux_faces={},
            convection_faces={},
            material_k_e_alpha=(0.15, 1000.0, 1.5e-5),
            volumetric_power_points_xyz=np.array([[hot_x, 4.0, 3.0]], dtype=np.float64),
            volumetric_power_values=np.array([q_mag], dtype=np.float64),
        )
        # Note scalar parameter vector intentionally does NOT encode load_face or hot_x,
        # so only the continuous vector-BC + power-source tokens distinguish the BC shift!
        t_mag = float(np.linalg.norm(traction))
        return (
            mesh,
            spec,
            res.nodal_temperature_k,
            res.nodal_displacement,
            [t_mag, q_mag],
        )

    train_cases = [
        _solve_case("x_max", (0.0, -0.4, 0.0), 12.0, 0.03),
        _solve_case("x_max", (0.0, -1.0, 0.0), 12.0, 0.07),
        _solve_case("y_max", (0.0, 0.0, 0.4), 6.0, 0.03),
        _solve_case("y_max", (0.0, 0.0, 1.0), 6.0, 0.07),
        _solve_case("z_max", (0.4, 0.0, 0.0), 10.0, 0.03),
        _solve_case("z_max", (1.0, 0.0, 0.0), 10.0, 0.07),
    ]
    test_cases = [
        _solve_case("x_max", (0.0, -0.7, 0.0), 12.0, 0.05),
        _solve_case("y_max", (0.0, 0.0, 0.7), 6.0, 0.05),
        _solve_case("z_max", (0.7, 0.0, 0.0), 10.0, 0.05),
    ]
    del nodes
    return train_cases, test_cases


def test_vector_bc_multiphysics_operator_beats_scalar_baseline_by_50_percent() -> None:
    """Gate G-V5-5b: Vector-BC operator cuts BC/power-map shift error >= 50% vs. scalar baseline."""
    train_cases, test_cases = _make_bc_shift_dataset()

    # 1) Train Vector-BC Multiphysics3DOperator jointly on (T, u)
    torch.manual_seed(0)
    mp_op = Multiphysics3DOperator(
        Multiphysics3DConfig(
            parameter_dim=2, hidden_dim=32, slice_count=6, num_blocks=2
        )
    )
    opt_mp = torch.optim.Adam(mp_op.parameters(), lr=6.0e-3)

    # 2) Train scalar-only baseline (geometry coords/normals only, blind to vector-BC & power-source location)
    torch.manual_seed(0)
    base_op = Surface3DEnergyOperator(
        Surface3DConfig(
            parameter_dim=2,
            surface_feature_dim=6,
            hidden_dim=32,
            slice_count=6,
            num_blocks=2,
        )
    )
    opt_base = torch.optim.Adam(base_op.parameters(), lr=6.0e-3)

    # Pre-build tensors
    mp_train = []
    base_train = []
    for mesh, spec, t_true, u_true, p_vals in train_cases:
        c_t, n_t, a_t, f_t, q_t = build_vector_bc_surface_tokens_3d(mesh, spec)  # type: ignore[arg-type]
        pts = torch.tensor(mesh.nodes, dtype=torch.float32).unsqueeze(0)  # type: ignore[attr-defined]
        p_t = torch.tensor([p_vals], dtype=torch.float32)
        t_t = torch.tensor(t_true, dtype=torch.float32).unsqueeze(0)
        u_t = torch.tensor(u_true, dtype=torch.float32).unsqueeze(0)
        mp_train.append((c_t, a_t, f_t, q_t, pts, p_t, t_t, u_t))
        base_train.append((c_t, n_t, a_t, torch.cat((c_t, n_t), dim=-1), pts, p_t, u_t))

    n_tr = float(len(mp_train))
    for _ in range(280):
        opt_mp.zero_grad()
        loss_mp = torch.zeros(())
        for c_t, a_t, f_t, q_t, pts, p_t, t_t, u_t in mp_train:
            enc = mp_op.encode(c_t, a_t, f_t, q_t)
            t_hat, u_hat = mp_op.predict_fields(enc, pts, p_t, reference_temp_k=293.15)
            loss_mp = (
                loss_mp
                + torch.mean(((t_hat - t_t) / 20.0) ** 2)
                + torch.mean((u_hat - u_t) ** 2) * 1000.0
            )
        (loss_mp / n_tr).backward()  # type: ignore[no-untyped-call]
        opt_mp.step()

        opt_base.zero_grad()
        loss_b = torch.zeros(())
        for c_t, n_t, a_t, bf_t, pts, p_t, u_t in base_train:
            enc_b = base_op.encode_surface(c_t, n_t, a_t, bf_t)
            u_hat_b = base_op.displacement(
                encoding=enc_b, query_xyz=pts, parameters=p_t
            )
            loss_b = loss_b + torch.mean((u_hat_b - u_t) ** 2) * 1000.0
        (loss_b / n_tr).backward()  # type: ignore[no-untyped-call]
        opt_base.step()

    mp_op.eval()
    base_op.eval()
    mp_u_errs: list[float] = []
    base_u_errs: list[float] = []
    mp_t_errs: list[float] = []
    with torch.no_grad():
        t_start = time.perf_counter()
        for mesh, spec, t_true, u_true, p_vals in test_cases:
            c_t, n_t, a_t, f_t, q_t = build_vector_bc_surface_tokens_3d(mesh, spec)  # type: ignore[arg-type]
            pts = torch.tensor(mesh.nodes, dtype=torch.float32).unsqueeze(0)  # type: ignore[attr-defined]
            p_t = torch.tensor([p_vals], dtype=torch.float32)

            enc = mp_op.encode(c_t, a_t, f_t, q_t)
            t_hat, u_hat = mp_op.predict_fields(enc, pts, p_t, reference_temp_k=293.15)
            mp_u_errs.append(float(np.linalg.norm(u_hat[0].numpy() - u_true)))
            mp_t_errs.append(float(np.mean(np.abs(t_hat[0].numpy() - t_true))))

            enc_b = base_op.encode_surface(c_t, n_t, a_t, torch.cat((c_t, n_t), dim=-1))
            u_hat_b = base_op.displacement(
                encoding=enc_b, query_xyz=pts, parameters=p_t
            )
            base_u_errs.append(float(np.linalg.norm(u_hat_b[0].numpy() - u_true)))
        infer_s = (time.perf_counter() - t_start) / len(test_cases)

    mean_mp_u = float(np.mean(mp_u_errs))
    mean_base_u = float(np.mean(base_u_errs))
    reduction = 1.0 - (mean_mp_u / max(mean_base_u, 1.0e-12))
    assert reduction >= 0.50, (
        f"Expected >= 50% BC-shift error reduction vs scalar baseline, got {reduction:.1%} "
        f"(mp={mean_mp_u:.5f}, base={mean_base_u:.5f})"
    )
    assert float(np.mean(mp_t_errs)) < 5.0
    assert infer_s < 0.1


def test_guarded_transient_3d_stepper_triggers_fem_fallback_on_drift() -> None:
    """Gate G-V5-5b: Guarded transient rollout catches drift & falls back to exact FEM."""
    mesh = make_brick_tet4_mesh(20.0, 6.0, 6.0, nx=4, ny=1, nz=1)
    mat = ElectroThermalMaterialSpec(
        part_id="bar",
        conductivity_xyz_w_per_mm_k=(0.05, 0.05, 0.05),
        volumetric_heat_capacity_j_per_mm3_k=3.0e-3,
        thermal_expansion_per_k=1.2e-5,
        reference_temp_k=300.0,
        mechanical=MaterialSpec(
            name="bar-m",
            model="compressible_neo_hookean",
            properties={
                "elastic_modulus": Quantity(1000.0, Unit.MPA).to_canonical(),
                "poisson_ratio": Quantity(0.30, Unit.ONE).to_canonical(),
            },
        ),
    )
    dt_s = 1.0
    k_mat, c_mat, _, _, _, _ = _assemble_multimat_thermal_matrices(
        mesh, [mat] * len(mesh.cells)
    )
    lhs = (c_mat / dt_s + k_mat).tocsr()
    rhs_mat = (c_mat / dt_s).tocsr()
    tbcs = (
        TemperatureBC3D("left", ("x_min",), 300.0),
        TemperatureBC3D("right", ("x_max",), 350.0),
    )

    from piginot.fem.backend_3d import face_node_indices_3d

    fixed = set(face_node_indices_3d(mesh, ("x_min", "x_max")))
    free = np.array(
        [i for i in range(len(mesh.nodes)) if i not in fixed], dtype=np.int64
    )

    def _exact_step(t_prev: np.ndarray, step: int) -> np.ndarray:
        del step
        t_next, _ = solve_3d_multi_material_heat_conduction(
            mesh,
            [mat],
            temperature_bcs=tbcs,
            dt_s=dt_s,
            num_steps=1,
            initial_temperature_k=t_prev,
            crank_nicolson=False,
        )
        return t_next

    def _drifting_neural_step(t_prev: np.ndarray, step: int) -> np.ndarray:
        exact = _exact_step(t_prev, step)
        # Inject exponential instability on steps 3 and 5
        if step in (3, 5):
            return exact + 80.0 * np.sin(np.arange(len(t_prev), dtype=np.float64))
        return exact

    stepper = GuardedTransient3DStepper(
        neural_step_fn=_drifting_neural_step,
        exact_fem_step_fn=_exact_step,
        lhs_matrix=lhs,
        rhs_matrix=rhs_mat,
        rhs_load=np.zeros(len(mesh.nodes), dtype=np.float64),
        free_dofs=free,
        max_relative_residual=0.05,
    )
    t0 = np.full(len(mesh.nodes), 300.0, dtype=np.float64)
    out = stepper.rollout(t0, num_steps=6)
    assert out.fem_fallback_count == 2
    assert out.step_records[2].used_fem_fallback is True
    assert out.step_records[4].used_fem_fallback is True
    assert out.step_records[0].used_fem_fallback is False
    assert out.max_recorded_residual <= 1.0e-8
    assert np.all(np.isfinite(out.temperature_trajectory_k[-1]))
    assert float(np.max(out.temperature_trajectory_k[-1])) <= 350.0 + 1.0e-6
