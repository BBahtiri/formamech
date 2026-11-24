"""3D linearized eigen-buckling & Crisfield Riks arc-length continuation (V4-3).

Provides two production structural stability solvers:

1. **Linearized 3D Eigen-Buckling** (:func:`solve_3d_linear_buckling`):
   Computes the pre-buckling elastic stress field ``sigma_0`` under reference
   loads, assembles the exact 3D geometric (initial-stress) stiffness matrix
   ``K_G(sigma_0)`` on ``TET4``/``TET10`` meshes, and solves
   ``-K_G phi = (1 / lambda) K_0 phi`` via sparse symmetric generalized
   eigensolve (guaranteed SPD mass matrix ``K_0``) to recover the lowest
   positive critical buckling load factors ``lambda_1 <= lambda_2 <= ...``.
2. **Crisfield Spherical Arc-Length Continuation** (:func:`solve_riks_continuation`,
   :func:`solve_3d_truss_riks`, :func:`solve_3d_riks_arc_length`):
   Tracks equilibrium paths past snap-through / limit points (``d_lambda / ds < 0``)
   governed by the explicit ``RIKS_SOLVER_MACHINE`` state machine.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh  # type: ignore[import-untyped]
from scipy.sparse import coo_matrix, csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import eigsh  # type: ignore[import-untyped]

from piginot.contracts import MaterialSpec
from piginot.contracts.state_machine import (
    StateMachineSpec,
    StateSnapshot,
    TransitionSpec,
    register_state_machine,
)
from piginot.fem.backend_3d import (
    DirichletBC3D,
    TorchFemBackend3D,
    TractionBC3D,
    assemble_surface_tractions_3d,
    face_node_indices_3d,
)
from piginot.fem.linear_solvers import (
    LinearSolverConfig,
    LinearSolverMethod,
    solve_linear_system,
)
from piginot.geometry.mesh3d import CanonicalMesh3D

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

BUCKLING_3D_RESULT_SCHEMA: Final = "formamech.buckling-3d-result-v1"
RIKS_3D_RESULT_SCHEMA: Final = "formamech.riks-3d-result-v1"


class RiksSolverState(StrEnum):
    """Explicit lifecycle state machine for Crisfield arc-length continuation."""

    INITIALIZED = "INITIALIZED"
    STEPPING = "STEPPING"
    LIMIT_POINT_PASSED = "LIMIT_POINT_PASSED"
    CONVERGED = "CONVERGED"
    STALLED = "STALLED"


RIKS_SOLVER_MACHINE: Final[StateMachineSpec[RiksSolverState]] = register_state_machine(
    StateMachineSpec(
        machine_id="formamech.fem.riks_arc_length",
        description=(
            "Crisfield spherical arc-length continuation lifecycle: "
            "INITIALIZED -> STEPPING -> LIMIT_POINT_PASSED -> CONVERGED | STALLED."
        ),
        states=tuple(RiksSolverState),
        initial_states=(RiksSolverState.INITIALIZED,),
        terminal_states=(RiksSolverState.CONVERGED, RiksSolverState.STALLED),
        transitions=(
            TransitionSpec(
                RiksSolverState.INITIALIZED,
                RiksSolverState.STEPPING,
                "start_continuation",
                "Begin spherical arc-length equilibrium path continuation.",
            ),
            TransitionSpec(
                RiksSolverState.STEPPING,
                RiksSolverState.LIMIT_POINT_PASSED,
                "cross_limit_point",
                "Detected snap-through / limit point (d_lambda/ds < 0).",
            ),
            TransitionSpec(
                RiksSolverState.STEPPING,
                RiksSolverState.CONVERGED,
                "finish_monotonic",
                "Reached target arc-length steps along primary equilibrium path.",
            ),
            TransitionSpec(
                RiksSolverState.LIMIT_POINT_PASSED,
                RiksSolverState.CONVERGED,
                "finish_post_buckling",
                "Completed post-buckling continuation past limit point.",
            ),
            TransitionSpec(
                RiksSolverState.STEPPING,
                RiksSolverState.STALLED,
                "stall_stepping",
                "Arc-length Newton corrector stalled before reaching target steps.",
            ),
            TransitionSpec(
                RiksSolverState.LIMIT_POINT_PASSED,
                RiksSolverState.STALLED,
                "stall_post_buckling",
                "Post-buckling corrector stalled on descending branch.",
            ),
        ),
    )
)


@dataclass(frozen=True, slots=True)
class Buckling3DResult:
    """Verified 3D linearized eigen-buckling output."""

    mesh_id: str
    load_factors: tuple[float, ...]
    mode_shapes_nodal: FloatArray  # (num_modes, N_nodes, 3)
    prebuckling_displacement_mm: FloatArray  # (N_nodes, 3)
    reference_load_norm_n: float
    wall_time_s: float

    def document(self) -> dict[str, Any]:
        return {
            "schema": BUCKLING_3D_RESULT_SCHEMA,
            "mesh_id": self.mesh_id,
            "load_factors": [float(v) for v in self.load_factors],
            "critical_load_factor": float(self.load_factors[0]),
            "reference_load_norm_n": float(self.reference_load_norm_n),
            "wall_time_s": float(self.wall_time_s),
        }


def _elastic_lame_from_material(material: MaterialSpec) -> tuple[float, float]:
    """Extract linear Lamé parameters ``(lam, mu)`` in MPa from a MaterialSpec."""
    e_q = material.properties.get("elastic_modulus")
    nu_q = material.properties.get("poisson_ratio")
    if e_q is None or nu_q is None:
        raise ValueError(
            "linear buckling requires 'elastic_modulus' and 'poisson_ratio' in MaterialSpec"
        )
    e_mpa = float(e_q.value)
    nu = float(nu_q.value)
    mu = e_mpa / (2.0 * (1.0 + nu))
    lam = e_mpa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def _assemble_3d_geometric_stiffness(
    u_lin: FloatArray,
    cells: IntArray,
    d_phys: FloatArray,
    w_vol: FloatArray,
    material: MaterialSpec,
    n_dofs: int,
) -> csr_matrix:
    """Assemble exact 3D initial-stress geometric stiffness ``K_G(sigma_0)``."""
    lam, mu = _elastic_lame_from_material(material)
    n_elem, n_loc = cells.shape
    u_nod = u_lin.reshape(-1, 3)
    u_e = u_nod[cells]  # (E, n_loc, 3)

    # Small-strain gradient grad_u[e, q, i, j] = sum_a u_e[e, a, i] * d_phys[e, q, a, j]
    grad_u = np.einsum("eai,eqaj->eqij", u_e, d_phys)
    eps = 0.5 * (grad_u + np.swapaxes(grad_u, -1, -2))
    tr_eps = eps[..., 0, 0] + eps[..., 1, 1] + eps[..., 2, 2]
    eye3 = np.eye(3, dtype=np.float64)[None, None, :, :]
    sigma = lam * tr_eps[..., None, None] * eye3 + 2.0 * mu * eps  # (E, Q, 3, 3)

    # Scalar geometric stiffness per element:
    # K_gs[e, a, b] = sum_q w_vol[e, q] * dN_a[e, q, i] * sigma[e, q, i, j] * dN_b[e, q, j]
    k_gs = np.einsum("eq,eqai,eqij,eqbj->eab", w_vol, d_phys, sigma, d_phys)

    ii_s = np.repeat(cells[:, :, None], n_loc, axis=2).reshape(-1)
    jj_s = np.repeat(cells[:, None, :], n_loc, axis=1).reshape(-1)
    n_nodes = n_dofs // 3
    k_s = coo_matrix((k_gs.reshape(-1), (ii_s, jj_s)), shape=(n_nodes, n_nodes)).tocsr()
    coo_s = k_s.tocoo()

    rows_all: list[IntArray] = []
    cols_all: list[IntArray] = []
    vals_all: list[FloatArray] = []
    for c in range(3):
        rows_all.append(3 * coo_s.row.astype(np.int64) + c)
        cols_all.append(3 * coo_s.col.astype(np.int64) + c)
        vals_all.append(coo_s.data.astype(np.float64))
    k_g = coo_matrix(
        (
            np.concatenate(vals_all),
            (np.concatenate(rows_all), np.concatenate(cols_all)),
        ),
        shape=(n_dofs, n_dofs),
    ).tocsr()
    return (0.5 * (k_g + k_g.T)).tocsr()


def solve_3d_linear_buckling(
    mesh: CanonicalMesh3D,
    material: MaterialSpec,
    *,
    dirichlet_bcs: Sequence[DirichletBC3D],
    traction_bcs: Sequence[TractionBC3D] = (),
    nodal_forces_n: FloatArray | None = None,
    num_modes: int = 3,
) -> Buckling3DResult:
    """Compute lowest positive critical buckling load factors ``(K_0 + lambda K_G) phi = 0``."""
    t0 = time.perf_counter()
    if type(num_modes) is not int or num_modes < 1:
        raise ValueError("num_modes must be >= 1")
    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes

    helper = TorchFemBackend3D()
    cells, d_phys, w_vol, _, rows, cols = helper._precompute_geometry(mesh)
    u_zero = np.zeros(n_dofs, dtype=np.float64)
    _, k_mat, _, _, _, _ = helper._assemble(
        u_zero, cells, d_phys, w_vol, rows, cols, material, n_dofs
    )
    k0 = (0.5 * (k_mat + k_mat.T)).tocsr()

    f_ref = np.zeros(n_dofs, dtype=np.float64)
    if traction_bcs:
        f_ref += assemble_surface_tractions_3d(mesh, traction_bcs).reshape(n_dofs)
    if nodal_forces_n is not None:
        f_ref += np.asarray(nodal_forces_n, dtype=np.float64).reshape(n_dofs)
    if float(np.linalg.norm(f_ref)) <= 1.0e-15:
        raise ValueError("non-zero reference load is required for buckling analysis")

    comp_idx = {"x": 0, "y": 1, "z": 2}
    constrained = np.zeros(n_dofs, dtype=bool)
    for bc in dirichlet_bcs:
        nids = face_node_indices_3d(mesh, bc.face_tags)
        for comp in bc.components:
            c = comp_idx[comp]
            for nid in nids:
                constrained[3 * nid + c] = True
    free = np.nonzero(~constrained)[0]
    if free.size <= num_modes:
        raise ValueError("free DOF count must exceed num_modes")

    k0_ff = k0[free][:, free].tocsr()
    u_f, rep = solve_linear_system(
        k0_ff,
        f_ref[free],
        config=LinearSolverConfig(method=LinearSolverMethod.DIRECT),
        is_symmetric=True,
    )
    if not rep.converged:
        raise RuntimeError(f"pre-buckling linear solve failed: {rep.reason}")
    u_lin = np.zeros(n_dofs, dtype=np.float64)
    u_lin[free] = u_f

    kg = _assemble_3d_geometric_stiffness(u_lin, cells, d_phys, w_vol, material, n_dofs)
    neg_kg_ff = (-kg[free][:, free]).tocsc()
    k0_ff_csc = k0_ff.tocsc()

    # Solve (-K_G) phi = mu K_0 phi with guaranteed SPD K_0; positive buckling
    # factors are lambda = 1 / mu for largest positive mu.
    if free.size <= 350:
        mu_all, vecs_all = eigh(neg_kg_ff.toarray(), k0_ff_csc.toarray())
        pos_mask = mu_all > 1.0e-12
        if int(np.sum(pos_mask)) < num_modes:
            raise ValueError(
                "fewer than num_modes positive compressive buckling modes found"
            )
        pos_idx = np.nonzero(pos_mask)[0]
        # Descending mu => ascending lambda = 1 / mu
        best = pos_idx[np.argsort(mu_all[pos_idx])[::-1][:num_modes]]
        mu_sel = mu_all[best]
        vecs_sel = vecs_all[:, best]
    else:
        k_req = min(max(num_modes + 2, 4), int(free.size) - 2)
        mu_raw, vecs_raw = eigsh(
            neg_kg_ff,
            k=k_req,
            M=k0_ff_csc,
            which="LA",
        )
        pos_idx = np.nonzero(mu_raw > 1.0e-12)[0]
        if pos_idx.size < num_modes:
            raise ValueError(
                "fewer than num_modes positive compressive buckling modes found"
            )
        order = pos_idx[np.argsort(mu_raw[pos_idx])[::-1][:num_modes]]
        mu_sel = np.asarray(mu_raw[order], dtype=np.float64)
        vecs_sel = np.asarray(vecs_raw[:, order], dtype=np.float64)

    lambdas = tuple(float(1.0 / m) for m in mu_sel)
    modes = np.zeros((num_modes, n_nodes, 3), dtype=np.float64)
    for k in range(num_modes):
        full_v = np.zeros(n_dofs, dtype=np.float64)
        v_k = vecs_sel[:, k]
        max_abs = float(np.max(np.abs(v_k)))
        if max_abs > 0.0:
            v_k = v_k / max_abs
        full_v[free] = v_k
        modes[k] = full_v.reshape(n_nodes, 3)

    return Buckling3DResult(
        mesh_id=mesh.mesh_id,
        load_factors=lambdas,
        mode_shapes_nodal=modes,
        prebuckling_displacement_mm=u_lin.reshape(n_nodes, 3),
        reference_load_norm_n=float(np.linalg.norm(f_ref)),
        wall_time_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Crisfield Spherical Arc-Length (Riks) Continuation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RiksConfig:
    """Configuration for Crisfield spherical arc-length continuation."""

    arc_length: float = 0.25
    psi: float = 1.0
    max_steps: int = 30
    max_newton_iters: int = 12
    tolerance: float = 1.0e-8
    desired_iters: int = 4
    min_arc_length: float = 1.0e-3
    max_arc_length: float = 2.0


@dataclass(frozen=True, slots=True)
class RiksStepRecord:
    """One converged equilibrium point along a Riks continuation path."""

    step_index: int
    load_factor: float
    displacement_norm: float
    residual_norm: float
    newton_iters: int
    arc_length_used: float


@dataclass(frozen=True, slots=True)
class RiksContinuationResult:
    """Auditable equilibrium path traced by Crisfield arc-length continuation."""

    state: RiksSolverState
    state_history: tuple[StateSnapshot[RiksSolverState], ...]
    limit_point_passed: bool
    peak_load_factor: float
    steps: tuple[RiksStepRecord, ...]
    final_displacement: FloatArray
    displacement_history: FloatArray  # (S+1, n_free)
    load_factor_history: FloatArray  # (S+1,)
    wall_time_s: float

    def document(self) -> dict[str, Any]:
        return {
            "schema": RIKS_3D_RESULT_SCHEMA,
            "state": self.state.value,
            "limit_point_passed": self.limit_point_passed,
            "peak_load_factor": float(self.peak_load_factor),
            "step_count": len(self.steps),
            "max_residual_norm": (
                max(s.residual_norm for s in self.steps) if self.steps else 0.0
            ),
            "load_factors": [float(v) for v in self.load_factor_history],
            "wall_time_s": float(self.wall_time_s),
        }


def solve_riks_continuation(
    internal_force_and_tangent: Callable[[FloatArray], tuple[FloatArray, FloatArray]],
    *,
    n_free: int,
    f_ref: FloatArray,
    config: RiksConfig | None = None,
) -> RiksContinuationResult:
    """Crisfield spherical arc-length continuation on an arbitrary nonlinear system.

    Traces ``f_int(u) - lambda * f_ref = 0`` with spherical constraint::

        ||du||^2 + psi^2 * (d_lambda)^2 * ||f_ref||^2 = ds^2
    """
    t0 = time.perf_counter()
    cfg = RiksConfig() if config is None else config
    f_vec = np.asarray(f_ref, dtype=np.float64).reshape(n_free)
    f_norm2 = float(np.dot(f_vec, f_vec))
    if f_norm2 <= 1.0e-20:
        raise ValueError("f_ref must be non-zero")
    psi2_f2 = (cfg.psi**2) * f_norm2

    state = RiksSolverState.INITIALIZED
    history: list[StateSnapshot[RiksSolverState]] = [
        RIKS_SOLVER_MACHINE.snapshot(state)
    ]
    RIKS_SOLVER_MACHINE.require_transition(state, RiksSolverState.STEPPING)
    state = RiksSolverState.STEPPING
    history.append(RIKS_SOLVER_MACHINE.snapshot(state))

    u: FloatArray = np.zeros(n_free, dtype=np.float64)
    lam = 0.0
    du_prev: FloatArray = np.zeros(n_free, dtype=np.float64)
    dlam_prev = 0.0
    ds = float(cfg.arc_length)
    limit_passed = False

    u_hist: list[FloatArray] = [u.copy()]
    lam_hist: list[float] = [lam]
    step_records: list[RiksStepRecord] = []

    for step in range(1, cfg.max_steps + 1):
        f_int0, k_t0 = internal_force_and_tangent(u)
        # Tangent predictor du_t = K_t^{-1} f_ref
        try:
            du_t = np.linalg.solve(k_t0, f_vec)
        except np.linalg.LinAlgError:
            k_reg = k_t0 + 1.0e-10 * np.eye(n_free, dtype=np.float64)
            du_t = np.linalg.solve(k_reg, f_vec)

        denom_pred = math.sqrt(float(np.dot(du_t, du_t)) + psi2_f2)
        dlam_p = ds / max(denom_pred, 1.0e-15)
        du_p = dlam_p * du_t

        # Consistent orientation with previous converged increment
        if step > 1:
            dot_prev = float(np.dot(du_p, du_prev)) + psi2_f2 * dlam_p * dlam_prev
            if dot_prev < 0.0:
                dlam_p = -dlam_p
                du_p = -du_p
        else:
            # First step increases load factor
            if dlam_p < 0.0:
                dlam_p = -dlam_p
                du_p = -du_p

        du = du_p.copy()
        dlam = dlam_p
        converged_step = False
        res_norm = float("inf")
        iters_used = 0

        for it in range(1, cfg.max_newton_iters + 1):
            iters_used = it
            u_try = u + du
            lam_try = lam + dlam
            f_int, k_t = internal_force_and_tangent(u_try)
            r_vec = lam_try * f_vec - f_int
            res_norm = float(np.linalg.norm(r_vec)) / max(math.sqrt(f_norm2), 1.0)
            if res_norm <= cfg.tolerance:
                converged_step = True
                break

            try:
                du_r = np.linalg.solve(k_t, r_vec)
                du_f = np.linalg.solve(k_t, f_vec)
            except np.linalg.LinAlgError:
                break

            # Crisfield spherical quadratic: a * ddlam^2 + b * ddlam + c = 0
            a_q = float(np.dot(du_f, du_f)) + psi2_f2
            du_bar = du + du_r
            b_q = 2.0 * (float(np.dot(du_bar, du_f)) + psi2_f2 * dlam)
            c_q = float(np.dot(du_bar, du_bar)) + psi2_f2 * (dlam**2) - ds * ds

            disc = b_q * b_q - 4.0 * a_q * c_q
            if disc >= 0.0 and a_q > 1.0e-20:
                sq = math.sqrt(disc)
                r1 = (-b_q + sq) / (2.0 * a_q)
                r2 = (-b_q - sq) / (2.0 * a_q)
                # Choose root with maximal alignment to current increment (du, dlam)
                cand1_u = du_bar + r1 * du_f
                cand2_u = du_bar + r2 * du_f
                cos1 = float(np.dot(du, cand1_u)) + psi2_f2 * dlam * (dlam + r1)
                cos2 = float(np.dot(du, cand2_u)) + psi2_f2 * dlam * (dlam + r2)
                ddlam = r1 if cos1 >= cos2 else r2
            else:
                # Linearized orthogonal corrector fallback
                denom_lin = float(np.dot(du, du_f)) + psi2_f2 * dlam
                if abs(denom_lin) <= 1.0e-15:
                    ddlam = 0.0
                else:
                    ddlam = -float(np.dot(du, du_r)) / denom_lin

            du = du_bar + ddlam * du_f
            dlam = dlam + ddlam

        if not converged_step:
            # Cut arc length and retry once
            ds_next = max(0.5 * ds, cfg.min_arc_length)
            if ds_next < ds - 1.0e-12:
                ds = ds_next
                continue
            RIKS_SOLVER_MACHINE.require_transition(state, RiksSolverState.STALLED)
            state = RiksSolverState.STALLED
            history.append(RIKS_SOLVER_MACHINE.snapshot(state))
            break

        u = np.asarray(u + du, dtype=np.float64)
        lam = lam + dlam
        du_prev = np.asarray(du, dtype=np.float64).copy()
        dlam_prev = dlam

        if dlam < -1.0e-9 and not limit_passed:
            limit_passed = True
            RIKS_SOLVER_MACHINE.require_transition(
                state, RiksSolverState.LIMIT_POINT_PASSED
            )
            state = RiksSolverState.LIMIT_POINT_PASSED
            history.append(RIKS_SOLVER_MACHINE.snapshot(state))

        u_hist.append(u.copy())
        lam_hist.append(lam)
        step_records.append(
            RiksStepRecord(
                step_index=step,
                load_factor=float(lam),
                displacement_norm=float(np.linalg.norm(u)),
                residual_norm=float(res_norm),
                newton_iters=iters_used,
                arc_length_used=float(ds),
            )
        )
        # Adaptive arc-length scaling
        scale = math.sqrt(cfg.desired_iters / max(iters_used, 1))
        ds = min(cfg.max_arc_length, max(cfg.min_arc_length, ds * scale))

    if state in (RiksSolverState.STEPPING, RiksSolverState.LIMIT_POINT_PASSED):
        RIKS_SOLVER_MACHINE.require_transition(state, RiksSolverState.CONVERGED)
        state = RiksSolverState.CONVERGED
        history.append(RIKS_SOLVER_MACHINE.snapshot(state))

    lam_arr = np.asarray(lam_hist, dtype=np.float64)
    return RiksContinuationResult(
        state=state,
        state_history=tuple(history),
        limit_point_passed=limit_passed,
        peak_load_factor=float(np.max(lam_arr)),
        steps=tuple(step_records),
        final_displacement=u.copy(),
        displacement_history=np.vstack(u_hist),
        load_factor_history=lam_arr,
        wall_time_s=time.perf_counter() - t0,
    )


def solve_3d_truss_riks(
    nodes_mm: FloatArray,
    bars: IntArray,
    *,
    ea_n: float,
    constrained_dofs: Sequence[int],
    f_ref_n: FloatArray,
    config: RiksConfig | None = None,
) -> RiksContinuationResult:
    """Solve geometrically nonlinear 3D Green-Lagrange truss/arch snap-through via Riks."""
    nodes = np.asarray(nodes_mm, dtype=np.float64)
    elems = np.asarray(bars, dtype=np.int64)
    n_nodes = int(nodes.shape[0])
    n_dofs = 3 * n_nodes
    fixed = np.zeros(n_dofs, dtype=bool)
    for d in constrained_dofs:
        fixed[int(d)] = True
    free = np.nonzero(~fixed)[0]
    f_full = np.asarray(f_ref_n, dtype=np.float64).reshape(n_dofs)

    def _fint_and_kt(u_f: FloatArray) -> tuple[FloatArray, FloatArray]:
        u_full = np.zeros(n_dofs, dtype=np.float64)
        u_full[free] = u_f
        u_nod = u_full.reshape(n_nodes, 3)
        f_int = np.zeros(n_dofs, dtype=np.float64)
        k_t = np.zeros((n_dofs, n_dofs), dtype=np.float64)
        for bar in elems:
            n0, n1 = int(bar[0]), int(bar[1])
            dx0 = nodes[n1] - nodes[n0]
            l0_sq = float(np.dot(dx0, dx0))
            l0 = math.sqrt(l0_sq)
            dx = dx0 + (u_nod[n1] - u_nod[n0])
            # Green-Lagrange axial strain E_gl = (L^2 - L0^2) / (2 L0^2)
            e_gl = 0.5 * (float(np.dot(dx, dx)) - l0_sq) / l0_sq
            n_ax = ea_n * e_gl  # 2nd PK axial force * L0 / L
            # Internal nodal force: f1 = (N / L0) * dx, f0 = -f1
            f_e = (n_ax / l0) * dx
            s0, s1 = 3 * n0, 3 * n1
            f_int[s0 : s0 + 3] -= f_e
            f_int[s1 : s1 + 3] += f_e
            # Exact tangent: K_e = (EA / L0^3) (dx dx^T) + (N / L0) I_3
            k_blk = (ea_n / (l0**3)) * np.outer(dx, dx) + (n_ax / l0) * np.eye(3)
            k_t[s0 : s0 + 3, s0 : s0 + 3] += k_blk
            k_t[s1 : s1 + 3, s1 : s1 + 3] += k_blk
            k_t[s0 : s0 + 3, s1 : s1 + 3] -= k_blk
            k_t[s1 : s1 + 3, s0 : s0 + 3] -= k_blk
        return f_int[free], k_t[np.ix_(free, free)]

    return solve_riks_continuation(
        _fint_and_kt,
        n_free=int(free.size),
        f_ref=f_full[free],
        config=config,
    )


def solve_3d_riks_arc_length(
    mesh: CanonicalMesh3D,
    material: MaterialSpec,
    *,
    dirichlet_bcs: Sequence[DirichletBC3D],
    traction_bcs: Sequence[TractionBC3D],
    config: RiksConfig | None = None,
) -> RiksContinuationResult:
    """Run Crisfield spherical arc-length continuation on a 3D hyperelastic solid."""
    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes
    helper = TorchFemBackend3D()
    cells, d_phys, w_vol, _, rows, cols = helper._precompute_geometry(mesh)
    f_ref = assemble_surface_tractions_3d(mesh, traction_bcs).reshape(n_dofs)

    comp_idx = {"x": 0, "y": 1, "z": 2}
    constrained = np.zeros(n_dofs, dtype=bool)
    for bc in dirichlet_bcs:
        nids = face_node_indices_3d(mesh, bc.face_tags)
        for comp in bc.components:
            c = comp_idx[comp]
            for nid in nids:
                constrained[3 * nid + c] = True
    free = np.nonzero(~constrained)[0]

    def _fint_and_kt(u_f: FloatArray) -> tuple[FloatArray, FloatArray]:
        u_full = np.zeros(n_dofs, dtype=np.float64)
        u_full[free] = u_f
        f_int, k_mat, _, _, _, min_det = helper._assemble(
            u_full, cells, d_phys, w_vol, rows, cols, material, n_dofs
        )
        if min_det <= 1.0e-8:
            raise ValueError("inverted element encountered during 3D Riks continuation")
        return f_int[free], k_mat[free][:, free].toarray()

    return solve_riks_continuation(
        _fint_and_kt,
        n_free=int(free.size),
        f_ref=f_ref[free],
        config=config,
    )


__all__ = (
    "BUCKLING_3D_RESULT_SCHEMA",
    "RIKS_3D_RESULT_SCHEMA",
    "RIKS_SOLVER_MACHINE",
    "Buckling3DResult",
    "RiksConfig",
    "RiksContinuationResult",
    "RiksSolverState",
    "RiksStepRecord",
    "solve_3d_linear_buckling",
    "solve_3d_riks_arc_length",
    "solve_3d_truss_riks",
    "solve_riks_continuation",
)
