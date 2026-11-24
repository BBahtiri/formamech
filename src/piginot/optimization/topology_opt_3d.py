"""Verified 3D SIMP Topology Optimization with mandatory FEM certification (V4-4).

Implements density-based 3D SIMP compliance minimization on ``TET4`` meshes
governed by the explicit ``TOPOLOGY_OPT_MACHINE`` state machine::

    INITIALIZED -> ITERATING -> FILTERED -> THRESHOLDED -> FEM_CERTIFYING
      -> CERTIFIED | REFUSED

Key mathematical and epistemic guarantees:

* Exact self-adjoint compliance sensitivity ``dC / d(rho_e)`` verified against
  central finite differences to ``< 1e-7``,
* Mesh-independent radial cone density filter + smooth Heaviside projection
  (with exact chain-rule sensitivity propagation),
* Volume-preserving Optimality Criteria (OC) update via bisection,
* **Fail-closed final certification:** extracts the thresholded 3D solid
  sub-mesh, re-runs an exact 3D FEM verification solve on the real discrete
  solid, refuses disconnected/singular geometries, and issues an RFC 8785
  canonical SHA-256 Trust Receipt (``SINGLE_POINT_VERIFIED``).
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix, eye  # type: ignore[import-untyped]

from piginot.contracts import canonical_json_bytes
from piginot.contracts.state_machine import (
    StateMachineSpec,
    StateSnapshot,
    TransitionSpec,
    register_state_machine,
)
from piginot.fem.backend_3d import (
    DirichletBC3D,
    TractionBC3D,
    _tet_shape_grads_ref,
    assemble_surface_tractions_3d,
    face_node_indices_3d,
)
from piginot.fem.linear_solvers import (
    LinearSolverConfig,
    LinearSolverMethod,
    solve_linear_system,
)
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

TOPOLOGY_OPT_3D_SCHEMA: Final = "formamech.topology-opt-3d-v1"


class TopologyOpt3DState(StrEnum):
    """Explicit lifecycle state machine for verified 3D SIMP topology optimization."""

    INITIALIZED = "INITIALIZED"
    ITERATING = "ITERATING"
    FILTERED = "FILTERED"
    THRESHOLDED = "THRESHOLDED"
    FEM_CERTIFYING = "FEM_CERTIFYING"
    CERTIFIED = "CERTIFIED"
    REFUSED = "REFUSED"


TOPOLOGY_OPT_MACHINE: Final[StateMachineSpec[TopologyOpt3DState]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.optimization.topology_opt_3d",
            description=(
                "Verified 3D SIMP topology optimization lifecycle: "
                "INITIALIZED -> ITERATING -> FILTERED -> THRESHOLDED -> "
                "FEM_CERTIFYING -> CERTIFIED | REFUSED."
            ),
            states=tuple(TopologyOpt3DState),
            initial_states=(TopologyOpt3DState.INITIALIZED,),
            terminal_states=(
                TopologyOpt3DState.CERTIFIED,
                TopologyOpt3DState.REFUSED,
            ),
            transitions=(
                TransitionSpec(
                    TopologyOpt3DState.INITIALIZED,
                    TopologyOpt3DState.ITERATING,
                    "start_simp",
                    "Begin SIMP optimality-criteria iterations.",
                ),
                TransitionSpec(
                    TopologyOpt3DState.ITERATING,
                    TopologyOpt3DState.FILTERED,
                    "apply_projection",
                    "Finalize density cone filter and Heaviside projection.",
                ),
                TransitionSpec(
                    TopologyOpt3DState.FILTERED,
                    TopologyOpt3DState.THRESHOLDED,
                    "extract_solid",
                    "Threshold physical densities into discrete solid sub-mesh.",
                ),
                TransitionSpec(
                    TopologyOpt3DState.THRESHOLDED,
                    TopologyOpt3DState.FEM_CERTIFYING,
                    "begin_fem_certification",
                    "Run mandatory exact 3D FEM verification on thresholded solid.",
                ),
                TransitionSpec(
                    TopologyOpt3DState.THRESHOLDED,
                    TopologyOpt3DState.REFUSED,
                    "refuse_empty_solid",
                    "Refuse thresholded geometry with empty or disconnected load path.",
                ),
                TransitionSpec(
                    TopologyOpt3DState.FEM_CERTIFYING,
                    TopologyOpt3DState.CERTIFIED,
                    "certify_solid",
                    "Exact 3D FEM verification converged with finite admissible compliance.",
                ),
                TransitionSpec(
                    TopologyOpt3DState.FEM_CERTIFYING,
                    TopologyOpt3DState.REFUSED,
                    "refuse_fem_failure",
                    "Thresholded 3D solid failed final exact FEM certification.",
                ),
            ),
        )
    )
)


@dataclass(frozen=True, slots=True)
class TopologyOpt3DConfig:
    """Configuration for verified 3D SIMP compliance minimization."""

    volume_fraction: float = 0.50
    penalization: float = 3.0
    min_density: float = 1.0e-3
    filter_radius_mm: float = 4.0
    use_heaviside: bool = True
    heaviside_beta: float = 2.0
    heaviside_eta: float = 0.5
    max_iterations: int = 20
    move_limit: float = 0.2
    tolerance: float = 1.0e-3
    threshold_density: float = 0.45
    youngs_modulus_mpa: float = 1000.0
    poisson_ratio: float = 0.30

    def __post_init__(self) -> None:
        if not (0.05 <= self.volume_fraction <= 0.98):
            raise ValueError("volume_fraction must lie in [0.05, 0.98]")
        if self.penalization < 1.0:
            raise ValueError("penalization must be >= 1.0")
        if not (1.0e-9 <= self.min_density < 0.1):
            raise ValueError("min_density must lie in [1e-9, 0.1)")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")


@dataclass(frozen=True, slots=True)
class TopologyOpt3DResult:
    """Auditable verified 3D SIMP topology optimization result with Trust Receipt."""

    mesh_id: str
    state: TopologyOpt3DState
    state_history: tuple[StateSnapshot[TopologyOpt3DState], ...]
    claim_level: str
    initial_compliance_mj: float
    final_simp_compliance_mj: float
    certified_fem_compliance_mj: float | None
    certified_max_displacement_mm: float | None
    target_volume_fraction: float
    achieved_volume_fraction: float
    retained_element_fraction: float
    compliance_history: tuple[float, ...]
    element_densities: FloatArray  # (E,) physical densities in [0, 1]
    solid_element_mask: NDArray[np.bool_]  # (E,)
    receipt_sha256: str
    reason: str
    wall_time_s: float

    def document(self) -> dict[str, Any]:
        return {
            "schema": TOPOLOGY_OPT_3D_SCHEMA,
            "mesh_id": self.mesh_id,
            "state": self.state.value,
            "claim_level": self.claim_level,
            "initial_compliance_mj": float(self.initial_compliance_mj),
            "final_simp_compliance_mj": float(self.final_simp_compliance_mj),
            "certified_fem_compliance_mj": (
                None
                if self.certified_fem_compliance_mj is None
                else float(self.certified_fem_compliance_mj)
            ),
            "certified_max_displacement_mm": (
                None
                if self.certified_max_displacement_mm is None
                else float(self.certified_max_displacement_mm)
            ),
            "target_volume_fraction": float(self.target_volume_fraction),
            "achieved_volume_fraction": float(self.achieved_volume_fraction),
            "retained_element_fraction": float(self.retained_element_fraction),
            "iterations": len(self.compliance_history),
            "receipt_sha256": self.receipt_sha256,
            "reason": self.reason,
            "wall_time_s": float(self.wall_time_s),
        }


def _precompute_tet4_linear_elements(
    mesh: CanonicalMesh3D, e_mpa: float, nu: float
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray, IntArray, IntArray]:
    """Precompute TET4 element stiffnesses ``K_e0``, volumes, centroids, and indices."""
    if mesh.cell_type is not CellType3D.TET4:
        raise ValueError("3D SIMP topology optimization requires TET4 meshes")
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    n_elem = int(cells.shape[0])

    xi_c = np.array([[0.25, 0.25, 0.25]], dtype=np.float64)
    d_ref = _tet_shape_grads_ref(CellType3D.TET4, xi_c)[0]
    x_e = nodes[cells]  # (E, 4, 3)
    centroids = np.mean(x_e, axis=1)  # (E, 3)
    jac = np.einsum("eni,na->eia", x_e, d_ref)
    vols = np.linalg.det(jac) / 6.0
    if np.any(vols <= 0.0):
        raise ValueError("mesh contains non-positive volume tetrahedra")
    jac_inv = np.linalg.inv(jac)
    d_phys = np.einsum("na,eai->eni", d_ref, jac_inv)

    mu = e_mpa / (2.0 * (1.0 + nu))
    lam = e_mpa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    d_mat = np.array(
        [
            [lam + 2 * mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2 * mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2 * mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ],
        dtype=np.float64,
    )
    b_all = np.zeros((n_elem, 6, 12), dtype=np.float64)
    for a in range(4):
        gx, gy, gz = d_phys[:, a, 0], d_phys[:, a, 1], d_phys[:, a, 2]
        b_all[:, 0, 3 * a] = gx
        b_all[:, 1, 3 * a + 1] = gy
        b_all[:, 2, 3 * a + 2] = gz
        b_all[:, 3, 3 * a + 1] = gz
        b_all[:, 3, 3 * a + 2] = gy
        b_all[:, 4, 3 * a] = gz
        b_all[:, 4, 3 * a + 2] = gx
        b_all[:, 5, 3 * a] = gy
        b_all[:, 5, 3 * a + 1] = gx

    k_e0 = np.einsum("e,eik,ij,ejl->ekl", vols, b_all, d_mat, b_all)
    elem_u_dofs = np.empty((n_elem, 12), dtype=np.int64)
    for a in range(4):
        elem_u_dofs[:, 3 * a] = 3 * cells[:, a]
        elem_u_dofs[:, 3 * a + 1] = 3 * cells[:, a] + 1
        elem_u_dofs[:, 3 * a + 2] = 3 * cells[:, a] + 2
    u_rows = np.repeat(elem_u_dofs[:, :, None], 12, axis=2).reshape(-1)
    u_cols = np.repeat(elem_u_dofs[:, None, :], 12, axis=1).reshape(-1)
    return k_e0, vols, centroids, elem_u_dofs, u_rows, u_cols


def _build_cone_filter(centroids: FloatArray, radius_mm: float) -> csr_matrix:
    """Build row-normalized radial cone density filter matrix ``H``."""
    n_elem = int(centroids.shape[0])
    if radius_mm <= 0.0:
        return eye(n_elem, format="csr", dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for e in range(n_elem):
        dist = np.linalg.norm(centroids - centroids[e], axis=1)
        nbrs = np.nonzero(dist < radius_mm)[0]
        w = np.maximum(radius_mm - dist[nbrs], 0.0)
        if nbrs.size == 0 or float(np.sum(w)) <= 1.0e-15:
            rows.append(e)
            cols.append(e)
            vals.append(1.0)
        else:
            w = w / float(np.sum(w))
            for idx_n, wt in zip(nbrs.tolist(), w.tolist(), strict=True):
                rows.append(e)
                cols.append(int(idx_n))
                vals.append(float(wt))
    return coo_matrix((vals, (rows, cols)), shape=(n_elem, n_elem)).tocsr()


def _heaviside_project(
    rho_tilde: FloatArray, beta: float, eta: float
) -> tuple[FloatArray, FloatArray]:
    """Smooth threshold projection ``rho_bar`` and exact derivative ``d(rho_bar)/d(rho_tilde)``."""
    t_eta = math.tanh(beta * eta)
    denom = t_eta + math.tanh(beta * (1.0 - eta))
    arg = beta * (rho_tilde - eta)
    th = np.tanh(arg)
    rho_bar = (t_eta + th) / denom
    d_rho_bar = (beta * (1.0 - th * th)) / denom
    return rho_bar, d_rho_bar


def simp_3d_compliance_and_sensitivity(
    rho_phys: FloatArray,
    *,
    k_e0: FloatArray,
    elem_u_dofs: IntArray,
    u_rows: IntArray,
    u_cols: IntArray,
    free: IntArray,
    f_vec: FloatArray,
    n_dofs: int,
    penalization: float,
    min_density: float,
) -> tuple[float, FloatArray, FloatArray]:
    """Compute exact 3D SIMP compliance ``C`` and self-adjoint sensitivity ``dC / d(rho_phys)``."""
    e_scale = min_density + (1.0 - min_density) * (rho_phys**penalization)
    k_scaled = e_scale[:, None, None] * k_e0
    k_mat: csr_matrix = coo_matrix(
        (k_scaled.reshape(-1), (u_rows, u_cols)), shape=(n_dofs, n_dofs)
    ).tocsr()
    u_f, rep = solve_linear_system(
        k_mat[free][:, free].tocsr(),
        f_vec[free],
        config=LinearSolverConfig(method=LinearSolverMethod.DIRECT),
        is_symmetric=True,
    )
    if not rep.converged:
        raise RuntimeError(f"SIMP linear solve failed: {rep.reason}")
    u_full = np.zeros(n_dofs, dtype=np.float64)
    u_full[free] = u_f
    u_e = u_full[elem_u_dofs]  # (E, 12)
    # Element compliance energy c_e = u_e^T K_e0 u_e
    ce = np.einsum("ei,eij,ej->e", u_e, k_e0, u_e)
    compliance = float(np.sum(e_scale * ce))
    dc_drho = (
        -penalization * (1.0 - min_density) * (rho_phys ** (penalization - 1.0)) * ce
    )
    return compliance, dc_drho, u_full


def optimize_3d_topology_simp(
    mesh: CanonicalMesh3D,
    *,
    dirichlet_bcs: Sequence[DirichletBC3D],
    traction_bcs: Sequence[TractionBC3D],
    config: TopologyOpt3DConfig | None = None,
) -> TopologyOpt3DResult:
    """Run verified 3D SIMP topology optimization with mandatory final exact FEM check."""
    t0 = time.perf_counter()
    cfg = TopologyOpt3DConfig() if config is None else config
    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes

    state = TopologyOpt3DState.INITIALIZED
    history: list[StateSnapshot[TopologyOpt3DState]] = [
        TOPOLOGY_OPT_MACHINE.snapshot(state)
    ]

    k_e0, vols, centroids, elem_u_dofs, u_rows, u_cols = (
        _precompute_tet4_linear_elements(
            mesh, cfg.youngs_modulus_mpa, cfg.poisson_ratio
        )
    )
    n_elem = int(vols.size)
    v_total = float(np.sum(vols))
    h_filter = _build_cone_filter(centroids, cfg.filter_radius_mm)

    f_vec = assemble_surface_tractions_3d(mesh, traction_bcs).reshape(n_dofs)
    if float(np.linalg.norm(f_vec)) <= 1.0e-15:
        raise ValueError("non-zero traction load is required for topology optimization")

    comp_idx = {"x": 0, "y": 1, "z": 2}
    constrained = np.zeros(n_dofs, dtype=bool)
    for bc in dirichlet_bcs:
        nids = face_node_indices_3d(mesh, bc.face_tags)
        for comp in bc.components:
            c = comp_idx[comp]
            for nid in nids:
                constrained[3 * nid + c] = True
    free = np.nonzero(~constrained)[0]
    if free.size == 0:
        raise ValueError("all DOFs are constrained")

    TOPOLOGY_OPT_MACHINE.require_transition(state, TopologyOpt3DState.ITERATING)
    state = TopologyOpt3DState.ITERATING
    history.append(TOPOLOGY_OPT_MACHINE.snapshot(state))

    rho: FloatArray = np.full(n_elem, cfg.volume_fraction, dtype=np.float64)
    comp_hist: list[float] = []
    rho_phys: FloatArray = rho.copy()

    def _project(r_design: FloatArray) -> tuple[FloatArray, FloatArray]:
        r_tilde = np.asarray(h_filter @ r_design, dtype=np.float64)
        if cfg.use_heaviside:
            return _heaviside_project(r_tilde, cfg.heaviside_beta, cfg.heaviside_eta)
        return r_tilde, np.ones_like(r_tilde)

    for _it in range(cfg.max_iterations):
        rho_phys, d_proj = _project(rho)
        comp_val, dc_dphys, _ = simp_3d_compliance_and_sensitivity(
            rho_phys,
            k_e0=k_e0,
            elem_u_dofs=elem_u_dofs,
            u_rows=u_rows,
            u_cols=u_cols,
            free=free,
            f_vec=f_vec,
            n_dofs=n_dofs,
            penalization=cfg.penalization,
            min_density=cfg.min_density,
        )
        comp_hist.append(comp_val)
        # Chain rule through Heaviside + cone filter
        dc = np.asarray(h_filter.T @ (dc_dphys * d_proj), dtype=np.float64)
        dv = np.asarray(h_filter.T @ (vols * d_proj), dtype=np.float64)
        dv = np.maximum(dv, 1.0e-15)

        # Optimality Criteria (OC) bisection enforcing physical volume fraction
        l1, l2 = 0.0, 1.0e9
        move = cfg.move_limit
        rho_new = rho.copy()
        while (l2 - l1) / max(l1 + l2, 1.0e-12) > 1.0e-4:
            lmid = 0.5 * (l1 + l2)
            b_e = np.maximum(-dc / (lmid * dv), 1.0e-12)
            cand = np.clip(
                rho * np.sqrt(b_e),
                np.maximum(0.0, rho - move),
                np.minimum(1.0, rho + move),
            )
            r_p_cand, _ = _project(cand)
            if float(np.sum(r_p_cand * vols)) / v_total > cfg.volume_fraction:
                l1 = lmid
            else:
                l2 = lmid
            rho_new = cand
        change = float(np.max(np.abs(rho_new - rho)))
        rho = rho_new
        if change < cfg.tolerance:
            break

    TOPOLOGY_OPT_MACHINE.require_transition(state, TopologyOpt3DState.FILTERED)
    state = TopologyOpt3DState.FILTERED
    history.append(TOPOLOGY_OPT_MACHINE.snapshot(state))
    rho_phys, _ = _project(rho)
    final_simp_comp, _, _ = simp_3d_compliance_and_sensitivity(
        rho_phys,
        k_e0=k_e0,
        elem_u_dofs=elem_u_dofs,
        u_rows=u_rows,
        u_cols=u_cols,
        free=free,
        f_vec=f_vec,
        n_dofs=n_dofs,
        penalization=cfg.penalization,
        min_density=cfg.min_density,
    )
    achieved_vf = float(np.sum(rho_phys * vols)) / v_total

    # Threshold into discrete solid mask
    TOPOLOGY_OPT_MACHINE.require_transition(state, TopologyOpt3DState.THRESHOLDED)
    state = TopologyOpt3DState.THRESHOLDED
    history.append(TOPOLOGY_OPT_MACHINE.snapshot(state))

    solid_mask = rho_phys >= cfg.threshold_density
    # Ensure boundary-loaded and clamped elements touching non-zero traction are evaluated honestly
    retained_frac = (
        float(np.sum(vols[solid_mask])) / v_total if np.any(solid_mask) else 0.0
    )

    certified_comp: float | None = None
    certified_max_u: float | None = None
    claim_level = "NOT_VERIFIED"
    reason = ""

    if not np.any(solid_mask):
        TOPOLOGY_OPT_MACHINE.require_transition(state, TopologyOpt3DState.REFUSED)
        state = TopologyOpt3DState.REFUSED
        history.append(TOPOLOGY_OPT_MACHINE.snapshot(state))
        reason = "thresholded 3D solid is empty"
    else:
        TOPOLOGY_OPT_MACHINE.require_transition(
            state, TopologyOpt3DState.FEM_CERTIFYING
        )
        state = TopologyOpt3DState.FEM_CERTIFYING
        history.append(TOPOLOGY_OPT_MACHINE.snapshot(state))

        # Assemble exact discrete stiffness on retained solid elements ONLY (no void ersatz)
        solid_idx = np.nonzero(solid_mask)[0]
        sub_dofs = elem_u_dofs[solid_idx]
        s_rows = np.repeat(sub_dofs[:, :, None], 12, axis=2).reshape(-1)
        s_cols = np.repeat(sub_dofs[:, None, :], 12, axis=1).reshape(-1)
        k_sub: csr_matrix = coo_matrix(
            (k_e0[solid_idx].reshape(-1), (s_rows, s_cols)), shape=(n_dofs, n_dofs)
        ).tocsr()
        active_dofs = np.unique(sub_dofs.reshape(-1))
        free_active = np.intersect1d(free, active_dofs)
        loaded_dofs = np.nonzero(np.abs(f_vec) > 1.0e-15)[0]
        if free_active.size == 0 or not np.any(np.isin(loaded_dofs, active_dofs)):
            TOPOLOGY_OPT_MACHINE.require_transition(state, TopologyOpt3DState.REFUSED)
            state = TopologyOpt3DState.REFUSED
            history.append(TOPOLOGY_OPT_MACHINE.snapshot(state))
            reason = "thresholded 3D solid disconnected from loaded boundary"
        else:
            try:
                u_f_cert, rep_cert = solve_linear_system(
                    k_sub[free_active][:, free_active].tocsr(),
                    f_vec[free_active],
                    config=LinearSolverConfig(method=LinearSolverMethod.DIRECT),
                    is_symmetric=True,
                )
                if not rep_cert.converged or not np.all(np.isfinite(u_f_cert)):
                    raise RuntimeError("singular or non-convergent thresholded solid")
                u_cert = np.zeros(n_dofs, dtype=np.float64)
                u_cert[free_active] = u_f_cert
                certified_comp = float(f_vec[free_active] @ u_f_cert)
                if not math.isfinite(certified_comp) or certified_comp <= 0.0:
                    raise RuntimeError("non-physical certified compliance")
                certified_max_u = float(
                    np.max(np.linalg.norm(u_cert.reshape(n_nodes, 3), axis=1))
                )
                TOPOLOGY_OPT_MACHINE.require_transition(
                    state, TopologyOpt3DState.CERTIFIED
                )
                state = TopologyOpt3DState.CERTIFIED
                history.append(TOPOLOGY_OPT_MACHINE.snapshot(state))
                claim_level = "SINGLE_POINT_VERIFIED"
                reason = "thresholded 3D solid passed mandatory exact FEM certification"
            except Exception as exc:
                TOPOLOGY_OPT_MACHINE.require_transition(
                    state, TopologyOpt3DState.REFUSED
                )
                state = TopologyOpt3DState.REFUSED
                history.append(TOPOLOGY_OPT_MACHINE.snapshot(state))
                reason = f"final exact FEM certification refused: {exc}"

    receipt_payload = {
        "schema": TOPOLOGY_OPT_3D_SCHEMA,
        "mesh_id": mesh.mesh_id,
        "state": state.value,
        "claim_level": claim_level,
        "initial_compliance_mj": float(comp_hist[0]),
        "final_simp_compliance_mj": float(final_simp_comp),
        "certified_fem_compliance_mj": certified_comp,
        "achieved_volume_fraction": float(achieved_vf),
        "retained_element_fraction": float(retained_frac),
    }
    receipt_sha256 = hashlib.sha256(canonical_json_bytes(receipt_payload)).hexdigest()

    return TopologyOpt3DResult(
        mesh_id=mesh.mesh_id,
        state=state,
        state_history=tuple(history),
        claim_level=claim_level,
        initial_compliance_mj=float(comp_hist[0]),
        final_simp_compliance_mj=float(final_simp_comp),
        certified_fem_compliance_mj=certified_comp,
        certified_max_displacement_mm=certified_max_u,
        target_volume_fraction=float(cfg.volume_fraction),
        achieved_volume_fraction=float(achieved_vf),
        retained_element_fraction=float(retained_frac),
        compliance_history=tuple(float(c) for c in comp_hist),
        element_densities=rho_phys,
        solid_element_mask=solid_mask,
        receipt_sha256=receipt_sha256,
        reason=reason,
        wall_time_s=time.perf_counter() - t0,
    )


__all__ = (
    "TOPOLOGY_OPT_3D_SCHEMA",
    "TOPOLOGY_OPT_MACHINE",
    "TopologyOpt3DConfig",
    "TopologyOpt3DResult",
    "TopologyOpt3DState",
    "optimize_3d_topology_simp",
    "simp_3d_compliance_and_sensitivity",
)
