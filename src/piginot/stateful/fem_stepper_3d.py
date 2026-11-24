"""Incremental 3D stateful FEM trajectory solver & GuardedStatefulStepper3D (V3-4B).

Drives any :class:`~piginot.stateful.constitutive.ConstitutiveModel` (3D J2
isotropic/combined/finite-strain plasticity, 3D Generalized Maxwell
viscoelasticity) on ``CanonicalMesh3D`` ``TET4``/``TET10`` meshes using Phase 1
iterative linear solvers, BLAKE2b quadrature-state hashing, and residual-
triggered 3D FEM fallback for ``Surface3DEnergyOperator``.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import coo_matrix  # type: ignore[import-untyped]

from piginot.fem.backend_3d import (
    DirichletBC3D,
    TorchFemBackend3D,
    TractionBC3D,
    assemble_surface_tractions_3d,
    face_node_indices_3d,
)
from piginot.fem.linear_solvers import LinearSolverConfig, solve_linear_system
from piginot.geometry.mesh3d import CanonicalMesh3D
from piginot.stateful.constitutive import ConstitutiveModel

FloatArray = NDArray[np.float64]
STATEFUL_3D_TRAJECTORY_SCHEMA: Final = "formamech.stateful-3d-trajectory-v1"


@dataclass(frozen=True, slots=True)
class LoadStep3D:
    """One incremental load step in a 3D stateful trajectory."""

    step_index: int
    time_s: float
    dt_s: float
    dirichlet_bcs: tuple[DirichletBC3D, ...]
    traction_bcs: tuple[TractionBC3D, ...] = ()
    temperature_k: float | None = None


@dataclass(frozen=True, slots=True)
class StatefulStep3DRecord:
    """Auditable per-step record for a 3D stateful FEM or guarded surrogate rollout."""

    step_index: int
    time_s: float
    used_fem_fallback: bool
    converged: bool
    residual_norm: float
    newton_iterations: int
    dissipation_total: float
    min_det_f: float
    max_displacement_mm: float
    state_blake2b: str
    nodal_displacement: FloatArray


@dataclass(frozen=True, slots=True)
class Stateful3DTrajectory:
    """Immutable trajectory result with per-step BLAKE2b state digests."""

    mesh_id: str
    model_id: str
    steps: tuple[StatefulStep3DRecord, ...]
    final_state: Mapping[str, torch.Tensor]
    wall_time_s: float

    def document(self) -> dict[str, Any]:
        return {
            "schema": STATEFUL_3D_TRAJECTORY_SCHEMA,
            "mesh_id": self.mesh_id,
            "model_id": self.model_id,
            "step_count": len(self.steps),
            "fallback_steps": sum(1 for s in self.steps if s.used_fem_fallback),
            "wall_time_s": float(self.wall_time_s),
            "steps": [
                {
                    "step_index": s.step_index,
                    "time_s": float(s.time_s),
                    "used_fem_fallback": s.used_fem_fallback,
                    "converged": s.converged,
                    "residual_norm": float(s.residual_norm),
                    "newton_iterations": s.newton_iterations,
                    "dissipation_total": float(s.dissipation_total),
                    "min_det_f": float(s.min_det_f),
                    "max_displacement_mm": float(s.max_displacement_mm),
                    "state_blake2b": s.state_blake2b,
                }
                for s in self.steps
            ],
        }


def _hash_state_dict(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.blake2b(digest_size=32)
    for k in sorted(state.keys()):
        h.update(k.encode())
        h.update(b"\x00")
        h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def solve_incremental_fem_3d_trajectory(
    mesh: CanonicalMesh3D,
    model: ConstitutiveModel,
    load_steps: Sequence[LoadStep3D],
    *,
    solver_tolerance: float = 1.0e-7,
    max_newton_iterations: int = 25,
    linear_solver: LinearSolverConfig | None = None,
    surrogate_predictor: Callable[[LoadStep3D, FloatArray], FloatArray] | None = None,
    guard_residual_threshold: float | None = None,
) -> Stateful3DTrajectory:
    """Run incremental 3D ``TET4``/``TET10`` FEM (or guarded surrogate + FEM fallback) rollout."""
    t0 = time.perf_counter()
    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes
    helper = TorchFemBackend3D()
    cells, d_phys, w_vol, _, rows, cols = helper._precompute_geometry(mesh)
    n_elem, n_loc = cells.shape
    n_q = d_phys.shape[1]
    n_edof = 3 * n_loc
    lin_cfg = LinearSolverConfig() if linear_solver is None else linear_solver
    comp_idx = {"x": 0, "y": 1, "z": 2}

    state_curr = model.initial_state((n_elem, n_q), dtype=torch.float64)
    u_prev: FloatArray = np.zeros(n_dofs, dtype=np.float64)
    f_old = torch.eye(3, dtype=torch.float64).expand((n_elem, n_q, 3, 3)).clone()
    records: list[StatefulStep3DRecord] = []

    def _eval_step_state(
        u_vec: FloatArray,
        dt: float,
        temp_k: float | None,
    ) -> tuple[FloatArray, Any, Mapping[str, torch.Tensor], float, float, bool]:
        u_e = u_vec.reshape(n_nodes, 3)[cells]
        f_np = (
            np.einsum("eni,eqnj->eqij", u_e, d_phys)
            + np.eye(3, dtype=np.float64)[None, None, :, :]
        )
        f_t = torch.from_numpy(np.ascontiguousarray(f_np))
        temp_t = (
            torch.full((n_elem, n_q), float(temp_k), dtype=torch.float64)
            if temp_k is not None
            else None
        )
        upd = model.update(f_old, f_t, state_curr, dt, temperature=temp_t)
        p_np = upd.stress.detach().numpy()
        c_np = upd.algorithmic_tangent.detach().numpy()
        f_elem = np.einsum("eq,eqij,eqnj->eni", w_vol, p_np, d_phys)
        f_int = np.zeros((n_nodes, 3), dtype=np.float64)
        np.add.at(f_int, cells, f_elem)
        tmp = np.einsum("eq,eqijkl,eqnj->eqnikl", w_vol, c_np, d_phys)
        k_elem = np.einsum("eqnikl,eqml->enimk", tmp, d_phys)
        k_mat = coo_matrix(
            (k_elem.reshape(n_elem, n_edof, n_edof).reshape(-1), (rows, cols)),
            shape=(n_dofs, n_dofs),
        ).tocsr()
        diss_tot = float(np.sum(upd.dissipation.detach().numpy() * w_vol))
        min_det = float(upd.det_f.detach().min().item())
        return (
            f_int.reshape(-1),
            k_mat,
            upd.state_new,
            diss_tot,
            min_det,
            upd.admissible,
        )

    for step in load_steps:
        f_ext = assemble_surface_tractions_3d(mesh, step.traction_bcs).reshape(-1)
        target_d: dict[int, float] = {}
        for bc in step.dirichlet_bcs:
            nids = face_node_indices_3d(mesh, bc.face_tags)
            for comp, val in zip(bc.components, bc.values_mm, strict=True):
                c = comp_idx[comp]
                for nid in nids:
                    target_d[3 * nid + c] = float(val)
        constrained = np.zeros(n_dofs, dtype=bool)
        for dof in target_d:
            constrained[dof] = True
        free_dofs = np.nonzero(~constrained)[0]

        # Guarded surrogate path: accept iff admissible & equilibrium residual <= guard_threshold
        used_fallback = True
        u_sol: FloatArray = u_prev.copy()
        for dof, val in target_d.items():
            u_sol[dof] = val

        if surrogate_predictor is not None and guard_residual_threshold is not None:
            u_cand = np.asarray(
                surrogate_predictor(step, u_prev.reshape(n_nodes, 3)), dtype=np.float64
            ).reshape(n_dofs)
            for dof, val in target_d.items():
                u_cand[dof] = val
            f_int_s, _, st_s, diss_s, det_s, adm_s = _eval_step_state(
                u_cand, step.dt_s, step.temperature_k
            )
            res_s = float(np.linalg.norm((f_int_s - f_ext)[free_dofs]))
            if adm_s and res_s <= guard_residual_threshold:
                used_fallback = False
                u_sol = u_cand
                state_curr = {k: v.detach().clone() for k, v in st_s.items()}
                u_e_new = u_sol.reshape(n_nodes, 3)[cells]
                f_old = torch.from_numpy(
                    np.ascontiguousarray(
                        np.einsum("eni,eqnj->eqij", u_e_new, d_phys)
                        + np.eye(3, dtype=np.float64)[None, None, :, :]
                    )
                )
                u_prev = u_sol.copy()
                records.append(
                    StatefulStep3DRecord(
                        step_index=step.step_index,
                        time_s=step.time_s,
                        used_fem_fallback=False,
                        converged=True,
                        residual_norm=res_s,
                        newton_iterations=0,
                        dissipation_total=diss_s,
                        min_det_f=det_s,
                        max_displacement_mm=float(
                            np.max(np.linalg.norm(u_sol.reshape(n_nodes, 3), axis=1))
                        ),
                        state_blake2b=_hash_state_dict(state_curr),
                        nodal_displacement=u_sol.reshape(n_nodes, 3).copy(),
                    )
                )
                continue

        # Authoritative incremental 3D Newton FEM solve with linearized tangent
        # predictor for incremental Dirichlet/traction loads and Armijo-style
        # residual-decreasing backtracking line search.
        constrained_dofs = np.nonzero(constrained)[0]
        du_dirichlet = u_sol[constrained_dofs] - u_prev[constrained_dofs]
        if len(free_dofs) > 0 and (
            np.any(du_dirichlet != 0.0) or np.any(f_ext[free_dofs] != 0.0)
        ):
            f_pred, k_pred, _, _, _, _ = _eval_step_state(
                u_prev, step.dt_s, step.temperature_k
            )
            rhs_pred = (
                f_ext[free_dofs]
                - f_pred[free_dofs]
                - k_pred[free_dofs][:, constrained_dofs] @ du_dirichlet
            )
            du_pred, pred_rep = solve_linear_system(
                k_pred[free_dofs][:, free_dofs],
                rhs_pred,
                config=lin_cfg,
                is_symmetric=None,
            )
            if pred_rep.converged and np.all(np.isfinite(du_pred)):
                for alpha_p in (1.0, 0.5, 0.25):
                    u_try = u_sol.copy()
                    u_try[free_dofs] = u_prev[free_dofs] + alpha_p * du_pred
                    u_e_try = u_try.reshape(n_nodes, 3)[cells]
                    f_try = (
                        np.einsum("eni,eqnj->eqij", u_e_try, d_phys)
                        + np.eye(3, dtype=np.float64)[None, None, :, :]
                    )
                    if float(np.min(np.linalg.det(f_try))) > 1.0e-4:
                        u_sol = u_try
                        break

        converged = False
        n_iters = 0
        f_int, k_mat, last_state, last_diss, last_det, _ = _eval_step_state(
            u_sol, step.dt_s, step.temperature_k
        )
        res_free = (f_int - f_ext)[free_dofs]
        last_res = float(np.linalg.norm(res_free))
        if last_res <= solver_tolerance:
            converged = True
        else:
            for it in range(1, max_newton_iterations + 1):
                n_iters = it
                du_free, lin_rep = solve_linear_system(
                    k_mat[free_dofs][:, free_dofs],
                    -res_free,
                    config=lin_cfg,
                    is_symmetric=None,
                )
                if not lin_rep.converged or not np.all(np.isfinite(du_free)):
                    break
                accepted = False
                best_cand: (
                    tuple[
                        FloatArray,
                        FloatArray,
                        Any,
                        Mapping[str, torch.Tensor],
                        float,
                        float,
                        float,
                    ]
                    | None
                ) = None
                for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    u_ls = u_sol.copy()
                    u_ls[free_dofs] += alpha * du_free
                    f_c, k_c, st_c, diss_c, det_c, _ = _eval_step_state(
                        u_ls, step.dt_s, step.temperature_k
                    )
                    res_c_vec = (f_c - f_ext)[free_dofs]
                    res_c = float(np.linalg.norm(res_c_vec))
                    if det_c > 1.0e-5 and np.isfinite(res_c):
                        if best_cand is None or res_c < best_cand[6]:
                            best_cand = (
                                u_ls,
                                res_c_vec,
                                k_c,
                                st_c,
                                diss_c,
                                det_c,
                                res_c,
                            )
                        if (
                            res_c < (1.0 - 1.0e-4 * alpha) * last_res
                            or res_c <= solver_tolerance
                        ):
                            accepted = True
                            u_sol = u_ls
                            res_free = res_c_vec
                            k_mat = k_c
                            last_state = st_c
                            last_diss = diss_c
                            last_det = det_c
                            last_res = res_c
                            break
                if not accepted:
                    if best_cand is None:
                        break
                    (
                        u_sol,
                        res_free,
                        k_mat,
                        last_state,
                        last_diss,
                        last_det,
                        last_res,
                    ) = best_cand
                if last_res <= solver_tolerance:
                    converged = True
                    break

        state_curr = {k: v.detach().clone() for k, v in last_state.items()}
        u_e_new = u_sol.reshape(n_nodes, 3)[cells]
        f_old = torch.from_numpy(
            np.ascontiguousarray(
                np.einsum("eni,eqnj->eqij", u_e_new, d_phys)
                + np.eye(3, dtype=np.float64)[None, None, :, :]
            )
        )
        u_prev = u_sol.copy()
        records.append(
            StatefulStep3DRecord(
                step_index=step.step_index,
                time_s=step.time_s,
                used_fem_fallback=used_fallback,
                converged=converged,
                residual_norm=last_res,
                newton_iterations=n_iters,
                dissipation_total=last_diss,
                min_det_f=last_det,
                max_displacement_mm=float(
                    np.max(np.linalg.norm(u_sol.reshape(n_nodes, 3), axis=1))
                ),
                state_blake2b=_hash_state_dict(state_curr),
                nodal_displacement=u_sol.reshape(n_nodes, 3).copy(),
            )
        )

    return Stateful3DTrajectory(
        mesh_id=mesh.mesh_id,
        model_id=model.state_schema.model_id,
        steps=tuple(records),
        final_state=state_curr,
        wall_time_s=time.perf_counter() - t0,
    )


__all__ = (
    "STATEFUL_3D_TRAJECTORY_SCHEMA",
    "LoadStep3D",
    "Stateful3DTrajectory",
    "StatefulStep3DRecord",
    "solve_incremental_fem_3d_trajectory",
)
