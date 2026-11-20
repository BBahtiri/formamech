"""Incremental 2D plane-stress viscoelastic FEM reference stepper (Feature F0/F1).

Solves path-dependent 2D finite-strain plane-stress viscoelasticity on any
``CanonicalMesh`` over a ``TimeGrid`` and ``LoadSegment`` sequence using
``scikit-fem`` with quadrature-point internal variables, exact condensed
algorithmic tangent, Newton iteration, adaptive substep halving, and
content-hashed ``BoundTrajectoryState`` tracking.

Optional dependency: requires ``.[verify]`` (``scikit-fem`` / ``scipy``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse.linalg import spsolve  # type: ignore[import-untyped]
from skfem import (  # type: ignore[import-untyped]
    BilinearForm,
    CellBasis,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    LinearForm,
    MeshTri,
    asm,
)
from skfem.helpers import grad  # type: ignore[import-untyped]

from piginot.contracts import CanonicalMesh
from piginot.contracts.stateful import (
    BoundTrajectoryState,
    IntegratorSpec,
    LoadSegment,
    TimeGrid,
    evaluate_load_path,
)
from piginot.fem.binding import canonical_facet_map, facets_for_tag
from piginot.stateful.stepper import STEP_CONTROLLER_VERSION, hash_state_tensors
from piginot.stateful.viscoelastic import GeneralizedMaxwellViscoelastic

FloatArray = NDArray[np.float64]


@LinearForm  # type: ignore[untyped-decorator]
def _internal_form(test: Any, data: Any) -> Any:
    return np.einsum("ij...,ij...->...", data.stress, grad(test))


@BilinearForm  # type: ignore[untyped-decorator]
def _tangent_form(increment: Any, test: Any, data: Any) -> Any:
    return np.einsum(
        "ij...,ijkl...,kl...->...",
        grad(test),
        data.modulus,
        grad(increment),
    )


@dataclass(frozen=True, slots=True)
class DirichletHistoryBC:
    """Time-dependent component-wise Dirichlet BC bound to a named boundary group."""

    name: str
    boundary_group: str
    component: str  # "x" or "y"
    load_key: str | None = None  # None => fixed zero value


@dataclass(frozen=True, slots=True)
class IncrementalFemStep:
    """Converged state and diagnostics of one incremental viscoelastic FEM step."""

    time_index: int
    time_value: float
    dt: float
    substeps: int
    newton_iterations: int
    residual_norm: float
    displacement: FloatArray
    reactions: dict[str, float]
    total_free_energy: float
    step_dissipation: float
    min_det_f: float
    bound_state: BoundTrajectoryState


@dataclass(frozen=True, slots=True)
class IncrementalFemTrajectory:
    """Complete deterministic trajectory record of an incremental FEM solve."""

    converged: bool
    mesh_id: str
    element_order: int
    steps: tuple[IncrementalFemStep, ...]
    final_state: dict[str, torch.Tensor]
    reason: str


def _build_mesh_and_basis(
    mesh: CanonicalMesh, element_order: int
) -> tuple[Any, Any, Any]:
    if element_order not in (1, 2):
        raise ValueError("element_order must be 1 or 2")
    points = np.asarray(mesh.nodes, dtype=np.float64).T
    triangles = np.asarray(mesh.cells, dtype=np.int64).T
    skmesh = MeshTri(points, triangles)
    scalar_elem = ElementTriP1() if element_order == 1 else ElementTriP2()
    basis = CellBasis(skmesh, ElementVector(scalar_elem))
    scalar_basis = CellBasis(skmesh, scalar_elem)
    return skmesh, basis, scalar_basis


def _constrained_dofs(
    basis: Any,
    skmesh: Any,
    mesh: CanonicalMesh,
    bcs: Sequence[DirichletHistoryBC],
) -> tuple[dict[str, NDArray[np.int64]], NDArray[np.int64], NDArray[np.int64]]:
    facet_map = canonical_facet_map(skmesh)
    comp_map = {"x": "u^1", "y": "u^2"}
    by_name: dict[str, NDArray[np.int64]] = {}
    constrained_set: set[int] = set()
    for bc in bcs:
        if bc.component not in comp_map:
            raise ValueError(f"component must be 'x' or 'y', got {bc.component!r}")
        facet_indices = facets_for_tag(mesh, facet_map, bc.boundary_group)
        dof_groups = basis.get_dofs(facets=facet_indices)
        dofs = np.asarray(dof_groups.nodal[comp_map[bc.component]], dtype=np.int64)
        if hasattr(dof_groups, "facet") and comp_map[bc.component] in dof_groups.facet:
            dofs = np.unique(
                np.concatenate(
                    (
                        dofs,
                        np.asarray(
                            dof_groups.facet[comp_map[bc.component]], dtype=np.int64
                        ),
                    )
                )
            )
        by_name[bc.name] = np.unique(dofs)
        constrained_set.update(int(d) for d in dofs)
    constrained = np.asarray(sorted(constrained_set), dtype=np.int64)
    all_dofs = np.arange(basis.N, dtype=np.int64)
    free = np.setdiff1d(all_dofs, constrained, assume_unique=True)
    return by_name, constrained, free


def _solve_single_increment(
    *,
    basis: Any,
    model: GeneralizedMaxwellViscoelastic,
    u_old: FloatArray,
    state_old: Mapping[str, torch.Tensor],
    f33_old: torch.Tensor,
    target_dirichlet: Mapping[int, float],
    free_dofs: NDArray[np.int64],
    dt: float,
    integrator: IntegratorSpec,
) -> tuple[
    bool,
    FloatArray,
    dict[str, torch.Tensor],
    torch.Tensor,
    FloatArray,
    float,
    float,
    float,
    float,
    int,
    str,
]:
    u = u_old.copy()
    for dof, val in target_dirichlet.items():
        u[dof] = val

    grad_old = np.asarray(basis.interpolate(u_old).grad, dtype=np.float64)
    f2d_old = torch.from_numpy(
        np.ascontiguousarray(np.transpose(grad_old, (2, 3, 0, 1)) + np.eye(2))
    )
    f33_guess = f33_old
    dx = np.asarray(basis.dx, dtype=np.float64)

    for it in range(1, integrator.max_newton_iterations + 1):
        grad_new = np.asarray(basis.interpolate(u).grad, dtype=np.float64)
        f2d_new = torch.from_numpy(
            np.ascontiguousarray(np.transpose(grad_new, (2, 3, 0, 1)) + np.eye(2))
        )
        ps = model.plane_stress_update(
            f2d_old,
            f2d_new,
            state_old,
            dt,
            f33_guess=f33_guess,
            max_iterations=integrator.plane_stress_max_iterations,
        )
        if not ps.closure_converged or not ps.update_3d.admissible:
            return (
                False,
                u,
                dict(state_old),
                f33_guess,
                np.zeros_like(u),
                float("inf"),
                0.0,
                0.0,
                float(ps.update_3d.det_f.min().item()),
                it,
                "plane-stress closure or admissibility failed",
            )
        f33_guess = ps.f33
        stress_2d = ps.update_3d.stress[..., :2, :2].detach().cpu().numpy()
        modulus_2d = ps.condensed_tangent_2d.detach().cpu().numpy()
        stress_field = np.transpose(stress_2d, (2, 3, 0, 1))
        modulus_field = np.transpose(modulus_2d, (2, 3, 4, 5, 0, 1))
        internal = np.asarray(
            asm(_internal_form, basis, stress=stress_field), dtype=np.float64
        )
        res_free = internal[free_dofs]
        res_norm = float(np.linalg.norm(res_free))
        if res_norm <= integrator.residual_tolerance:
            free_e = float(np.sum(ps.update_3d.free_energy.detach().cpu().numpy() * dx))
            step_diss = float(
                np.sum(ps.update_3d.dissipation.detach().cpu().numpy() * dx)
            )
            min_det = float(ps.update_3d.det_f.min().item())
            return (
                True,
                u,
                {k: v.detach().clone() for k, v in ps.update_3d.state_new.items()},
                ps.f33.detach().clone(),
                internal,
                res_norm,
                free_e,
                step_diss,
                min_det,
                it,
                "converged",
            )

        tangent = asm(_tangent_form, basis, modulus=modulus_field).tocsr()
        k_ff = tangent[free_dofs][:, free_dofs].tocsc()
        du_free = np.asarray(spsolve(k_ff, -res_free), dtype=np.float64)
        if not np.all(np.isfinite(du_free)):
            return (
                False,
                u,
                dict(state_old),
                f33_guess,
                internal,
                res_norm,
                0.0,
                0.0,
                float(ps.update_3d.det_f.min().item()),
                it,
                "linear solve produced non-finite Newton increment",
            )
        u[free_dofs] += du_free

    return (
        False,
        u,
        dict(state_old),
        f33_guess,
        np.zeros_like(u),
        float("inf"),
        0.0,
        0.0,
        0.0,
        integrator.max_newton_iterations,
        "maximum Newton iterations exceeded",
    )


def solve_incremental_fem_trajectory(
    mesh: CanonicalMesh,
    model: GeneralizedMaxwellViscoelastic,
    boundary_conditions: Sequence[DirichletHistoryBC],
    time_grid: TimeGrid,
    load_segments: Sequence[LoadSegment],
    *,
    problem_spec_hash: str,
    load_path_hash: str,
    parameters: Mapping[str, float] | None = None,
    element_order: int = 1,
    integrator: IntegratorSpec | None = None,
) -> IncrementalFemTrajectory:
    """Solve a 2D plane-stress viscoelastic load-path history with adaptive substepping."""
    integ = IntegratorSpec() if integrator is None else integrator
    skmesh, basis, _ = _build_mesh_and_basis(mesh, element_order)
    by_name, _, free_dofs = _constrained_dofs(basis, skmesh, mesh, boundary_conditions)
    q_shape = tuple(int(d) for d in basis.dx.shape)

    u_curr = np.zeros(basis.N, dtype=np.float64)
    state_curr = model.initial_state(q_shape, dtype=torch.float64)
    f33_curr = torch.ones(q_shape, dtype=torch.float64)

    times = time_grid.fixed_times()
    steps: list[IncrementalFemStep] = []

    for step_idx in range(1, len(times)):
        t_prev = times[step_idx - 1]
        t_next = times[step_idx]
        dt_total = t_next - t_prev

        solved_step = False
        last_reason = ""
        for halving in range(integ.max_substep_halvings + 1):
            n_sub = 2**halving
            sub_dt = dt_total / n_sub
            u_sub = u_curr.copy()
            s_sub = {k: v.clone() for k, v in state_curr.items()}
            f33_sub = f33_curr.clone()
            total_diss = 0.0
            total_iters = 0
            min_det = float("inf")
            last_internal = np.zeros_like(u_curr)
            last_res = 0.0
            last_free_e = 0.0
            sub_ok = True

            for sub in range(1, n_sub + 1):
                t_sub = t_prev + sub * sub_dt
                loads = evaluate_load_path(load_segments, t_sub, parameters)
                target_d: dict[int, float] = {}
                for bc in boundary_conditions:
                    val = 0.0 if bc.load_key is None else float(loads[bc.load_key])
                    for dof in by_name[bc.name]:
                        target_d[int(dof)] = val

                (
                    ok,
                    u_next,
                    s_next,
                    f33_next,
                    internal,
                    res_norm,
                    free_e,
                    step_diss,
                    det_min,
                    iters,
                    reason,
                ) = _solve_single_increment(
                    basis=basis,
                    model=model,
                    u_old=u_sub,
                    state_old=s_sub,
                    f33_old=f33_sub,
                    target_dirichlet=target_d,
                    free_dofs=free_dofs,
                    dt=sub_dt,
                    integrator=integ,
                )
                if not ok:
                    sub_ok = False
                    last_reason = reason
                    break
                u_sub = u_next
                s_sub = s_next
                f33_sub = f33_next
                total_diss += step_diss
                total_iters += iters
                min_det = min(min_det, det_min)
                last_internal = internal
                last_res = res_norm
                last_free_e = free_e

            if sub_ok:
                u_curr = u_sub
                state_curr = s_sub
                f33_curr = f33_sub
                reactions = {
                    bc.name: float(np.sum(last_internal[by_name[bc.name]]))
                    for bc in boundary_conditions
                }
                bound = BoundTrajectoryState(
                    problem_spec_hash=problem_spec_hash,
                    mesh_hash=mesh.mesh_id,
                    material_schema_hash=model.state_schema.schema_hash,
                    load_path_hash=load_path_hash,
                    time_index=step_idx,
                    time_value=t_next,
                    step_controller_version=STEP_CONTROLLER_VERSION,
                    state_hash=hash_state_tensors(state_curr),
                )
                steps.append(
                    IncrementalFemStep(
                        time_index=step_idx,
                        time_value=t_next,
                        dt=dt_total,
                        substeps=n_sub,
                        newton_iterations=total_iters,
                        residual_norm=last_res,
                        displacement=u_curr.copy(),
                        reactions=reactions,
                        total_free_energy=last_free_e,
                        step_dissipation=total_diss,
                        min_det_f=min_det,
                        bound_state=bound,
                    )
                )
                solved_step = True
                break

        if not solved_step:
            return IncrementalFemTrajectory(
                converged=False,
                mesh_id=mesh.mesh_id,
                element_order=element_order,
                steps=tuple(steps),
                final_state=state_curr,
                reason=f"step {step_idx} at t={t_next} failed: {last_reason}",
            )

    return IncrementalFemTrajectory(
        converged=True,
        mesh_id=mesh.mesh_id,
        element_order=element_order,
        steps=tuple(steps),
        final_state=state_curr,
        reason="all incremental viscoelastic FEM steps converged",
    )


__all__ = (
    "DirichletHistoryBC",
    "IncrementalFemStep",
    "IncrementalFemTrajectory",
    "solve_incremental_fem_trajectory",
)
