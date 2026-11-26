"""3D goal-oriented Dual-Weighted Residual (DWR) linear adjoint certification (V5-1).

Provides single-linearization 3D QoI error estimation without executing a full
nonlinear Newton FEM solve, so 3D ``certified`` serving stays fast and only
escalates to warm-started ``exact`` 3D FEM when an error tolerance or physical
guard is tripped.

Supported 3D differentiable QoIs:
* :class:`ComplianceQoi3D`: external work ``f_ext^T u`` (mJ = N*mm).
* :class:`ReactionForceQoi3D`: summed reaction component on a Dirichlet BC face.
* :class:`TotalStrainEnergyQoi3D`: volumetric strain energy ``int W(F) dV``.
* :class:`SmoothedPeakVonMisesQoi3D`: differentiable Lp-norm proxy over
  unsmoothed Gauss-point Cauchy von Mises stress (preserving stress hotspots).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from piginot.certify.certificate import CertificateVerdict
from piginot.certify.guards import coercive_inertia
from piginot.fem.backend_3d import (
    FemRequest3D,
    TorchFemBackend3D,
    assemble_surface_tractions_3d,
    face_node_indices_3d,
)
from piginot.fem.linear_solvers import LinearSolverConfig, solve_linear_system
from piginot.physics import consistent_tangent

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DWR_3D_CERTIFICATION_SCHEMA: Final = "formamech.dwr-3d-certification-v1"


@dataclass(frozen=True, slots=True)
class Linearized3DState:
    """Single-point discrete 3D FEM linearization around a surrogate field."""

    request: FemRequest3D
    u_full: FloatArray  # (3*N,) with exact Dirichlet imposed
    free_dofs: IntArray
    bc_comp_dofs: dict[str, tuple[IntArray, IntArray, IntArray]]
    cells: IntArray  # (E, n_loc)
    d_phys: FloatArray  # (E, Q, n_loc, 3)
    w_vol: FloatArray  # (E, Q)
    f_int: FloatArray  # (3*N,)
    f_ext: FloatArray  # (3*N,)
    residual: FloatArray  # (3*N,) = f_int - f_ext
    tangent: csr_matrix  # (3*N, 3*N)
    first_piola: FloatArray  # (E, Q, 3, 3)
    deformation_gradient: FloatArray  # (E, Q, 3, 3)
    strain_energy: float
    min_det_f: float
    dirichlet_deviation: float


class DifferentiableQoi3D(Protocol):
    """Differentiable scalar functional on one :class:`Linearized3DState`."""

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def unit(self) -> str: ...

    def evaluate(self, state: Linearized3DState) -> float: ...

    def free_gradient(self, state: Linearized3DState) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class ComplianceQoi3D:
    """External work / compliance ``f_ext^T u`` (N*mm = mJ)."""

    name: str = "compliance"
    unit: str = "mJ"

    @property
    def kind(self) -> str:
        return "compliance"

    def evaluate(self, state: Linearized3DState) -> float:
        return float(np.dot(state.f_ext, state.u_full))

    def free_gradient(self, state: Linearized3DState) -> FloatArray:
        return np.asarray(state.f_ext[state.free_dofs], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class TotalStrainEnergyQoi3D:
    """Total volumetric strain energy ``int_Omega W(F) dV`` (exact gradient = ``f_int``)."""

    name: str = "total_strain_energy"
    unit: str = "mJ"

    @property
    def kind(self) -> str:
        return "total_strain_energy"

    def evaluate(self, state: Linearized3DState) -> float:
        return float(state.strain_energy)

    def free_gradient(self, state: Linearized3DState) -> FloatArray:
        return np.asarray(state.f_int[state.free_dofs], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ReactionForceQoi3D:
    """Component reaction force summed over one named 3D Dirichlet BC."""

    name: str
    boundary_condition_name: str
    component: str = "x"
    unit: str = "N"

    def __post_init__(self) -> None:
        if self.component not in ("x", "y", "z"):
            raise ValueError(f"component must be x/y/z, got {self.component!r}")

    @property
    def kind(self) -> str:
        return "reaction_force"

    def _dofs(self, state: Linearized3DState) -> IntArray:
        if self.boundary_condition_name not in state.bc_comp_dofs:
            raise KeyError(
                f"unknown 3D Dirichlet BC {self.boundary_condition_name!r}; "
                f"available: {sorted(state.bc_comp_dofs)}"
            )
        comp_idx = {"x": 0, "y": 1, "z": 2}[self.component]
        dofs = state.bc_comp_dofs[self.boundary_condition_name][comp_idx]
        if dofs.size == 0:
            raise ValueError(
                f"BC {self.boundary_condition_name!r} has no constrained {self.component!r} DOFs"
            )
        return dofs

    def evaluate(self, state: Linearized3DState) -> float:
        dofs = self._dofs(state)
        return float(np.sum(state.residual[dofs]))

    def free_gradient(self, state: Linearized3DState) -> FloatArray:
        dofs = self._dofs(state)
        block = state.tangent[dofs][:, state.free_dofs]
        ones = np.ones(dofs.size, dtype=np.float64)
        return np.asarray(block.T @ ones, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SmoothedPeakVonMisesQoi3D:
    """Differentiable Lp-norm proxy over unsmoothed 3D Gauss-point Cauchy von Mises stress."""

    name: str = "smoothed_peak_von_mises_3d"
    p_norm: float = 8.0
    smooth_eps: float = 1.0e-12
    unit: str = "MPa"

    def __post_init__(self) -> None:
        if not math.isfinite(self.p_norm) or self.p_norm < 2.0:
            raise ValueError("p_norm must be finite and >= 2.0")
        if not math.isfinite(self.smooth_eps) or self.smooth_eps <= 0.0:
            raise ValueError("smooth_eps must be finite and positive")

    @property
    def kind(self) -> str:
        return "smoothed_peak_stress"

    def _vm_torch(self, f_t: torch.Tensor, p_t: torch.Tensor) -> torch.Tensor:
        j = torch.linalg.det(f_t)
        sigma = torch.einsum("...ij,...kj->...ik", p_t, f_t) / j[..., None, None]
        tr = sigma[..., 0, 0] + sigma[..., 1, 1] + sigma[..., 2, 2]
        eye3 = torch.eye(3, dtype=f_t.dtype, device=f_t.device)
        dev = sigma - (tr / 3.0)[..., None, None] * eye3
        vm_sq = 1.5 * torch.sum(dev * dev, dim=(-2, -1)) + self.smooth_eps**2
        return torch.sqrt(vm_sq)

    def evaluate(self, state: Linearized3DState) -> float:
        with torch.no_grad():
            f_t = torch.from_numpy(np.ascontiguousarray(state.deformation_gradient))
            p_t = torch.from_numpy(np.ascontiguousarray(state.first_piola))
            vm = self._vm_torch(f_t, p_t).numpy()
        vol = max(float(np.sum(state.w_vol)), 1.0e-15)
        scale = max(float(np.max(vm)), self.smooth_eps)
        mean_p = float(np.sum(((vm / scale) ** self.p_norm) * state.w_vol) / vol)
        return float(scale * (mean_p ** (1.0 / self.p_norm)))

    def free_gradient(self, state: Linearized3DState) -> FloatArray:
        vol = max(float(np.sum(state.w_vol)), 1.0e-15)
        f_t = torch.from_numpy(
            np.ascontiguousarray(state.deformation_gradient)
        ).requires_grad_(True)
        p_t = torch.from_numpy(np.ascontiguousarray(state.first_piola)).requires_grad_(
            True
        )
        w_t = torch.from_numpy(np.ascontiguousarray(state.w_vol))
        vm = self._vm_torch(f_t, p_t)
        scale = float(vm.detach().max().item())
        scale = max(scale, self.smooth_eps)
        mean_p = torch.sum(((vm / scale) ** self.p_norm) * w_t) / vol
        q_val = scale * (mean_p ** (1.0 / self.p_norm))
        dq_df_direct, dq_dp = torch.autograd.grad(q_val, (f_t, p_t))
        with torch.no_grad():
            c_tan = consistent_tangent(f_t.detach(), state.request.material)
            dq_df_total = dq_df_direct + torch.einsum("eqij,eqijkl->eqkl", dq_dp, c_tan)
        dq_df_np = dq_df_total.detach().numpy()
        # dQ / du_{e,n,k} = sum_{q,l} dq_df_{e,q,k,l} * dN_{e,q,n,l}
        g_elem = np.einsum("eqkl,eqnl->enk", dq_df_np, state.d_phys)
        n_nodes = state.u_full.size // 3
        g_full = np.zeros((n_nodes, 3), dtype=np.float64)
        np.add.at(g_full, state.cells, g_elem)
        return np.asarray(g_full.reshape(-1)[state.free_dofs], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class DwrEstimate3D:
    """Per-QoI 3D DWR adjoint error estimate."""

    qoi_name: str
    qoi_kind: str
    unit: str
    verdict: CertificateVerdict
    qoi_value: float | None
    dwr_estimate: float | None
    relative_dwr_estimate: float | None
    corrected_qoi_value: float | None
    adjoint_residual_norm: float | None
    reason: str = ""

    def document(self) -> dict[str, Any]:
        return {
            "qoi_name": self.qoi_name,
            "qoi_kind": self.qoi_kind,
            "unit": self.unit,
            "verdict": self.verdict.value,
            "qoi_value": self.qoi_value,
            "dwr_estimate": self.dwr_estimate,
            "relative_dwr_estimate": self.relative_dwr_estimate,
            "corrected_qoi_value": self.corrected_qoi_value,
            "adjoint_residual_norm": self.adjoint_residual_norm,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class Dwr3DCertificationResult:
    """Complete single-linearization 3D DWR certificate across requested QoIs."""

    mesh_id: str
    verdict: CertificateVerdict
    min_det_f: float
    dirichlet_deviation: float
    residual_norm: float
    eta_energy: float | None
    one_step_disp_rel_l2: float | None
    positive_definite: bool | None
    estimates: tuple[DwrEstimate3D, ...]
    should_escalate: bool
    escalation_tolerance: float | None
    wall_time_s: float
    reason: str

    def document(self) -> dict[str, Any]:
        return {
            "schema": DWR_3D_CERTIFICATION_SCHEMA,
            "mesh_id": self.mesh_id,
            "verdict": self.verdict.value,
            "min_det_f": float(self.min_det_f),
            "dirichlet_deviation": float(self.dirichlet_deviation),
            "residual_norm": float(self.residual_norm),
            "eta_energy": self.eta_energy,
            "one_step_disp_rel_l2": self.one_step_disp_rel_l2,
            "positive_definite": self.positive_definite,
            "estimates": [e.document() for e in self.estimates],
            "should_escalate": self.should_escalate,
            "escalation_tolerance": self.escalation_tolerance,
            "wall_time_s": float(self.wall_time_s),
            "reason": self.reason,
        }


def linearize_3d_state(
    request: FemRequest3D,
    nodal_displacement: FloatArray,
) -> Linearized3DState:
    """Build one 3D discrete linearization at ``nodal_displacement`` (zero Newton loop)."""
    mesh = request.mesh
    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes
    u_raw = np.asarray(nodal_displacement, dtype=np.float64).reshape(n_dofs)

    comp_idx = {"x": 0, "y": 1, "z": 2}
    target_d: dict[int, float] = {}
    bc_comp_dofs: dict[str, tuple[IntArray, IntArray, IntArray]] = {}
    for bc in request.dirichlet_bcs:
        nids = face_node_indices_3d(mesh, bc.face_tags)
        lists: tuple[list[int], list[int], list[int]] = ([], [], [])
        for comp, val in zip(bc.components, bc.values_mm, strict=True):
            c = comp_idx[comp]
            for nid in nids:
                dof = 3 * nid + c
                target_d[dof] = float(val)
                lists[c].append(dof)
        bc_comp_dofs[bc.name] = (
            np.asarray(lists[0], dtype=np.int64),
            np.asarray(lists[1], dtype=np.int64),
            np.asarray(lists[2], dtype=np.int64),
        )

    u_lifted = u_raw.copy()
    max_dev = 0.0
    constrained = np.zeros(n_dofs, dtype=bool)
    for dof, val in target_d.items():
        constrained[dof] = True
        max_dev = max(max_dev, abs(float(u_raw[dof]) - val))
        u_lifted[dof] = val
    free_dofs = np.nonzero(~constrained)[0].astype(np.int64)

    backend = TorchFemBackend3D()
    cells, d_phys, w_vol, _, rows, cols = backend._precompute_geometry(mesh)
    f_ext = assemble_surface_tractions_3d(mesh, request.traction_bcs).reshape(n_dofs)
    f_int, k_mat, p_np, f_np, energy, min_det = backend._assemble(
        u_lifted, cells, d_phys, w_vol, rows, cols, request.material, n_dofs
    )
    return Linearized3DState(
        request=request,
        u_full=u_lifted,
        free_dofs=free_dofs,
        bc_comp_dofs=bc_comp_dofs,
        cells=cells,
        d_phys=d_phys,
        w_vol=w_vol,
        f_int=f_int,
        f_ext=f_ext,
        residual=f_int - f_ext,
        tangent=k_mat,
        first_piola=p_np,
        deformation_gradient=f_np,
        strain_energy=energy,
        min_det_f=min_det,
        dirichlet_deviation=max_dev,
    )


def estimate_qoi_dwr_3d(
    request: FemRequest3D,
    nodal_displacement: FloatArray,
    qois: Sequence[DifferentiableQoi3D] | None = None,
    *,
    linear_solver: LinearSolverConfig | None = None,
    escalation_tolerance: float | None = None,
    check_inertia: bool = True,
) -> Dwr3DCertificationResult:
    """Compute 3D linear DWR adjoint error indicators for one surrogate prediction."""
    t0 = time.perf_counter()
    active_qois: tuple[DifferentiableQoi3D, ...] = (
        (ComplianceQoi3D(), TotalStrainEnergyQoi3D(), SmoothedPeakVonMisesQoi3D())
        if qois is None
        else tuple(qois)
    )
    if not active_qois:
        raise ValueError("qois must be non-empty")

    state = linearize_3d_state(request, nodal_displacement)
    r_free = state.residual[state.free_dofs]
    res_norm = float(np.linalg.norm(r_free))

    if state.min_det_f <= 1.0e-8 or not math.isfinite(state.strain_energy):
        return Dwr3DCertificationResult(
            mesh_id=request.mesh.mesh_id,
            verdict=CertificateVerdict.INADMISSIBLE,
            min_det_f=state.min_det_f,
            dirichlet_deviation=state.dirichlet_deviation,
            residual_norm=float("inf"),
            eta_energy=None,
            one_step_disp_rel_l2=None,
            positive_definite=None,
            estimates=tuple(
                DwrEstimate3D(
                    qoi_name=q.name,
                    qoi_kind=q.kind,
                    unit=q.unit,
                    verdict=CertificateVerdict.INADMISSIBLE,
                    qoi_value=None,
                    dwr_estimate=None,
                    relative_dwr_estimate=None,
                    corrected_qoi_value=None,
                    adjoint_residual_norm=None,
                    reason=f"folded/inadmissible 3D state (min_det_f={state.min_det_f:.3e})",
                )
                for q in active_qois
            ),
            should_escalate=True,
            escalation_tolerance=escalation_tolerance,
            wall_time_s=time.perf_counter() - t0,
            reason=f"inadmissible 3D deformation (min_det_f={state.min_det_f:.3e})",
        )

    k_ff = state.tangent[state.free_dofs][:, state.free_dofs].tocsr()
    pos_def: bool | None = None
    if check_inertia and k_ff.shape[0] <= 12_000:
        pos_def = coercive_inertia(k_ff).positive_definite
        if pos_def is False:
            return Dwr3DCertificationResult(
                mesh_id=request.mesh.mesh_id,
                verdict=CertificateVerdict.INDEFINITE,
                min_det_f=state.min_det_f,
                dirichlet_deviation=state.dirichlet_deviation,
                residual_norm=res_norm,
                eta_energy=None,
                one_step_disp_rel_l2=None,
                positive_definite=False,
                estimates=tuple(
                    DwrEstimate3D(
                        qoi_name=q.name,
                        qoi_kind=q.kind,
                        unit=q.unit,
                        verdict=CertificateVerdict.INDEFINITE,
                        qoi_value=q.evaluate(state),
                        dwr_estimate=None,
                        relative_dwr_estimate=None,
                        corrected_qoi_value=None,
                        adjoint_residual_norm=None,
                        reason="3D tangent stiffness K_ff is indefinite (buckling/instability)",
                    )
                    for q in active_qois
                ),
                should_escalate=True,
                escalation_tolerance=escalation_tolerance,
                wall_time_s=time.perf_counter() - t0,
                reason="3D tangent stiffness K_ff is not positive definite",
            )

    lin_cfg = request.linear_solver if linear_solver is None else linear_solver
    du_free, rep_du = solve_linear_system(
        k_ff, -r_free, config=lin_cfg, is_symmetric=True
    )
    if not rep_du.converged or not np.all(np.isfinite(du_free)):
        return Dwr3DCertificationResult(
            mesh_id=request.mesh.mesh_id,
            verdict=CertificateVerdict.INDEFINITE,
            min_det_f=state.min_det_f,
            dirichlet_deviation=state.dirichlet_deviation,
            residual_norm=res_norm,
            eta_energy=None,
            one_step_disp_rel_l2=None,
            positive_definite=pos_def,
            estimates=(),
            should_escalate=True,
            escalation_tolerance=escalation_tolerance,
            wall_time_s=time.perf_counter() - t0,
            reason=f"3D linear Newton step solve failed ({rep_du.reason})",
        )

    en_quad = float(du_free @ (k_ff @ du_free))
    if en_quad < -1.0e-12:
        pos_def = False
        return Dwr3DCertificationResult(
            mesh_id=request.mesh.mesh_id,
            verdict=CertificateVerdict.INDEFINITE,
            min_det_f=state.min_det_f,
            dirichlet_deviation=state.dirichlet_deviation,
            residual_norm=res_norm,
            eta_energy=None,
            one_step_disp_rel_l2=None,
            positive_definite=False,
            estimates=(),
            should_escalate=True,
            escalation_tolerance=escalation_tolerance,
            wall_time_s=time.perf_counter() - t0,
            reason="negative tangent energy norm du^T K du < 0",
        )
    eta_energy = math.sqrt(max(en_quad, 0.0))
    u_corr_full = state.u_full.copy()
    u_corr_full[state.free_dofs] += du_free
    one_step_rel = float(
        np.linalg.norm(du_free) / max(float(np.linalg.norm(u_corr_full)), 1.0e-12)
    )

    estimates: list[DwrEstimate3D] = []
    max_rel = 0.0
    any_failed = False
    for qoi in active_qois:
        q_val = qoi.evaluate(state)
        g_q = qoi.free_gradient(state)
        z_free, rep_adj = solve_linear_system(
            k_ff.T.tocsr(), g_q, config=lin_cfg, is_symmetric=True
        )
        if not rep_adj.converged or not np.all(np.isfinite(z_free)):
            any_failed = True
            estimates.append(
                DwrEstimate3D(
                    qoi_name=qoi.name,
                    qoi_kind=qoi.kind,
                    unit=qoi.unit,
                    verdict=CertificateVerdict.INDEFINITE,
                    qoi_value=q_val,
                    dwr_estimate=None,
                    relative_dwr_estimate=None,
                    corrected_qoi_value=None,
                    adjoint_residual_norm=None,
                    reason=f"3D adjoint solve failed ({rep_adj.reason})",
                )
            )
            continue
        adj_res = float(np.linalg.norm(k_ff.T @ z_free - g_q))
        eta_dwr = float(-np.dot(z_free, r_free))
        rel_dwr = abs(eta_dwr) / max(abs(q_val), 1.0e-12)
        max_rel = max(max_rel, rel_dwr)
        estimates.append(
            DwrEstimate3D(
                qoi_name=qoi.name,
                qoi_kind=qoi.kind,
                unit=qoi.unit,
                verdict=CertificateVerdict.OK,
                qoi_value=q_val,
                dwr_estimate=eta_dwr,
                relative_dwr_estimate=rel_dwr,
                corrected_qoi_value=q_val + eta_dwr,
                adjoint_residual_norm=adj_res,
                reason="ok",
            )
        )

    exceeded_tol = escalation_tolerance is not None and max_rel > float(
        escalation_tolerance
    )
    should_escalate = any_failed or exceeded_tol
    verdict = CertificateVerdict.INDEFINITE if any_failed else CertificateVerdict.OK
    reason = (
        "3D DWR adjoint estimate exceeded escalation_tolerance"
        if exceeded_tol and not any_failed
        else ("3D adjoint solve failed" if any_failed else "3D linear DWR certified")
    )
    return Dwr3DCertificationResult(
        mesh_id=request.mesh.mesh_id,
        verdict=verdict,
        min_det_f=state.min_det_f,
        dirichlet_deviation=state.dirichlet_deviation,
        residual_norm=res_norm,
        eta_energy=eta_energy,
        one_step_disp_rel_l2=one_step_rel,
        positive_definite=True if pos_def is None else pos_def,
        estimates=tuple(estimates),
        should_escalate=should_escalate,
        escalation_tolerance=escalation_tolerance,
        wall_time_s=time.perf_counter() - t0,
        reason=reason,
    )


__all__ = (
    "DWR_3D_CERTIFICATION_SCHEMA",
    "ComplianceQoi3D",
    "DifferentiableQoi3D",
    "Dwr3DCertificationResult",
    "DwrEstimate3D",
    "Linearized3DState",
    "ReactionForceQoi3D",
    "SmoothedPeakVonMisesQoi3D",
    "TotalStrainEnergyQoi3D",
    "estimate_qoi_dwr_3d",
    "linearize_3d_state",
)
