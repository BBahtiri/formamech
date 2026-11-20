"""Goal-oriented Dual-Weighted Residual (DWR) / adjoint estimators for smooth QoIs.

Provides exact discrete sensitivities and linearized dual solves for differentiable
scalar quantities of interest on any ``FemLinearization``:

* :class:`ReactionForceQoi`: total work-conjugate reaction force on a declared
  essential boundary condition group.
* :class:`RegionalAverageStressQoi`: area-weighted regional average of a first
  Piola stress component or smoothed Frobenius stress norm.

The DWR estimate ``eta_dwr = -z^T r_free(u)`` (where ``K_free(u)^T z = dQ/du_free``)
estimates the discrete error ``Q(u_FEM) - Q(u)`` on the same mesh. It is an
error indicator against the discrete FEM solution, not a continuum error bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse.linalg import spsolve  # type: ignore[import-untyped]
from skfem import LinearForm, asm  # type: ignore[import-untyped]
from skfem.helpers import grad  # type: ignore[import-untyped]

from piginot.certify.certificate import CertificateVerdict
from piginot.certify.guards import symmetric_inertia
from piginot.fem.linearization import FemLinearization, LinearizedState
from piginot.fem.models import FemSolveFailure
from piginot.physics import consistent_tangent

FloatArray = NDArray[np.float64]


@LinearForm  # type: ignore[untyped-decorator]
def _weighted_stress_grad_form(test: Any, data: Any) -> Any:
    return np.einsum("ij...,ij...->...", data.weight_tensor, grad(test))


class DifferentiableQoi(Protocol):
    """Differentiable scalar functional on one ``FemLinearization``."""

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def unit(self) -> str: ...

    def evaluate(
        self, linearization: FemLinearization, state: LinearizedState
    ) -> float: ...

    def free_gradient(
        self, linearization: FemLinearization, state: LinearizedState
    ) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class ReactionForceQoi:
    """Discrete reaction force summed over one named Dirichlet condition."""

    name: str
    boundary_condition_name: str
    unit: str = "N"

    @property
    def kind(self) -> str:
        return "reaction_force"

    def _dofs(self, linearization: FemLinearization) -> NDArray[np.int64]:
        by_name = linearization.constraints.by_name
        if self.boundary_condition_name not in by_name:
            raise KeyError(
                f"unknown boundary condition {self.boundary_condition_name!r}; "
                f"available: {sorted(by_name)}"
            )
        dofs = by_name[self.boundary_condition_name]
        if dofs.size == 0:
            raise ValueError(
                f"boundary condition {self.boundary_condition_name!r} has zero DOFs"
            )
        return dofs

    def evaluate(
        self, linearization: FemLinearization, state: LinearizedState
    ) -> float:
        dofs = self._dofs(linearization)
        return float(np.sum(state.residual[dofs]))

    def free_gradient(
        self, linearization: FemLinearization, state: LinearizedState
    ) -> FloatArray:
        dofs = self._dofs(linearization)
        free = linearization.constraints.free
        # R_bc(u) = sum_{i in dofs} (f_int_i(u) - f_ext_i), so dR_bc/du_free = (1^T K[dofs, free])^T.
        block = state.assembly.tangent[dofs][:, free]
        ones = np.ones(dofs.size, dtype=np.float64)
        return np.asarray(block.T @ ones, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class RegionalAverageStressQoi:
    """Area-weighted regional average first-Piola component or smooth Frobenius norm."""

    name: str
    component: tuple[int, int] | None = (0, 0)
    region_id: int | None = None
    smooth_eps: float = 1e-12
    unit: str = "MPa"

    def __post_init__(self) -> None:
        if self.component is not None:
            r, c = self.component
            if r not in (0, 1) or c not in (0, 1):
                raise ValueError("component indices must lie in {0, 1}")
        if not math.isfinite(self.smooth_eps) or self.smooth_eps <= 0.0:
            raise ValueError("smooth_eps must be finite and positive")

    @property
    def kind(self) -> str:
        return "regional_average_stress"

    def _cell_mask(self, linearization: FemLinearization) -> NDArray[np.bool_]:
        regions = np.asarray(linearization.request.mesh.region_ids, dtype=np.int64)
        if self.region_id is None:
            mask = np.ones(regions.shape, dtype=bool)
        else:
            mask = regions == self.region_id
        if not np.any(mask):
            raise ValueError(f"region_id={self.region_id} selects zero cells")
        return mask

    def evaluate(
        self, linearization: FemLinearization, state: LinearizedState
    ) -> float:
        mask = self._cell_mask(linearization)
        dx = np.asarray(linearization.basis.dx, dtype=np.float64)[mask]
        area = float(np.sum(dx))
        p2d = np.asarray(state.assembly.first_piola, dtype=np.float64)[mask, :, :2, :2]
        if self.component is not None:
            r, c = self.component
            vals = p2d[..., r, c]
        else:
            vals = np.sqrt(np.sum(p2d * p2d, axis=(-2, -1)) + self.smooth_eps**2)
        return float(np.sum(vals * dx) / area)

    def free_gradient(
        self, linearization: FemLinearization, state: LinearizedState
    ) -> FloatArray:
        mask = self._cell_mask(linearization)
        dx_all = np.asarray(linearization.basis.dx, dtype=np.float64)
        area = float(np.sum(dx_all[mask]))

        f3d = torch.from_numpy(
            np.ascontiguousarray(state.assembly.deformation_gradient)
        )
        tangent_3d = consistent_tangent(f3d, linearization.material)
        denom = tangent_3d[..., 2, 2, 2, 2]
        left = tangent_3d[..., :2, :2, 2, 2]
        right = tangent_3d[..., 2, 2, :2, :2]
        condensed = (
            (
                tangent_3d[..., :2, :2, :2, :2]
                - (
                    left[..., :, :, None, None]
                    * right[..., None, None, :, :]
                    / denom[..., None, None, None, None]
                )
            )
            .detach()
            .cpu()
            .numpy()
        )

        p2d = np.asarray(state.assembly.first_piola, dtype=np.float64)[..., :2, :2]
        n_cells, n_qpts = p2d.shape[:2]
        dq_dp = np.zeros((n_cells, n_qpts, 2, 2), dtype=np.float64)
        if self.component is not None:
            r, c = self.component
            dq_dp[mask, :, r, c] = 1.0 / area
        else:
            p_sub = p2d[mask]
            norm = np.sqrt(np.sum(p_sub * p_sub, axis=(-2, -1)) + self.smooth_eps**2)
            dq_dp[mask] = (p_sub / norm[..., None, None]) / area

        # Chain rule: dQ/dF_{kl} = sum_{i,j} (dQ/dP_{ij}) * A_cond_{ijkl}.
        dq_df = np.einsum("cqij,cqijkl->cqkl", dq_dp, condensed)
        weight_field = np.transpose(dq_df, (2, 3, 0, 1))
        full_grad = np.asarray(
            asm(
                _weighted_stress_grad_form,
                linearization.basis,
                weight_tensor=weight_field,
            ),
            dtype=np.float64,
        )
        return np.asarray(full_grad[linearization.constraints.free], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class DwrEstimate:
    """Goal-oriented adjoint/DWR error estimate for one QoI on one discrete mesh."""

    qoi_name: str
    qoi_kind: str
    unit: str
    verdict: CertificateVerdict
    qoi_value: float | None
    dwr_estimate: float | None
    newton_qoi_change: float | None
    corrected_qoi_value: float | None
    eta_energy: float | None
    adjoint_residual_norm: float | None
    dirichlet_deviation: float
    guards: dict[str, bool | None]
    reason: str = ""


def estimate_qoi_dwr(
    linearization: FemLinearization,
    nodal_displacement: FloatArray,
    qoi: DifferentiableQoi,
) -> DwrEstimate:
    """Solve the linearized adjoint problem and compute ``eta_dwr = -z^T r_free``."""
    raw = linearization.vector(nodal_displacement)
    deviation = linearization.dirichlet_deviation(raw)
    vector = linearization.with_exact_dirichlet(raw)
    try:
        before = linearization.state(vector)
    except FemSolveFailure as exc:
        return DwrEstimate(
            qoi_name=qoi.name,
            qoi_kind=qoi.kind,
            unit=qoi.unit,
            verdict=CertificateVerdict.INADMISSIBLE,
            qoi_value=None,
            dwr_estimate=None,
            newton_qoi_change=None,
            corrected_qoi_value=None,
            eta_energy=None,
            adjoint_residual_norm=None,
            dirichlet_deviation=deviation,
            guards={"admissible": False},
            reason=f"{exc.code.value}: {exc}",
        )

    guards: dict[str, bool | None] = {"admissible": True}
    free = linearization.constraints.free
    k_free = linearization.free_tangent(before)
    inertia = symmetric_inertia(k_free)
    guards["positive_definite"] = inertia.positive_definite
    if not inertia.positive_definite:
        return DwrEstimate(
            qoi_name=qoi.name,
            qoi_kind=qoi.kind,
            unit=qoi.unit,
            verdict=CertificateVerdict.INDEFINITE,
            qoi_value=qoi.evaluate(linearization, before),
            dwr_estimate=None,
            newton_qoi_change=None,
            corrected_qoi_value=None,
            eta_energy=None,
            adjoint_residual_norm=None,
            dirichlet_deviation=deviation,
            guards=guards,
            reason="free tangent is not positive definite",
        )

    q_val = qoi.evaluate(linearization, before)
    grad_q = qoi.free_gradient(linearization, before)
    adjoint_free = np.asarray(spsolve(k_free.T.tocsc(), grad_q), dtype=np.float64)
    if not np.all(np.isfinite(adjoint_free)):
        guards["adjoint_solved"] = False
        return DwrEstimate(
            qoi_name=qoi.name,
            qoi_kind=qoi.kind,
            unit=qoi.unit,
            verdict=CertificateVerdict.INDEFINITE,
            qoi_value=q_val,
            dwr_estimate=None,
            newton_qoi_change=None,
            corrected_qoi_value=None,
            eta_energy=None,
            adjoint_residual_norm=None,
            dirichlet_deviation=deviation,
            guards=guards,
            reason="adjoint solve produced non-finite values",
        )

    guards["adjoint_solved"] = True
    adj_res = float(np.linalg.norm(k_free.T @ adjoint_free - grad_q))
    r_free = before.residual[free]
    eta_dwr = float(-adjoint_free @ r_free)

    try:
        increment = linearization.newton_increment(before)
        after = linearization.state(before.displacement + increment)
        guards["corrected_admissible"] = True
    except FemSolveFailure as exc:
        guards["corrected_admissible"] = False
        return DwrEstimate(
            qoi_name=qoi.name,
            qoi_kind=qoi.kind,
            unit=qoi.unit,
            verdict=CertificateVerdict.INADMISSIBLE,
            qoi_value=q_val,
            dwr_estimate=eta_dwr,
            newton_qoi_change=None,
            corrected_qoi_value=None,
            eta_energy=None,
            adjoint_residual_norm=adj_res,
            dirichlet_deviation=deviation,
            guards=guards,
            reason=f"corrected state inadmissible: {exc}",
        )

    q_corr = qoi.evaluate(linearization, after)
    eta_energy = linearization.energy_norm(increment, before)
    return DwrEstimate(
        qoi_name=qoi.name,
        qoi_kind=qoi.kind,
        unit=qoi.unit,
        verdict=CertificateVerdict.OK,
        qoi_value=q_val,
        dwr_estimate=eta_dwr,
        newton_qoi_change=q_corr - q_val,
        corrected_qoi_value=q_corr,
        eta_energy=eta_energy,
        adjoint_residual_norm=adj_res,
        dirichlet_deviation=deviation,
        guards=guards,
        reason="adjoint DWR and Newton QoI indicators evaluated",
    )


__all__ = (
    "DifferentiableQoi",
    "DwrEstimate",
    "ReactionForceQoi",
    "RegionalAverageStressQoi",
    "estimate_qoi_dwr",
)
