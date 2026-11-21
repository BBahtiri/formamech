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
from skfem import LinearForm, asm  # type: ignore[import-untyped]
from skfem.helpers import grad  # type: ignore[import-untyped]

from piginot.certify.certificate import CertificateVerdict
from piginot.certify.guards import coercive_inertia
from piginot.fem.linearization import LinearizedState
from piginot.fem.models import FemSolveFailure
from piginot.fem.protocol import DiscreteLinearization
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
        self, linearization: DiscreteLinearization, state: LinearizedState
    ) -> float: ...

    def free_gradient(
        self, linearization: DiscreteLinearization, state: LinearizedState
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

    def _dofs(self, linearization: DiscreteLinearization) -> NDArray[np.int64]:
        by_name = linearization.constraints.by_name
        if self.boundary_condition_name not in by_name:
            raise KeyError(
                f"unknown boundary condition {self.boundary_condition_name!r}; "
                f"available: {sorted(by_name)}"
            )
        dofs: NDArray[np.int64] = by_name[self.boundary_condition_name]
        if dofs.size == 0:
            raise ValueError(
                f"boundary condition {self.boundary_condition_name!r} has zero DOFs"
            )
        return dofs

    def evaluate(
        self, linearization: DiscreteLinearization, state: LinearizedState
    ) -> float:
        dofs = self._dofs(linearization)
        return float(np.sum(state.residual[dofs]))

    def free_gradient(
        self, linearization: DiscreteLinearization, state: LinearizedState
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

    def _cell_mask(self, linearization: DiscreteLinearization) -> NDArray[np.bool_]:
        regions = np.asarray(linearization.request.mesh.region_ids, dtype=np.int64)
        if self.region_id is None:
            mask = np.ones(regions.shape, dtype=bool)
        else:
            mask = regions == self.region_id
        if not np.any(mask):
            raise ValueError(f"region_id={self.region_id} selects zero cells")
        return mask

    def evaluate(
        self, linearization: DiscreteLinearization, state: LinearizedState
    ) -> float:
        mask = self._cell_mask(linearization)
        dx = np.asarray(linearization.quadrature_weights(), dtype=np.float64)[mask]
        area = float(np.sum(dx))
        p2d = np.asarray(state.assembly.first_piola, dtype=np.float64)[mask, :, :2, :2]
        if self.component is not None:
            r, c = self.component
            vals = p2d[..., r, c]
        else:
            vals = np.sqrt(np.sum(p2d * p2d, axis=(-2, -1)) + self.smooth_eps**2)
        return float(np.sum(vals * dx) / area)

    def free_gradient(self, linearization: Any, state: LinearizedState) -> FloatArray:
        mask = self._cell_mask(linearization)
        dx_all = np.asarray(linearization.quadrature_weights(), dtype=np.float64)
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
        if hasattr(linearization, "assemble_grad_weighted_linear_form"):
            full_grad = np.asarray(
                linearization.assemble_grad_weighted_linear_form(dq_df),
                dtype=np.float64,
            )
        else:
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
class TotalStrainEnergyQoi:
    """Total discrete strain energy ``int_Omega W(F) dA`` (exact gradient = ``f_int``)."""

    name: str = "total_strain_energy"
    unit: str = "mJ"

    @property
    def kind(self) -> str:
        return "total_strain_energy"

    def evaluate(
        self, linearization: DiscreteLinearization, state: LinearizedState
    ) -> float:
        return float(state.assembly.strain_energy)

    def free_gradient(
        self, linearization: DiscreteLinearization, state: LinearizedState
    ) -> FloatArray:
        free = linearization.constraints.free
        return np.asarray(state.assembly.internal_force[free], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SmoothedPeakVonMisesQoi:
    """Smooth Lp-norm proxy for peak in-plane equivalent stress over a region.

    Evaluates ``((1/|Omega_r|) int_{Omega_r} sigma_eq(P)^p dA)^(1/p)`` where
    ``sigma_eq(P) = sqrt(P11^2 - P11*P22 + P22^2 + 3*((P12+P21)/2)^2 + eps^2)``.
    Smooth and differentiable for adjoint DWR while approximating peak von Mises
    stress on non-singular regions.
    """

    name: str = "smoothed_peak_von_mises"
    p_norm: float = 8.0
    region_id: int | None = None
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

    def _cell_mask(self, linearization: DiscreteLinearization) -> NDArray[np.bool_]:
        regions = np.asarray(linearization.request.mesh.region_ids, dtype=np.int64)
        mask = (
            np.ones(regions.shape, dtype=bool)
            if self.region_id is None
            else (regions == self.region_id)
        )
        if not np.any(mask):
            raise ValueError(f"region_id={self.region_id} selects zero cells")
        return mask

    def _sigma_eq(self, p2d: FloatArray) -> FloatArray:
        p11 = p2d[..., 0, 0]
        p22 = p2d[..., 1, 1]
        ps = 0.5 * (p2d[..., 0, 1] + p2d[..., 1, 0])
        val = p11 * p11 - p11 * p22 + p22 * p22 + 3.0 * ps * ps + self.smooth_eps**2
        return np.asarray(np.sqrt(val), dtype=np.float64)

    def evaluate(
        self, linearization: DiscreteLinearization, state: LinearizedState
    ) -> float:
        mask = self._cell_mask(linearization)
        dx = np.asarray(linearization.quadrature_weights(), dtype=np.float64)[mask]
        area = float(np.sum(dx))
        p2d = np.asarray(state.assembly.first_piola, dtype=np.float64)[mask, :, :2, :2]
        seq = self._sigma_eq(p2d)
        scale = max(float(np.max(seq)), self.smooth_eps)
        mean_p = float(np.sum(((seq / scale) ** self.p_norm) * dx) / area)
        return float(scale * (mean_p ** (1.0 / self.p_norm)))

    def free_gradient(self, linearization: Any, state: LinearizedState) -> FloatArray:
        mask = self._cell_mask(linearization)
        dx_all = np.asarray(linearization.quadrature_weights(), dtype=np.float64)
        dx = dx_all[mask]
        area = float(np.sum(dx))

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

        p2d_all = np.asarray(state.assembly.first_piola, dtype=np.float64)[..., :2, :2]
        p_sub = p2d_all[mask]
        seq = self._sigma_eq(p_sub)
        scale = max(float(np.max(seq)), self.smooth_eps)
        p_exp = self.p_norm
        mean_p = float(np.sum(((seq / scale) ** p_exp) * dx) / area)
        # dQ/d(seq) = (1 / area) * (mean_p^(1/p - 1)) * (seq / scale)^(p - 1)
        dq_dseq = (
            (mean_p ** (1.0 / p_exp - 1.0)) * ((seq / scale) ** (p_exp - 1.0)) / area
        )

        p11 = p_sub[..., 0, 0]
        p22 = p_sub[..., 1, 1]
        ps = 0.5 * (p_sub[..., 0, 1] + p_sub[..., 1, 0])
        dq_dp = np.zeros_like(p2d_all)
        factor = dq_dseq / seq
        dq_dp[mask, :, 0, 0] = factor * (p11 - 0.5 * p22)
        dq_dp[mask, :, 1, 1] = factor * (p22 - 0.5 * p11)
        dq_dp[mask, :, 0, 1] = factor * (1.5 * ps)
        dq_dp[mask, :, 1, 0] = factor * (1.5 * ps)

        dq_df = np.einsum("cqij,cqijkl->cqkl", dq_dp, condensed)
        if hasattr(linearization, "assemble_grad_weighted_linear_form"):
            full_grad = np.asarray(
                linearization.assemble_grad_weighted_linear_form(dq_df),
                dtype=np.float64,
            )
        else:
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
    linearization: DiscreteLinearization,
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
    coercive = coercive_inertia(k_free)
    guards["positive_definite"] = coercive.positive_definite
    if not coercive.tangent_symmetric:
        guards["tangent_symmetric"] = False
    if not coercive.positive_definite:
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
    adjoint_free = linearization.solve_adjoint(before, grad_q)
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
    "SmoothedPeakVonMisesQoi",
    "TotalStrainEnergyQoi",
    "estimate_qoi_dwr",
)
