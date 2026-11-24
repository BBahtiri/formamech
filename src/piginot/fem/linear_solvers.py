"""Pluggable 3D sparse linear solvers, preconditioners, and neural deflation (V3-1).

Removes the ~30k-50k DOF CPU direct-LU memory ceiling in 3D finite-strain FEM by
providing symmetric and unsymmetric Krylov solvers with pure-MIT/BSD
preconditioners (3x3 nodal block-Jacobi, symmetric Gauss-Seidel/SSOR, ILU,
optional ``pyamg`` smoothed aggregation AMG, and two-level neural coarse-space
deflation built from operator snapshots).
"""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import (  # type: ignore[import-untyped]
    csc_matrix,
    csr_matrix,
    diags,
    tril,
    triu,
)
from scipy.sparse.linalg import (  # type: ignore[import-untyped]
    LinearOperator,
    cg,
    gmres,
    spilu,
    splu,
    spsolve,
)

from piginot.contracts.state_machine import (
    StateMachineSpec,
    StateSnapshot,
    TransitionSpec,
    register_state_machine,
)

FloatArray = NDArray[np.float64]
LINEAR_SOLVER_CONFIG_SCHEMA: Final = "formamech.linear-solver-config-v1"
LINEAR_SOLVE_REPORT_SCHEMA: Final = "formamech.linear-solve-report-v1"


class LinearSolverMethod(StrEnum):
    """Supported sparse linear solver strategies for 3D FEM tangents."""

    AUTO = "auto"
    DIRECT = "direct"
    PCG_JACOBI = "pcg-jacobi"
    PCG_SSOR = "pcg-ssor"
    PCG_ILU = "pcg-ilu"
    PCG_AMG = "pcg-amg"
    GMRES_ILU = "gmres-ilu"


class LinearSolverState(StrEnum):
    """Explicit lifecycle states of one 3D linear solve."""

    INITIALIZED = "INITIALIZED"
    FACTORIZING_PRECOND = "FACTORIZING_PRECOND"
    ITERATING = "ITERATING"
    STAGNATED_FALLBACK = "STAGNATED_FALLBACK"
    CONVERGED = "CONVERGED"
    FAILED = "FAILED"


LINEAR_SOLVER_MACHINE: Final[StateMachineSpec[LinearSolverState]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.fem.linear_solver",
            description=(
                "Lifecycle of a 3D sparse tangent linear solve from initialization "
                "through preconditioning, Krylov iteration, optional direct fallback, "
                "to terminal CONVERGED or FAILED."
            ),
            states=tuple(LinearSolverState),
            initial_states=(LinearSolverState.INITIALIZED,),
            terminal_states=(LinearSolverState.CONVERGED, LinearSolverState.FAILED),
            transitions=(
                TransitionSpec(
                    LinearSolverState.INITIALIZED,
                    LinearSolverState.FACTORIZING_PRECOND,
                    "build_preconditioner",
                    "Construct iterative preconditioner or sparse LU factor.",
                ),
                TransitionSpec(
                    LinearSolverState.INITIALIZED,
                    LinearSolverState.CONVERGED,
                    "direct_converged",
                    "Zero-DOF or exact direct solve completed immediately.",
                ),
                TransitionSpec(
                    LinearSolverState.INITIALIZED,
                    LinearSolverState.FAILED,
                    "reject_invalid",
                    "Matrix/RHS invalid or symmetry contract violated.",
                ),
                TransitionSpec(
                    LinearSolverState.FACTORIZING_PRECOND,
                    LinearSolverState.ITERATING,
                    "start_krylov",
                    "Begin Krylov iterations with built preconditioner.",
                ),
                TransitionSpec(
                    LinearSolverState.FACTORIZING_PRECOND,
                    LinearSolverState.CONVERGED,
                    "direct_converged",
                    "Direct sparse factorization solved system within tolerance.",
                ),
                TransitionSpec(
                    LinearSolverState.FACTORIZING_PRECOND,
                    LinearSolverState.STAGNATED_FALLBACK,
                    "precond_fallback",
                    "Preconditioner setup failed; falling back to direct solve.",
                ),
                TransitionSpec(
                    LinearSolverState.FACTORIZING_PRECOND,
                    LinearSolverState.FAILED,
                    "precond_failed",
                    "Preconditioner setup failed with fallback disabled.",
                ),
                TransitionSpec(
                    LinearSolverState.ITERATING,
                    LinearSolverState.CONVERGED,
                    "krylov_converged",
                    "Krylov solver satisfied relative/absolute residual tolerance.",
                ),
                TransitionSpec(
                    LinearSolverState.ITERATING,
                    LinearSolverState.STAGNATED_FALLBACK,
                    "krylov_fallback",
                    "Krylov solver stagnated; escalating to exact direct fallback.",
                ),
                TransitionSpec(
                    LinearSolverState.ITERATING,
                    LinearSolverState.FAILED,
                    "krylov_failed",
                    "Krylov solver failed to converge with fallback disabled.",
                ),
                TransitionSpec(
                    LinearSolverState.STAGNATED_FALLBACK,
                    LinearSolverState.CONVERGED,
                    "fallback_converged",
                    "Direct fallback produced a finite verified increment.",
                ),
                TransitionSpec(
                    LinearSolverState.STAGNATED_FALLBACK,
                    LinearSolverState.FAILED,
                    "fallback_failed",
                    "Direct fallback failed or produced non-finite output.",
                ),
            ),
        )
    )
)


@dataclass(frozen=True, slots=True)
class LinearSolverConfig:
    """Declarative sparse linear solver configuration for 3D FEM backends."""

    method: LinearSolverMethod = LinearSolverMethod.AUTO
    direct_dof_threshold: int = 15_000
    rtol: float = 1.0e-10
    atol: float = 1.0e-12
    max_iterations: int = 2_000
    gmres_restart: int = 50
    ilu_drop_tol: float = 1.0e-4
    ilu_fill_factor: float = 10.0
    ssor_omega: float = 1.0
    allow_fallback_to_direct: bool = True
    use_neural_coarse_deflation: bool = False
    block_size: int = 3

    def __post_init__(self) -> None:
        method = LinearSolverMethod(self.method)
        object.__setattr__(self, "method", method)
        if type(self.direct_dof_threshold) is not int or self.direct_dof_threshold < 0:
            raise ValueError("direct_dof_threshold must be a nonnegative integer")
        if not math.isfinite(self.rtol) or self.rtol <= 0.0:
            raise ValueError("rtol must be finite and positive")
        if not math.isfinite(self.atol) or self.atol < 0.0:
            raise ValueError("atol must be finite and nonnegative")
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if type(self.gmres_restart) is not int or self.gmres_restart < 1:
            raise ValueError("gmres_restart must be >= 1")
        if not math.isfinite(self.ilu_drop_tol) or self.ilu_drop_tol <= 0.0:
            raise ValueError("ilu_drop_tol must be finite and positive")
        if not math.isfinite(self.ilu_fill_factor) or self.ilu_fill_factor < 1.0:
            raise ValueError("ilu_fill_factor must be >= 1.0")
        if not math.isfinite(self.ssor_omega) or not (0.0 < self.ssor_omega < 2.0):
            raise ValueError("ssor_omega must lie strictly in (0, 2)")
        if type(self.allow_fallback_to_direct) is not bool:
            raise TypeError("allow_fallback_to_direct must be bool")
        if type(self.use_neural_coarse_deflation) is not bool:
            raise TypeError("use_neural_coarse_deflation must be bool")
        if type(self.block_size) is not int or self.block_size < 1:
            raise ValueError("block_size must be >= 1")

    def document(self) -> dict[str, Any]:
        return {
            "schema": LINEAR_SOLVER_CONFIG_SCHEMA,
            "method": self.method.value,
            "direct_dof_threshold": self.direct_dof_threshold,
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "max_iterations": self.max_iterations,
            "gmres_restart": self.gmres_restart,
            "ilu_drop_tol": float(self.ilu_drop_tol),
            "ilu_fill_factor": float(self.ilu_fill_factor),
            "ssor_omega": float(self.ssor_omega),
            "allow_fallback_to_direct": self.allow_fallback_to_direct,
            "use_neural_coarse_deflation": self.use_neural_coarse_deflation,
            "block_size": self.block_size,
        }


@dataclass(frozen=True, slots=True)
class LinearSolveReport:
    """Auditable diagnostic record for one 3D tangent linear solve."""

    requested_method: LinearSolverMethod
    method_used: LinearSolverMethod
    state: LinearSolverState
    converged: bool
    iterations: int
    relative_residual: float
    n_dofs: int
    wall_time_s: float
    used_fallback: bool
    used_neural_deflation: bool
    reason: str

    def state_snapshot(self) -> StateSnapshot[LinearSolverState]:
        return LINEAR_SOLVER_MACHINE.snapshot(
            self.state,
            blockers=() if self.converged else (self.reason,),
            metadata={
                "method_used": self.method_used.value,
                "iterations": self.iterations,
                "relative_residual": self.relative_residual,
            },
        )

    def document(self) -> dict[str, Any]:
        return {
            "schema": LINEAR_SOLVE_REPORT_SCHEMA,
            "requested_method": self.requested_method.value,
            "method_used": self.method_used.value,
            "state": self.state.value,
            "converged": self.converged,
            "iterations": self.iterations,
            "relative_residual": (
                float(self.relative_residual)
                if math.isfinite(self.relative_residual)
                else None
            ),
            "n_dofs": self.n_dofs,
            "wall_time_s": float(self.wall_time_s),
            "used_fallback": self.used_fallback,
            "used_neural_deflation": self.used_neural_deflation,
            "reason": self.reason,
        }


class NeuralCoarsePreconditioner:
    """Two-level SPD coarse-space correction built from neural operator snapshots.

    Given a low-dimensional orthonormal basis ``Z`` of shape ``(n_dofs, r)``
    extracted via SVD from neural operator displacement predictions, defines the
    exact Galerkin coarse operator ``K_c = Z^T K Z`` and the symmetric two-level
    preconditioner::

        B = P_c + (I - P_c K) M^{-1} (I - K P_c),   P_c = Z K_c^{-1} Z^T

    which preserves symmetry and positive definiteness for PCG whenever ``K``
    and the fine smoother ``M^{-1}`` are SPD.
    """

    def __init__(
        self,
        basis: FloatArray,
        *,
        regularization: float = 1.0e-12,
    ) -> None:
        arr = np.asarray(basis, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
            raise ValueError("basis must be a non-empty 2D array of shape (n_dofs, r)")
        if not np.all(np.isfinite(arr)):
            raise ValueError("basis must contain only finite entries")
        if not math.isfinite(regularization) or regularization < 0.0:
            raise ValueError("regularization must be finite and nonnegative")
        # Orthonormalize columns via reduced QR
        q_mat, _ = np.linalg.qr(arr, mode="reduced")
        self._basis: FloatArray = np.ascontiguousarray(q_mat, dtype=np.float64)
        self._regularization = float(regularization)

    @classmethod
    def from_snapshots(
        cls,
        snapshots: Sequence[FloatArray],
        *,
        free_dofs: NDArray[np.intp] | None = None,
        max_modes: int = 16,
        energy_threshold: float = 1.0 - 1.0e-10,
    ) -> NeuralCoarsePreconditioner:
        """Build an orthonormal coarse basis from operator displacement snapshots."""
        if not snapshots:
            raise ValueError("at least one displacement snapshot is required")
        if type(max_modes) is not int or max_modes < 1:
            raise ValueError("max_modes must be >= 1")
        cols: list[FloatArray] = []
        for snap in snapshots:
            vec: FloatArray = np.asarray(snap, dtype=np.float64).reshape(-1)
            if free_dofs is not None:
                vec = np.asarray(vec[free_dofs], dtype=np.float64)
            if float(np.linalg.norm(vec)) > 1.0e-14:
                cols.append(vec)
        if not cols:
            raise ValueError("snapshots are all numerically zero on free DOFs")
        mat = np.column_stack(cols)
        u_mat, s_vals, _ = np.linalg.svd(mat, full_matrices=False)
        s_sq = s_vals * s_vals
        total = float(np.sum(s_sq))
        if total <= 0.0:
            raise ValueError("snapshot SVD has zero energy")
        cumulative = np.cumsum(s_sq) / total
        keep = int(np.searchsorted(cumulative, energy_threshold)) + 1
        keep = max(1, min(keep, max_modes, u_mat.shape[1]))
        return cls(u_mat[:, :keep])

    @property
    def n_dofs(self) -> int:
        return int(self._basis.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self._basis.shape[1])

    def wrap(
        self,
        matrix: csr_matrix,
        fine_preconditioner: LinearOperator,
    ) -> LinearOperator:
        """Combine fine smoother ``M^{-1}`` with exact coarse correction on ``matrix``."""
        n = int(matrix.shape[0])
        if n != self.n_dofs:
            raise ValueError(
                f"NeuralCoarsePreconditioner DOF mismatch: basis has {self.n_dofs}, "
                f"matrix has {n}"
            )
        z = self._basis  # (n, r)
        kz = np.asarray(matrix @ z, dtype=np.float64)  # (n, r)
        k_c = z.T @ kz  # (r, r)
        k_c = 0.5 * (k_c + k_c.T) + self._regularization * np.eye(
            self.n_modes, dtype=np.float64
        )
        # Solve coarse system via stable dense LU/Cholesky
        try:
            chol = np.linalg.cholesky(k_c)

            def _coarse_solve(vec_r: FloatArray) -> FloatArray:
                y = np.linalg.solve(chol, vec_r)
                return np.asarray(np.linalg.solve(chol.T, y), dtype=np.float64)

        except np.linalg.LinAlgError:

            def _coarse_solve(vec_r: FloatArray) -> FloatArray:
                return np.asarray(
                    np.linalg.lstsq(k_c, vec_r, rcond=None)[0], dtype=np.float64
                )

        def _matvec(r: FloatArray) -> FloatArray:
            r_arr = np.asarray(r, dtype=np.float64)
            # Symmetric two-level multiplicative/additive hybrid:
            # y_c1 = Z K_c^{-1} Z^T r
            c1 = _coarse_solve(z.T @ r_arr)
            y_c1 = z @ c1
            # r_perp = r - K y_c1 = r - (K Z) c1
            r_perp = r_arr - kz @ c1
            # fine correction on orthogonal complement
            y_f = np.asarray(fine_preconditioner.matvec(r_perp), dtype=np.float64)
            # project fine correction back: y_f - Z K_c^{-1} (Z^T K) y_f
            c2 = _coarse_solve(kz.T @ y_f)
            return np.asarray(y_c1 + y_f - z @ c2, dtype=np.float64)

        return LinearOperator((n, n), matvec=_matvec, dtype=np.float64)


@runtime_checkable
class LinearSolver3D(Protocol):
    """Protocol for pluggable 3D sparse linear solvers."""

    def solve(
        self,
        matrix: csr_matrix | csc_matrix,
        rhs: FloatArray,
        *,
        is_symmetric: bool = True,
        neural_preconditioner: NeuralCoarsePreconditioner | None = None,
    ) -> tuple[FloatArray, LinearSolveReport]: ...


def is_pyamg_available() -> bool:
    """Return True if optional ``pyamg`` smoothed aggregation AMG is importable."""
    try:
        importlib.import_module("pyamg")
    except ImportError:
        return False
    return True


def _build_block_jacobi_preconditioner(
    matrix: csr_matrix, block_size: int
) -> LinearOperator:
    """Build an exact ``block_size x block_size`` block-diagonal inverse (or diagonal fallback)."""
    n = int(matrix.shape[0])
    if block_size > 1 and n % block_size == 0:
        n_blocks = n // block_size
        inv_blocks = np.zeros((n_blocks, block_size, block_size), dtype=np.float64)
        eye_b = np.eye(block_size, dtype=np.float64)
        # Extract contiguous diagonal blocks directly from CSR
        for b in range(n_blocks):
            start = b * block_size
            end = start + block_size
            sub = matrix[start:end, start:end].toarray()
            sub = 0.5 * (sub + sub.T)
            try:
                inv_blocks[b] = np.linalg.inv(sub)
            except np.linalg.LinAlgError:
                diag_vals = np.diag(sub)
                safe = np.where(np.abs(diag_vals) > 1.0e-14, diag_vals, 1.0)
                inv_blocks[b] = np.diag(1.0 / safe)
                if not np.all(np.isfinite(inv_blocks[b])):
                    inv_blocks[b] = eye_b

        def _block_matvec(v: FloatArray) -> FloatArray:
            v_b = np.asarray(v, dtype=np.float64).reshape(n_blocks, block_size)
            out = np.einsum("bij,bj->bi", inv_blocks, v_b)
            return np.asarray(out.reshape(n), dtype=np.float64)

        return LinearOperator((n, n), matvec=_block_matvec, dtype=np.float64)

    diag = np.asarray(matrix.diagonal(), dtype=np.float64)
    inv_diag = np.where(np.abs(diag) > 1.0e-14, 1.0 / diag, 1.0)
    return LinearOperator(
        (n, n),
        matvec=lambda v: np.asarray(
            inv_diag * np.asarray(v, dtype=np.float64), dtype=np.float64
        ),
        dtype=np.float64,
    )


def _build_ssor_preconditioner(matrix: csr_matrix, omega: float) -> LinearOperator:
    """Build a strictly SPD Symmetric Gauss-Seidel (SSOR) preconditioner for PCG.

    Applies ``M^{-1} = omega*(2-omega) * (D + omega*U)^{-1} D (D + omega*L)^{-1}``
    using zero-fill compiled SuperLU triangular solves for C-level speed.
    """
    n = int(matrix.shape[0])
    diag = np.asarray(matrix.diagonal(), dtype=np.float64)
    safe_diag = np.where(np.abs(diag) > 1.0e-14, diag, 1.0)
    d_mat = diags(safe_diag, format="csc")
    lower = (d_mat + omega * tril(matrix, k=-1, format="csc")).tocsc()
    upper = (d_mat + omega * triu(matrix, k=1, format="csc")).tocsc()
    lu_lower = splu(lower, permc_spec="NATURAL", diag_pivot_thresh=0.0)
    lu_upper = splu(upper, permc_spec="NATURAL", diag_pivot_thresh=0.0)
    scale = omega * (2.0 - omega)

    def _ssor_matvec(v: FloatArray) -> FloatArray:
        v_arr = np.asarray(v, dtype=np.float64)
        y = np.asarray(lu_lower.solve(v_arr), dtype=np.float64)
        z = safe_diag * y
        x = np.asarray(lu_upper.solve(z), dtype=np.float64)
        return np.asarray(scale * x, dtype=np.float64)

    return LinearOperator((n, n), matvec=_ssor_matvec, dtype=np.float64)


def _build_ilu_preconditioner(
    matrix: csc_matrix, *, drop_tol: float, fill_factor: float
) -> LinearOperator:
    """Build a SuperLU incomplete LU preconditioner."""
    n = int(matrix.shape[0])
    ilu = spilu(matrix, drop_tol=drop_tol, fill_factor=fill_factor)
    return LinearOperator(
        (n, n),
        matvec=lambda v: np.asarray(
            ilu.solve(np.asarray(v, dtype=np.float64)), dtype=np.float64
        ),
        dtype=np.float64,
    )


def _build_amg_preconditioner(matrix: csr_matrix) -> LinearOperator:
    """Build an optional ``pyamg`` smoothed aggregation V-cycle preconditioner."""
    pyamg = importlib.import_module("pyamg")
    ml = pyamg.smoothed_aggregation_solver(matrix)
    op: LinearOperator = ml.aspreconditioner(cycle="V")
    return op


def _check_matrix_symmetry(matrix: csr_matrix, tol: float = 1.0e-8) -> bool:
    """Fast Frobenius-norm symmetry check for tangent matrices."""
    diff = matrix - matrix.T
    if diff.nnz == 0:
        return True
    norm_a = float(np.sqrt(np.sum(matrix.data * matrix.data)))
    norm_d = float(np.sqrt(np.sum(diff.data * diff.data)))
    return norm_d <= tol * max(norm_a, 1.0)


def _resolve_effective_method(
    config: LinearSolverConfig,
    *,
    n_dofs: int,
    is_symmetric: bool,
) -> LinearSolverMethod:
    """Resolve AUTO dispatch and enforce symmetry safety."""
    method = config.method
    if method is LinearSolverMethod.AUTO:
        if n_dofs <= config.direct_dof_threshold:
            return LinearSolverMethod.DIRECT
        if not is_symmetric:
            return LinearSolverMethod.GMRES_ILU
        if is_pyamg_available():
            return LinearSolverMethod.PCG_AMG
        return LinearSolverMethod.GMRES_ILU
    if not is_symmetric and method in (
        LinearSolverMethod.PCG_JACOBI,
        LinearSolverMethod.PCG_SSOR,
        LinearSolverMethod.PCG_AMG,
    ):
        raise ValueError(
            f"symmetric CG method {method.value!r} cannot be used on an "
            f"unsymmetric tangent matrix; use 'gmres-ilu', 'direct', or 'auto'"
        )
    return method


def solve_linear_system(
    matrix: csr_matrix | csc_matrix,
    rhs: FloatArray,
    *,
    config: LinearSolverConfig | None = None,
    is_symmetric: bool | None = None,
    neural_preconditioner: NeuralCoarsePreconditioner | None = None,
) -> tuple[FloatArray, LinearSolveReport]:
    """Solve ``matrix @ x = rhs`` with explicit state transitions and fallback control."""
    t0 = time.perf_counter()
    cfg = LinearSolverConfig() if config is None else config
    rhs_arr = np.asarray(rhs, dtype=np.float64).reshape(-1)
    n_dofs = int(rhs_arr.shape[0])
    state = LinearSolverState.INITIALIZED

    if matrix.shape != (n_dofs, n_dofs):
        state = LINEAR_SOLVER_MACHINE.require_transition(
            state, LinearSolverState.FAILED, trigger="reject_invalid"
        ).to_state
        raise ValueError(
            f"linear system shape mismatch: matrix={matrix.shape}, rhs=({n_dofs},)"
        )
    if not np.all(np.isfinite(rhs_arr)):
        state = LINEAR_SOLVER_MACHINE.require_transition(
            state, LinearSolverState.FAILED, trigger="reject_invalid"
        ).to_state
        return (
            np.full(n_dofs, np.nan, dtype=np.float64),
            LinearSolveReport(
                requested_method=cfg.method,
                method_used=cfg.method,
                state=state,
                converged=False,
                iterations=0,
                relative_residual=float("inf"),
                n_dofs=n_dofs,
                wall_time_s=time.perf_counter() - t0,
                used_fallback=False,
                used_neural_deflation=False,
                reason="non-finite right-hand side",
            ),
        )
    if n_dofs == 0:
        state = LINEAR_SOLVER_MACHINE.require_transition(
            state, LinearSolverState.CONVERGED, trigger="direct_converged"
        ).to_state
        return (
            np.zeros(0, dtype=np.float64),
            LinearSolveReport(
                requested_method=cfg.method,
                method_used=LinearSolverMethod.DIRECT,
                state=state,
                converged=True,
                iterations=0,
                relative_residual=0.0,
                n_dofs=0,
                wall_time_s=time.perf_counter() - t0,
                used_fallback=False,
                used_neural_deflation=False,
                reason="empty-system",
            ),
        )

    rhs_norm = float(np.linalg.norm(rhs_arr))
    if rhs_norm <= cfg.atol:
        state = LINEAR_SOLVER_MACHINE.require_transition(
            state, LinearSolverState.CONVERGED, trigger="direct_converged"
        ).to_state
        return (
            np.zeros(n_dofs, dtype=np.float64),
            LinearSolveReport(
                requested_method=cfg.method,
                method_used=LinearSolverMethod.DIRECT,
                state=state,
                converged=True,
                iterations=0,
                relative_residual=0.0,
                n_dofs=n_dofs,
                wall_time_s=time.perf_counter() - t0,
                used_fallback=False,
                used_neural_deflation=False,
                reason="zero-rhs",
            ),
        )

    mat_csr = matrix.tocsr()
    sym = (
        _check_matrix_symmetry(mat_csr) if is_symmetric is None else bool(is_symmetric)
    )
    effective = _resolve_effective_method(cfg, n_dofs=n_dofs, is_symmetric=sym)

    def _run_direct(
        curr_state: LinearSolverState, *, used_fb: bool
    ) -> tuple[FloatArray, LinearSolveReport]:
        sol = np.asarray(spsolve(mat_csr.tocsc(), rhs_arr), dtype=np.float64)
        finite = bool(np.all(np.isfinite(sol)))
        rel_res = (
            float(np.linalg.norm(mat_csr @ sol - rhs_arr)) / rhs_norm
            if finite
            else float("inf")
        )
        if finite:
            trig = "fallback_converged" if used_fb else "direct_converged"
            end_s = LINEAR_SOLVER_MACHINE.require_transition(
                curr_state, LinearSolverState.CONVERGED, trigger=trig
            ).to_state
            return (
                sol,
                LinearSolveReport(
                    requested_method=cfg.method,
                    method_used=LinearSolverMethod.DIRECT,
                    state=end_s,
                    converged=True,
                    iterations=1,
                    relative_residual=rel_res,
                    n_dofs=n_dofs,
                    wall_time_s=time.perf_counter() - t0,
                    used_fallback=used_fb,
                    used_neural_deflation=False,
                    reason="direct-fallback-converged"
                    if used_fb
                    else "direct-converged",
                ),
            )
        trig = "fallback_failed" if used_fb else "precond_failed"
        end_s = LINEAR_SOLVER_MACHINE.require_transition(
            curr_state, LinearSolverState.FAILED, trigger=trig
        ).to_state
        return (
            sol,
            LinearSolveReport(
                requested_method=cfg.method,
                method_used=LinearSolverMethod.DIRECT,
                state=end_s,
                converged=False,
                iterations=1,
                relative_residual=float("inf"),
                n_dofs=n_dofs,
                wall_time_s=time.perf_counter() - t0,
                used_fallback=used_fb,
                used_neural_deflation=False,
                reason="direct solve returned non-finite values",
            ),
        )

    state = LINEAR_SOLVER_MACHINE.require_transition(
        state, LinearSolverState.FACTORIZING_PRECOND, trigger="build_preconditioner"
    ).to_state
    if effective is LinearSolverMethod.DIRECT:
        return _run_direct(state, used_fb=False)

    # Build preconditioner
    use_deflation = (
        neural_preconditioner is not None and cfg.use_neural_coarse_deflation
    )
    precond: LinearOperator | None = None
    precond_error: str | None = None
    try:
        if effective is LinearSolverMethod.PCG_JACOBI:
            precond = _build_block_jacobi_preconditioner(mat_csr, cfg.block_size)
        elif effective is LinearSolverMethod.PCG_SSOR:
            precond = _build_ssor_preconditioner(mat_csr, cfg.ssor_omega)
        elif effective in (LinearSolverMethod.PCG_ILU, LinearSolverMethod.GMRES_ILU):
            precond = _build_ilu_preconditioner(
                mat_csr.tocsc(),
                drop_tol=cfg.ilu_drop_tol,
                fill_factor=cfg.ilu_fill_factor,
            )
        elif effective is LinearSolverMethod.PCG_AMG:
            if not is_pyamg_available():
                raise ImportError(
                    "pyamg is not installed; install formamech[scale] for pcg-amg"
                )
            precond = _build_amg_preconditioner(mat_csr)
    except Exception as exc:  # noqa: BLE001
        precond_error = str(exc)

    if precond is None:
        if cfg.allow_fallback_to_direct:
            state = LINEAR_SOLVER_MACHINE.require_transition(
                state,
                LinearSolverState.STAGNATED_FALLBACK,
                trigger="precond_fallback",
            ).to_state
            return _run_direct(state, used_fb=True)
        state = LINEAR_SOLVER_MACHINE.require_transition(
            state, LinearSolverState.FAILED, trigger="precond_failed"
        ).to_state
        return (
            np.full(n_dofs, np.nan, dtype=np.float64),
            LinearSolveReport(
                requested_method=cfg.method,
                method_used=effective,
                state=state,
                converged=False,
                iterations=0,
                relative_residual=float("inf"),
                n_dofs=n_dofs,
                wall_time_s=time.perf_counter() - t0,
                used_fallback=False,
                used_neural_deflation=False,
                reason=f"preconditioner failed: {precond_error}",
            ),
        )

    if use_deflation and neural_preconditioner is not None:
        precond = neural_preconditioner.wrap(mat_csr, precond)

    state = LINEAR_SOLVER_MACHINE.require_transition(
        state, LinearSolverState.ITERATING, trigger="start_krylov"
    ).to_state

    iters = 0

    def _count_iter(_: Any) -> None:
        nonlocal iters
        iters += 1

    # PCG_ILU uses GMRES internally because ILU(K) is non-symmetric even when K is SPD,
    # while PCG_JACOBI / PCG_SSOR / PCG_AMG preserve exact symmetry for CG.
    if effective in (
        LinearSolverMethod.PCG_JACOBI,
        LinearSolverMethod.PCG_SSOR,
        LinearSolverMethod.PCG_AMG,
    ):
        sol_raw, info = cg(
            mat_csr,
            rhs_arr,
            rtol=cfg.rtol,
            atol=cfg.atol,
            maxiter=cfg.max_iterations,
            M=precond,
            callback=_count_iter,
        )
    else:
        sol_raw, info = gmres(
            mat_csr,
            rhs_arr,
            rtol=cfg.rtol,
            atol=cfg.atol,
            restart=min(cfg.gmres_restart, max(n_dofs, 1)),
            maxiter=cfg.max_iterations,
            M=precond,
            callback=_count_iter,
            callback_type="pr_norm",
        )

    sol = np.asarray(sol_raw, dtype=np.float64)
    finite = bool(np.all(np.isfinite(sol)))
    rel_res = (
        float(np.linalg.norm(mat_csr @ sol - rhs_arr)) / rhs_norm
        if finite
        else float("inf")
    )
    ok = (
        finite
        and int(info) == 0
        and (rel_res <= cfg.rtol * 10.0 or rel_res * rhs_norm <= cfg.atol * 10.0)
    )

    if ok:
        state = LINEAR_SOLVER_MACHINE.require_transition(
            state, LinearSolverState.CONVERGED, trigger="krylov_converged"
        ).to_state
        return (
            sol,
            LinearSolveReport(
                requested_method=cfg.method,
                method_used=effective,
                state=state,
                converged=True,
                iterations=max(iters, 1),
                relative_residual=rel_res,
                n_dofs=n_dofs,
                wall_time_s=time.perf_counter() - t0,
                used_fallback=False,
                used_neural_deflation=use_deflation,
                reason="krylov-converged",
            ),
        )

    if cfg.allow_fallback_to_direct:
        state = LINEAR_SOLVER_MACHINE.require_transition(
            state, LinearSolverState.STAGNATED_FALLBACK, trigger="krylov_fallback"
        ).to_state
        return _run_direct(state, used_fb=True)

    state = LINEAR_SOLVER_MACHINE.require_transition(
        state, LinearSolverState.FAILED, trigger="krylov_failed"
    ).to_state
    return (
        sol,
        LinearSolveReport(
            requested_method=cfg.method,
            method_used=effective,
            state=state,
            converged=False,
            iterations=iters,
            relative_residual=rel_res,
            n_dofs=n_dofs,
            wall_time_s=time.perf_counter() - t0,
            used_fallback=False,
            used_neural_deflation=use_deflation,
            reason=f"krylov did not converge (info={info}, rel_res={rel_res:.3e})",
        ),
    )


class SparseLinearSolver3D:
    """Concrete :class:`LinearSolver3D` wrapping :func:`solve_linear_system`."""

    def __init__(self, config: LinearSolverConfig | None = None) -> None:
        self.config = LinearSolverConfig() if config is None else config

    def solve(
        self,
        matrix: csr_matrix | csc_matrix,
        rhs: FloatArray,
        *,
        is_symmetric: bool = True,
        neural_preconditioner: NeuralCoarsePreconditioner | None = None,
    ) -> tuple[FloatArray, LinearSolveReport]:
        return solve_linear_system(
            matrix,
            rhs,
            config=self.config,
            is_symmetric=is_symmetric,
            neural_preconditioner=neural_preconditioner,
        )


__all__ = (
    "LINEAR_SOLVER_CONFIG_SCHEMA",
    "LINEAR_SOLVER_MACHINE",
    "LINEAR_SOLVE_REPORT_SCHEMA",
    "LinearSolveReport",
    "LinearSolver3D",
    "LinearSolverConfig",
    "LinearSolverMethod",
    "LinearSolverState",
    "NeuralCoarsePreconditioner",
    "SparseLinearSolver3D",
    "is_pyamg_available",
    "solve_linear_system",
)
