"""The backend's discrete problem, linearized at an arbitrary displacement (S21).

``FemLinearization`` rebuilds exactly what ``ScikitFemBackend`` solves for a
request — basis, essential constraints, dead loads, kernel material — and
exposes it at any displacement: residual, tangent, potential energy, one
Newton increment, and the recovered first Piola stress. The S21 certificate
uses it to measure how far a surrogate's displacement is from the converged
discrete solution, without a second FEM implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from piginot.fem.assembly import AssemblyState, assemble_state, recover_stress
from piginot.fem.backend import (
    ScikitFemBackend,
    _build_basis,
    _evaluate_qois,
    _kernel_material,
    _solve_increment,
)
from piginot.fem.binding import BoundConstraints, assemble_dead_loads, bind_constraints
from piginot.fem.models import FemRequest

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LinearizedState:
    """Residual, tangent and energy of one displacement at full load."""

    displacement: FloatArray
    assembly: AssemblyState
    residual: FloatArray
    potential_energy: float


class FemLinearization:
    """The discrete problem of one request, at full load (load factor 1)."""

    def __init__(
        self, request: FemRequest, backend: ScikitFemBackend | None = None
    ) -> None:
        backend = backend if backend is not None else ScikitFemBackend()
        self.request = request
        self.skmesh, self.basis, self.scalar_basis = _build_basis(request)
        self.constraints: BoundConstraints = bind_constraints(
            self.basis,
            self.skmesh,
            request.mesh,
            request.boundary_conditions,
            provider_ids=request.boundary_field_provider_ids,
            providers=backend._dirichlet_field_providers,
            rigid_mode_constraint=request.rigid_mode_constraint,
        )
        self.material = _kernel_material(request.material)
        self.external_force: FloatArray = assemble_dead_loads(
            self.basis,
            self.skmesh,
            request.mesh,
            request.loads,
            thickness=1.0,
            provider_ids=request.body_force_provider_ids,
            providers=backend._body_force_providers,
            traction_provider_ids=request.traction_field_provider_ids,
            traction_providers=backend._traction_field_providers,
        )
        x_dofs, y_dofs = (np.asarray(v) for v in self.basis.split_indices())
        self._x_dofs, self._y_dofs = x_dofs, y_dofs

    # -- layout -------------------------------------------------------------

    @property
    def dof_coordinates(self) -> FloatArray:
        """Reference coordinates of the displacement nodes (P1 or P2)."""

        return np.asarray(self.scalar_basis.doflocs, dtype=np.float64).T

    @property
    def has_global_constraints(self) -> bool:
        return bool(self.constraints.global_matrix.shape[0])

    def vector(self, nodal: FloatArray) -> FloatArray:
        """Full DOF vector from nodal ``(n, 2)`` displacements."""

        values = np.asarray(nodal, dtype=np.float64)
        if values.shape != (self._x_dofs.size, 2):
            raise ValueError(
                f"nodal displacement must have shape ({self._x_dofs.size}, 2)"
            )
        result = np.zeros(self.basis.N, dtype=np.float64)
        result[self._x_dofs] = values[:, 0]
        result[self._y_dofs] = values[:, 1]
        return result

    def nodal(self, vector: FloatArray) -> FloatArray:
        return np.column_stack((vector[self._x_dofs], vector[self._y_dofs]))

    def dirichlet_deviation(self, vector: FloatArray) -> float:
        """Largest deviation from the prescribed values on constrained DOFs."""

        constrained = self.constraints.constrained
        if not constrained.size:
            return 0.0
        return float(
            np.max(
                np.abs(
                    vector[constrained] - self.constraints.target_values[constrained]
                )
            )
        )

    def with_exact_dirichlet(self, vector: FloatArray) -> FloatArray:
        result = np.array(vector, dtype=np.float64, copy=True)
        constrained = self.constraints.constrained
        result[constrained] = self.constraints.target_values[constrained]
        return result

    # -- physics ------------------------------------------------------------

    def state(self, vector: FloatArray) -> LinearizedState:
        """Assemble at ``vector`` (raises ``FemSolveFailure`` if inadmissible)."""

        assembly = assemble_state(
            self.basis,
            vector,
            self.material,
            closure_iterations=self.request.config.plane_stress_max_iterations,
            thickness=1.0,
        )
        residual = assembly.internal_force - self.external_force
        potential = assembly.strain_energy - float(self.external_force @ vector)
        return LinearizedState(vector, assembly, residual, potential)

    def newton_increment(self, state: LinearizedState) -> FloatArray:
        """Full-length increment of one Newton step at full load."""

        free_increment = _solve_increment(
            state.assembly, state.residual, state.displacement, self.constraints, 1.0
        )
        increment = np.zeros(self.basis.N, dtype=np.float64)
        increment[self.constraints.free] = free_increment
        return increment

    def free_tangent(self, state: LinearizedState) -> Any:
        free = self.constraints.free
        return state.assembly.tangent[free][:, free]

    def energy_norm(self, difference: FloatArray, state: LinearizedState) -> float:
        """``sqrt(d^T K d)`` on the free DOFs, ``K`` the tangent of ``state``."""

        free = self.constraints.free
        part = difference[free]
        value = float(part @ (self.free_tangent(state) @ part))
        return float(np.sqrt(max(value, 0.0)))

    def recovered_stress(self, state: LinearizedState) -> FloatArray:
        """Nodal first Piola stress (L2 projection), shape ``(n, 3, 3)``."""

        return np.asarray(
            recover_stress(self.scalar_basis, state.assembly.first_piola),
            dtype=np.float64,
        )

    def qois(self, state: LinearizedState) -> dict[str, float]:
        """The request's QoIs, evaluated as the backend evaluates them."""

        return _evaluate_qois(
            self.request, self.recovered_stress(state), self.scalar_basis, self.skmesh
        )

    def quadrature_weights(self) -> FloatArray:
        return np.asarray(self.basis.dx, dtype=np.float64)


__all__ = ("FemLinearization", "LinearizedState")
