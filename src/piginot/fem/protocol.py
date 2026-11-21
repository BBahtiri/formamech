"""Pluggable backend protocols for interchangeable FEM implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from piginot.fem.models import FemAttempt, FemRequest

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Declared numerical and licensing capabilities of a :class:`SolverBackend`."""

    spatial_dimensions: tuple[int, ...] = (2,)
    physics_modes: tuple[str, ...] = ("2d-plane-stress-finite-strain",)
    element_orders: tuple[int, ...] = (1, 2)
    supports_warm_start: bool = True
    supports_linearization: bool = True
    supports_dead_tractions: bool = True
    supports_follower_pressure: bool = False
    supports_contact: bool = False
    supports_gpu: bool = False
    supports_mpi: bool = False
    license_spdx: str = "MIT"

    def document(self) -> dict[str, Any]:
        return {
            "spatial_dimensions": list(self.spatial_dimensions),
            "physics_modes": list(self.physics_modes),
            "element_orders": list(self.element_orders),
            "supports_warm_start": self.supports_warm_start,
            "supports_linearization": self.supports_linearization,
            "supports_dead_tractions": self.supports_dead_tractions,
            "supports_follower_pressure": self.supports_follower_pressure,
            "supports_contact": self.supports_contact,
            "supports_gpu": self.supports_gpu,
            "supports_mpi": self.supports_mpi,
            "license_spdx": self.license_spdx,
        }


@runtime_checkable
class DiscreteLinearization(Protocol):
    """Backend-agnostic linearized discrete problem at an arbitrary displacement."""

    request: FemRequest
    constraints: Any

    @property
    def dof_coordinates(self) -> FloatArray: ...

    @property
    def dof_count(self) -> int: ...

    @property
    def has_global_constraints(self) -> bool: ...

    @property
    def is_tangent_symmetric(self) -> bool: ...

    def vector(self, nodal: FloatArray) -> FloatArray: ...

    def nodal(self, vector: FloatArray) -> FloatArray: ...

    def dirichlet_deviation(self, vector: FloatArray) -> float: ...

    def with_exact_dirichlet(self, vector: FloatArray) -> FloatArray: ...

    def state(self, vector: FloatArray) -> Any: ...

    def newton_increment(self, state: Any) -> FloatArray: ...

    def free_tangent(self, state: Any) -> Any: ...

    def solve_adjoint(self, state: Any, rhs_free: FloatArray) -> FloatArray: ...

    def energy_norm(self, difference: FloatArray, state: Any) -> float: ...

    def recovered_stress(self, state: Any) -> FloatArray: ...

    def qois(self, state: Any) -> Mapping[str, float]: ...

    def quadrature_weights(self) -> FloatArray: ...


@runtime_checkable
class FemBackend(Protocol):
    backend_name: str

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        """Return one visible outcome for every request."""


@runtime_checkable
class SolverBackend(FemBackend, Protocol):
    """Pluggable solver contract decoupling serving, DWR, and compiler from skfem."""

    backend_name: str
    backend_id: str
    capabilities: BackendCapabilities

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        """Return one visible outcome for every request."""

    def solve_from(
        self,
        request: FemRequest,
        initial: FloatArray,
        load_factors: Sequence[float] = (1.0,),
    ) -> FemAttempt:
        """Warm-start nonlinear solve from ``initial`` across ``load_factors``."""

    def linearize(self, request: FemRequest) -> DiscreteLinearization:
        """Build a reusable discrete linearization for certification and DWR."""


@runtime_checkable
class IndependentSolverAdapter(Protocol):
    """Optional boundary for verification with a separately implemented solver.

    An adapter translates canonical :class:`FemRequest` values to an independent
    solver and translates every outcome back to :class:`FemAttempt`, preserving
    input order and failed attempts.  ``adapter_id`` must identify the adapter and
    independent solver version used for provenance.  Core intentionally provides
    this protocol only; an external package owns any concrete implementation.
    """

    adapter_id: str

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        """Return one independently computed outcome for every request."""


__all__ = (
    "BackendCapabilities",
    "DiscreteLinearization",
    "FemBackend",
    "IndependentSolverAdapter",
    "SolverBackend",
)
