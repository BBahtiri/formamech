"""Typed S7 finite-element solver contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    ExecutionProfile,
    LoadSpec,
    MaterialSpec,
    QoISpec,
    hash_spec,
)
from piginot.contracts.specs import FrozenMap, SpecContract


class FemFailureCode(StrEnum):
    """Stable reason codes recorded by S7 attempts."""

    NEGATIVE_JACOBIAN = "NEGATIVE_JACOBIAN"
    CLOSURE_FAILURE = "CLOSURE_FAILURE"
    CONSTRAINT_RANK_LOSS = "CONSTRAINT_RANK_LOSS"
    NEWTON_NON_CONVERGENCE = "NEWTON_NON_CONVERGENCE"
    RESOURCE_EXHAUSTION = "RESOURCE_EXHAUSTION"
    QOI_REGULARITY_REJECTED = "QOI_REGULARITY_REJECTED"
    QOI_RESOLUTION_REJECTED = "QOI_RESOLUTION_REJECTED"
    UNSUPPORTED_REQUEST = "UNSUPPORTED_REQUEST"


@dataclass(frozen=True, slots=True)
class FemSolverConfig(SpecContract):
    """Newton/load-continuation settings that close over recipe identity."""

    element_order: int = 1
    solver_tolerance: float = 1e-10
    max_newton_iterations: int = 16
    plane_stress_max_iterations: int = 32
    load_steps: int = 1
    max_dofs: int = 100_000

    def __post_init__(self) -> None:
        if self.element_order not in (1, 2):
            raise ValueError("element_order must be 1 or 2")
        if self.solver_tolerance <= 0.0:
            raise ValueError("solver_tolerance must be positive")
        if self.max_newton_iterations < 0:
            raise ValueError("max_newton_iterations must be non-negative")
        if self.plane_stress_max_iterations < 0:
            raise ValueError("plane_stress_max_iterations must be non-negative")
        if self.load_steps <= 0:
            raise ValueError("load_steps must be positive")
        if self.max_dofs <= 0:
            raise ValueError("max_dofs must be positive")

    def with_resource_limit(self, *, max_dofs: int) -> FemSolverConfig:
        return replace(self, max_dofs=max_dofs)


@dataclass(frozen=True, slots=True)
class FemRequest(SpecContract):
    """Content-addressed S7 solve request."""

    request_id: str
    mesh: CanonicalMesh
    material: MaterialSpec
    boundary_conditions: tuple[BoundaryConditionSpec, ...]
    loads: tuple[LoadSpec, ...]
    qois: tuple[QoISpec, ...]
    config: FemSolverConfig
    execution_profile: ExecutionProfile
    body_force_provider_ids: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        providers: FrozenMap[str] = FrozenMap.copy_of(self.body_force_provider_ids)
        if any(not isinstance(value, str) or not value for value in providers.values()):
            raise ValueError("body-force provider IDs must be non-empty strings")
        object.__setattr__(self, "body_force_provider_ids", providers)

    @property
    def recipe_hash(self) -> str:
        return hash_spec(self, "recipe")

    def with_config(self, config: FemSolverConfig) -> FemRequest:
        return replace(self, config=config)

    def with_loads(self, loads: tuple[LoadSpec, ...]) -> FemRequest:
        return replace(self, loads=loads)

    def with_qois(self, qois: tuple[QoISpec, ...]) -> FemRequest:
        return replace(self, qois=qois)


@dataclass(frozen=True, slots=True)
class FemConvergenceEntry(SpecContract):
    load_step: int
    iteration: int
    residual_norm: float
    increment_norm: float


@dataclass(frozen=True, slots=True)
class FemResult(SpecContract):
    """Successful S7 result payload."""

    request_id: str
    mesh_id: str
    element_order: int
    displacement_dof_coordinates: tuple[tuple[float, float], ...]
    displacement: tuple[tuple[float, float], ...]
    quadrature_points: tuple[tuple[tuple[float, float], ...], ...]
    quadrature_weights: tuple[tuple[float, ...], ...]
    first_piola_stress: tuple[
        tuple[
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
                tuple[float, float, float],
            ],
            ...,
        ],
        ...,
    ]
    deformation_gradient: tuple[
        tuple[
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
                tuple[float, float, float],
            ],
            ...,
        ],
        ...,
    ]
    recovered_first_piola_stress: tuple[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
        ...,
    ]
    reactions: Mapping[str, tuple[float, float]]
    total_strain_energy: float
    external_work: float
    total_potential_energy: float
    convergence_history: tuple[FemConvergenceEntry, ...]
    qois: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reactions", FrozenMap.copy_of(self.reactions))
        object.__setattr__(self, "qois", FrozenMap.copy_of(self.qois))


@dataclass(frozen=True, slots=True)
class FemAttempt(SpecContract):
    """One visible attempt, successful or failed."""

    request: FemRequest
    recipe_hash: str
    success: bool
    result: FemResult | None
    result_hash: str | None
    store_content_hash: str | None
    failure_code: FemFailureCode | None
    message: str


class FemSolveFailure(RuntimeError):
    """Internal typed failure converted to a visible attempt."""

    def __init__(self, code: FemFailureCode, message: str) -> None:
        self.code = code
        super().__init__(message)


__all__ = (
    "FemAttempt",
    "FemConvergenceEntry",
    "FemFailureCode",
    "FemRequest",
    "FemResult",
    "FemSolveFailure",
    "FemSolverConfig",
)
