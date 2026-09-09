"""Immutable contracts for nondimensional S12 objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

import torch

from piginot.doe import EvidenceRole


class ObjectiveMode(StrEnum):
    """Registered combinations of supervised and physics objectives."""

    DATA_ONLY = "DATA_ONLY"
    PHYSICS_ONLY = "PHYSICS_ONLY"
    HYBRID = "HYBRID"


class EquilibriumForm(StrEnum):
    """Available equilibrium discretizations."""

    STRONG = "strong"
    WEAK = "weak"


class ObjectiveTerm(StrEnum):
    """Stable first-class loss-term identifiers."""

    DATA_U = "data_u"
    DATA_P = "data_P"
    EQUILIBRIUM_STRONG = "equilibrium_strong"
    EQUILIBRIUM_WEAK = "equilibrium_weak"
    CONSTITUTIVE = "constitutive"
    ANGULAR = "angular"
    DIRICHLET = "dirichlet"
    TRACTION_FREE = "traction_free"


@dataclass(frozen=True, slots=True)
class NondimensionalScales:
    """Fixed physical reference scales used by every objective term."""

    length_mm: float
    displacement_mm: float
    stress_mpa: float

    def __post_init__(self) -> None:
        for name in ("length_mm", "displacement_mm", "stress_mpa"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and strictly positive")
            object.__setattr__(self, name, float(value))

    @property
    def body_force_mpa_per_mm(self) -> float:
        return self.stress_mpa / self.length_mm

    @property
    def weak_force_per_thickness_n_per_mm(self) -> float:
        return self.stress_mpa * self.length_mm


@dataclass(frozen=True, slots=True)
class CoordinateConversion:
    """Per-axis conversion from supplied coordinate units to canonical mm."""

    to_mm: tuple[float, float]

    def __post_init__(self) -> None:
        if type(self.to_mm) is not tuple or len(self.to_mm) != 2:
            raise ValueError("to_mm must contain the x and y conversion factors")
        converted = tuple(float(value) for value in self.to_mm)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in self.to_mm
        ):
            raise ValueError("coordinate conversion factors must be positive and finite")
        object.__setattr__(self, "to_mm", converted)


@dataclass(frozen=True, slots=True)
class ObjectiveResiduals:
    """Named nondimensional residual tensors awaiting scalar reduction."""

    values: Mapping[ObjectiveTerm, torch.Tensor]

    def __post_init__(self) -> None:
        if not isinstance(self.values, Mapping):
            raise TypeError("values must be a mapping")
        copied: dict[ObjectiveTerm, torch.Tensor] = {}
        for term, value in self.values.items():
            if type(term) is not ObjectiveTerm:
                raise TypeError("objective residual keys must be ObjectiveTerm values")
            if not isinstance(value, torch.Tensor):
                raise TypeError("each objective residual must be a torch.Tensor")
            if value.layout is not torch.strided or value.dtype not in (
                torch.float32,
                torch.float64,
            ):
                raise ValueError("objective residuals must be dense floating tensors")
            if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
                raise ValueError("objective residuals must be finite and non-empty")
            copied[term] = value
        object.__setattr__(self, "values", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class ObjectiveEvaluation:
    """Scalar per-term losses plus their weighted total."""

    mode: ObjectiveMode
    equilibrium_form: EquilibriumForm
    losses: Mapping[ObjectiveTerm, torch.Tensor]
    total: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "losses", MappingProxyType(dict(self.losses)))


@dataclass(frozen=True, slots=True)
class ObjectiveRegistry:
    """Fail-closed objective selection and reduction policy."""

    mode: ObjectiveMode = ObjectiveMode.HYBRID
    equilibrium_form: EquilibriumForm = EquilibriumForm.WEAK
    term_weights: Mapping[ObjectiveTerm, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.mode) is not ObjectiveMode:
            raise TypeError("mode must be an ObjectiveMode")
        if type(self.equilibrium_form) is not EquilibriumForm:
            raise TypeError("equilibrium_form must be an EquilibriumForm")
        if not isinstance(self.term_weights, Mapping):
            raise TypeError("term_weights must be a mapping")
        weights = {term: 1.0 for term in ObjectiveTerm}
        for term, weight in self.term_weights.items():
            if type(term) is not ObjectiveTerm:
                raise TypeError("term weight keys must be ObjectiveTerm values")
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) <= 0.0
            ):
                raise ValueError("term weights must be finite and strictly positive")
            weights[term] = float(weight)
        object.__setattr__(self, "term_weights", MappingProxyType(weights))

    @property
    def reported_equilibrium_term(self) -> ObjectiveTerm:
        if self.equilibrium_form is EquilibriumForm.WEAK:
            return ObjectiveTerm.EQUILIBRIUM_WEAK
        return ObjectiveTerm.EQUILIBRIUM_STRONG

    @property
    def required_terms(self) -> tuple[ObjectiveTerm, ...]:
        data = (ObjectiveTerm.DATA_U, ObjectiveTerm.DATA_P)
        physics = (
            self.reported_equilibrium_term,
            ObjectiveTerm.CONSTITUTIVE,
            ObjectiveTerm.ANGULAR,
            ObjectiveTerm.DIRICHLET,
            ObjectiveTerm.TRACTION_FREE,
        )
        if self.mode is ObjectiveMode.DATA_ONLY:
            return data
        if self.mode is ObjectiveMode.PHYSICS_ONLY:
            return physics
        return (*data, *physics)

    def evaluate(
        self,
        residuals: ObjectiveResiduals,
        *,
        role: EvidenceRole,
    ) -> ObjectiveEvaluation:
        """Reduce residuals through the role-authorized registry boundary."""

        from piginot.objectives.registry import evaluate_objective

        return evaluate_objective(self, residuals, role=role)


__all__ = (
    "CoordinateConversion",
    "EquilibriumForm",
    "NondimensionalScales",
    "ObjectiveEvaluation",
    "ObjectiveMode",
    "ObjectiveRegistry",
    "ObjectiveResiduals",
    "ObjectiveTerm",
)
