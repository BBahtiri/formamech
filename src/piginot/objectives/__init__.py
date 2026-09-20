"""Nondimensional objective registry and verification primitives for S12."""

from __future__ import annotations

from typing import Any

from piginot.objectives.models import (
    CoordinateConversion,
    EquilibriumForm,
    NondimensionalScales,
    ObjectiveEvaluation,
    ObjectiveMode,
    ObjectiveRegistry,
    ObjectiveResiduals,
    ObjectiveTerm,
)

_LAZY_EXPORTS = {
    "GateStatus": ("piginot.objectives.gate", "GateStatus"),
    "S12GateCheck": ("piginot.objectives.gate", "S12GateCheck"),
    "S12GateEvaluation": ("piginot.objectives.gate", "S12GateEvaluation"),
    "S12GateMeasurements": ("piginot.objectives.gate", "S12GateMeasurements"),
    "evaluate_s12_gate": ("piginot.objectives.gate", "evaluate_s12_gate"),
    "PHYSICS_TERM_IDS": ("piginot.objectives.verification", "PHYSICS_TERM_IDS"),
    "S12VerificationReport": (
        "piginot.objectives.verification",
        "S12VerificationReport",
    ),
    "VERIFIED_GRADIENT_TERM_IDS": (
        "piginot.objectives.verification",
        "VERIFIED_GRADIENT_TERM_IDS",
    ),
    "VERIFIED_REACHABILITY_LOSS_PATHS": (
        "piginot.objectives.verification",
        "VERIFIED_REACHABILITY_LOSS_PATHS",
    ),
    "run_s12_verification": (
        "piginot.objectives.verification",
        "run_s12_verification",
    ),
    "angular_momentum_residual": (
        "piginot.objectives.terms",
        "angular_momentum_residual",
    ),
    "constitutive_residual": ("piginot.objectives.terms", "constitutive_residual"),
    "default_objective_registry": (
        "piginot.objectives.registry",
        "default_objective_registry",
    ),
    "dirichlet_residual": ("piginot.objectives.terms", "dirichlet_residual"),
    "displacement_gradient": (
        "piginot.objectives.kinematics",
        "displacement_gradient",
    ),
    "evaluate_objective": ("piginot.objectives.registry", "evaluate_objective"),
    "strong_equilibrium_residual": (
        "piginot.objectives.equilibrium",
        "strong_equilibrium_residual",
    ),
    "supervised_displacement_residual": (
        "piginot.objectives.terms",
        "supervised_displacement_residual",
    ),
    "supervised_stress_residual": (
        "piginot.objectives.terms",
        "supervised_stress_residual",
    ),
    "traction_free_residual": (
        "piginot.objectives.terms",
        "traction_free_residual",
    ),
    "weak_equilibrium_residual": (
        "piginot.objectives.equilibrium",
        "weak_equilibrium_residual",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazily load FEM-heavy objective helpers only when they are requested."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = (
    "CoordinateConversion",
    "EquilibriumForm",
    "GateStatus",
    "NondimensionalScales",
    "ObjectiveEvaluation",
    "ObjectiveMode",
    "ObjectiveRegistry",
    "ObjectiveResiduals",
    "ObjectiveTerm",
    "PHYSICS_TERM_IDS",
    "S12GateCheck",
    "S12GateEvaluation",
    "S12GateMeasurements",
    "S12VerificationReport",
    "VERIFIED_GRADIENT_TERM_IDS",
    "VERIFIED_REACHABILITY_LOSS_PATHS",
    "angular_momentum_residual",
    "constitutive_residual",
    "default_objective_registry",
    "dirichlet_residual",
    "displacement_gradient",
    "evaluate_objective",
    "evaluate_s12_gate",
    "run_s12_verification",
    "strong_equilibrium_residual",
    "supervised_displacement_residual",
    "supervised_stress_residual",
    "traction_free_residual",
    "weak_equilibrium_residual",
)
