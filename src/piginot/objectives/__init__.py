"""Nondimensional objective registry and verification primitives for S12."""

from __future__ import annotations

from typing import Any

from piginot.objectives.energy import (
    POTENTIAL_ENERGY_POLICY_ID,
    PotentialEnergyConfig,
    PotentialEnergyTerms,
    TriangleDomain,
    plane_stress_potential_energy,
    stratified_triangle_quadrature,
)
from piginot.objectives.equilibrium import (
    strong_equilibrium_residual,
    weak_equilibrium_residual,
)
from piginot.objectives.hierarchical_energy import (
    HIERARCHICAL_ENERGY_POLICY_ID,
    SUPPORTED_QUADRATURE_ORDERS,
    HierarchicalEnergyConfig,
    HierarchicalEnergyTerms,
    QuadratureConvergenceReport,
    check_quadrature_energy_convergence,
    deterministic_triangle_quadrature,
    hierarchical_plane_stress_energy,
    subdivide_triangle_domain,
)
from piginot.objectives.kinematics import displacement_gradient
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
from piginot.objectives.registry import (
    default_objective_registry,
    evaluate_objective,
)
from piginot.objectives.terms import (
    angular_momentum_residual,
    constitutive_residual,
    dirichlet_residual,
    supervised_displacement_residual,
    supervised_stress_residual,
    traction_free_residual,
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
}


def __getattr__(name: str) -> Any:
    """Lazily load verification exports that require optional FEM packages."""

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
    "HIERARCHICAL_ENERGY_POLICY_ID",
    "HierarchicalEnergyConfig",
    "HierarchicalEnergyTerms",
    "NondimensionalScales",
    "ObjectiveEvaluation",
    "ObjectiveMode",
    "ObjectiveRegistry",
    "ObjectiveResiduals",
    "ObjectiveTerm",
    "PHYSICS_TERM_IDS",
    "POTENTIAL_ENERGY_POLICY_ID",
    "PotentialEnergyConfig",
    "PotentialEnergyTerms",
    "QuadratureConvergenceReport",
    "S12GateCheck",
    "S12GateEvaluation",
    "S12GateMeasurements",
    "S12VerificationReport",
    "SUPPORTED_QUADRATURE_ORDERS",
    "TriangleDomain",
    "VERIFIED_GRADIENT_TERM_IDS",
    "VERIFIED_REACHABILITY_LOSS_PATHS",
    "angular_momentum_residual",
    "check_quadrature_energy_convergence",
    "constitutive_residual",
    "default_objective_registry",
    "deterministic_triangle_quadrature",
    "dirichlet_residual",
    "displacement_gradient",
    "evaluate_objective",
    "evaluate_s12_gate",
    "hierarchical_plane_stress_energy",
    "plane_stress_potential_energy",
    "run_s12_verification",
    "stratified_triangle_quadrature",
    "strong_equilibrium_residual",
    "subdivide_triangle_domain",
    "supervised_displacement_residual",
    "supervised_stress_residual",
    "traction_free_residual",
    "weak_equilibrium_residual",
)
