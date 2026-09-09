"""Nondimensional objective registry and verification primitives for S12."""

from piginot.objectives.equilibrium import (
    strong_equilibrium_residual,
    weak_equilibrium_residual,
)
from piginot.objectives.gate import (
    GateStatus,
    S12GateCheck,
    S12GateEvaluation,
    S12GateMeasurements,
    evaluate_s12_gate,
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
from piginot.objectives.verification import (
    PHYSICS_TERM_IDS,
    VERIFIED_GRADIENT_TERM_IDS,
    S12VerificationReport,
    run_s12_verification,
)

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
