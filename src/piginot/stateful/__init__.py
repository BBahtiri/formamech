"""Stateful nonlinear mechanics foundation: F0 infrastructure & F1 viscoelasticity.

Runtime-safe by default (no ``scikit-fem`` / ``scipy`` imported at top level);
incremental FEM reference stepper exports are lazily loaded under ``.[verify]``.
"""

from __future__ import annotations

from typing import Any

from piginot.stateful.artifact import (
    VISCOELASTIC_ARTIFACT_SCHEMA,
    LoadedViscoelasticArtifact,
    load_viscoelastic_artifact,
    run_viscoelastic_rollout,
    save_viscoelastic_artifact,
)
from piginot.stateful.constitutive import (
    ConstitutiveModel,
    MaterialUpdate,
    StatelessHyperelasticModel,
    TangentCheckResult,
    verify_algorithmic_tangent,
)
from piginot.stateful.j2_plasticity import (
    J2_PLASTICITY_MODEL_ID,
    J2IsotropicPlasticity,
    J2PlasticityConfig,
    analytical_uniaxial_j2_response,
    verify_plane_stress_tangent,
)
from piginot.stateful.plasticity_operator import (
    PLASTICITY_OPERATOR_ARCH_ID,
    HistoryAwarePlasticityOperator,
    PlasticityKinematics,
    PlasticityOperatorConfig,
    PlasticityTrainingSummary,
    evaluate_plasticity_discrete_residual,
    extract_local_plastic_features,
    project_plastic_state_to_nodes,
    summarize_plastic_state,
    train_plasticity_operator_variational,
)
from piginot.stateful.stepper import (
    STATEFUL_STEP_MACHINE,
    STEP_CONTROLLER_VERSION,
    TRAJECTORY_CHECKPOINT_SCHEMA,
    CandidateStepOutput,
    GuardedStatefulStepper,
    StatefulStepper,
    StepPhase,
    StepPredictor,
    StepResult,
    StepStatus,
    hash_state_tensors,
    load_trajectory_checkpoint,
    save_trajectory_checkpoint,
)
from piginot.stateful.viscoelastic import (
    VISCOELASTIC_MODEL_ID,
    GeneralizedMaxwellConfig,
    GeneralizedMaxwellViscoelastic,
    MaxwellBranch,
    PlaneStressMaterialUpdate,
)
from piginot.stateful.viscoelastic_operator import (
    VISCOELASTIC_OPERATOR_ARCH_ID,
    HistoryAwareViscoelasticOperator,
    ViscoelasticOperatorConfig,
    ViscoelasticTrainingSummary,
    extract_local_viscous_features,
    interpolate_nodal_features_in_cells,
    project_cell_state_to_nodes,
    summarize_viscous_state,
    train_viscoelastic_operator_energy,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DirichletHistoryBC": ("piginot.stateful.fem_stepper", "DirichletHistoryBC"),
    "IncrementalFemStep": ("piginot.stateful.fem_stepper", "IncrementalFemStep"),
    "IncrementalFemTrajectory": (
        "piginot.stateful.fem_stepper",
        "IncrementalFemTrajectory",
    ),
    "solve_incremental_fem_trajectory": (
        "piginot.stateful.fem_stepper",
        "solve_incremental_fem_trajectory",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    mod_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(mod_name), attr_name)
    globals()[name] = value
    return value


__all__ = (
    "J2_PLASTICITY_MODEL_ID",
    "PLASTICITY_OPERATOR_ARCH_ID",
    "STATEFUL_STEP_MACHINE",
    "STEP_CONTROLLER_VERSION",
    "TRAJECTORY_CHECKPOINT_SCHEMA",
    "VISCOELASTIC_ARTIFACT_SCHEMA",
    "VISCOELASTIC_MODEL_ID",
    "VISCOELASTIC_OPERATOR_ARCH_ID",
    "CandidateStepOutput",
    "ConstitutiveModel",
    "DirichletHistoryBC",
    "GeneralizedMaxwellConfig",
    "GeneralizedMaxwellViscoelastic",
    "GuardedStatefulStepper",
    "HistoryAwarePlasticityOperator",
    "HistoryAwareViscoelasticOperator",
    "IncrementalFemStep",
    "IncrementalFemTrajectory",
    "J2IsotropicPlasticity",
    "J2PlasticityConfig",
    "LoadedViscoelasticArtifact",
    "MaterialUpdate",
    "MaxwellBranch",
    "PlaneStressMaterialUpdate",
    "PlasticityKinematics",
    "PlasticityOperatorConfig",
    "PlasticityTrainingSummary",
    "StatefulStepper",
    "StatelessHyperelasticModel",
    "StepPhase",
    "StepPredictor",
    "StepResult",
    "StepStatus",
    "TangentCheckResult",
    "ViscoelasticOperatorConfig",
    "ViscoelasticTrainingSummary",
    "analytical_uniaxial_j2_response",
    "evaluate_plasticity_discrete_residual",
    "extract_local_plastic_features",
    "extract_local_viscous_features",
    "hash_state_tensors",
    "interpolate_nodal_features_in_cells",
    "load_trajectory_checkpoint",
    "load_viscoelastic_artifact",
    "project_cell_state_to_nodes",
    "project_plastic_state_to_nodes",
    "run_viscoelastic_rollout",
    "save_trajectory_checkpoint",
    "save_viscoelastic_artifact",
    "solve_incremental_fem_trajectory",
    "summarize_plastic_state",
    "summarize_viscous_state",
    "train_plasticity_operator_variational",
    "train_viscoelastic_operator_energy",
    "verify_algorithmic_tangent",
    "verify_plane_stress_tangent",
)
