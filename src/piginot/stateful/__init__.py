"""Stateful nonlinear mechanics foundation: F0 infrastructure & F1 viscoelasticity.

Runtime-safe by default (no ``scikit-fem`` / ``scipy`` imported at top level);
incremental FEM reference stepper exports are lazily loaded under ``.[verify]``.
"""

from __future__ import annotations

from typing import Any

from piginot.stateful.constitutive import (
    ConstitutiveModel,
    MaterialUpdate,
    StatelessHyperelasticModel,
    TangentCheckResult,
    verify_algorithmic_tangent,
)
from piginot.stateful.stepper import (
    STEP_CONTROLLER_VERSION,
    TRAJECTORY_CHECKPOINT_SCHEMA,
    CandidateStepOutput,
    GuardedStatefulStepper,
    StatefulStepper,
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
    "STEP_CONTROLLER_VERSION",
    "TRAJECTORY_CHECKPOINT_SCHEMA",
    "VISCOELASTIC_MODEL_ID",
    "VISCOELASTIC_OPERATOR_ARCH_ID",
    "CandidateStepOutput",
    "ConstitutiveModel",
    "DirichletHistoryBC",
    "GeneralizedMaxwellConfig",
    "GeneralizedMaxwellViscoelastic",
    "GuardedStatefulStepper",
    "HistoryAwareViscoelasticOperator",
    "IncrementalFemStep",
    "IncrementalFemTrajectory",
    "MaterialUpdate",
    "MaxwellBranch",
    "PlaneStressMaterialUpdate",
    "StatefulStepper",
    "StatelessHyperelasticModel",
    "StepPredictor",
    "StepResult",
    "StepStatus",
    "TangentCheckResult",
    "ViscoelasticOperatorConfig",
    "ViscoelasticTrainingSummary",
    "extract_local_viscous_features",
    "hash_state_tensors",
    "interpolate_nodal_features_in_cells",
    "load_trajectory_checkpoint",
    "project_cell_state_to_nodes",
    "save_trajectory_checkpoint",
    "solve_incremental_fem_trajectory",
    "summarize_viscous_state",
    "train_viscoelastic_operator_energy",
    "verify_algorithmic_tangent",
)
