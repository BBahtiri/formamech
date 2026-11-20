"""FEM-guarded differentiable design optimization (Feature E / P5)."""

from piginot.optimization.problem import (
    CONSTRAINT_KINDS,
    DESIGN_PROBLEM_SCHEMA,
    DESIGN_REPORT_SCHEMA,
    DesignConstraint,
    DesignParameter,
    DesignProblem,
)
from piginot.optimization.trust_region import (
    DesignIterationRecord,
    DesignOptimizationReport,
    DesignOptimizationStatus,
    FemDesignEvaluation,
    FemEvaluator,
    GradientConsistencyCheck,
    SurrogateDesignEvaluation,
    SurrogateEvaluator,
    VerifiedDesignSolution,
    run_fem_guarded_design_optimization,
)

__all__ = (
    "CONSTRAINT_KINDS",
    "DESIGN_PROBLEM_SCHEMA",
    "DESIGN_REPORT_SCHEMA",
    "DesignConstraint",
    "DesignIterationRecord",
    "DesignOptimizationReport",
    "DesignOptimizationStatus",
    "DesignParameter",
    "DesignProblem",
    "FemDesignEvaluation",
    "FemEvaluator",
    "GradientConsistencyCheck",
    "SurrogateDesignEvaluation",
    "SurrogateEvaluator",
    "VerifiedDesignSolution",
    "run_fem_guarded_design_optimization",
)
