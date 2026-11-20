"""FEM-guarded differentiable design optimization (Feature E / P5)."""

from __future__ import annotations

from typing import Any

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

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BundleDesignBridge": (
        "piginot.optimization.bundle_bridge",
        "BundleDesignBridge",
    ),
    "BundleDesignSpec": (
        "piginot.optimization.bundle_bridge",
        "BundleDesignSpec",
    ),
    "parse_bundle_design_document": (
        "piginot.optimization.bundle_bridge",
        "parse_bundle_design_document",
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
    "CONSTRAINT_KINDS",
    "DESIGN_PROBLEM_SCHEMA",
    "DESIGN_REPORT_SCHEMA",
    "BundleDesignBridge",
    "BundleDesignSpec",
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
    "parse_bundle_design_document",
    "run_fem_guarded_design_optimization",
)
