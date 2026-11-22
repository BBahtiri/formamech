"""Data-bound import, preparation, and parametric exploration of planar CAD profiles."""

from __future__ import annotations

from typing import Any

from piginot.cad.parametric import (
    BUILTIN_TEMPLATE_GENERATOR_ID,
    BUILTIN_TEMPLATES,
    GENERATOR_ENTRY_POINT_GROUP,
    PARAMETRIC_CAD_SCHEMA,
    PARAMETRIC_DOE_SCHEMA,
    ExpectedTopology,
    GeneratedCadMesh,
    GeometricBound,
    ParametricCadSpec,
    ParametricDoeReport,
    ParametricGeneratorFn,
    ParametricInstance,
    ParametricTopologyError,
    TopologyStabilityGuard,
    default_holed_plate_spec,
    default_notched_bracket_spec,
    get_parametric_generator,
    init_parametric_project,
    load_parametric_project,
    register_parametric_generator,
    sample_parametric_cad_doe,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "PARAMETRIC_BUNDLE_SCHEMA": (
        "piginot.cad.parametric_adapter",
        "PARAMETRIC_BUNDLE_SCHEMA",
    ),
    "PARAMETRIC_CAD": ("piginot.cad.parametric_adapter", "PARAMETRIC_CAD"),
    "PARAMETRIC_CAD_ADAPTER_ID": (
        "piginot.cad.parametric_adapter",
        "PARAMETRIC_CAD_ADAPTER_ID",
    ),
    "PARAMETRIC_PREDICTION_SCHEMA": (
        "piginot.cad.parametric_adapter",
        "PARAMETRIC_PREDICTION_SCHEMA",
    ),
    "PARAMETRIC_QUALIFICATION_SCHEMA": (
        "piginot.cad.parametric_adapter",
        "PARAMETRIC_QUALIFICATION_SCHEMA",
    ),
    "ParametricCadAdapter": ("piginot.cad.parametric_adapter", "ParametricCadAdapter"),
    "ParametricFieldMetrics": (
        "piginot.cad.parametric_adapter",
        "ParametricFieldMetrics",
    ),
    "ParametricPhysicsConfig": (
        "piginot.cad.parametric_adapter",
        "ParametricPhysicsConfig",
    ),
    "ParametricPredictionReport": (
        "piginot.cad.parametric_adapter",
        "ParametricPredictionReport",
    ),
    "ParametricQualificationReport": (
        "piginot.cad.parametric_adapter",
        "ParametricQualificationReport",
    ),
    "ParametricSurrogateBundle": (
        "piginot.cad.parametric_adapter",
        "ParametricSurrogateBundle",
    ),
    "ParametricTrainingConfig": (
        "piginot.cad.parametric_adapter",
        "ParametricTrainingConfig",
    ),
    "ParametricTrainingSummary": (
        "piginot.cad.parametric_adapter",
        "ParametricTrainingSummary",
    ),
    "evaluate_parametric_fem": (
        "piginot.cad.parametric_adapter",
        "evaluate_parametric_fem",
    ),
    "is_parametric_bundle_dir": (
        "piginot.cad.parametric_adapter",
        "is_parametric_bundle_dir",
    ),
    "load_parametric_bundle": (
        "piginot.cad.parametric_adapter",
        "load_parametric_bundle",
    ),
    "predict_parametric_cad": (
        "piginot.cad.parametric_adapter",
        "predict_parametric_cad",
    ),
    "qualify_parametric_bundle": (
        "piginot.cad.parametric_adapter",
        "qualify_parametric_bundle",
    ),
    "save_parametric_bundle": (
        "piginot.cad.parametric_adapter",
        "save_parametric_bundle",
    ),
    "train_parametric_cad_surrogate": (
        "piginot.cad.parametric_adapter",
        "train_parametric_cad_surrogate",
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
    "BUILTIN_TEMPLATES",
    "BUILTIN_TEMPLATE_GENERATOR_ID",
    "GENERATOR_ENTRY_POINT_GROUP",
    "PARAMETRIC_BUNDLE_SCHEMA",
    "PARAMETRIC_CAD",
    "PARAMETRIC_CAD_ADAPTER_ID",
    "PARAMETRIC_CAD_SCHEMA",
    "PARAMETRIC_DOE_SCHEMA",
    "PARAMETRIC_PREDICTION_SCHEMA",
    "PARAMETRIC_QUALIFICATION_SCHEMA",
    "ExpectedTopology",
    "GeneratedCadMesh",
    "GeometricBound",
    "ParametricCadAdapter",
    "ParametricCadSpec",
    "ParametricDoeReport",
    "ParametricFieldMetrics",
    "ParametricGeneratorFn",
    "ParametricInstance",
    "ParametricPhysicsConfig",
    "ParametricPredictionReport",
    "ParametricQualificationReport",
    "ParametricSurrogateBundle",
    "ParametricTopologyError",
    "ParametricTrainingConfig",
    "ParametricTrainingSummary",
    "TopologyStabilityGuard",
    "default_holed_plate_spec",
    "default_notched_bracket_spec",
    "evaluate_parametric_fem",
    "get_parametric_generator",
    "init_parametric_project",
    "is_parametric_bundle_dir",
    "load_parametric_bundle",
    "load_parametric_project",
    "predict_parametric_cad",
    "qualify_parametric_bundle",
    "register_parametric_generator",
    "sample_parametric_cad_doe",
    "save_parametric_bundle",
    "train_parametric_cad_surrogate",
)
