"""Data-bound import, preparation, and parametric exploration of planar CAD profiles."""

from __future__ import annotations

from typing import Any

_PARAMETRIC_SYMBOLS: tuple[str, ...] = (
    "BUILTIN_TEMPLATES",
    "BUILTIN_TEMPLATE_GENERATOR_ID",
    "GENERATOR_ENTRY_POINT_GROUP",
    "PARAMETRIC_CAD_SCHEMA",
    "PARAMETRIC_DOE_SCHEMA",
    "ExpectedTopology",
    "GeneratedCadMesh",
    "GeometricBound",
    "ParametricCadSpec",
    "ParametricDoeReport",
    "ParametricGeneratorFn",
    "ParametricInstance",
    "ParametricTopologyError",
    "TopologyStabilityGuard",
    "default_holed_plate_spec",
    "default_notched_bracket_spec",
    "get_parametric_generator",
    "init_parametric_project",
    "load_parametric_project",
    "register_parametric_generator",
    "sample_parametric_cad_doe",
)

_ADAPTER_SYMBOLS: tuple[str, ...] = (
    "PARAMETRIC_BUNDLE_SCHEMA",
    "PARAMETRIC_CAD",
    "PARAMETRIC_CAD_ADAPTER_ID",
    "PARAMETRIC_PREDICTION_SCHEMA",
    "PARAMETRIC_QUALIFICATION_SCHEMA",
    "ParametricCadAdapter",
    "ParametricFieldMetrics",
    "ParametricPhysicsConfig",
    "ParametricPredictionReport",
    "ParametricQualificationReport",
    "ParametricSurrogateBundle",
    "ParametricTrainingConfig",
    "ParametricTrainingSummary",
    "evaluate_parametric_fem",
    "is_parametric_bundle_dir",
    "load_parametric_bundle",
    "predict_parametric_cad",
    "qualify_parametric_bundle",
    "save_parametric_bundle",
    "train_parametric_cad_surrogate",
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    **{name: ("piginot.cad.parametric", name) for name in _PARAMETRIC_SYMBOLS},
    **{name: ("piginot.cad.parametric_adapter", name) for name in _ADAPTER_SYMBOLS},
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    mod_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(mod_name), attr_name)
    globals()[name] = value
    return value


__all__ = tuple(sorted(_LAZY_EXPORTS))
