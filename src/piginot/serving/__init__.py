"""Verified serving: FEM recipes and the fast, certified and exact modes (S22).

Importing this package needs only the runtime extra; the FEM stack is loaded
when a certified or exact request is served.
"""

from typing import Any

from piginot.serving.hybrid import (
    SERVING_FUNNEL_MACHINE,
    MissingExtraError,
    ServingMode,
    ServingPath,
    ServingStage,
    SolveAttempt,
    VerifiedPrediction,
    default_request_id,
    load_recipe,
    operator_displacement,
    recipe_path,
    serve,
    solve_ladder,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "SPECTRAL_REPAIR_MACHINE": (
        "piginot.serving.spectral_repair",
        "SPECTRAL_REPAIR_MACHINE",
    ),
    "SpectralRepairConfig": (
        "piginot.serving.spectral_repair",
        "SpectralRepairConfig",
    ),
    "SpectralRepairReport": (
        "piginot.serving.spectral_repair",
        "SpectralRepairReport",
    ),
    "SpectralRepairStatus": (
        "piginot.serving.spectral_repair",
        "SpectralRepairStatus",
    ),
    "TTA_LORA_MACHINE": (
        "piginot.serving.test_time_lora",
        "TTA_LORA_MACHINE",
    ),
    "TestTimeLoraConfig": (
        "piginot.serving.test_time_lora",
        "TestTimeLoraConfig",
    ),
    "TestTimeLoraReport": (
        "piginot.serving.test_time_lora",
        "TestTimeLoraReport",
    ),
    "TestTimeLoraStatus": (
        "piginot.serving.test_time_lora",
        "TestTimeLoraStatus",
    ),
    "adapt_operator_at_test_time": (
        "piginot.serving.test_time_lora",
        "adapt_operator_at_test_time",
    ),
    "spectrally_safe_repair": (
        "piginot.serving.spectral_repair",
        "spectrally_safe_repair",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = (
    "SERVING_FUNNEL_MACHINE",
    "SPECTRAL_REPAIR_MACHINE",
    "TTA_LORA_MACHINE",
    "MissingExtraError",
    "ServingMode",
    "ServingPath",
    "ServingStage",
    "SolveAttempt",
    "SpectralRepairConfig",
    "SpectralRepairReport",
    "SpectralRepairStatus",
    "TestTimeLoraConfig",
    "TestTimeLoraReport",
    "TestTimeLoraStatus",
    "VerifiedPrediction",
    "adapt_operator_at_test_time",
    "default_request_id",
    "load_recipe",
    "operator_displacement",
    "recipe_path",
    "serve",
    "solve_ladder",
    "spectrally_safe_repair",
)
