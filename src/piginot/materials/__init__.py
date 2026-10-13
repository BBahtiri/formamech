"""Hyperelastic material models beyond the closed-form Neo-Hookean (S25).

A model is a strain-energy formula (:mod:`piginot.contracts.energy_formula`):
a built-in one (Mooney-Rivlin, Yeoh, Gent) or a user's, registered by its
content name. :mod:`piginot.physics` dispatches every such model here; the
admission gate (:mod:`piginot.materials.admission`) decides whether a model
may be trained.
"""

from piginot.materials.hyperelastic import (
    FormulaMaterial,
    embed,
    invariants,
    material_values,
    torch_value,
)
from piginot.materials.registry import (
    formula_for,
    is_formula_model,
    known_models,
    register,
    resolve,
)

__all__ = (
    "FormulaMaterial",
    "embed",
    "formula_for",
    "invariants",
    "is_formula_model",
    "known_models",
    "material_values",
    "register",
    "resolve",
    "torch_value",
)
