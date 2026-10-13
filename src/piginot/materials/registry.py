"""Which strain-energy formula a material model name stands for (S25).

Built-in models are the formulas of
:data:`piginot.contracts.energy_formula.BUILTIN_FORMULAS`; a user formula is
registered under its content name ``energy-<16 hex>``, so the same formula
always has the same name and a name never changes meaning. The closed-form
Neo-Hookean model stays in :mod:`piginot.physics`.
"""

from __future__ import annotations

import threading
from functools import lru_cache

from piginot.contracts import MaterialSpec
from piginot.contracts.energy_formula import (
    BUILTIN_FORMULAS,
    CLOSED_FORM_MODELS,
    EnergyFormula,
)
from piginot.materials.hyperelastic import FormulaMaterial, material_values
from piginot.physics.errors import InvalidMaterialError

_LOCK = threading.Lock()
_USER: dict[str, EnergyFormula] = {}


def register(formula: EnergyFormula) -> str:
    """Make ``formula`` available as a material model; returns its name."""

    if not isinstance(formula, EnergyFormula):
        raise TypeError("formula must be an EnergyFormula")
    name = formula.model_name
    with _LOCK:
        _USER[name] = formula
    return name


def formula_for(model: str) -> EnergyFormula | None:
    """The formula behind ``model``; ``None`` for the closed form or unknown."""

    if model in BUILTIN_FORMULAS:
        return BUILTIN_FORMULAS[model]
    with _LOCK:
        return _USER.get(model)


def is_formula_model(model: object) -> bool:
    return (
        isinstance(model, str)
        and model not in CLOSED_FORM_MODELS
        and (formula_for(model) is not None)
    )


def known_models() -> tuple[str, ...]:
    with _LOCK:
        user = tuple(sorted(_USER))
    return (*CLOSED_FORM_MODELS, *sorted(BUILTIN_FORMULAS), *user)


@lru_cache(maxsize=256)
def _bound(model: str, items: tuple[tuple[str, float], ...]) -> FormulaMaterial:
    formula = formula_for(model)
    assert formula is not None
    return FormulaMaterial(formula, dict(items))


def resolve(material: MaterialSpec) -> FormulaMaterial:
    """The bound formula material of a material contract."""

    if not isinstance(material, MaterialSpec):
        raise InvalidMaterialError("material must be a MaterialSpec")
    formula = formula_for(material.model)
    if formula is None:
        raise InvalidMaterialError(f"unknown material model {material.model!r}")
    values = material_values(formula, material)
    return _bound(material.model, tuple(sorted(values.items())))


__all__ = ("formula_for", "is_formula_model", "known_models", "register", "resolve")
