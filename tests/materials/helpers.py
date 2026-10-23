"""Materials shared by the S25 tests."""

from __future__ import annotations

from piginot.contracts import CanonicalQuantity, MaterialSpec
from piginot.contracts.energy_formula import EnergyFormula
from piginot.contracts.units import Dimension, Unit

YOUNG, POISSON = 760.0, 0.23
SHEAR = YOUNG / (2.0 * (1.0 + POISSON))
LAME = YOUNG * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))

NEO_HOOKEAN_FORMULA = EnergyFormula.from_document(
    {
        "energy": "mu / 2 * (I1 - 3) - mu * log(J) + lam / 2 * log(J) ** 2",
        "parameters": {"mu": "MPa", "lam": "MPa"},
        "positive": ["mu"],
        "small_strain": {"shear_modulus": "mu", "bulk_modulus": "lam + 2 / 3 * mu"},
    }
)

_FIBRES = {
    "shear_modulus": 300.0,
    "bulk_modulus": 500.0,
    "k1": 50.0,
    "k2": 2.0,
    "fibre_angle": 30.0,
}
BUILTIN_VALUES = {
    "mooney_rivlin": {"c10": 120.0, "c01": 30.0, "bulk_modulus": 500.0},
    "yeoh": {"c10": 150.0, "c20": -5.0, "c30": 1.0, "bulk_modulus": 500.0},
    "gent": {"shear_modulus": 300.0, "jm": 20.0, "bulk_modulus": 500.0},
    "hgo": dict(_FIBRES),
    "hgo_single": dict(_FIBRES),
    "goh": {**_FIBRES, "kappa_d": 0.1},
}


def quantity(value: float, unit: Unit) -> CanonicalQuantity:
    dimension = {
        Unit.MPA: Dimension.STRESS,
        Unit.DEG: Dimension.ANGLE,
    }.get(unit, Dimension.DIMENSIONLESS)
    return CanonicalQuantity(value, unit, dimension)


def _unit(model: str, name: str) -> Unit:
    from piginot.materials import formula_for

    formula = formula_for(model)
    if formula is not None:
        return formula.parameters[name]
    return Unit.ONE if name == "poisson_ratio" else Unit.MPA


def material(model: str, values: dict[str, float]) -> MaterialSpec:
    return MaterialSpec(
        name=f"test-{model}",
        model=model,
        properties={
            name: quantity(value, _unit(model, name)) for name, value in values.items()
        },
    )


def neo_hookean() -> MaterialSpec:
    return material(
        "compressible_neo_hookean",
        {"elastic_modulus": YOUNG, "poisson_ratio": POISSON},
    )


def formula_neo_hookean() -> MaterialSpec:
    from piginot.materials import register

    return material(register(NEO_HOOKEAN_FORMULA), {"mu": SHEAR, "lam": LAME})
