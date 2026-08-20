"""Explicit dimensional quantities and conversion to PI-GINOT canonical units."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Final

from piginot.contracts.errors import (
    BareDimensionalValueError,
    ContractValidationError,
    NonFiniteNumberError,
    UnitMismatchError,
)


class Dimension(Enum):
    """Physical dimensions admitted by the S2 vocabulary."""

    DIMENSIONLESS = "dimensionless"
    LENGTH = "length"
    FORCE = "force"
    STRESS = "stress"
    MOMENT = "moment"


class Unit(Enum):
    """Finite unit registry; no unit inference or free-form parsing is allowed."""

    ONE = "1"
    MM = "mm"
    CM = "cm"
    M = "m"
    N = "N"
    KN = "kN"
    PA = "Pa"
    KPA = "kPa"
    MPA = "MPa"
    GPA = "GPa"
    N_MM = "N·mm"
    N_M = "N·m"
    J = "J"

    @property
    def dimension(self) -> Dimension:
        return _UNIT_DEFINITIONS[self][0]

    @property
    def canonical_unit(self) -> Unit:
        return _UNIT_DEFINITIONS[self][1]

    @property
    def canonical_factor(self) -> float:
        return _UNIT_DEFINITIONS[self][2]

    @property
    def is_canonical(self) -> bool:
        return self is self.canonical_unit


_UNIT_DEFINITIONS: Final[dict[Unit, tuple[Dimension, Unit, float]]] = {
    Unit.ONE: (Dimension.DIMENSIONLESS, Unit.ONE, 1.0),
    Unit.MM: (Dimension.LENGTH, Unit.MM, 1.0),
    Unit.CM: (Dimension.LENGTH, Unit.MM, 10.0),
    Unit.M: (Dimension.LENGTH, Unit.MM, 1000.0),
    Unit.N: (Dimension.FORCE, Unit.N, 1.0),
    Unit.KN: (Dimension.FORCE, Unit.N, 1000.0),
    Unit.PA: (Dimension.STRESS, Unit.MPA, 1e-6),
    Unit.KPA: (Dimension.STRESS, Unit.MPA, 1e-3),
    Unit.MPA: (Dimension.STRESS, Unit.MPA, 1.0),
    Unit.GPA: (Dimension.STRESS, Unit.MPA, 1000.0),
    Unit.N_MM: (Dimension.MOMENT, Unit.N_MM, 1.0),
    Unit.N_M: (Dimension.MOMENT, Unit.N_MM, 1000.0),
    Unit.J: (Dimension.MOMENT, Unit.N_MM, 1000.0),
}


def _validated_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError("quantity value must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise NonFiniteNumberError("quantity value must be finite")
    return number


@dataclass(frozen=True, slots=True)
class CanonicalQuantity:
    """A quantity already expressed in the declared internal unit system."""

    value: float
    unit: Unit

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _validated_number(self.value))
        if not isinstance(self.unit, Unit):
            raise ContractValidationError("canonical quantity unit must be a Unit")
        if not self.unit.is_canonical:
            raise ContractValidationError(
                f"{self.unit.value!r} is not a canonical internal unit"
            )

    @property
    def dimension(self) -> Dimension:
        return self.unit.dimension


@dataclass(frozen=True, slots=True)
class Quantity:
    """A DTO-boundary quantity whose source unit remains explicit."""

    value: float
    unit: Unit

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _validated_number(self.value))
        if not isinstance(self.unit, Unit):
            raise ContractValidationError("quantity unit must be a Unit")

    @property
    def dimension(self) -> Dimension:
        return self.unit.dimension

    def to_canonical(
        self, *, expected_dimension: Dimension | None = None
    ) -> CanonicalQuantity:
        return to_canonical(self, expected_dimension=expected_dimension)


def to_canonical(
    value: object, *, expected_dimension: Dimension | None = None
) -> CanonicalQuantity:
    """Explicitly cross the DTO boundary into the internal unit system."""

    if not isinstance(value, Quantity):
        raise BareDimensionalValueError(
            "dimensional values require an explicit Quantity and unit"
        )
    if expected_dimension is not None and value.dimension is not expected_dimension:
        raise UnitMismatchError(
            f"expected {expected_dimension.value}, got {value.dimension.value}"
        )
    converted = Decimal(str(value.value)) * Decimal(str(value.unit.canonical_factor))
    return CanonicalQuantity(value=float(converted), unit=value.unit.canonical_unit)
