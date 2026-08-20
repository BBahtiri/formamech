from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from piginot.contracts.errors import (
    BareDimensionalValueError,
    ContractValidationError,
    NonFiniteNumberError,
    UnitMismatchError,
)
from piginot.contracts.units import (
    CanonicalQuantity,
    Dimension,
    Quantity,
    Unit,
    to_canonical,
)


@pytest.mark.parametrize(
    ("unit", "dimension"),
    [
        (Unit.ONE, Dimension.DIMENSIONLESS),
        (Unit.MM, Dimension.LENGTH),
        (Unit.CM, Dimension.LENGTH),
        (Unit.M, Dimension.LENGTH),
        (Unit.N, Dimension.FORCE),
        (Unit.KN, Dimension.FORCE),
        (Unit.PA, Dimension.STRESS),
        (Unit.KPA, Dimension.STRESS),
        (Unit.MPA, Dimension.STRESS),
        (Unit.GPA, Dimension.STRESS),
        (Unit.N_MM, Dimension.MOMENT),
        (Unit.N_M, Dimension.MOMENT),
        (Unit.J, Dimension.MOMENT),
    ],
)
def test_each_supported_unit_declares_its_physical_dimension(
    unit: Unit, dimension: Dimension
) -> None:
    assert unit.dimension is dimension


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (Quantity(1.0, Unit.ONE), CanonicalQuantity(1.0, Unit.ONE)),
        (Quantity(2.5, Unit.MM), CanonicalQuantity(2.5, Unit.MM)),
        (Quantity(2.5, Unit.CM), CanonicalQuantity(25.0, Unit.MM)),
        (Quantity(2.5, Unit.M), CanonicalQuantity(2500.0, Unit.MM)),
        (Quantity(3.0, Unit.N), CanonicalQuantity(3.0, Unit.N)),
        (Quantity(0.125, Unit.KN), CanonicalQuantity(125.0, Unit.N)),
        (Quantity(3.0, Unit.PA), CanonicalQuantity(0.000003, Unit.MPA)),
        (Quantity(3.0, Unit.KPA), CanonicalQuantity(0.003, Unit.MPA)),
        (Quantity(2.1, Unit.MPA), CanonicalQuantity(2.1, Unit.MPA)),
        (Quantity(2.1, Unit.GPA), CanonicalQuantity(2100.0, Unit.MPA)),
        (Quantity(7.0, Unit.N_MM), CanonicalQuantity(7.0, Unit.N_MM)),
        (Quantity(1.5, Unit.N_M), CanonicalQuantity(1500.0, Unit.N_MM)),
        (Quantity(1.5, Unit.J), CanonicalQuantity(1500.0, Unit.N_MM)),
    ],
)
def test_conversion_to_the_internal_unit_system_is_explicit_and_exact(
    source: Quantity, expected: CanonicalQuantity
) -> None:
    assert source.to_canonical() == expected
    assert to_canonical(source) == expected


def test_equivalent_quantities_are_unequal_until_explicitly_canonicalized() -> None:
    metres = Quantity(1.0, Unit.M)
    millimetres = Quantity(1000.0, Unit.MM)

    assert metres != millimetres
    assert metres.to_canonical() == millimetres.to_canonical()


def test_equivalent_stress_conversions_have_exact_canonical_identity() -> None:
    from_kilopascals = Quantity(0.1, Unit.KPA).to_canonical()
    from_pascals = Quantity(100.0, Unit.PA).to_canonical()

    assert from_kilopascals == from_pascals


def test_explicit_conversion_can_enforce_the_expected_dimension() -> None:
    converted = to_canonical(
        Quantity(2.0, Unit.CM), expected_dimension=Dimension.LENGTH
    )

    assert converted == CanonicalQuantity(20.0, Unit.MM)


def test_wrong_expected_dimension_raises_a_typed_error() -> None:
    with pytest.raises(UnitMismatchError, match="length.*force|force.*length"):
        to_canonical(Quantity(4.0, Unit.N), expected_dimension=Dimension.LENGTH)


@pytest.mark.parametrize("bare_value", [0, 1.25, True, None])
def test_free_conversion_refuses_bare_dimensional_values(bare_value: Any) -> None:
    with pytest.raises(BareDimensionalValueError):
        to_canonical(bare_value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_quantity_refuses_non_finite_numbers(value: float) -> None:
    with pytest.raises(NonFiniteNumberError):
        Quantity(value, Unit.MM)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_quantity_refuses_non_finite_numbers(value: float) -> None:
    with pytest.raises(NonFiniteNumberError):
        CanonicalQuantity(value, Unit.MM)


def test_boolean_is_not_accepted_as_a_numeric_quantity() -> None:
    with pytest.raises(ContractValidationError):
        Quantity(True, Unit.MM)


def test_quantity_requires_a_unit_member_not_an_untyped_string() -> None:
    with pytest.raises(ContractValidationError):
        Quantity(1.0, "mm")  # type: ignore[arg-type]


def test_canonical_quantity_requires_a_unit_member_not_an_untyped_string() -> None:
    with pytest.raises(ContractValidationError):
        CanonicalQuantity(1.0, "mm")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "noncanonical",
    [Unit.CM, Unit.M, Unit.KN, Unit.PA, Unit.KPA, Unit.GPA, Unit.N_M, Unit.J],
)
def test_canonical_quantity_refuses_a_noncanonical_unit(noncanonical: Unit) -> None:
    with pytest.raises(ContractValidationError, match="canonical"):
        CanonicalQuantity(1.0, noncanonical)


def test_quantity_and_canonical_quantity_are_deeply_immutable() -> None:
    quantity = Quantity(1.0, Unit.M)
    canonical = quantity.to_canonical()

    with pytest.raises(FrozenInstanceError):
        quantity.value = 2.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        canonical.unit = Unit.N  # type: ignore[misc]


def test_quantities_are_hashable_value_objects() -> None:
    values = {
        Quantity(1.0, Unit.M),
        Quantity(1.0, Unit.M),
        Quantity(1000.0, Unit.MM),
    }

    assert len(values) == 2
