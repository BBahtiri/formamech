from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from piginot.contracts.errors import (
    ContractValidationError,
    ExpressionSyntaxError,
    UnitMismatchError,
    UnknownSymbolError,
)
from piginot.contracts.expressions import DerivedExpression
from piginot.contracts.specs import DerivedQuantitySpec, ParameterSpec
from piginot.contracts.units import Dimension, Quantity, Unit


def _length_parameter(name: str, lower: float, upper: float) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        lower_bound=Quantity(lower, Unit.MM).to_canonical(),
        upper_bound=Quantity(upper, Unit.MM).to_canonical(),
    )


def _centre_notch_parameters() -> tuple[ParameterSpec, ...]:
    return (
        _length_parameter("width", 16.0, 30.0),
        _length_parameter("notch_depth", 0.5, 5.0),
    )


def test_net_section_factor_has_a_stable_canonical_ast() -> None:
    expression = DerivedExpression.parse("width / (width - 2 * notch_depth)")

    assert expression.canonical_ast == (
        "binary",
        "/",
        ("name", "width"),
        (
            "binary",
            "-",
            ("name", "width"),
            (
                "binary",
                "*",
                ("number", 2),
                ("name", "notch_depth"),
            ),
        ),
    )


def test_whitespace_and_numeric_spelling_do_not_change_expression_identity() -> None:
    compact = DerivedExpression.parse("width/(width-2*notch_depth)")
    padded = DerivedExpression.parse(" width / ( width - 2.0e0 * notch_depth ) ")

    assert compact.canonical_ast == padded.canonical_ast
    assert compact == padded


def test_operator_precedence_is_preserved_in_the_canonical_ast() -> None:
    multiplied_first = DerivedExpression.parse("a + b * c")
    added_first = DerivedExpression.parse("(a + b) * c")

    assert multiplied_first.canonical_ast != added_first.canonical_ast


def test_canonical_ast_is_structural_not_algebraically_normalized() -> None:
    assert DerivedExpression.parse("a + b") != DerivedExpression.parse("b + a")
    assert DerivedExpression.parse("x") != DerivedExpression.parse("+x")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("-width", ("unary", "-", ("name", "width"))),
        ("+width", ("unary", "+", ("name", "width"))),
    ],
)
def test_unary_operators_are_part_of_the_small_grammar(
    source: str, expected: tuple[object, ...]
) -> None:
    assert DerivedExpression.parse(source).canonical_ast == expected


@pytest.mark.parametrize(
    "source",
    [
        "width ** 2",
        "abs(width)",
        "geometry.width",
        "geometry['width']",
        "width > notch_depth",
        "width and notch_depth",
        "width if flag else notch_depth",
        "[width]",
        "'width'",
        "lambda: width",
        "width +",
        "",
    ],
)
def test_anything_outside_arithmetic_names_literals_and_parentheses_is_rejected(
    source: str,
) -> None:
    with pytest.raises(ExpressionSyntaxError):
        DerivedExpression.parse(source)


def test_non_finite_numeric_literal_is_rejected() -> None:
    with pytest.raises(ContractValidationError, match="finite"):
        DerivedExpression.parse("width + 1e999")


def test_static_validation_resolves_names_and_infers_dimensionless_ratio() -> None:
    expression = DerivedExpression.parse("width / (width - 2 * notch_depth)")

    inferred = expression.validate(
        parameter_dimensions={
            "width": Dimension.LENGTH,
            "notch_depth": Dimension.LENGTH,
        },
        expected_dimension=Dimension.DIMENSIONLESS,
    )

    assert inferred is Dimension.DIMENSIONLESS


def test_static_validation_understands_force_times_length_as_moment() -> None:
    expression = DerivedExpression.parse("force * lever_arm")

    inferred = expression.validate(
        parameter_dimensions={
            "force": Dimension.FORCE,
            "lever_arm": Dimension.LENGTH,
        },
        expected_dimension=Dimension.MOMENT,
    )

    assert inferred is Dimension.MOMENT


def test_static_validation_keeps_declared_energy_and_moment_semantics_distinct() -> (
    None
):
    expression = DerivedExpression.parse("quantity")

    with pytest.raises(UnitMismatchError):
        expression.validate(
            parameter_dimensions={"quantity": Dimension.ENERGY},
            expected_dimension=Dimension.MOMENT,
        )
    with pytest.raises(UnitMismatchError):
        expression.validate(
            parameter_dimensions={"quantity": Dimension.MOMENT},
            expected_dimension=Dimension.ENERGY,
        )


def test_static_validation_preserves_dimension_through_unary_operators() -> None:
    expression = DerivedExpression.parse("-width")

    inferred = expression.validate(
        parameter_dimensions={"width": Dimension.LENGTH},
        expected_dimension=Dimension.LENGTH,
    )

    assert inferred is Dimension.LENGTH


def test_static_validation_refuses_unknown_parameter_names() -> None:
    expression = DerivedExpression.parse("width / missing_depth")

    with pytest.raises(UnknownSymbolError, match="missing_depth"):
        expression.validate(
            parameter_dimensions={"width": Dimension.LENGTH},
            expected_dimension=Dimension.DIMENSIONLESS,
        )


def test_addition_and_subtraction_require_matching_dimensions() -> None:
    expression = DerivedExpression.parse("width + load")

    with pytest.raises(UnitMismatchError):
        expression.validate(
            parameter_dimensions={
                "width": Dimension.LENGTH,
                "load": Dimension.FORCE,
            },
            expected_dimension=Dimension.LENGTH,
        )


def test_declared_dimension_must_match_the_statically_inferred_dimension() -> None:
    expression = DerivedExpression.parse("width / notch_depth")

    with pytest.raises(UnitMismatchError):
        expression.validate(
            parameter_dimensions={
                "width": Dimension.LENGTH,
                "notch_depth": Dimension.LENGTH,
            },
            expected_dimension=Dimension.LENGTH,
        )


def test_static_validation_refuses_dimensions_outside_the_contract_vocabulary() -> None:
    expression = DerivedExpression.parse("width * width")

    with pytest.raises(UnitMismatchError, match="unsupported derived dimension"):
        expression.validate(
            parameter_dimensions={"width": Dimension.LENGTH},
            expected_dimension=Dimension.LENGTH,
        )


@pytest.mark.parametrize(
    "canonical_ast",
    [
        ("invalid",),
        ("binary", "%", ("number", 1), ("number", 2)),
    ],
)
def test_static_validation_refuses_a_malformed_canonical_tree(
    canonical_ast: tuple[object, ...],
) -> None:
    expression = DerivedExpression(source="malformed", canonical_ast=canonical_ast)

    with pytest.raises(ExpressionSyntaxError):
        expression.validate(
            parameter_dimensions={},
            expected_dimension=Dimension.DIMENSIONLESS,
        )


def test_derived_quantity_validates_against_declared_primitive_parameters() -> None:
    derived = DerivedQuantitySpec(
        name="net_section_factor",
        expression=DerivedExpression.parse("width / (width - 2 * notch_depth)"),
        lower_bound=Quantity(1.07, Unit.ONE).to_canonical(),
        upper_bound=Quantity(2.65, Unit.ONE).to_canonical(),
        unit=Unit.ONE,
    )

    assert derived.validate_against(_centre_notch_parameters()) is None


def test_derived_quantity_rejects_an_unknown_primitive_parameter() -> None:
    derived = DerivedQuantitySpec(
        name="bad_ratio",
        expression=DerivedExpression.parse("width / missing_depth"),
        lower_bound=Quantity(1.0, Unit.ONE).to_canonical(),
        upper_bound=Quantity(3.0, Unit.ONE).to_canonical(),
        unit=Unit.ONE,
    )

    with pytest.raises(UnknownSymbolError, match="missing_depth"):
        derived.validate_against(_centre_notch_parameters())


def test_derived_quantity_bounds_must_match_its_explicit_unit() -> None:
    with pytest.raises(UnitMismatchError):
        DerivedQuantitySpec(
            name="bad_bounds",
            expression=DerivedExpression.parse("width"),
            lower_bound=Quantity(1.0, Unit.MM).to_canonical(),
            upper_bound=Quantity(2.0, Unit.MM).to_canonical(),
            unit=Unit.ONE,
        )


def test_derived_expression_dimension_must_match_its_explicit_unit() -> None:
    derived = DerivedQuantitySpec(
        name="bad_unit",
        expression=DerivedExpression.parse("width / notch_depth"),
        lower_bound=Quantity(1.0, Unit.MM).to_canonical(),
        upper_bound=Quantity(3.0, Unit.MM).to_canonical(),
        unit=Unit.MM,
    )

    with pytest.raises(UnitMismatchError):
        derived.validate_against(_centre_notch_parameters())


def test_expression_contract_is_immutable_and_has_no_evaluator() -> None:
    expression = DerivedExpression.parse("width / notch_depth")

    with pytest.raises(FrozenInstanceError):
        expression.source = "width"  # type: ignore[misc]
    assert not hasattr(expression, "evaluate")


def test_derived_quantity_contract_has_no_geometry_evaluator() -> None:
    derived = DerivedQuantitySpec(
        name="net_section_factor",
        expression=DerivedExpression.parse("width / notch_depth"),
        lower_bound=Quantity(1.0, Unit.ONE).to_canonical(),
        upper_bound=Quantity(3.0, Unit.ONE).to_canonical(),
        unit=Unit.ONE,
    )

    assert not hasattr(derived, "evaluate")
