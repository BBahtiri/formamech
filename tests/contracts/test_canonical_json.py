from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import pytest

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalQuantity,
    LoadSpec,
    Quantity,
    Unit,
)
from piginot.contracts.canonical import canonical_json_bytes, strict_json_loads
from piginot.contracts.errors import (
    ContractValidationError,
    DuplicateKeyError,
    NonFiniteNumberError,
)
from piginot.contracts.profile import ExecutionProfile


@pytest.mark.parametrize(
    "document",
    [
        '{"value": 1, "value": 2}',
        '{"outer": {"value": 1, "value": 2}}',
    ],
)
def test_strict_json_loads_rejects_duplicate_object_names(document: str) -> None:
    with pytest.raises(DuplicateKeyError, match="value"):
        strict_json_loads(document)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_loads_rejects_non_finite_numbers(token: str) -> None:
    with pytest.raises(NonFiniteNumberError, match="finite"):
        strict_json_loads(f'{{"value": {token}}}')


def test_strict_json_loads_rejects_number_overflow_as_non_finite() -> None:
    with pytest.raises(NonFiniteNumberError, match="finite"):
        strict_json_loads('{"value": 1e999}')


@pytest.mark.parametrize("integer", [str(2**53), "9" * 400])
def test_strict_json_loads_rejects_integers_outside_jcs_safe_domain(
    integer: str,
) -> None:
    with pytest.raises(ContractValidationError, match="safe integer domain"):
        strict_json_loads(f'{{"value": {integer}}}')


def test_strict_json_loads_accepts_text_and_utf8_bytes() -> None:
    document = '{"unit":"MPa","value":1.25}'

    assert strict_json_loads(document) == {"unit": "MPa", "value": 1.25}
    assert strict_json_loads(document.encode("utf-8")) == {
        "unit": "MPa",
        "value": 1.25,
    }


def test_strict_json_loads_rejects_invalid_utf8_bytes() -> None:
    with pytest.raises(ContractValidationError, match="valid UTF-8"):
        strict_json_loads(b'{"label":"\xff"}')


def test_canonical_json_bytes_matches_rfc_8785_number_and_escape_vector() -> None:
    payload = {
        "numbers": [
            333333333.33333329,
            1e30,
            4.50,
            2e-3,
            0.000000000000000000000000001,
        ],
        "string": '€$\x0f\nA\'B"\\\\"/',
        "literals": [None, True, False],
    }

    expected = (
        '{"literals":[null,true,false],'
        '"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
        '"string":"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'
    ).encode()

    canonical = canonical_json_bytes(payload)

    assert isinstance(canonical, bytes)
    assert canonical == expected


def test_canonical_json_bytes_sorts_names_by_utf16_code_units() -> None:
    payload = {
        "€": "Euro Sign",
        "\r": "Carriage Return",
        "דּ": "Hebrew Letter Dalet With Dagesh",
        "1": "One",
        "😀": "Emoji: Grinning Face",
        "\x80": "Control",
        "ö": "Latin Small Letter O With Diaeresis",
    }

    expected = (
        '{"\\r":"Carriage Return","1":"One","\x80":"Control",'
        '"ö":"Latin Small Letter O With Diaeresis","\u20ac":"Euro Sign",'
        '"😀":"Emoji: Grinning Face",'
        '"דּ":"Hebrew Letter Dalet With Dagesh"}'
    ).encode("utf-8")

    assert canonical_json_bytes(payload) == expected


def test_canonical_json_bytes_normalizes_negative_zero() -> None:
    assert canonical_json_bytes({"negative_zero": -0.0}) == b'{"negative_zero":0}'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_bytes_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(NonFiniteNumberError, match="finite"):
        canonical_json_bytes({"nested": [value]})


def test_canonical_json_bytes_rejects_non_string_mapping_names() -> None:
    with pytest.raises(ContractValidationError, match="names must be strings"):
        canonical_json_bytes({1: "one"})


def test_canonical_json_bytes_rejects_mapping_type_tag_spoofing() -> None:
    with pytest.raises(ContractValidationError, match="reserved"):
        canonical_json_bytes(
            {
                "$type": "piginot.contracts.CanonicalQuantity",
                "dimension": "length",
                "unit": "mm",
                "value": 1,
            }
        )


@pytest.mark.parametrize("value", [b"bytes are not JSON", {"sets", "are", "not JSON"}])
def test_canonical_json_bytes_rejects_unsupported_value_types(value: object) -> None:
    with pytest.raises(ContractValidationError, match="unsupported canonical JSON"):
        canonical_json_bytes(value)


@pytest.mark.parametrize("value", [2**53, -(2**53)])
def test_canonical_json_bytes_rejects_integers_outside_jcs_safe_domain(
    value: int,
) -> None:
    with pytest.raises(ContractValidationError, match="safe integer domain"):
        canonical_json_bytes(value)


def test_canonical_json_bytes_accepts_jcs_safe_integer_boundary() -> None:
    assert canonical_json_bytes(2**53 - 1) == b"9007199254740991"


def test_canonical_json_bytes_has_stable_public_quantity_type_discriminators() -> None:
    quantity = Quantity(1.0, Unit.MM)
    canonical_quantity = CanonicalQuantity(1.0, Unit.MM)

    assert canonical_json_bytes(quantity) == (
        b'{"$type":"piginot.contracts.Quantity","dimension":"length",'
        b'"unit":"mm","value":1}'
    )
    assert canonical_json_bytes(canonical_quantity) == (
        b'{"$type":"piginot.contracts.CanonicalQuantity","dimension":"length",'
        b'"unit":"mm","value":1}'
    )


def test_canonical_json_bytes_distinguishes_semantically_distinct_spec_types() -> None:
    boundary = BoundaryConditionSpec(
        name="edge_action",
        boundary_tag="loaded_edge",
        kind="traction",
    )
    load = LoadSpec(
        name="edge_action",
        boundary_tag="loaded_edge",
        kind="traction",
    )

    assert canonical_json_bytes(boundary) == (
        b'{"$type":"piginot.contracts.BoundaryConditionSpec",'
        b'"boundary_tag":"loaded_edge","components":[],"kind":"traction",'
        b'"name":"edge_action","values":[]}'
    )
    assert canonical_json_bytes(load) == (
        b'{"$type":"piginot.contracts.LoadSpec",'
        b'"boundary_tag":"loaded_edge","components":[],"kind":"traction",'
        b'"name":"edge_action","values":[]}'
    )


def test_dataclass_canonical_fields_use_explicit_metadata_not_repr_or_compare() -> None:
    @dataclass(frozen=True)
    class CanonicalFieldControls:
        included: int = field(repr=False, compare=False)
        excluded: str = field(metadata={"canonical": False})

    canonical = canonical_json_bytes(CanonicalFieldControls(7, "transient"))
    decoded = strict_json_loads(canonical)

    assert isinstance(decoded, dict)
    assert decoded["included"] == 7
    assert "excluded" not in decoded
    assert "$type" in decoded


def test_canonical_json_bytes_serializes_frozen_maps_as_json_objects() -> None:
    profile = ExecutionProfile(
        python_version="3.13.11",
        library_versions={"numpy": "2.3.2", "rfc8785": "0.1.4"},
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="PCG64",
        seed=20260820,
    )

    canonical = canonical_json_bytes(profile)
    decoded = strict_json_loads(canonical)

    assert isinstance(decoded, dict)
    assert decoded["library_versions"] == {"numpy": "2.3.2", "rfc8785": "0.1.4"}
    assert b'"_items"' not in canonical


def test_canonical_json_bytes_recursively_normalizes_frozen_dataclasses() -> None:
    @dataclass(frozen=True)
    class Bounds:
        __canonical_namespace__: ClassVar[str] = "tests.contracts"

        upper: float
        lower: float

    @dataclass(frozen=True)
    class QuantityContract:
        __canonical_namespace__: ClassVar[str] = "tests.contracts"

        unit: str
        name: str
        bounds: Bounds
        samples: tuple[int, ...]

    contract = QuantityContract(
        unit="mm",
        name="width",
        bounds=Bounds(upper=10.0, lower=2.0),
        samples=(2, 4, 10),
    )

    assert canonical_json_bytes(contract) == (
        b'{"$type":"tests.contracts.test_canonical_json_bytes_recursively_'
        b'normalizes_frozen_dataclasses.<locals>.QuantityContract",'
        b'"bounds":{"$type":"tests.contracts.test_canonical_json_bytes_'
        b'recursively_normalizes_frozen_dataclasses.<locals>.Bounds",'
        b'"lower":2,"upper":10},"name":"width","samples":[2,4,10],'
        b'"unit":"mm"}'
    )
