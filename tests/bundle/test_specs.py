from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
import rfc8785

from campaigns.dogbone_v1.s14_envelope import _family, dogbone_envelope
from piginot.bundle import ConfidenceSpec, QoiSpec, ScaledParameter, SpecDecodeError
from piginot.bundle.specs import (
    ConditioningInput,
    decode_envelope,
    decode_membership,
    decode_mesh,
    decode_model_spec,
    decode_morph_map,
)
from piginot.contracts import canonical_json_bytes
from tests.bundle.helpers import tiny_model_spec


def test_family_specs_round_trip_exactly() -> None:
    family = _family()
    envelope = dogbone_envelope()

    assert decode_mesh(canonical_json_bytes(family.reference_mesh)) == (
        family.reference_mesh
    )
    assert decode_morph_map(canonical_json_bytes(family.morph_map)) == family.morph_map
    assert decode_membership(canonical_json_bytes(family.membership)) == (
        family.membership
    )
    assert decode_envelope(envelope.to_bytes()) == envelope


def _edited(payload: bytes, edit: Any) -> bytes:
    document = json.loads(payload)
    edit(document)
    return rfc8785.dumps(document)


def test_non_canonical_or_foreign_specs_are_refused() -> None:
    family = _family()
    mesh = canonical_json_bytes(family.reference_mesh)
    morph = canonical_json_bytes(family.morph_map)
    envelope = dogbone_envelope().to_bytes()

    with pytest.raises(SpecDecodeError):
        decode_mesh(mesh.replace(b",", b", ", 1))
    with pytest.raises(SpecDecodeError):
        decode_mesh(morph)
    with pytest.raises(SpecDecodeError):
        decode_mesh(b"[1]")
    with pytest.raises(SpecDecodeError):
        decode_mesh(_edited(mesh, lambda d: d["nodes"][0].__setitem__(0, 1.2345)))
    with pytest.raises(SpecDecodeError):
        decode_morph_map(_edited(morph, lambda d: d.update({"$type": "x.Other"})))
    with pytest.raises(SpecDecodeError):
        decode_morph_map(morph + b" ")
    with pytest.raises(SpecDecodeError):
        decode_envelope(morph)
    with pytest.raises(SpecDecodeError):
        decode_envelope(envelope[:-1] + b" }")


def test_model_spec_round_trip_and_inputs() -> None:
    spec = tiny_model_spec()
    assert decode_model_spec(spec.to_bytes()) == spec
    assert spec.inputs == (
        "L_total",
        "R_fillet",
        "W_gauge",
        "W_grip",
        "elastic_modulus",
        "nominal_axial_strain",
        "poisson_ratio",
    )
    assert ScaledParameter(("a", "b"), 0.5).evaluate({"a": 3.0, "b": 4.0}) == 6.0

    with pytest.raises(SpecDecodeError, match="round-trip"):
        decode_model_spec(spec.to_bytes() + b" ")
    with pytest.raises(SpecDecodeError, match="incomplete"):
        decode_model_spec(_edited(spec.to_bytes(), lambda d: d.pop("dtype")))
    with pytest.raises(SpecDecodeError, match="two entries"):
        decode_model_spec(
            _edited(spec.to_bytes(), lambda d: d["frame"]["extent"].pop())
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"conditioning": ()},
        {
            "conditioning": tuple(
                reversed(tiny_model_spec().conditioning)  # order differs
            )
        },
        {"lifting_kind": "affine"},
        {"dtype": "float32"},
        {"query_chunk": 0},
        {"boundary_conditions": {"bottom": ("z",)}},
        {"served_stress": ""},
        {
            "qois": (
                QoiSpec("q", "P_served", "bottom", (), "MPa"),
                QoiSpec("q", "sigma", "bottom", (), "MPa"),
            )
        },
    ],
)
def test_invalid_model_specs_are_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(SpecDecodeError):
        tiny_model_spec(**overrides)


def test_invalid_parts_are_refused() -> None:
    with pytest.raises(SpecDecodeError):
        ConditioningInput("a", "a", 0.0, 0.0)
    with pytest.raises(SpecDecodeError):
        ConditioningInput("", "a", 0.0, 1.0)
    duplicate = tiny_model_spec().conditioning[0]
    with pytest.raises(SpecDecodeError):
        tiny_model_spec(conditioning=(duplicate, duplicate))
    with pytest.raises(SpecDecodeError):
        QoiSpec("q", "P_material", "bottom", (), "MPa")
    good = tiny_model_spec().confidence
    for change in (
        {"variant": "pointwise"},
        {"quantile": 1.0},
        {"values": ()},
        {"values": (0.3, 0.2, 0.1)},
        {"upper_edges": (0.3, 0.2, 0.1)},
        {"reference_stress_mpa": 0.0},
    ):
        with pytest.raises(SpecDecodeError):
            dataclasses.replace(good, **change)
    assert isinstance(good, ConfidenceSpec)
