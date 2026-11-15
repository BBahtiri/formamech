from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from types import MappingProxyType
from typing import Any

import pytest

from piginot.contracts import CanonicalQuantity, Unit, canonical_json_bytes, hash_spec
from piginot.geometry import canonicalize_mesh
from piginot.morphing import FixedGeometryMap, apply_morph
from piginot.morphing.fixed import BoundFixedGeometryDeformation
from piginot.morphing.membership import FamilyMembershipPredicate
from piginot.morphing.models import (
    MorphingError,
    MorphReasonCode,
    MorphValidityCriteria,
)
from tests.morphing.helpers import rectangle_payload


def test_fixed_map_is_an_immutable_parameter_free_recipe() -> None:
    mapping = FixedGeometryMap()
    assert fields(mapping) == ()
    assert mapping.parameter_names == ()
    assert mapping.kind == "fixed-geometry-v1"
    assert canonical_json_bytes(mapping) == (
        b'{"$type":"piginot.morphing.FixedGeometryMap"}'
    )
    with pytest.raises((AttributeError, TypeError)):
        mapping.kind = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("theta", ({}, MappingProxyType({})))
def test_empty_geometry_parameters_bind_deterministically(
    theta: Mapping[str, CanonicalQuantity],
) -> None:
    first = FixedGeometryMap().bind(theta)
    second = FixedGeometryMap().bind(theta)
    assert isinstance(first, BoundFixedGeometryDeformation)
    assert first == second
    assert first.kind == FixedGeometryMap.kind
    assert first.parameter_hash == hash_spec({}, "content")


@pytest.mark.parametrize("theta", (None, (), [], "", 0))
def test_bind_requires_an_explicit_mapping(theta: Any) -> None:
    with pytest.raises(MorphingError) as caught:
        FixedGeometryMap().bind(theta)
    assert caught.value.reason_code is MorphReasonCode.INVALID_PARAMETER_VALUE


@pytest.mark.parametrize(
    "theta",
    (
        {"width": CanonicalQuantity(3.0, Unit.MM)},
        {"strain": CanonicalQuantity(0.1, Unit.ONE)},
        {"modulus": CanonicalQuantity(760.0, Unit.MPA)},
        {"unknown": None},
    ),
)
def test_nonempty_parameters_are_never_silently_ignored(theta: Any) -> None:
    with pytest.raises(MorphingError) as caught:
        FixedGeometryMap().bind(theta)
    assert caught.value.reason_code is MorphReasonCode.UNKNOWN_PARAMETER


@pytest.mark.parametrize("point", ((0.0, 0.0), (-3.0, 1.25), (123.0, -4.5)))
def test_fixed_deformation_and_inverse_are_exact_identity(point) -> None:
    deformation = FixedGeometryMap().bind({})
    assert deformation.map_point(point) == point
    assert deformation.inverse_point(point) == point
    assert deformation.jacobian(point) == ((1.0, 0.0), (0.0, 1.0))


@pytest.mark.parametrize(
    "point",
    (
        (),
        (1.0,),
        (1.0, 2.0, 3.0),
        [1.0, 2.0],
        (True, 1.0),
        (1.0, False),
        ("1", 2.0),
        (1.0, None),
        (float("nan"), 1.0),
        (1.0, float("inf")),
    ),
)
@pytest.mark.parametrize("method", ("map_point", "inverse_point", "jacobian"))
def test_deformation_refuses_malformed_or_nonfinite_points(point, method) -> None:
    deformation = FixedGeometryMap().bind({})
    with pytest.raises(MorphingError) as caught:
        getattr(deformation, method)(point)
    assert caught.value.reason_code is MorphReasonCode.INVALID_MAP_CONFIGURATION


def test_load_and_material_variation_use_only_the_empty_geometry_subset() -> None:
    mesh = canonicalize_mesh(rectangle_payload(width=3.0, height=1.5, nx=3, ny=2))
    criteria = MorphValidityCriteria(jacobian_margin=0.01)
    membership = FamilyMembershipPredicate(
        "fixed-profile",
        (),
        (),
        mesh.mesh_id,
        mesh.topology_signature,
        {name: len(indices) for name, indices in mesh.semantic_tags.items()},
        criteria,
        hash_spec(criteria, "content"),
    )
    mapping = FixedGeometryMap()
    for strain, modulus in ((0.01, 760.0), (0.02, 1000.0)):
        request = {
            "strain": CanonicalQuantity(strain, Unit.ONE),
            "modulus": CanonicalQuantity(modulus, Unit.MPA),
        }
        theta = {name: request[name] for name in mapping.parameter_names}
        result = apply_morph(mesh, mapping, membership, theta)
        assert result.accepted
        assert result.mesh is not None
        assert result.mesh.mesh_id == mesh.mesh_id
        assert result.mesh.nodes == mesh.nodes
        assert result.mesh.cells == mesh.cells
        assert result.mesh.semantic_tags == mesh.semantic_tags
        assert result.report.minimum_jacobian_determinant == 1.0


def test_new_geometry_parameters_are_refused_by_the_fixed_membership_path() -> None:
    mesh = canonicalize_mesh(rectangle_payload())
    criteria = MorphValidityCriteria()
    membership = FamilyMembershipPredicate(
        "fixed-profile",
        (),
        (),
        mesh.mesh_id,
        mesh.topology_signature,
        {name: len(indices) for name, indices in mesh.semantic_tags.items()},
        criteria,
        hash_spec(criteria, "content"),
    )
    result = apply_morph(
        mesh, FixedGeometryMap(), membership, {"width": CanonicalQuantity(5.0, Unit.MM)}
    )
    assert not result.accepted
    assert result.reason_codes == (MorphReasonCode.UNKNOWN_PARAMETER,)
