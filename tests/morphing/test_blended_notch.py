from __future__ import annotations

import itertools
import math

import pytest

from piginot.bundle.specs import decode_morph_map
from piginot.contracts import (
    CanonicalQuantity,
    DerivedExpression,
    DerivedQuantitySpec,
    Dimension,
    GeometryFamilySpec,
    ParameterSpec,
    Unit,
    canonical_json_bytes,
)
from piginot.morphing import (
    BlendedNotchMap,
    BlendedNotchMeshSpec,
    BlendedNotchParameters,
    FamilyMembershipPredicate,
    MorphingError,
    MorphReasonCode,
    MorphValidityCriteria,
    apply_morph,
    build_blended_notch_reference_mesh,
)

REFERENCE = BlendedNotchParameters(25.0, 23.0, 5.0, 3.0)
SPEC = BlendedNotchMeshSpec(12, 12, 8, "bottom", "right_end", "notch", "top", "left")
MAP = BlendedNotchMap(REFERENCE, "width", "notch_radius", "notch_depth")


def _length(value: float) -> CanonicalQuantity:
    return CanonicalQuantity(value, Unit.MM, Dimension.LENGTH)


def _theta(width: float, radius: float, depth: float) -> dict[str, CanonicalQuantity]:
    return {
        "width": _length(width),
        "notch_radius": _length(radius),
        "notch_depth": _length(depth),
    }


def test_the_profile_has_the_declared_depth_root_radius_and_c2_blend() -> None:
    p = BlendedNotchParameters(25.0, 20.0, 4.0, 2.5)
    b = p.blend_length_mm
    assert b == pytest.approx(math.sqrt(60.0))
    assert p.profile_height_mm(0.0) == pytest.approx(10.0 - 2.5)
    assert p.profile_height_mm(b) == 10.0 and p.profile_height_mm(25.0) == 10.0
    step = 1.0e-4
    curvature = (
        p.profile_height_mm(step)
        - 2.0 * p.profile_height_mm(0.0)
        + p.profile_height_mm(step)  # symmetric about the root
    ) / (step * step)
    assert curvature == pytest.approx(1.0 / 4.0, rel=1e-4)
    for x in (0.5, 3.0, b - 1e-3):
        numeric = (p.profile_height_mm(x + 1e-6) - p.profile_height_mm(x - 1e-6)) / 2e-6
        assert p.profile_slope(x) == pytest.approx(numeric, rel=1e-6, abs=1e-9)
    left = (p.profile_slope(b - 1e-5) - p.profile_slope(b - 2e-5)) / 1e-5
    assert abs(left) < 1e-3 and p.profile_slope(b) == 0.0  # C1 and C2 at the knot


@pytest.mark.parametrize(
    "changes",
    [
        {"width_mm": -1.0},
        {"notch_depth_mm": 12.0},
        {"notch_radius_mm": 40.0},
        {"half_length_mm": float("nan")},
    ],
)
def test_invalid_geometry_fails_closed(changes: dict[str, float]) -> None:
    values = {
        "half_length_mm": 25.0,
        "width_mm": 23.0,
        "notch_radius_mm": 5.0,
        "notch_depth_mm": 3.0,
    } | changes
    with pytest.raises(MorphingError):
        BlendedNotchParameters(**values)
    with pytest.raises(MorphingError):
        REFERENCE.profile_height_mm(26.0)


def test_reference_mesh_is_structured_tagged_and_deterministic() -> None:
    mesh = build_blended_notch_reference_mesh(
        REFERENCE, SPEC, source_provenance={"generator": "test"}
    )
    again = build_blended_notch_reference_mesh(
        REFERENCE, SPEC, source_provenance={"generator": "test"}
    )
    assert mesh == again
    assert len(mesh.nodes) == 25 * 9 and len(mesh.cells) == 2 * 24 * 8
    counts = {tag: len(facets) for tag, facets in mesh.semantic_tags.items()}
    assert counts == {"bottom": 24, "right_end": 8, "notch": 12, "top": 12, "left": 8}
    assert mesh.topology_signature.endswith("components=1;holes=0")
    root = min(mesh.nodes, key=lambda node: (node[0], -node[1]))
    assert root == (0.0, 23.0 / 2.0 - 3.0)
    with pytest.raises(TypeError):
        build_blended_notch_reference_mesh(
            object(),
            SPEC,
            source_provenance={},  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        build_blended_notch_reference_mesh(REFERENCE, SPEC, source_provenance={"": "x"})
    with pytest.raises(MorphingError):
        BlendedNotchMeshSpec(1, 1, 1, "a", "a", "b", "c", "d")


def test_the_map_is_identity_at_the_reference_and_exact_on_the_profile() -> None:
    identity = MAP.bind(_theta(23.0, 5.0, 3.0))
    for point in ((0.0, 8.0), (4.0, 3.0), (20.0, 11.5), (25.0, 0.0)):
        assert identity.map_point(point) == pytest.approx(point, abs=1e-12)
    target = BlendedNotchParameters(25.0, 18.0, 2.5, 4.0)
    bound = MAP.bind(_theta(18.0, 2.5, 4.0))
    for x in (0.0, 2.0, REFERENCE.blend_length_mm, 15.0, 25.0):
        mapped_x, mapped_y = bound.map_point((x, REFERENCE.profile_height_mm(x)))
        assert mapped_y == pytest.approx(target.profile_height_mm(mapped_x), rel=1e-12)
    assert bound.map_point((25.0, 0.0)) == (25.0, 0.0)


def test_the_analytic_jacobian_matches_finite_differences() -> None:
    bound = MAP.bind(_theta(17.0, 3.0, 4.5))
    step = 1.0e-6
    for x, y in ((0.3, 2.0), (6.0, 4.0), (12.0, 5.0), (22.0, 10.0)):
        jacobian = bound.jacobian((x, y))
        right = bound.map_point((x + step, y))
        left = bound.map_point((x - step, y))
        up = bound.map_point((x, y + step))
        down = bound.map_point((x, y - step))
        assert jacobian[0][0] == pytest.approx(
            (right[0] - left[0]) / (2 * step), rel=1e-6
        )
        assert jacobian[1][0] == pytest.approx(
            (right[1] - left[1]) / (2 * step), rel=1e-5, abs=1e-8
        )
        assert jacobian[1][1] == pytest.approx((up[1] - down[1]) / (2 * step), rel=1e-6)
        assert jacobian[0][1] == 0.0


def test_binding_refuses_bad_parameters_and_points() -> None:
    cases = [
        ({"width": _length(20.0)}, MorphReasonCode.MISSING_PARAMETER),
        (
            _theta(20.0, 3.0, 2.0) | {"extra": _length(1.0)},
            MorphReasonCode.UNKNOWN_PARAMETER,
        ),
        (
            _theta(20.0, 3.0, 2.0) | {"width": 20.0},
            MorphReasonCode.INVALID_PARAMETER_VALUE,
        ),
        (
            _theta(20.0, 3.0, 2.0) | {"width": CanonicalQuantity(0.2, Unit.ONE)},
            MorphReasonCode.PARAMETER_UNIT_MISMATCH,
        ),
        (_theta(20.0, 3.0, 11.0), MorphReasonCode.INVALID_PARAMETER_VALUE),
    ]
    for theta, code in cases:
        with pytest.raises(MorphingError) as error:
            MAP.bind(theta)  # type: ignore[arg-type]
        assert error.value.reason_code is code
    with pytest.raises(MorphingError):
        MAP.bind([])  # type: ignore[arg-type]
    bound = MAP.bind(_theta(20.0, 3.0, 2.0))
    for point in ((30.0, 1.0), (1.0, 20.0), (1.0,)):
        with pytest.raises(MorphingError):
            bound.map_point(point)  # type: ignore[arg-type]
    with pytest.raises(MorphingError):
        BlendedNotchMap(REFERENCE, "width", "width", "notch_depth")
    with pytest.raises(MorphingError):
        BlendedNotchMap(object(), "a", "b", "c")  # type: ignore[arg-type]


def test_every_corner_of_the_box_morphs_validly_and_round_trips() -> None:
    mesh = build_blended_notch_reference_mesh(
        REFERENCE, SPEC, source_provenance={"generator": "test"}
    )
    family = GeometryFamilySpec(
        name="blended-notch-test",
        parameters=(
            ParameterSpec("width", _length(16.0), _length(30.0)),
            ParameterSpec("notch_radius", _length(2.0), _length(8.0)),
            ParameterSpec("notch_depth", _length(1.0), _length(5.0)),
        ),
        derived_quantities=(
            DerivedQuantitySpec(
                "notch_sharpness",
                DerivedExpression.parse("notch_depth / notch_radius"),
                CanonicalQuantity(0.125, Unit.ONE, Dimension.DIMENSIONLESS),
                CanonicalQuantity(2.5, Unit.ONE, Dimension.DIMENSIONLESS),
                Unit.ONE,
                "depth over root radius",
            ),
        ),
        topology_signature=mesh.topology_signature,
        semantic_tags=tuple(mesh.semantic_tags),
    )
    predicate = FamilyMembershipPredicate.compile(
        family, mesh, MorphValidityCriteria(0.01, 0.05, 5.0, 20.0)
    )
    for width, radius, depth in itertools.product((16.0, 30.0), (2.0, 8.0), (1.0, 5.0)):
        result = apply_morph(mesh, MAP, predicate, _theta(width, radius, depth))
        assert result.accepted, (width, radius, depth, result.reason_codes)
    assert decode_morph_map(canonical_json_bytes(MAP)) == MAP
