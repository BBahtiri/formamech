from __future__ import annotations

import math

import pytest

from piginot.contracts import CanonicalQuantity, Dimension, Unit, canonical_json_bytes
from piginot.morphing import (
    CircularTransitionMap,
    CircularTransitionMeshSpec,
    CircularTransitionParameters,
    build_circular_transition_reference_mesh,
)
from piginot.morphing.models import MorphingError, MorphReasonCode


def _quantity(value: float) -> CanonicalQuantity:
    return CanonicalQuantity(value, Unit.MM, Dimension.LENGTH)


def _reference() -> CircularTransitionParameters:
    return CircularTransitionParameters(
        total_length_mm=54.0,
        outer_width_mm=20.0,
        inner_width_mm=10.0,
        transition_radius_mm=12.0,
    )


def _map() -> CircularTransitionMap:
    return CircularTransitionMap(
        reference=_reference(),
        total_length_parameter="total_length",
        outer_width_parameter="outer_width",
        inner_width_parameter="inner_width",
        transition_radius_parameter="transition_radius",
        minimum_inner_length_mm=2.0,
    )


def _theta(
    *,
    total_length: float = 54.0,
    outer_width: float = 20.0,
    inner_width: float = 10.0,
    transition_radius: float = 12.0,
) -> dict[str, CanonicalQuantity]:
    return {
        "total_length": _quantity(total_length),
        "outer_width": _quantity(outer_width),
        "inner_width": _quantity(inner_width),
        "transition_radius": _quantity(transition_radius),
    }


def _flatten(
    matrix: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, ...]:
    return tuple(value for row in matrix for value in row)


def test_default_parameters_are_an_identity_map() -> None:
    deformation = _map().bind(_theta())
    reference = _reference()
    points = (
        (0.0, 0.0),
        (reference.transition_start_x_mm, reference.inner_half_width_mm),
        (reference.half_length_mm, reference.outer_half_width_mm),
        (reference.transition_start_x_mm / 2.0, 1.25),
    )

    for point in points:
        assert deformation.map_point(point) == pytest.approx(point, abs=1.0e-12)
        assert _flatten(deformation.jacobian(point)) == pytest.approx(
            (1.0, 0.0, 0.0, 1.0), abs=1.0e-12
        )


def test_map_reproduces_the_target_transition_and_boundary_exactly() -> None:
    target = CircularTransitionParameters(
        total_length_mm=66.0,
        outer_width_mm=24.0,
        inner_width_mm=8.0,
        transition_radius_mm=18.0,
    )
    deformation = _map().bind(
        _theta(
            total_length=target.total_length_mm,
            outer_width=target.outer_width_mm,
            inner_width=target.inner_width_mm,
            transition_radius=target.transition_radius_mm,
        )
    )
    reference = _reference()

    mapped_transition = deformation.map_point(
        (reference.transition_start_x_mm, reference.inner_half_width_mm)
    )
    mapped_loaded_corner = deformation.map_point(
        (reference.half_length_mm, reference.outer_half_width_mm)
    )

    assert mapped_transition == pytest.approx(
        (target.transition_start_x_mm, target.inner_half_width_mm), abs=1.0e-12
    )
    assert mapped_loaded_corner == pytest.approx(
        (target.half_length_mm, target.outer_half_width_mm), abs=1.0e-12
    )

    for fraction in (0.1, 0.3, 0.6, 0.9):
        reference_x = reference.transition_start_x_mm + fraction * (
            reference.half_length_mm - reference.transition_start_x_mm
        )
        reference_y = reference.profile_height_mm(reference_x)
        mapped_x, mapped_y = deformation.map_point((reference_x, reference_y))
        assert mapped_y == pytest.approx(
            target.profile_height_mm(mapped_x), abs=1.0e-11
        )


def test_analytic_jacobian_matches_centered_finite_difference() -> None:
    deformation = _map().bind(
        _theta(
            total_length=62.0,
            outer_width=23.0,
            inner_width=9.0,
            transition_radius=17.0,
        )
    )
    reference = _reference()
    point = (
        reference.transition_start_x_mm
        + 0.4 * (reference.half_length_mm - reference.transition_start_x_mm),
        0.45
        * reference.profile_height_mm(
            reference.transition_start_x_mm
            + 0.4 * (reference.half_length_mm - reference.transition_start_x_mm)
        ),
    )
    step = 1.0e-6
    analytical = deformation.jacobian(point)
    plus_x = deformation.map_point((point[0] + step, point[1]))
    minus_x = deformation.map_point((point[0] - step, point[1]))
    plus_y = deformation.map_point((point[0], point[1] + step))
    minus_y = deformation.map_point((point[0], point[1] - step))
    numerical = (
        (
            (plus_x[0] - minus_x[0]) / (2.0 * step),
            (plus_y[0] - minus_y[0]) / (2.0 * step),
        ),
        (
            (plus_x[1] - minus_x[1]) / (2.0 * step),
            (plus_y[1] - minus_y[1]) / (2.0 * step),
        ),
    )

    assert _flatten(analytical) == pytest.approx(
        _flatten(numerical), rel=2.0e-8, abs=2.0e-8
    )
    determinant = (
        analytical[0][0] * analytical[1][1] - analytical[0][1] * analytical[1][0]
    )
    assert determinant > 0.0


def test_inverse_map_round_trips_points_on_both_profile_pieces() -> None:
    deformation = _map().bind(
        _theta(
            total_length=62.0,
            outer_width=23.0,
            inner_width=9.0,
            transition_radius=17.0,
        )
    )
    reference = _reference()
    points = (
        (0.2 * reference.transition_start_x_mm, 1.0),
        (reference.transition_start_x_mm, reference.inner_half_width_mm),
        (
            reference.transition_start_x_mm
            + 0.7 * (reference.half_length_mm - reference.transition_start_x_mm),
            2.5,
        ),
        (reference.half_length_mm, reference.outer_half_width_mm),
    )

    for point in points:
        assert deformation.inverse_point(deformation.map_point(point)) == pytest.approx(
            point, rel=1.0e-12, abs=1.0e-12
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"outer_width_mm": 10.0, "inner_width_mm": 10.0},
        {"outer_width_mm": 26.0, "inner_width_mm": 6.0, "transition_radius_mm": 10.0},
        {
            "total_length_mm": 20.0,
            "outer_width_mm": 26.0,
            "inner_width_mm": 6.0,
            "transition_radius_mm": 20.0,
        },
    ),
)
def test_invalid_transition_geometry_fails_closed(changes: dict[str, float]) -> None:
    values = {
        "total_length_mm": 54.0,
        "outer_width_mm": 20.0,
        "inner_width_mm": 10.0,
        "transition_radius_mm": 12.0,
    }
    values.update(changes)

    with pytest.raises(MorphingError) as caught:
        CircularTransitionParameters(**values)

    assert caught.value.reason_code is MorphReasonCode.INVALID_MAP_CONFIGURATION


def test_reference_mesh_is_canonical_tagged_and_deterministic() -> None:
    spec = CircularTransitionMeshSpec(
        inner_columns=8,
        transition_columns=6,
        transverse_rows=5,
        lower_tag="lower",
        loaded_end_tag="loaded",
        transition_profile_tag="transition",
        inner_profile_tag="inner",
        left_end_tag="left",
    )
    first = build_circular_transition_reference_mesh(
        _reference(), spec, source_provenance={"source": "unit-test"}
    )
    second = build_circular_transition_reference_mesh(
        _reference(), spec, source_provenance={"source": "unit-test"}
    )

    assert first == second
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert set(first.semantic_tags) == {
        "inner",
        "left",
        "loaded",
        "lower",
        "transition",
    }
    assert first.topology_signature
    assert len(first.cells) == 2 * (8 + 6) * 5
    assert math.isclose(
        max(point[0] for point in first.nodes), _reference().half_length_mm
    )
