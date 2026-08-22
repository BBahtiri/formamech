from __future__ import annotations

import pytest

from piginot.geometry import (
    BoundaryTagError,
    GeometryReasonCode,
    GeometryValidationError,
    canonicalize_mesh,
)
from tests.geometry.helpers import payload_with, permute_payload, square_payload


def _capture_error(payload) -> GeometryValidationError:
    with pytest.raises(GeometryValidationError) as caught:
        canonicalize_mesh(payload)
    return caught.value


def test_duplicate_nodes_are_exact_and_signed_zero_is_not_distinct() -> None:
    base = square_payload()
    signed_zero_duplicate = payload_with(
        base,
        nodes=(*base.nodes, (-0.0, -0.0)),
    )

    error = _capture_error(signed_zero_duplicate)

    assert error.reason_code is GeometryReasonCode.DUPLICATE_NODE
    assert len(error.entity_ids) == 2


def test_nearby_but_bitwise_distinct_nodes_are_not_silently_merged() -> None:
    base = square_payload()
    distinct_unused_node = payload_with(
        base,
        nodes=(*base.nodes, (1.0e-13, 0.0)),
    )

    error = _capture_error(distinct_unused_node)

    assert error.reason_code is GeometryReasonCode.UNUSED_NODE
    assert error.reason_code is not GeometryReasonCode.DUPLICATE_NODE
    assert error.entity_ids


@pytest.mark.parametrize(
    ("nodes", "cells", "reason_code"),
    [
        (
            ((0.0, 0.0), (1.0, 0.0), (0.5, 0.0)),
            ((0, 1, 2),),
            GeometryReasonCode.DEGENERATE_ELEMENT,
        ),
        (
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            ((0, 2, 1),),
            GeometryReasonCode.INVERTED_ELEMENT,
        ),
    ],
)
def test_degenerate_and_inverted_elements_have_distinct_failures(
    nodes: tuple[tuple[float, float], ...],
    cells: tuple[tuple[int, int, int], ...],
    reason_code: GeometryReasonCode,
) -> None:
    base = square_payload()
    invalid = payload_with(
        base,
        nodes=nodes,
        cells=cells,
        boundary_facets=((0, 1), (1, 2), (2, 0)),
        boundary_tags={"outer": (0, 1, 2)},
        region_ids=(7,),
    )

    error = _capture_error(invalid)

    assert error.reason_code is reason_code
    assert error.entity_ids == (0,)
    assert invalid.cells == cells


def test_unused_nodes_and_cyclic_duplicate_cells_are_rejected() -> None:
    base = square_payload()
    unused = payload_with(base, nodes=(*base.nodes, (2.0, 2.0)))
    duplicate = payload_with(
        base,
        nodes=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        cells=((0, 1, 2), (1, 2, 0)),
        region_ids=(7, 7),
        boundary_facets=((0, 1), (1, 2), (2, 0)),
        boundary_tags={"outer": (0, 1, 2)},
    )

    unused_error = _capture_error(unused)
    duplicate_error = _capture_error(duplicate)

    assert unused_error.reason_code is GeometryReasonCode.UNUSED_NODE
    assert unused_error.entity_ids == (4,)
    assert duplicate_error.reason_code is GeometryReasonCode.DUPLICATE_CELL
    assert duplicate_error.entity_ids == (0, 1)


def test_region_ids_must_be_positive_integers() -> None:
    base = square_payload()
    missing = payload_with(base, region_ids=(7, 0))

    error = _capture_error(missing)

    assert error.reason_code is GeometryReasonCode.MISSING_REGION_TAG
    assert error.entity_ids == (1,)


def test_non_manifold_and_same_direction_shared_edges_are_distinct_and_stable() -> None:
    base = square_payload()
    non_manifold = payload_with(
        base,
        nodes=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (0.5, -1.0), (0.5, 2.0)),
        cells=((0, 1, 2), (1, 0, 3), (0, 1, 4)),
        boundary_facets=((1, 2), (2, 0), (0, 3), (3, 1), (1, 4), (4, 0)),
        boundary_tags={"outer": (0, 1, 2, 3, 4, 5)},
        region_ids=(7, 7, 7),
    )
    same_direction = payload_with(
        base,
        nodes=((0.0, 0.0), (1.0, 0.0), (0.25, 1.0), (0.75, 1.0)),
        cells=((0, 1, 2), (0, 1, 3)),
        boundary_facets=((1, 2), (2, 0), (1, 3), (3, 0)),
        boundary_tags={"outer": (0, 1, 2, 3)},
        region_ids=(7, 7),
    )

    non_manifold_error = _capture_error(non_manifold)
    orientation_error = _capture_error(same_direction)

    assert non_manifold_error.reason_code is GeometryReasonCode.NON_MANIFOLD_EDGE
    assert non_manifold_error.entity_ids
    assert (
        orientation_error.reason_code
        is GeometryReasonCode.INCONSISTENT_EDGE_ORIENTATION
    )
    assert orientation_error.entity_ids

    permuted_non_manifold = permute_payload(
        non_manifold,
        node_order=(4, 2, 0, 3, 1),
        cell_order=(2, 0, 1),
        facet_order=(5, 2, 0, 4, 1, 3),
        cell_rotations=(2, 1, 2),
    )
    permuted_orientation = permute_payload(
        same_direction,
        node_order=(2, 0, 3, 1),
        cell_order=(1, 0),
        facet_order=(3, 1, 0, 2),
        cell_rotations=(1, 2),
    )
    assert _capture_error(permuted_non_manifold).entity_ids == (
        non_manifold_error.entity_ids
    )
    assert _capture_error(permuted_orientation).entity_ids == (
        orientation_error.entity_ids
    )


def test_boundary_facets_must_match_the_topological_boundary_exactly() -> None:
    base = square_payload()
    extra_interior_facet = payload_with(
        base,
        boundary_facets=(*base.boundary_facets, (0, 2)),
        boundary_tags={**dict(base.boundary_tags), "interior": (4,)},
    )
    reordered = permute_payload(
        extra_interior_facet,
        node_order=(2, 0, 3, 1),
        facet_order=(4, 2, 0, 3, 1),
    )

    first_error = _capture_error(extra_interior_facet)
    second_error = _capture_error(reordered)

    assert first_error.reason_code is GeometryReasonCode.BOUNDARY_MISMATCH
    assert first_error.entity_ids
    assert second_error.entity_ids == first_error.entity_ids


@pytest.mark.parametrize(
    ("tags", "reason_code", "expected_facet_ids"),
    [
        (
            {"bottom": (0,), "top": (2,), "left": (3,)},
            GeometryReasonCode.UNCLASSIFIED_BOUNDARY_FACET,
            (2,),
        ),
        (
            {
                "bottom": (0,),
                "right": (1,),
                "also_right": (1,),
                "top": (2,),
                "left": (3,),
            },
            GeometryReasonCode.AMBIGUOUS_BOUNDARY_TAG,
            (2,),
        ),
    ],
)
def test_boundary_tags_are_an_exact_partition_with_stable_facet_ids(
    tags: dict[str, tuple[int, ...]],
    reason_code: GeometryReasonCode,
    expected_facet_ids: tuple[int, ...],
) -> None:
    invalid = square_payload(boundary_tags=tags)
    reordered = permute_payload(
        invalid,
        node_order=(2, 0, 3, 1),
        facet_order=(2, 0, 3, 1),
    )

    first_error = _capture_error(invalid)
    second_error = _capture_error(reordered)

    assert isinstance(first_error, BoundaryTagError)
    assert first_error.reason_code is reason_code
    assert first_error.facet_ids == expected_facet_ids
    assert second_error.reason_code is reason_code
    assert second_error.facet_ids == expected_facet_ids
