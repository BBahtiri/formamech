from __future__ import annotations

from piginot.geometry import canonicalize_mesh
from tests.geometry.helpers import canonical_content, permute_payload, square_payload


def test_canonicalization_ignores_all_source_numbering_and_ordering() -> None:
    source = square_payload()
    equivalent = permute_payload(
        source,
        node_order=(2, 0, 3, 1),
        cell_order=(1, 0),
        facet_order=(2, 0, 3, 1),
        cell_rotations=(1, 2),
        reverse_facets=True,
    )

    canonical = canonicalize_mesh(source)
    permuted = canonicalize_mesh(equivalent)

    assert canonical_content(permuted) == canonical_content(canonical)
    assert permuted.source_provenance != canonical.source_provenance


def test_canonicalization_does_not_mutate_or_heal_the_source_payload() -> None:
    payload = square_payload()
    before = (
        payload.nodes,
        payload.cells,
        payload.boundary_facets,
        payload.boundary_tags,
        payload.region_ids,
        payload.source_provenance,
    )

    canonical = canonicalize_mesh(payload)

    assert (
        payload.nodes,
        payload.cells,
        payload.boundary_facets,
        payload.boundary_tags,
        payload.region_ids,
        payload.source_provenance,
    ) == before
    assert canonical is not payload


def test_canonicalization_normalizes_signed_zero_without_rounding_coordinates() -> None:
    positive = square_payload()
    signed_zero = permute_payload(positive)
    signed_zero = type(signed_zero)(
        nodes=tuple(
            (-0.0 if x == 0.0 else x, -0.0 if y == 0.0 else y)
            for x, y in signed_zero.nodes
        ),
        cells=signed_zero.cells,
        cell_type=signed_zero.cell_type,
        boundary_facets=signed_zero.boundary_facets,
        boundary_tags=signed_zero.boundary_tags,
        region_ids=signed_zero.region_ids,
        unit=signed_zero.unit,
        source_provenance=signed_zero.source_provenance,
    )

    positive_mesh = canonicalize_mesh(positive)
    signed_zero_mesh = canonicalize_mesh(signed_zero)

    assert canonical_content(signed_zero_mesh) == canonical_content(positive_mesh)
    assert all(
        coordinate == 0.0 and coordinate.hex() == "0x0.0p+0"
        for node in signed_zero_mesh.nodes
        for coordinate in node
        if coordinate == 0.0
    )
