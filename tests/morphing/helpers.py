from __future__ import annotations

from piginot.contracts import GeometryFamilySpec, ParameterSpec, Quantity, Unit
from piginot.geometry import CellType, MeshPayload
from piginot.morphing.models import FFDControlPoint, MorphValidityCriteria


def rectangle_payload(
    *,
    width: float = 4.0,
    height: float = 2.0,
    nx: int = 4,
    ny: int = 2,
) -> MeshPayload:
    nodes = tuple(
        (width * ix / nx, height * iy / ny)
        for iy in range(ny + 1)
        for ix in range(nx + 1)
    )

    def node_id(ix: int, iy: int) -> int:
        return iy * (nx + 1) + ix

    cells: list[tuple[int, int, int]] = []
    for iy in range(ny):
        for ix in range(nx):
            lower_left = node_id(ix, iy)
            lower_right = node_id(ix + 1, iy)
            upper_left = node_id(ix, iy + 1)
            upper_right = node_id(ix + 1, iy + 1)
            cells.append((lower_left, lower_right, upper_right))
            cells.append((lower_left, upper_right, upper_left))

    boundary_facets: list[tuple[int, int]] = []
    bottom: list[int] = []
    right: list[int] = []
    top: list[int] = []
    left: list[int] = []
    for ix in range(nx):
        bottom.append(len(boundary_facets))
        boundary_facets.append((node_id(ix, 0), node_id(ix + 1, 0)))
    for iy in range(ny):
        right.append(len(boundary_facets))
        boundary_facets.append((node_id(nx, iy), node_id(nx, iy + 1)))
    for ix in reversed(range(nx)):
        top.append(len(boundary_facets))
        boundary_facets.append((node_id(ix + 1, ny), node_id(ix, ny)))
    for iy in reversed(range(ny)):
        left.append(len(boundary_facets))
        boundary_facets.append((node_id(0, iy + 1), node_id(0, iy)))

    return MeshPayload(
        nodes=nodes,
        cells=tuple(cells),
        cell_type=CellType.TRIANGLE,
        boundary_facets=tuple(boundary_facets),
        boundary_tags={
            "bottom": tuple(bottom),
            "right": tuple(right),
            "top": tuple(top),
            "left": tuple(left),
        },
        region_ids=tuple(7 for _ in cells),
        unit=Unit.MM,
        source_provenance={"adapter": "fixture", "source": "s6-rectangle"},
    )


def length(value: float):
    return Quantity(value, Unit.MM).to_canonical()


def stretch_family(reference, lower: float = -5.0, upper: float = 1.0):
    return GeometryFamilySpec(
        name="stretch_family",
        parameters=(ParameterSpec("stretch", length(lower), length(upper)),),
        topology_signature=reference.topology_signature,
        semantic_tags=tuple(reference.semantic_tags),
    )


def stretch_controls() -> tuple[FFDControlPoint, ...]:
    return (
        FFDControlPoint(None, None),
        FFDControlPoint(None, None),
        FFDControlPoint("stretch", None),
        FFDControlPoint("stretch", None),
    )


def gate_criteria() -> MorphValidityCriteria:
    return MorphValidityCriteria(
        jacobian_margin=0.01,
        minimum_feature_size_mm=0.02,
        minimum_angle_degrees=5.0,
        maximum_aspect_ratio=20.0,
    )
