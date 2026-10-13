"""A small rectangular plate problem for the S24 ProblemSpec tests."""

from __future__ import annotations

import copy
from typing import Any

from piginot.contracts import CanonicalMesh, Unit

MESH_ID = "a" * 64
MORPH = "b" * 64


def plate_document(lifting: str = "quarter") -> dict[str, Any]:
    """A plate ``[0, width] x [0, height]`` pulled at ``x = width``."""

    document: dict[str, Any] = {
        "schema": "piginot.problem-spec-v1",
        "name": "plate-v1",
        "physics": "2d-plane-stress-finite-strain",
        "parameters": [
            {
                "name": "width",
                "role": "geometry",
                "lo": 10,
                "hi": 20,
                "unit": "mm",
                "distribution": "uniform",
            },
            {
                "name": "height",
                "role": "geometry",
                "lo": 4,
                "hi": 8,
                "unit": "mm",
                "distribution": "uniform",
            },
            {
                "name": "strain",
                "role": "load",
                "lo": 0.05,
                "hi": 0.1,
                "unit": "1",
                "distribution": "uniform",
            },
            {
                "name": "modulus",
                "role": "material",
                "lo": 760,
                "hi": 760,
                "unit": "MPa",
                "distribution": "fixed",
            },
            {
                "name": "poisson",
                "role": "material",
                "lo": 0.3,
                "hi": 0.3,
                "unit": "1",
                "distribution": "fixed",
            },
        ],
        "constants": [{"name": "unit_length", "value": 1, "unit": "mm"}],
        "constraints": [
            {
                "name": "aspect",
                "expression": "width / height",
                "lo": 1.5,
                "hi": 4.5,
                "unit": "1",
            }
        ],
        "geometry": {
            "reference_mesh_id": MESH_ID,
            "morph_map_sha256": MORPH,
            "regions": ["bottom", "left", "right", "top"],
            "frame": {"origin": ["0", "0"], "extent": ["width", "height"]},
        },
        "material": {
            "model": "compressible_neo_hookean",
            "parameters": {"elastic_modulus": "modulus", "poisson_ratio": "poisson"},
        },
        "boundary_conditions": [
            {
                "name": "left_x",
                "region": "left",
                "kind": "symmetry",
                "components": ["x"],
            },
            {
                "name": "bottom_y",
                "region": "bottom",
                "kind": "symmetry",
                "components": ["y"],
            },
            {
                "name": "pull",
                "region": "right",
                "kind": "dirichlet",
                "components": ["x"],
                "values": [{"nominal_strain": "strain"}],
            },
            {"name": "top_free", "region": "top", "kind": "traction_free"},
        ],
        "lifting": {"kind": "quarter", "displacement_scale": "strain * width"},
        "qois": [
            {
                "name": "peak",
                "field": "P_served",
                "reduction": "max-frobenius",
                "region": "top",
                "exclude_regions": ["right"],
            }
        ],
        "verification": {
            "quantile": 0.9,
            "tolerances": {
                "absolute_peak_error": 0.05,
                "p_relative_l2": 0.05,
                "u_relative_l2": 0.02,
            },
        },
        "split": {
            "pilot": 4,
            "train": 32,
            "development": 8,
            "holdout": 8,
            "reserve": 4,
        },
        "fem": {
            "material_name": "plate-material",
            "element_order": 2,
            "load_steps": 8,
            "max_dofs": 100000,
            "max_newton_iterations": 20,
            "plane_stress_max_iterations": 32,
            "solver_tolerance": 1e-10,
        },
    }
    if lifting == "adf":
        document["lifting"] = adf_lifting()
    return document


def adf_lifting(trimmed: bool = True) -> dict[str, Any]:
    def line(start: list[str], end: list[str]) -> dict[str, Any]:
        piece: dict[str, Any] = {"line": {"start": start, "end": end}}
        if not trimmed:
            piece["trimmed"] = False
        return piece

    return {
        "kind": "adf",
        "displacement_scale": "strain * width",
        "pieces": {
            "left_x": [line(["0", "height"], ["0", "0"])],
            "bottom_y": [line(["0", "0"], ["width", "0"])],
            "pull": [line(["width", "0"], ["width", "height"])],
        },
    }


def edited(
    document: dict[str, Any], path: tuple[Any, ...], value: Any
) -> dict[str, Any]:
    """A deep copy with ``document[path] = value``; ``...`` deletes the key."""

    result = copy.deepcopy(document)
    target = result
    for key in path[:-1]:
        target = target[key]
    if value is ...:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return result


def plate_mesh(
    width: float = 16.0,
    height: float = 6.0,
    columns: int = 4,
    rows: int = 2,
    mesh_id: str = MESH_ID,
) -> CanonicalMesh:
    """A structured triangle mesh of the plate with tagged boundary facets."""

    def node(column: int, row: int) -> int:
        return row * (columns + 1) + column

    nodes = tuple(
        (width * column / columns, height * row / rows)
        for row in range(rows + 1)
        for column in range(columns + 1)
    )
    cells = []
    for row in range(rows):
        for column in range(columns):
            a, b = node(column, row), node(column + 1, row)
            c, d = node(column + 1, row + 1), node(column, row + 1)
            cells.extend([(a, b, c), (a, c, d)])
    facets: list[tuple[int, int]] = []
    tags: dict[str, list[int]] = {"bottom": [], "right": [], "top": [], "left": []}

    def add(tag: str, facet: tuple[int, int]) -> None:
        tags[tag].append(len(facets))
        facets.append(facet)

    for column in range(columns):
        add("bottom", (node(column, 0), node(column + 1, 0)))
        add("top", (node(column + 1, rows), node(column, rows)))
    for row in range(rows):
        add("right", (node(columns, row), node(columns, row + 1)))
        add("left", (node(0, row + 1), node(0, row)))
    return CanonicalMesh(
        mesh_id=mesh_id,
        nodes=nodes,
        cells=tuple(cells),
        cell_type="triangle",
        boundary_facets=tuple(facets),
        semantic_tags={tag: tuple(items) for tag, items in tags.items()},
        region_ids=tuple(1 for _ in cells),
        unit=Unit.MM,
        orientation="counterclockwise",
        topology_signature="planar-cell-complex-v1:components=1;holes=0",
        quality_metrics={},
        source_provenance={},
    )
