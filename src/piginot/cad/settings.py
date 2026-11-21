"""Pure CAD physics settings and straight-boundary lifting declarations."""

from __future__ import annotations

import math
from typing import Any

from piginot.cad.adapter import morph_map_sha256
from piginot.cad.resource import GeometryResource
from piginot.compiler._validation import array, integer, object_fields
from piginot.compiler.config_types import TrainingConfig
from piginot.contracts import CanonicalMesh, ProblemSpec, strict_json_loads

SCHEMA = "piginot.cad-settings-v1"
SCHEMA_V2 = "formamech.cad-settings-v2"
SUPPORTED_SCHEMAS = (SCHEMA, SCHEMA_V2)
REQUIRED = (
    "schema",
    "name",
    "parameters",
    "material",
    "boundary_conditions",
    "qois",
    "verification",
    "displacement_scale",
    "training",
    "sampling",
)
OPTIONAL = ("constants", "constraints", "fem", "model", "objective", "reference")
TRAINING_FIELDS = (
    "optimizer_steps",
    "case_batch_size",
    "checkpoint_interval",
    "learning_rate",
    "final_learning_rate",
    "constant_steps",
    "seed",
)
FEM_DEFAULTS = {
    "material_name": "cad-material",
    "element_order": 2,
    "load_steps": 64,
    "max_dofs": 100000,
    "max_newton_iterations": 20,
    "plane_stress_max_iterations": 32,
    "solver_tolerance": 1e-10,
}


def load_settings(payload: bytes) -> dict[str, Any]:
    result = object_fields(
        strict_json_loads(payload), "CAD settings", REQUIRED, OPTIONAL
    )
    if result["schema"] not in SUPPORTED_SCHEMAS:
        raise ValueError("unsupported CAD settings schema")
    sampling = object_fields(
        result["sampling"], "sampling", ("fit_cases", "screen_cases")
    )
    integer(sampling["fit_cases"], "sampling.fit_cases", minimum=2)
    integer(sampling["screen_cases"], "sampling.screen_cases")
    TrainingConfig(**object_fields(result["training"], "training", TRAINING_FIELDS))
    return result


def _chain_nodes(mesh: CanonicalMesh, region: str) -> list[int]:
    facets = tuple(mesh.boundary_facets[index] for index in mesh.semantic_tags[region])
    successors = dict(facets)
    ends = {end for _, end in facets}
    starts = set(successors) - ends
    if len(successors) != len(facets) or len(ends) != len(facets) or len(starts) != 1:
        raise ValueError(
            f"{region}: essential line group must be a connected unbranched chain"
        )
    start = next(iter(starts))
    chain = [start]
    while chain[-1] in successors and len(chain) <= len(facets):
        chain.append(successors[chain[-1]])
    if len(chain) != len(facets) + 1 or chain[-1] in successors:
        raise ValueError(f"{region}: essential line group must be connected")
    return chain


def _line_piece(mesh: CanonicalMesh, region: str) -> dict[str, Any]:
    chain = _chain_nodes(mesh, region)
    first, last = mesh.nodes[chain[0]], mesh.nodes[chain[-1]]
    dx, dy = last[0] - first[0], last[1] - first[1]
    length = math.hypot(dx, dy)
    if length == 0 or any(
        abs(
            dx * (mesh.nodes[node][1] - first[1])
            - dy * (mesh.nodes[node][0] - first[0])
        )
        > 1e-12 * max(1.0, length * length)
        for node in chain
    ):
        raise ValueError(f"{region}: essential boundary must be a straight line")
    return {"line": {"start": [str(x) for x in first], "end": [str(x) for x in last]}}


def _arc_piece(mesh: CanonicalMesh, region: str) -> dict[str, Any]:
    """Fit an exact circular arc piece for ADF lifting on a curved CAD boundary."""
    chain = _chain_nodes(mesh, region)
    if len(chain) < 3:
        raise ValueError(f"{region}: circular arc essential group needs >= 3 nodes")
    p1 = mesh.nodes[chain[0]]
    p2 = mesh.nodes[chain[len(chain) // 2]]
    p3 = mesh.nodes[chain[-1]]
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) <= 1e-12:
        raise ValueError(f"{region}: arc nodes are collinear; cannot determine center")
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    radius = math.hypot(ax - ux, ay - uy)
    if radius <= 0.0 or any(
        abs(math.hypot(mesh.nodes[n][0] - ux, mesh.nodes[n][1] - uy) - radius)
        > 1e-5 * max(1.0, radius)
        for n in chain
    ):
        raise ValueError(f"{region}: essential arc nodes do not lie on a single circle")
    return {
        "arc": {
            "start": [str(x) for x in p1],
            "end": [str(x) for x in p3],
            "center": [str(ux), str(uy)],
        }
    }


def _boundaries(
    resource: GeometryResource,
    mesh: CanonicalMesh,
    supplied: object,
    *,
    schema: str = SCHEMA,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    conditions = [
        object_fields(
            item,
            "boundary condition",
            ("name", "region", "kind"),
            ("components", "values"),
        )
        for item in array(supplied, "boundary_conditions")
    ]
    regions = {condition["region"] for condition in conditions}
    if regions != set(mesh.semantic_tags):
        raise ValueError(
            "boundary conditions must explicitly cover exactly the prepared boundary groups"
        )
    pieces = {}
    for condition in conditions:
        kind, region = condition["kind"], condition["region"]
        if kind not in ("dirichlet", "symmetry", "traction_free"):
            raise ValueError(
                "CAD preparation supports Dirichlet, symmetry and traction_free; Neumann is unsupported"
            )
        if kind != "traction_free":
            bkind = resource.boundary_kinds[region]
            if bkind == "line":
                pieces[condition["name"]] = [_line_piece(mesh, region)]
            elif schema == SCHEMA_V2 and bkind == "arc":
                pieces[condition["name"]] = [_arc_piece(mesh, region)]
            else:
                raise ValueError(
                    "essential CAD boundaries must be straight line groups"
                )
    return conditions, pieces


def _constants(mesh: CanonicalMesh, settings: dict[str, Any]) -> list[dict[str, Any]]:
    origin = tuple(min(point[axis] for point in mesh.nodes) for axis in (0, 1))
    extent = tuple(
        max(point[axis] for point in mesh.nodes) - origin[axis] for axis in (0, 1)
    )
    if any(number <= 0 for number in extent):
        raise ValueError("CAD mesh frame must have positive extent on both axes")
    automatic = dict(
        zip(
            ("cad_origin_x", "cad_origin_y", "cad_extent_x", "cad_extent_y"),
            (*origin, *extent),
            strict=True,
        )
    )
    supplied = [
        object_fields(item, "constant", ("name", "value", "unit"))
        for item in array(settings.get("constants", []), "constants")
    ]
    parameters = array(settings["parameters"], "parameters")
    if set(automatic) & {
        item.get("name") for item in (*supplied, *parameters) if isinstance(item, dict)
    }:
        raise ValueError(
            "parameter/constant name collides with a reserved CAD frame constant"
        )
    return [
        *supplied,
        *(
            {"name": name, "value": value, "unit": "mm"}
            for name, value in automatic.items()
        ),
    ]


def build_problem(resource: GeometryResource, settings: dict[str, Any]) -> ProblemSpec:
    mesh = resource.mesh(1)
    conditions, pieces = _boundaries(
        resource,
        mesh,
        settings["boundary_conditions"],
        schema=str(settings.get("schema", SCHEMA)),
    )
    sampling = settings["sampling"]
    fem = object_fields(settings.get("fem", {}), "fem", (), tuple(FEM_DEFAULTS))
    spec = ProblemSpec.from_document(
        {
            "schema": "piginot.problem-spec-v1",
            "name": settings["name"],
            "physics": "2d-plane-stress-finite-strain",
            "parameters": settings["parameters"],
            "constants": _constants(mesh, settings),
            "constraints": settings.get("constraints", []),
            "geometry": {
                "reference_mesh_id": mesh.mesh_id,
                "morph_map_sha256": morph_map_sha256(),
                "regions": sorted(mesh.semantic_tags),
                "frame": {
                    "origin": ["cad_origin_x", "cad_origin_y"],
                    "extent": ["cad_extent_x", "cad_extent_y"],
                },
            },
            "material": settings["material"],
            "boundary_conditions": conditions,
            "lifting": {
                "kind": "adf",
                "displacement_scale": settings["displacement_scale"],
                "pieces": pieces,
            },
            "qois": settings["qois"],
            "verification": settings["verification"],
            "split": {
                "pilot": 0,
                "train": sampling["fit_cases"] + sampling["screen_cases"],
                "development": 1,
                "holdout": 1,
                "reserve": 0,
            },
            "fem": {**FEM_DEFAULTS, **fem},
        }
    )
    if any(parameter.role not in ("material", "load") for parameter in spec.parameters):
        raise ValueError("fixed CAD parameters must have material or load roles")
    if not any(parameter.lo < parameter.hi for parameter in spec.parameters):
        raise ValueError(
            "CAD sampling requires at least one varying material or load range"
        )
    spec.check_mesh(mesh)
    return spec
