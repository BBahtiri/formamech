"""Private geometric and counter-based primitives for S10 collocation."""

from __future__ import annotations

import hashlib
import math

from piginot.contracts import CanonicalMesh, canonical_json_bytes
from piginot.datasets.collocation_models import (
    CollocationKind,
    CollocationStream,
    SemanticCollocationQuota,
    _WeightedEntity,
)

_U53_DENOMINATOR = float(1 << 53)


def _triangle_area(
    nodes: tuple[tuple[float, float], ...], cell: tuple[int, ...]
) -> float:
    a, b, c = (nodes[index] for index in cell)
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * 0.5


def _edge_length(
    nodes: tuple[tuple[float, float], ...], edge: tuple[int, int]
) -> float:
    a, b = (nodes[index] for index in edge)
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _edge_cells(mesh: CanonicalMesh) -> dict[tuple[int, int], tuple[int, ...]]:
    incident: dict[tuple[int, int], list[int]] = {}
    for cell_index, cell in enumerate(mesh.cells):
        for first, second in (
            (cell[0], cell[1]),
            (cell[1], cell[2]),
            (cell[2], cell[0]),
        ):
            incident.setdefault(_edge_key(first, second), []).append(cell_index)
    if any(len(cell_ids) > 2 for cell_ids in incident.values()):
        raise ValueError("non-manifold mesh edges cannot be sampled")
    return {edge: tuple(cell_ids) for edge, cell_ids in incident.items()}


def _edge_key(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


def _validate_triangles(mesh: CanonicalMesh) -> None:
    if str(mesh.cell_type).casefold() != "triangle" or any(
        len(cell) != 3 for cell in mesh.cells
    ):
        raise ValueError("collocation currently requires a triangular mesh")


def _interior_entities(mesh: CanonicalMesh) -> tuple[_WeightedEntity, ...]:
    _validate_triangles(mesh)
    entities = tuple(
        _WeightedEntity(
            index,
            tuple(cell),
            (mesh.region_ids[index],),
            _triangle_area(mesh.nodes, cell),
        )
        for index, cell in enumerate(mesh.cells)
    )
    if any(entity.weight <= 0.0 for entity in entities):
        raise ValueError("interior sampling requires positive-area triangles")
    return tuple(
        sorted(
            entities, key=lambda entity: (tuple(sorted(entity.node_ids)), entity.index)
        )
    )


def _boundary_entities(mesh: CanonicalMesh) -> tuple[_WeightedEntity, ...]:
    incident = _edge_cells(mesh)
    entities: list[_WeightedEntity] = []
    for index, facet in enumerate(mesh.boundary_facets):
        edge = _edge_key(facet[0], facet[1])
        cell_ids = incident.get(edge, ())
        regions = tuple(sorted({mesh.region_ids[cell_id] for cell_id in cell_ids}))
        entities.append(
            _WeightedEntity(index, edge, regions, _edge_length(mesh.nodes, edge))
        )
    if any(entity.weight <= 0.0 for entity in entities):
        raise ValueError("boundary sampling requires positive-length facets")
    return tuple(sorted(entities, key=lambda entity: (entity.node_ids, entity.index)))


def _interface_entities(mesh: CanonicalMesh) -> tuple[_WeightedEntity, ...]:
    incident = _edge_cells(mesh)
    entities: list[_WeightedEntity] = []
    for edge, cell_ids in incident.items():
        if len(cell_ids) != 2:
            continue
        regions = tuple(sorted({mesh.region_ids[cell_id] for cell_id in cell_ids}))
        if len(regions) != 2:
            continue
        entities.append(
            _WeightedEntity(
                len(entities), edge, regions, _edge_length(mesh.nodes, edge)
            )
        )
    if any(entity.weight <= 0.0 for entity in entities):
        raise ValueError("interface sampling requires positive-length edges")
    return tuple(sorted(entities, key=lambda entity: entity.node_ids))


def _quota_entities(
    mesh: CanonicalMesh,
    quota: SemanticCollocationQuota,
    interior: tuple[_WeightedEntity, ...],
    boundary: tuple[_WeightedEntity, ...],
) -> tuple[_WeightedEntity, ...]:
    if quota.kind is CollocationKind.BOUNDARY:
        tag = quota.semantic_tag
        if tag is None:
            raise ValueError("boundary QoI quota requires a semantic tag")
        facet_ids = set(mesh.semantic_tags[tag])
        result = tuple(entity for entity in boundary if entity.index in facet_ids)
    else:
        result = tuple(
            entity for entity in interior if entity.region_ids == (quota.region_id,)
        )
    _require_candidates(result, quota.count, f"QoI quota {quota.quota_id!r}")
    return result


def _require_candidates(
    entities: tuple[_WeightedEntity, ...], count: int, label: str
) -> None:
    if count and not entities:
        raise ValueError(
            f"requested {label} collocation but the mesh has no {label} entities"
        )


def _u53(
    stream: CollocationStream,
    kind: CollocationKind,
    counter: int,
    lane: int,
    quota_id: str | None = None,
) -> float:
    frame = canonical_json_bytes(
        {
            "algorithm_id": stream.algorithm_id,
            "schema_version": stream.schema_version,
            "case_id": stream.case_id,
            "mesh_content_hash": stream.mesh_content_hash,
            "semantic_spec_hash": stream.semantic_spec_hash,
            "quota_policy_hash": stream.quota_policy_hash,
            "quota_id": quota_id,
            "kind": kind.value,
            "counter": counter,
            "lane": lane,
        }
    )
    digest = hashlib.blake2b(
        frame,
        key=bytes.fromhex(stream.replay_seed_hex),
        digest_size=8,
        person=b"piginot-s10-u53",
    ).digest()
    return (int.from_bytes(digest, "big") >> 11) / _U53_DENOMINATOR


def _choose(
    stream: CollocationStream,
    kind: CollocationKind,
    counter: int,
    entities: tuple[_WeightedEntity, ...],
    quota_id: str | None = None,
) -> _WeightedEntity:
    target = _u53(stream, kind, counter, 0, quota_id) * sum(
        entity.weight for entity in entities
    )
    cumulative = 0.0
    for entity in entities:
        cumulative += entity.weight
        if target < cumulative:
            return entity
    return entities[-1]


def _sample_triangle(
    stream: CollocationStream,
    counter: int,
    mesh: CanonicalMesh,
    entity: _WeightedEntity,
    quota_id: str | None = None,
) -> tuple[float, float]:
    a, b, c = (mesh.nodes[index] for index in entity.node_ids)
    root = math.sqrt(_u53(stream, CollocationKind.INTERIOR, counter, 1, quota_id))
    second = _u53(stream, CollocationKind.INTERIOR, counter, 2, quota_id)
    weights = (1.0 - root, root * (1.0 - second), root * second)
    return (
        weights[0] * a[0] + weights[1] * b[0] + weights[2] * c[0],
        weights[0] * a[1] + weights[1] * b[1] + weights[2] * c[1],
    )


def _sample_edge(
    stream: CollocationStream,
    kind: CollocationKind,
    counter: int,
    mesh: CanonicalMesh,
    entity: _WeightedEntity,
    quota_id: str | None = None,
) -> tuple[float, float]:
    a, b = (mesh.nodes[index] for index in entity.node_ids)
    fraction = _u53(stream, kind, counter, 1, quota_id)
    return (
        a[0] + fraction * (b[0] - a[0]),
        a[1] + fraction * (b[1] - a[1]),
    )
