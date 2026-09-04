"""Deterministic, traceable S10 collocation sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    LoadSpec,
    canonical_json_bytes,
    hash_spec,
)
from piginot.datasets.collocation_models import (
    MAX_COLLOCATION_POINTS,
    SAMPLER_ALGORITHM_ID,
    SEED_COMMITMENT_ALGORITHM_ID,
    STREAM_SCHEMA_VERSION,
    CollocationBatch,
    CollocationKind,
    CollocationPoint,
    CollocationStream,
    SemanticBCClass,
    SemanticCollocationQuota,
    _decode_seed,
    _derived_public_seed_hex,
    _require_digest,
    _seed_commitment,
    _WeightedEntity,
)
from piginot.datasets.collocation_sampling import (
    _boundary_entities,
    _choose,
    _interface_entities,
    _interior_entities,
    _quota_entities,
    _require_candidates,
    _sample_edge,
    _sample_triangle,
)
from piginot.datasets.snapshots import DatasetCaseRecord, RoleDatasetSnapshot
from piginot.fem import FemRequest
from piginot.store import ObjectStore

_DIRICHLET_KINDS = frozenset({"dirichlet", "essential", "symmetry"})
_NEUMANN_KINDS = frozenset({"neumann", "traction"})
_FREE_KINDS = frozenset({"free"})
_MIXED_KINDS = frozenset({"mixed"})


@dataclass(frozen=True, slots=True)
class StoredCollocationReference:
    """Content identities retained after verified collocation persistence."""

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    snapshot_content_hash: str
    sampler_stream_hash: str
    batch_content_hash: str
    store_content_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "snapshot_content_hash",
            "sampler_stream_hash",
            "batch_content_hash",
            "store_content_hash",
        ):
            _require_digest(getattr(self, field_name), field_name)


def classify_boundary_facet(
    mesh: CanonicalMesh,
    facet_index: int,
    boundary_conditions: Sequence[BoundaryConditionSpec],
    loads: Sequence[LoadSpec],
) -> tuple[SemanticBCClass, tuple[str, ...]]:
    """Classify one boundary facet using its semantic tags and physics specs."""

    if isinstance(facet_index, bool) or not isinstance(facet_index, int):
        raise ValueError("facet_index must be an integer")
    if facet_index < 0 or facet_index >= len(mesh.boundary_facets):
        raise ValueError("facet_index is outside the boundary facet table")
    conditions, normalized_loads = _normalized_specs(boundary_conditions, loads)
    _validate_semantic_targets(mesh, conditions, normalized_loads)
    tags = tuple(
        sorted(
            tag for tag, facets in mesh.semantic_tags.items() if facet_index in facets
        )
    )
    condition_kinds = {
        condition.kind.casefold()
        for condition in conditions
        if condition.boundary_tag in tags
    }
    load_kinds = {
        load.kind.casefold() for load in normalized_loads if load.boundary_tag in tags
    }
    has_dirichlet = bool(condition_kinds & (_DIRICHLET_KINDS | _MIXED_KINDS))
    has_neumann = bool(
        condition_kinds & (_NEUMANN_KINDS | _MIXED_KINDS)
        or load_kinds & (_NEUMANN_KINDS | _MIXED_KINDS)
    )
    if has_dirichlet and has_neumann:
        return SemanticBCClass.MIXED, tags
    if has_dirichlet:
        return SemanticBCClass.DIRICHLET, tags
    if has_neumann:
        return SemanticBCClass.NEUMANN, tags
    return SemanticBCClass.FREE, tags


def sample_collocation(
    snapshot: RoleDatasetSnapshot,
    case_id: str,
    mesh: CanonicalMesh,
    boundary_conditions: Sequence[BoundaryConditionSpec],
    loads: Sequence[LoadSpec],
    *,
    stream_index: int,
    interior_count: int,
    boundary_count: int,
    interface_count: int,
    counter_start: int = 0,
    qoi_quotas: Sequence[SemanticCollocationQuota] = (),
) -> CollocationBatch:
    """Draw deterministic physical-measure-weighted collocation points."""

    case = _validated_snapshot_case(snapshot, case_id, mesh)
    conditions, normalized_loads = _normalized_specs(boundary_conditions, loads)
    normalized_quotas = _normalized_quotas(qoi_quotas)
    _validate_semantic_targets(mesh, conditions, normalized_loads)
    _validate_quotas(mesh, normalized_quotas)
    stream = _make_stream(
        case,
        snapshot,
        mesh,
        conditions,
        normalized_loads,
        stream_index=stream_index,
        interior_count=interior_count,
        boundary_count=boundary_count,
        interface_count=interface_count,
        counter_start=counter_start,
        qoi_quotas=normalized_quotas,
    )
    interior = _interior_entities(mesh)
    boundary = _boundary_entities(mesh)
    interface = _interface_entities(mesh)
    _require_candidates(interior, stream.interior_count, "interior")
    _require_candidates(boundary, stream.boundary_count, "boundary")
    _require_candidates(interface, stream.interface_count, "interface")

    stream_hash = stream.stream_hash
    points: list[CollocationPoint] = []
    counter = stream.counter_start
    for _ in range(stream.interior_count):
        entity = _choose(stream, CollocationKind.INTERIOR, counter, interior)
        point = _sample_triangle(stream, counter, mesh, entity)
        points.append(
            _point(
                case,
                stream_hash,
                CollocationKind.INTERIOR,
                SemanticBCClass.INTERIOR,
                counter,
                entity,
                point,
            )
        )
        counter += 1
    for _ in range(stream.boundary_count):
        entity = _choose(stream, CollocationKind.BOUNDARY, counter, boundary)
        point = _sample_edge(stream, CollocationKind.BOUNDARY, counter, mesh, entity)
        semantic_class, tags = classify_boundary_facet(
            mesh, entity.index, conditions, normalized_loads
        )
        points.append(
            _point(
                case,
                stream_hash,
                CollocationKind.BOUNDARY,
                semantic_class,
                counter,
                entity,
                point,
                tags,
            )
        )
        counter += 1
    for _ in range(stream.interface_count):
        entity = _choose(stream, CollocationKind.INTERFACE, counter, interface)
        point = _sample_edge(stream, CollocationKind.INTERFACE, counter, mesh, entity)
        points.append(
            _point(
                case,
                stream_hash,
                CollocationKind.INTERFACE,
                SemanticBCClass.INTERFACE,
                counter,
                entity,
                point,
            )
        )
        counter += 1
    for quota in stream.qoi_quotas:
        entities = _quota_entities(mesh, quota, interior, boundary)
        for _ in range(quota.count):
            entity = _choose(stream, quota.kind, counter, entities, quota.quota_id)
            if quota.kind is CollocationKind.BOUNDARY:
                point = _sample_edge(
                    stream,
                    quota.kind,
                    counter,
                    mesh,
                    entity,
                    quota.quota_id,
                )
                semantic_class, tags = classify_boundary_facet(
                    mesh, entity.index, conditions, normalized_loads
                )
            else:
                point = _sample_triangle(
                    stream,
                    counter,
                    mesh,
                    entity,
                    quota.quota_id,
                )
                semantic_class, tags = SemanticBCClass.INTERIOR, ()
            points.append(
                _point(
                    case,
                    stream_hash,
                    quota.kind,
                    semantic_class,
                    counter,
                    entity,
                    point,
                    tags,
                    quota.quota_id,
                )
            )
            counter += 1
    return CollocationBatch(stream, stream_hash, tuple(points))


def sample_request_collocation(
    snapshot: RoleDatasetSnapshot,
    case_id: str,
    request: FemRequest,
    *,
    stream_index: int,
    interior_count: int,
    boundary_count: int,
    interface_count: int,
    counter_start: int = 0,
    qoi_quotas: Sequence[SemanticCollocationQuota],
) -> CollocationBatch:
    """Sample a FEM request while guaranteeing coverage of every declared QoI tag."""

    _validated_request_case(snapshot, case_id, request)
    normalized_quotas = _normalized_quotas(qoi_quotas)
    _validate_request_qoi_quotas(request, normalized_quotas)
    return sample_collocation(
        snapshot,
        case_id,
        request.mesh,
        request.boundary_conditions,
        request.loads,
        stream_index=stream_index,
        interior_count=interior_count,
        boundary_count=boundary_count,
        interface_count=interface_count,
        counter_start=counter_start,
        qoi_quotas=normalized_quotas,
    )


def _validate_request_qoi_quotas(
    request: FemRequest,
    quotas: tuple[SemanticCollocationQuota, ...],
) -> None:
    covered_tags = {
        quota.semantic_tag for quota in quotas if quota.kind is CollocationKind.BOUNDARY
    }
    missing_tags = tuple(
        sorted({qoi.region_tag for qoi in request.qois} - covered_tags)
    )
    if missing_tags:
        raise ValueError(
            "every declared QoI region requires a measure-independent boundary "
            f"quota; missing QoI region tag(s): {', '.join(missing_tags)}"
        )


def replay_collocation(
    stream: CollocationStream,
    snapshot: RoleDatasetSnapshot,
    case_id: str,
    mesh: CanonicalMesh,
    boundary_conditions: Sequence[BoundaryConditionSpec],
    loads: Sequence[LoadSpec],
) -> CollocationBatch:
    """Recreate a batch solely from its public stream and bound public inputs."""

    if type(stream) is not CollocationStream:
        raise TypeError("stream must be an exact CollocationStream")
    case = _validated_snapshot_case(snapshot, case_id, mesh)
    conditions, normalized_loads = _normalized_specs(boundary_conditions, loads)
    if (
        snapshot.campaign_hash != stream.campaign_hash
        or snapshot.public_case_manifest_hash != stream.public_case_manifest_hash
        or snapshot.content_hash != stream.snapshot_content_hash
    ):
        raise ValueError("snapshot does not match the recorded collocation stream")
    if (
        case.case_id != stream.case_id
        or case.leakage_group_id != stream.leakage_group_id
        or case.role is not stream.role
    ):
        raise ValueError("case does not match the recorded collocation stream")
    if (
        mesh.mesh_id != stream.mesh_id
        or hash_spec(mesh, "content") != stream.mesh_content_hash
    ):
        raise ValueError("mesh does not match the recorded collocation stream")
    if _semantic_spec_hash(conditions, normalized_loads) != stream.semantic_spec_hash:
        raise ValueError("semantic specifications do not match the recorded stream")
    return sample_collocation(
        snapshot,
        case_id,
        mesh,
        conditions,
        normalized_loads,
        stream_index=stream.stream_index,
        interior_count=stream.interior_count,
        boundary_count=stream.boundary_count,
        interface_count=stream.interface_count,
        counter_start=stream.counter_start,
        qoi_quotas=stream.qoi_quotas,
    )


def verify_collocation_batch(
    batch: CollocationBatch,
    snapshot: RoleDatasetSnapshot,
    case_id: str,
    mesh: CanonicalMesh,
    boundary_conditions: Sequence[BoundaryConditionSpec],
    loads: Sequence[LoadSpec],
) -> CollocationBatch:
    """Accept a batch only when deterministic replay reproduces every byte."""

    if type(batch) is not CollocationBatch:
        raise TypeError("batch must be an exact CollocationBatch")
    replayed = replay_collocation(
        batch.stream,
        snapshot,
        case_id,
        mesh,
        boundary_conditions,
        loads,
    )
    if replayed.to_bytes() != batch.to_bytes():
        raise ValueError("collocation batch does not match deterministic replay")
    return batch


def verify_request_collocation_batch(
    batch: CollocationBatch,
    snapshot: RoleDatasetSnapshot,
    case_id: str,
    request: FemRequest,
) -> CollocationBatch:
    """Replay a batch only against the exact snapshot-bound FEM request."""

    if type(batch) is not CollocationBatch:
        raise TypeError("batch must be an exact CollocationBatch")
    _validated_request_case(snapshot, case_id, request)
    _validate_request_qoi_quotas(request, batch.stream.qoi_quotas)
    return verify_collocation_batch(
        batch,
        snapshot,
        case_id,
        request.mesh,
        request.boundary_conditions,
        request.loads,
    )


def persist_collocation_batch(
    batch: CollocationBatch,
    snapshot: RoleDatasetSnapshot,
    case_id: str,
    request: FemRequest,
    store: ObjectStore,
) -> StoredCollocationReference:
    """Persist only a request-bound, replay-verified collocation batch."""

    if type(store) is not ObjectStore:
        raise TypeError("store must be an exact ObjectStore")
    verified = verify_request_collocation_batch(
        batch,
        snapshot,
        case_id,
        request,
    )
    store_hash = store.put(verified.to_bytes())
    return StoredCollocationReference(
        snapshot_content_hash=snapshot.content_hash,
        sampler_stream_hash=verified.sampler_stream_hash,
        batch_content_hash=verified.content_hash,
        store_content_hash=store_hash,
    )


def _validated_snapshot_case(
    snapshot: object,
    case_id: object,
    mesh: object,
) -> DatasetCaseRecord:
    if type(snapshot) is not RoleDatasetSnapshot:
        raise TypeError("collocation requires an exact RoleDatasetSnapshot")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be non-empty text")
    if type(mesh) is not CanonicalMesh:
        raise TypeError("mesh must be an exact CanonicalMesh")
    matches = tuple(case for case in snapshot.cases if case.case_id == case_id)
    if len(matches) != 1:
        raise ValueError("case_id must identify exactly one snapshot case")
    case = matches[0]
    if not case.success or case.sample_trace is None:
        raise ValueError("collocation requires a successful snapshot case")
    if case.mesh_id != mesh.mesh_id or case.mesh_content_hash != hash_spec(
        mesh, "content"
    ):
        raise ValueError("mesh does not match the successful snapshot case")
    return case


def _validated_request_case(
    snapshot: RoleDatasetSnapshot,
    case_id: str,
    request: FemRequest,
) -> DatasetCaseRecord:
    if type(request) is not FemRequest:
        raise TypeError("request collocation requires an exact FemRequest")
    if request.request_id != case_id:
        raise ValueError("FEM request identity does not match the snapshot case")
    case = _validated_snapshot_case(snapshot, case_id, request.mesh)
    if (
        request.recipe_hash != case.request_recipe_hash
        or hash_spec(request, "content") != case.request_content_hash
    ):
        raise ValueError("FEM request does not match the successful snapshot case")
    return case


def _normalized_specs(
    boundary_conditions: Sequence[BoundaryConditionSpec],
    loads: Sequence[LoadSpec],
) -> tuple[tuple[BoundaryConditionSpec, ...], tuple[LoadSpec, ...]]:
    conditions = tuple(boundary_conditions)
    normalized_loads = tuple(loads)
    if any(type(value) is not BoundaryConditionSpec for value in conditions):
        raise TypeError("boundary_conditions must contain BoundaryConditionSpec values")
    if any(type(value) is not LoadSpec for value in normalized_loads):
        raise TypeError("loads must contain LoadSpec values")
    return (
        tuple(sorted(conditions, key=canonical_json_bytes)),
        tuple(sorted(normalized_loads, key=canonical_json_bytes)),
    )


def _normalized_quotas(
    quotas: Sequence[SemanticCollocationQuota],
) -> tuple[SemanticCollocationQuota, ...]:
    result = tuple(quotas)
    if any(type(quota) is not SemanticCollocationQuota for quota in result):
        raise TypeError("qoi_quotas must contain SemanticCollocationQuota values")
    quota_ids = tuple(quota.quota_id for quota in result)
    if len(quota_ids) != len(set(quota_ids)):
        raise ValueError("qoi_quotas must have unique quota IDs")
    return tuple(sorted(result, key=lambda quota: (quota.kind.value, quota.quota_id)))


def _validate_quotas(
    mesh: CanonicalMesh,
    quotas: tuple[SemanticCollocationQuota, ...],
) -> None:
    for quota in quotas:
        if quota.kind is CollocationKind.BOUNDARY:
            tag = quota.semantic_tag
            if tag is None or tag not in mesh.semantic_tags:
                raise ValueError(f"QoI quota targets unknown semantic tag {tag!r}")
            if not mesh.semantic_tags[tag]:
                raise ValueError("QoI quota semantic tag has no boundary facets")
        elif quota.region_id not in mesh.region_ids:
            raise ValueError(f"QoI quota targets unknown region_id {quota.region_id!r}")


def _validate_semantic_targets(
    mesh: CanonicalMesh,
    conditions: tuple[BoundaryConditionSpec, ...],
    loads: tuple[LoadSpec, ...],
) -> None:
    tags = set(mesh.semantic_tags)
    for condition in conditions:
        kind = condition.kind.casefold()
        if kind not in _DIRICHLET_KINDS | _NEUMANN_KINDS | _FREE_KINDS | _MIXED_KINDS:
            raise ValueError(f"unsupported boundary semantic kind {condition.kind!r}")
        if condition.boundary_tag not in tags:
            raise ValueError(
                f"boundary condition targets unknown semantic tag {condition.boundary_tag!r}"
            )
    for load in loads:
        if load.boundary_tag is None:
            continue
        kind = load.kind.casefold()
        if kind not in _NEUMANN_KINDS | _MIXED_KINDS:
            raise ValueError(f"unsupported boundary load semantic kind {load.kind!r}")
        if load.boundary_tag not in tags:
            raise ValueError(f"load targets unknown semantic tag {load.boundary_tag!r}")


def _semantic_spec_hash(
    conditions: tuple[BoundaryConditionSpec, ...],
    loads: tuple[LoadSpec, ...],
) -> str:
    return hash_spec(
        {"boundary_conditions": conditions, "loads": loads},
        "content",
    )


def _make_stream(
    case: DatasetCaseRecord,
    snapshot: RoleDatasetSnapshot,
    mesh: CanonicalMesh,
    conditions: tuple[BoundaryConditionSpec, ...],
    loads: tuple[LoadSpec, ...],
    *,
    stream_index: int,
    interior_count: int,
    boundary_count: int,
    interface_count: int,
    counter_start: int,
    qoi_quotas: tuple[SemanticCollocationQuota, ...],
) -> CollocationStream:
    total = _validated_total(
        interior_count,
        boundary_count,
        interface_count,
        sum(quota.count for quota in qoi_quotas),
    )
    if (
        isinstance(counter_start, bool)
        or not isinstance(counter_start, int)
        or counter_start < 0
    ):
        raise ValueError("counter_start must be a non-negative integer")
    if (
        isinstance(stream_index, bool)
        or not isinstance(stream_index, int)
        or stream_index < 0
    ):
        raise ValueError("stream_index must be a public non-negative integer")
    mesh_content_hash = hash_spec(mesh, "content")
    semantic_spec_hash = _semantic_spec_hash(conditions, loads)
    quota_policy_hash = hash_spec(qoi_quotas, "content")
    replay_seed_hex = _derived_public_seed_hex(
        algorithm_id=SAMPLER_ALGORITHM_ID,
        seed_commitment_algorithm_id=SEED_COMMITMENT_ALGORITHM_ID,
        schema_version=STREAM_SCHEMA_VERSION,
        campaign_hash=snapshot.campaign_hash,
        public_case_manifest_hash=snapshot.public_case_manifest_hash,
        snapshot_content_hash=snapshot.content_hash,
        case_id=case.case_id,
        leakage_group_id=case.leakage_group_id,
        role=case.role,
        mesh_id=mesh.mesh_id,
        mesh_content_hash=mesh_content_hash,
        semantic_spec_hash=semantic_spec_hash,
        stream_index=stream_index,
        counter_start=counter_start,
        interior_count=interior_count,
        boundary_count=boundary_count,
        interface_count=interface_count,
        qoi_quotas=qoi_quotas,
        quota_policy_hash=quota_policy_hash,
    )
    seed = _decode_seed(replay_seed_hex)
    return CollocationStream(
        algorithm_id=SAMPLER_ALGORITHM_ID,
        seed_commitment_algorithm_id=SEED_COMMITMENT_ALGORITHM_ID,
        schema_version=STREAM_SCHEMA_VERSION,
        replay_seed_hex=replay_seed_hex,
        seed_commitment=_seed_commitment(seed),
        campaign_hash=snapshot.campaign_hash,
        public_case_manifest_hash=snapshot.public_case_manifest_hash,
        snapshot_content_hash=snapshot.content_hash,
        case_id=case.case_id,
        leakage_group_id=case.leakage_group_id,
        role=case.role,
        mesh_id=mesh.mesh_id,
        mesh_content_hash=mesh_content_hash,
        semantic_spec_hash=semantic_spec_hash,
        stream_index=stream_index,
        counter_start=counter_start,
        counter_stop=counter_start + total,
        interior_count=interior_count,
        boundary_count=boundary_count,
        interface_count=interface_count,
        qoi_quotas=qoi_quotas,
        quota_policy_hash=quota_policy_hash,
    )


def _validated_total(
    interior: int,
    boundary: int,
    interface: int,
    quota: int,
) -> int:
    counts: tuple[int, ...] = (interior, boundary, interface, quota)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts
    ):
        raise ValueError("collocation counts must be non-negative integers")
    total = sum(counts)
    if total <= 0:
        raise ValueError("at least one collocation point must be requested")
    if total > MAX_COLLOCATION_POINTS:
        raise ValueError(
            f"collocation population exceeds maximum {MAX_COLLOCATION_POINTS}"
        )
    return total


def _point(
    case: DatasetCaseRecord,
    stream_hash: str,
    kind: CollocationKind,
    semantic_class: SemanticBCClass,
    counter: int,
    entity: _WeightedEntity,
    coordinates: tuple[float, float],
    tags: tuple[str, ...] = (),
    qoi_quota_id: str | None = None,
) -> CollocationPoint:
    return CollocationPoint(
        sample_id=f"{case.case_id}:collocation:{counter:016x}",
        case_id=case.case_id,
        leakage_group_id=case.leakage_group_id,
        role=case.role,
        kind=kind,
        semantic_class=semantic_class,
        coordinates=coordinates,
        source_entity_index=entity.index,
        source_node_ids=entity.node_ids,
        region_ids=entity.region_ids,
        semantic_tags=tags,
        qoi_quota_id=qoi_quota_id,
        counter=counter,
        sampler_stream_hash=stream_hash,
    )


__all__ = (
    "CollocationBatch",
    "CollocationKind",
    "CollocationPoint",
    "CollocationStream",
    "MAX_COLLOCATION_POINTS",
    "SAMPLER_ALGORITHM_ID",
    "SEED_COMMITMENT_ALGORITHM_ID",
    "STREAM_SCHEMA_VERSION",
    "SemanticBCClass",
    "SemanticCollocationQuota",
    "StoredCollocationReference",
    "classify_boundary_facet",
    "persist_collocation_batch",
    "replay_collocation",
    "sample_collocation",
    "sample_request_collocation",
    "verify_collocation_batch",
    "verify_request_collocation_batch",
)
