"""Immutable contracts for deterministic S10 collocation streams."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from piginot.contracts import Unit, canonical_json_bytes, hash_spec
from piginot.doe.roles import EvidenceRole

SAMPLER_ALGORITHM_ID = "blake2b-keyed-counter-u53-v1"
SEED_COMMITMENT_ALGORITHM_ID = "blake2b-256-piginot-s10-collocation-seed-v1"
STREAM_SCHEMA_VERSION = "1.0"
MAX_COLLOCATION_POINTS = 100_000
_SEED_HEX = re.compile(r"[0-9a-f]{64}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_SAFE_INTEGER = (1 << 53) - 1
_PUBLIC_ROLES = frozenset(
    {
        EvidenceRole.PILOT,
        EvidenceRole.TRAIN,
        EvidenceRole.DEVELOPMENT,
        EvidenceRole.LEGACY,
    }
)


def _decode_seed(value: object) -> bytes:
    if not isinstance(value, str) or _SEED_HEX.fullmatch(value) is None:
        raise ValueError(
            "replay_seed_hex must be exactly 64 lowercase hexadecimal characters"
        )
    return bytes.fromhex(value)


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 256-bit digest")
    return value


def _require_public_role(value: object) -> EvidenceRole:
    if isinstance(value, EvidenceRole):
        role = value
    elif isinstance(value, str):
        role = EvidenceRole(value)
    else:
        raise ValueError("collocation role must be an EvidenceRole")
    if role not in _PUBLIC_ROLES:
        raise ValueError("collocation evidence must use a public role")
    return role


def _seed_commitment(seed: bytes) -> str:
    return hashlib.blake2b(
        seed,
        digest_size=32,
        person=b"piginot-s10-seed",
    ).hexdigest()


def _derived_public_seed_hex(
    *,
    algorithm_id: str,
    seed_commitment_algorithm_id: str,
    schema_version: str,
    campaign_hash: str,
    public_case_manifest_hash: str,
    snapshot_content_hash: str,
    case_id: str,
    leakage_group_id: str,
    role: EvidenceRole,
    mesh_id: str,
    mesh_content_hash: str,
    semantic_spec_hash: str,
    stream_index: int,
    counter_start: int,
    interior_count: int,
    boundary_count: int,
    interface_count: int,
    qoi_quotas: tuple[SemanticCollocationQuota, ...],
    quota_policy_hash: str,
) -> str:
    """Derive replay material solely from committed public inputs."""

    preimage = canonical_json_bytes(
        {
            "algorithm_id": algorithm_id,
            "seed_commitment_algorithm_id": seed_commitment_algorithm_id,
            "schema_version": schema_version,
            "campaign_hash": campaign_hash,
            "public_case_manifest_hash": public_case_manifest_hash,
            "snapshot_content_hash": snapshot_content_hash,
            "case_id": case_id,
            "leakage_group_id": leakage_group_id,
            "role": role,
            "mesh_id": mesh_id,
            "mesh_content_hash": mesh_content_hash,
            "semantic_spec_hash": semantic_spec_hash,
            "stream_index": stream_index,
            "counter_start": counter_start,
            "interior_count": interior_count,
            "boundary_count": boundary_count,
            "interface_count": interface_count,
            "qoi_quotas": qoi_quotas,
            "quota_policy_hash": quota_policy_hash,
        }
    )
    return hashlib.blake2b(
        preimage,
        digest_size=32,
        person=b"piginot-s10-rng",
    ).hexdigest()


class CollocationKind(StrEnum):
    """Geometric support used to draw a collocation point."""

    INTERIOR = "INTERIOR"
    BOUNDARY = "BOUNDARY"
    INTERFACE = "INTERFACE"


class SemanticBCClass(StrEnum):
    """Physics-facing semantic classification for a collocation point."""

    DIRICHLET = "DIRICHLET"
    NEUMANN = "NEUMANN"
    MIXED = "MIXED"
    FREE = "FREE"
    INTERFACE = "INTERFACE"
    INTERIOR = "INTERIOR"


@dataclass(frozen=True, slots=True)
class SemanticCollocationQuota:
    """A guaranteed QoI-region count independent of entity measure."""

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    quota_id: str
    kind: CollocationKind
    count: int
    semantic_tag: str | None = None
    region_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.quota_id, str) or not self.quota_id:
            raise ValueError("quota_id must be non-empty text")
        kind = CollocationKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count <= 0
        ):
            raise ValueError(
                "semantic collocation quota count must be a positive integer"
            )
        if kind is CollocationKind.BOUNDARY:
            if not isinstance(self.semantic_tag, str) or not self.semantic_tag:
                raise ValueError("boundary QoI quota requires a semantic_tag")
            if self.region_id is not None:
                raise ValueError("boundary QoI quota must not declare region_id")
        elif kind is CollocationKind.INTERIOR:
            if isinstance(self.region_id, bool) or not isinstance(self.region_id, int):
                raise ValueError("interior QoI quota requires an integer region_id")
            if self.semantic_tag is not None:
                raise ValueError("interior QoI quota must not declare semantic_tag")
        else:
            raise ValueError("QoI quotas support BOUNDARY or INTERIOR entities")


@dataclass(frozen=True, slots=True)
class CollocationStream:
    """Self-contained public replay recipe and RNG lineage."""

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    algorithm_id: str
    seed_commitment_algorithm_id: str
    schema_version: str
    replay_seed_hex: str
    seed_commitment: str
    campaign_hash: str
    public_case_manifest_hash: str
    snapshot_content_hash: str
    case_id: str
    leakage_group_id: str
    role: EvidenceRole
    mesh_id: str
    mesh_content_hash: str
    semantic_spec_hash: str
    stream_index: int
    counter_start: int
    counter_stop: int
    interior_count: int
    boundary_count: int
    interface_count: int
    qoi_quotas: tuple[SemanticCollocationQuota, ...]
    quota_policy_hash: str

    def __post_init__(self) -> None:
        if self.algorithm_id != SAMPLER_ALGORITHM_ID:
            raise ValueError("unsupported collocation sampler algorithm")
        if self.seed_commitment_algorithm_id != SEED_COMMITMENT_ALGORITHM_ID:
            raise ValueError("unsupported seed commitment algorithm")
        if self.schema_version != STREAM_SCHEMA_VERSION:
            raise ValueError("unsupported collocation stream schema")
        for value, label in (
            (self.case_id, "case_id"),
            (self.leakage_group_id, "leakage_group_id"),
            (self.mesh_id, "mesh_id"),
        ):
            _require_text(value, label)
        for value, label in (
            (self.campaign_hash, "campaign_hash"),
            (self.public_case_manifest_hash, "public_case_manifest_hash"),
            (self.snapshot_content_hash, "snapshot_content_hash"),
            (self.mesh_content_hash, "mesh_content_hash"),
            (self.semantic_spec_hash, "semantic_spec_hash"),
            (self.quota_policy_hash, "quota_policy_hash"),
        ):
            _require_digest(value, label)
        object.__setattr__(self, "role", _require_public_role(self.role))
        quotas = tuple(self.qoi_quotas)
        if any(type(quota) is not SemanticCollocationQuota for quota in quotas):
            raise TypeError("qoi_quotas must contain SemanticCollocationQuota values")
        expected_quotas = tuple(
            sorted(quotas, key=lambda quota: (quota.kind.value, quota.quota_id))
        )
        if quotas != expected_quotas:
            raise ValueError("qoi_quotas must use canonical order")
        quota_ids = tuple(quota.quota_id for quota in quotas)
        if len(quota_ids) != len(set(quota_ids)):
            raise ValueError("qoi_quotas must have unique quota IDs")
        if self.quota_policy_hash != hash_spec(quotas, "content"):
            raise ValueError("quota_policy_hash does not match qoi_quotas")
        object.__setattr__(self, "qoi_quotas", quotas)
        if (
            isinstance(self.stream_index, bool)
            or not isinstance(self.stream_index, int)
            or self.stream_index < 0
        ):
            raise ValueError("stream_index must be a public non-negative integer")
        counts = (self.interior_count, self.boundary_count, self.interface_count)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("collocation counts must be non-negative integers")
        point_count = sum(counts) + self.qoi_quota_count
        if point_count <= 0:
            raise ValueError("at least one collocation point must be requested")
        if point_count > MAX_COLLOCATION_POINTS:
            raise ValueError(
                f"collocation population exceeds maximum {MAX_COLLOCATION_POINTS}"
            )
        if (
            isinstance(self.counter_start, bool)
            or not isinstance(self.counter_start, int)
            or self.counter_start < 0
        ):
            raise ValueError("counter_start must be a non-negative integer")
        expected_stop = self.counter_start + sum(counts) + self.qoi_quota_count
        if (
            isinstance(self.counter_stop, bool)
            or not isinstance(self.counter_stop, int)
            or self.counter_stop != expected_stop
        ):
            raise ValueError("counter_stop must equal counter_start plus point counts")
        if self.counter_stop > _MAX_SAFE_INTEGER:
            raise ValueError(
                "counter interval exceeds the canonical safe-integer domain"
            )
        expected_seed_hex = _derived_public_seed_hex(
            algorithm_id=self.algorithm_id,
            seed_commitment_algorithm_id=self.seed_commitment_algorithm_id,
            schema_version=self.schema_version,
            campaign_hash=self.campaign_hash,
            public_case_manifest_hash=self.public_case_manifest_hash,
            snapshot_content_hash=self.snapshot_content_hash,
            case_id=self.case_id,
            leakage_group_id=self.leakage_group_id,
            role=self.role,
            mesh_id=self.mesh_id,
            mesh_content_hash=self.mesh_content_hash,
            semantic_spec_hash=self.semantic_spec_hash,
            stream_index=self.stream_index,
            counter_start=self.counter_start,
            interior_count=self.interior_count,
            boundary_count=self.boundary_count,
            interface_count=self.interface_count,
            qoi_quotas=self.qoi_quotas,
            quota_policy_hash=self.quota_policy_hash,
        )
        if self.replay_seed_hex != expected_seed_hex:
            raise ValueError(
                "replay_seed_hex does not match derived public replay seed"
            )
        seed = _decode_seed(self.replay_seed_hex)
        if self.seed_commitment != _seed_commitment(seed):
            raise ValueError("seed commitment does not match replay_seed_hex")
        _require_digest(self.seed_commitment, "seed_commitment")

    @property
    def stream_hash(self) -> str:
        """Return the content identity referenced by every sampled point."""

        return hash_spec(self, "content")

    @property
    def qoi_quota_count(self) -> int:
        return sum(quota.count for quota in self.qoi_quotas)


@dataclass(frozen=True, slots=True)
class CollocationPoint:
    """One collocation sample with exact sampler-stream traceability."""

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    sample_id: str
    case_id: str
    leakage_group_id: str
    role: EvidenceRole
    kind: CollocationKind
    semantic_class: SemanticBCClass
    coordinates: tuple[float, float]
    source_entity_index: int
    source_node_ids: tuple[int, ...]
    region_ids: tuple[int, ...]
    semantic_tags: tuple[str, ...]
    qoi_quota_id: str | None
    counter: int
    sampler_stream_hash: str
    coordinate_unit: Unit = Unit.MM

    def __post_init__(self) -> None:
        for value, label in (
            (self.sample_id, "sample_id"),
            (self.case_id, "case_id"),
            (self.leakage_group_id, "leakage_group_id"),
        ):
            _require_text(value, label)
        object.__setattr__(self, "role", _require_public_role(self.role))
        kind = CollocationKind(self.kind)
        semantic_class = SemanticBCClass(self.semantic_class)
        if (
            kind is CollocationKind.INTERIOR
            and semantic_class is not SemanticBCClass.INTERIOR
        ):
            raise ValueError("interior collocation must use INTERIOR semantics")
        if (
            kind is CollocationKind.INTERFACE
            and semantic_class is not SemanticBCClass.INTERFACE
        ):
            raise ValueError("interface collocation must use INTERFACE semantics")
        if kind is CollocationKind.BOUNDARY and semantic_class in {
            SemanticBCClass.INTERIOR,
            SemanticBCClass.INTERFACE,
        }:
            raise ValueError(
                "boundary collocation requires boundary-condition semantics"
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "semantic_class", semantic_class)
        if self.coordinate_unit is not Unit.MM:
            raise ValueError("collocation coordinate_unit must be canonical mm")
        raw_coordinates = tuple(self.coordinates)
        if len(raw_coordinates) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in raw_coordinates
        ):
            raise ValueError("collocation coordinates must be a finite 2D point")
        coordinates = tuple(float(value) for value in raw_coordinates)
        object.__setattr__(self, "coordinates", coordinates)
        if (
            isinstance(self.source_entity_index, bool)
            or not isinstance(self.source_entity_index, int)
            or self.source_entity_index < 0
        ):
            raise ValueError("source_entity_index must be a non-negative integer")
        node_ids = tuple(self.source_node_ids)
        expected_nodes = 3 if kind is CollocationKind.INTERIOR else 2
        if (
            len(node_ids) != expected_nodes
            or len(node_ids) != len(set(node_ids))
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in node_ids
            )
        ):
            raise ValueError("source_node_ids do not match the collocation entity")
        region_ids = tuple(sorted(set(self.region_ids)))
        if not region_ids or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in region_ids
        ):
            raise ValueError("region_ids must contain integer region identities")
        tags = tuple(sorted(set(self.semantic_tags)))
        if any(not isinstance(tag, str) or not tag for tag in tags):
            raise ValueError("semantic_tags must contain non-empty text")
        if (
            isinstance(self.counter, bool)
            or not isinstance(self.counter, int)
            or not 0 <= self.counter <= _MAX_SAFE_INTEGER
        ):
            raise ValueError("counter must be a canonical non-negative integer")
        _require_digest(self.sampler_stream_hash, "sampler_stream_hash")
        object.__setattr__(self, "source_node_ids", node_ids)
        object.__setattr__(self, "region_ids", region_ids)
        object.__setattr__(self, "semantic_tags", tags)
        if self.qoi_quota_id is not None and (
            not isinstance(self.qoi_quota_id, str) or not self.qoi_quota_id
        ):
            raise ValueError("qoi_quota_id must be None or non-empty text")


@dataclass(frozen=True, slots=True)
class CollocationBatch:
    """Canonical byte representation of one deterministic sampler stream."""

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    stream: CollocationStream
    sampler_stream_hash: str
    points: tuple[CollocationPoint, ...]

    def __post_init__(self) -> None:
        if type(self.stream) is not CollocationStream:
            raise TypeError("stream must be a CollocationStream")
        if self.sampler_stream_hash != self.stream.stream_hash:
            raise ValueError("sampler_stream_hash does not match stream")
        points = tuple(self.points)
        if len(points) != self.stream.counter_stop - self.stream.counter_start:
            raise ValueError("point count does not match stream counter interval")
        if any(type(point) is not CollocationPoint for point in points):
            raise TypeError("points must contain CollocationPoint values")
        expected_counters = tuple(
            range(self.stream.counter_start, self.stream.counter_stop)
        )
        if tuple(point.counter for point in points) != expected_counters:
            raise ValueError("point counters must cover the recorded interval exactly")
        if any(
            point.sampler_stream_hash != self.sampler_stream_hash for point in points
        ):
            raise ValueError("every point must reference the recorded sampler stream")
        if any(
            point.case_id != self.stream.case_id
            or point.leakage_group_id != self.stream.leakage_group_id
            or point.role is not self.stream.role
            for point in points
        ):
            raise ValueError("every point must match the stream case provenance")
        expected_sample_ids = tuple(
            f"{self.stream.case_id}:collocation:{counter:016x}"
            for counter in expected_counters
        )
        if tuple(point.sample_id for point in points) != expected_sample_ids:
            raise ValueError("point sample IDs must match the stream counters")
        background = tuple(point for point in points if point.qoi_quota_id is None)
        background_counts = {
            kind: sum(point.kind is kind for point in background)
            for kind in CollocationKind
        }
        expected_background = {
            CollocationKind.INTERIOR: self.stream.interior_count,
            CollocationKind.BOUNDARY: self.stream.boundary_count,
            CollocationKind.INTERFACE: self.stream.interface_count,
        }
        if background_counts != expected_background:
            raise ValueError("background point kinds do not match stream counts")
        quota_by_id = {quota.quota_id: quota for quota in self.stream.qoi_quotas}
        for quota_id, quota in quota_by_id.items():
            selected = tuple(
                point for point in points if point.qoi_quota_id == quota_id
            )
            if len(selected) != quota.count or any(
                point.kind is not quota.kind for point in selected
            ):
                raise ValueError("QoI point counts do not match stream quotas")
        if any(
            point.qoi_quota_id is not None and point.qoi_quota_id not in quota_by_id
            for point in points
        ):
            raise ValueError("point references an unknown QoI quota")
        object.__setattr__(self, "points", points)

    def to_bytes(self) -> bytes:
        """Return the RFC 8785 canonical batch representation."""

        return canonical_json_bytes(self)

    @property
    def content_hash(self) -> str:
        return hash_spec(self, "content")


@dataclass(frozen=True, slots=True)
class _WeightedEntity:
    index: int
    node_ids: tuple[int, ...]
    region_ids: tuple[int, ...]
    weight: float


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
)
