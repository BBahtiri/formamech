"""Immutable evidence contracts for DogBone S10 dataset closeout."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import ClassVar, cast

from piginot.contracts import (
    CanonicalQuantity,
    Dimension,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.datasets import (
    CollocationKind,
    CrossMeshMethod,
    SemanticCollocationQuota,
)
from piginot.doe import EvidenceRole
from piginot.fem import FemFailureCode

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SCHEMA_VERSION = "1.0"
_STRESS_FIELD_NAME = "first_piola_stress"
_COMPONENT_SHAPE = (3, 3)
S10_FAILURE_CANARY_ID = "snapshot-failure-visibility-canary-v1"

DOGBONE_S10_QUERY_POINTS = tuple(
    (x, y) for x in (1.0, 5.0, 9.0, 13.0, 17.0) for y in (0.75, 1.5, 2.25)
)


def require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 256-bit digest")
    return value


def require_revision(value: object, label: str) -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 40-character Git revision")
    return value


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class S10DatasetPolicy:
    """Finite common-query, collocation, and scaling policy."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    query_points: tuple[tuple[float, float], ...]
    interior_count: int
    boundary_count: int
    interface_count: int
    qoi_quotas: tuple[SemanticCollocationQuota, ...]
    reference_scale: CanonicalQuantity
    stream_index: int = 0
    field_name: str = _STRESS_FIELD_NAME
    component_shape: tuple[int, ...] = _COMPONENT_SHAPE
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        points = tuple(
            tuple(float(coordinate) for coordinate in point)
            for point in self.query_points
        )
        if not points or any(
            len(point) != 2 or any(not math.isfinite(value) for value in point)
            for point in points
        ):
            raise ValueError("query_points must contain finite physical 2D points")
        if len(points) != len(set(points)):
            raise ValueError("query_points must be unique")
        for field_name in (
            "interior_count",
            "boundary_count",
            "interface_count",
            "stream_index",
        ):
            _count(getattr(self, field_name), field_name)
        raw_quotas = tuple(self.qoi_quotas)
        if any(type(quota) is not SemanticCollocationQuota for quota in raw_quotas):
            raise TypeError("qoi_quotas must contain SemanticCollocationQuota values")
        quotas = tuple(
            sorted(raw_quotas, key=lambda item: (item.kind.value, item.quota_id))
        )
        if (
            self.interior_count
            + self.boundary_count
            + self.interface_count
            + sum(quota.count for quota in quotas)
            <= 0
        ):
            raise ValueError("collocation policy must request at least one point")
        if type(self.reference_scale) is not CanonicalQuantity:
            raise TypeError("reference_scale must be a CanonicalQuantity")
        if (
            self.reference_scale.unit is not Unit.MPA
            or self.reference_scale.dimension is not Dimension.STRESS
            or self.reference_scale.value <= 0.0
        ):
            raise ValueError("reference_scale must be a positive MPa stress")
        if self.field_name != _STRESS_FIELD_NAME:
            raise ValueError(f"field_name must be {_STRESS_FIELD_NAME!r}")
        if tuple(self.component_shape) != _COMPONENT_SHAPE:
            raise ValueError("S10 stress projection requires a 3x3 component shape")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("unsupported S10 dataset policy schema")
        object.__setattr__(
            self,
            "query_points",
            cast(tuple[tuple[float, float], ...], points),
        )
        object.__setattr__(self, "qoi_quotas", quotas)
        object.__setattr__(self, "component_shape", tuple(self.component_shape))

    @property
    def collocation_points_per_case(self) -> int:
        return (
            self.interior_count
            + self.boundary_count
            + self.interface_count
            + sum(quota.count for quota in self.qoi_quotas)
        )


DOGBONE_S10_DATASET_POLICY = S10DatasetPolicy(
    query_points=DOGBONE_S10_QUERY_POINTS,
    interior_count=128,
    boundary_count=64,
    interface_count=0,
    qoi_quotas=(
        SemanticCollocationQuota(
            quota_id="stress-critical-right-arc-v1",
            kind=CollocationKind.BOUNDARY,
            count=64,
            semantic_tag="right_arc",
        ),
    ),
    reference_scale=CanonicalQuantity(760.0, Unit.MPA),
)


@dataclass(frozen=True, slots=True)
class S10MutationEvidence:
    """Independent test evidence supplied to the finite G10 reducer."""

    drop_projection_error_mutant_fired: bool
    stress_node_pairing_mutant_fired: bool
    normalize_on_all_splits_mutant_fired: bool
    code_coverage_percent: float | None

    def __post_init__(self) -> None:
        for field_name in (
            "drop_projection_error_mutant_fired",
            "stress_node_pairing_mutant_fired",
            "normalize_on_all_splits_mutant_fired",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be boolean")
        if self.code_coverage_percent is not None:
            value = float(self.code_coverage_percent)
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise ValueError("code_coverage_percent must lie in [0, 100]")
            object.__setattr__(self, "code_coverage_percent", value)


@dataclass(frozen=True, slots=True)
class RoleSnapshotEvidence:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    role: EvidenceRole
    case_count: int
    successful_case_count: int
    failed_case_count: int
    snapshot_content_hash: str
    store_content_hash: str

    def __post_init__(self) -> None:
        role = EvidenceRole(self.role)
        if role is EvidenceRole.HOLDOUT:
            raise ValueError("HOLDOUT cannot enter public S10 evidence")
        for name in ("case_count", "successful_case_count", "failed_case_count"):
            _count(getattr(self, name), name)
        if self.successful_case_count + self.failed_case_count != self.case_count:
            raise ValueError("role snapshot success and failure counts must reconcile")
        require_digest(self.snapshot_content_hash, "snapshot_content_hash")
        require_digest(self.store_content_hash, "store_content_hash")
        object.__setattr__(self, "role", role)


@dataclass(frozen=True, slots=True)
class CaseMaterializationEvidence:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    case_id: str
    role: EvidenceRole
    fem_result_hash: str
    fem_result_store_content_hash: str
    extraction_content_hash: str
    extraction_store_content_hash: str
    stress_field_content_hash: str
    stress_field_store_content_hash: str
    projection_content_hash: str
    projection_store_content_hash: str
    transfer_error_content_hash: str
    collocation_stream_hash: str
    collocation_batch_content_hash: str
    collocation_store_content_hash: str
    collocation_point_count: int
    projection_source_verified: bool
    collocation_replay_verified: bool

    def __post_init__(self) -> None:
        require_text(self.case_id, "case_id")
        role = EvidenceRole(self.role)
        if role is EvidenceRole.HOLDOUT:
            raise ValueError("HOLDOUT cannot enter public S10 evidence")
        for name in (
            "fem_result_hash",
            "fem_result_store_content_hash",
            "extraction_content_hash",
            "extraction_store_content_hash",
            "stress_field_content_hash",
            "stress_field_store_content_hash",
            "projection_content_hash",
            "projection_store_content_hash",
            "transfer_error_content_hash",
            "collocation_stream_hash",
            "collocation_batch_content_hash",
            "collocation_store_content_hash",
        ):
            require_digest(getattr(self, name), name)
        _count(self.collocation_point_count, "collocation_point_count")
        if self.collocation_point_count == 0:
            raise ValueError("collocation_point_count must be positive")
        if type(self.projection_source_verified) is not bool:
            raise TypeError("projection_source_verified must be boolean")
        if type(self.collocation_replay_verified) is not bool:
            raise TypeError("collocation_replay_verified must be boolean")
        object.__setattr__(self, "role", role)


@dataclass(frozen=True, slots=True)
class CrossMeshComparisonEvidence:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    anchor_case_id: str
    compared_case_id: str
    method: CrossMeshMethod
    artifact_content_hash: str
    store_content_hash: str
    left_source_field_hash: str
    right_source_field_hash: str
    relative_l2_difference: float
    source_verified: bool

    def __post_init__(self) -> None:
        require_text(self.anchor_case_id, "anchor_case_id")
        require_text(self.compared_case_id, "compared_case_id")
        if self.anchor_case_id == self.compared_case_id:
            raise ValueError("cross-mesh comparison requires two cases")
        method = CrossMeshMethod(self.method)
        if method is not CrossMeshMethod.PHYSICAL_DOMAIN_INTEGRAL:
            raise ValueError("cross-mesh evidence must use physical integrals")
        for name in (
            "artifact_content_hash",
            "store_content_hash",
            "left_source_field_hash",
            "right_source_field_hash",
        ):
            require_digest(getattr(self, name), name)
        value = float(self.relative_l2_difference)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("relative_l2_difference must be finite and non-negative")
        if type(self.source_verified) is not bool:
            raise TypeError("source_verified must be boolean")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "relative_l2_difference", value)


@dataclass(frozen=True, slots=True)
class NormalizedSampleEvidence:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    case_id: str
    fem_result_hash: str
    normalized_content_hash: str
    store_content_hash: str
    round_trip_verified: bool

    def __post_init__(self) -> None:
        require_text(self.case_id, "case_id")
        for name in (
            "fem_result_hash",
            "normalized_content_hash",
            "store_content_hash",
        ):
            require_digest(getattr(self, name), name)
        if type(self.round_trip_verified) is not bool:
            raise TypeError("round_trip_verified must be boolean")


@dataclass(frozen=True, slots=True)
class NormalizationEvidence:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    training_case_ids: tuple[str, ...]
    training_snapshot_hash: str
    statistics_content_hash: str
    statistics_store_content_hash: str
    statistic_count: int
    non_train_source_count: int
    non_active_source_count: int
    snapshot_binding_verified: bool
    normalized_samples: tuple[NormalizedSampleEvidence, ...]

    def __post_init__(self) -> None:
        case_ids = tuple(self.training_case_ids)
        if not case_ids or case_ids != tuple(sorted(set(case_ids))):
            raise ValueError("training_case_ids must be sorted, unique, and non-empty")
        for case_id in case_ids:
            require_text(case_id, "training case ID")
        for name in (
            "training_snapshot_hash",
            "statistics_content_hash",
            "statistics_store_content_hash",
        ):
            require_digest(getattr(self, name), name)
        for name in (
            "statistic_count",
            "non_train_source_count",
            "non_active_source_count",
        ):
            _count(getattr(self, name), name)
        samples = tuple(self.normalized_samples)
        if any(type(sample) is not NormalizedSampleEvidence for sample in samples):
            raise TypeError("normalized_samples must contain evidence values")
        if tuple(sample.case_id for sample in samples) != case_ids:
            raise ValueError("normalized samples must exactly match training case IDs")
        if type(self.snapshot_binding_verified) is not bool:
            raise TypeError("snapshot_binding_verified must be boolean")
        object.__setattr__(self, "training_case_ids", case_ids)
        object.__setattr__(self, "normalized_samples", samples)

    @property
    def round_trip_verified_count(self) -> int:
        return sum(sample.round_trip_verified for sample in self.normalized_samples)


@dataclass(frozen=True, slots=True)
class FailedCaseEvidence:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    case_id: str
    role: EvidenceRole
    failure_code: FemFailureCode
    message: str
    snapshot_content_hash: str

    def __post_init__(self) -> None:
        require_text(self.case_id, "case_id")
        role = EvidenceRole(self.role)
        if role is EvidenceRole.HOLDOUT:
            raise ValueError("HOLDOUT cannot enter public S10 evidence")
        code = FemFailureCode(self.failure_code)
        if self.message != f"failed:{code.value}":
            raise ValueError("failed case message must be sanitized")
        require_digest(self.snapshot_content_hash, "snapshot_content_hash")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "failure_code", code)


@dataclass(frozen=True, slots=True)
class FailureVisibilityCanaryEvidence:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    canary_id: str
    case_id: str
    role: EvidenceRole
    is_production_failure: bool
    failure_code: FemFailureCode
    represented_failure_count: int
    message: str
    snapshot_content_hash: str
    store_content_hash: str

    def __post_init__(self) -> None:
        if self.canary_id != S10_FAILURE_CANARY_ID:
            raise ValueError("unsupported failure visibility canary")
        require_text(self.case_id, "case_id")
        role = EvidenceRole(self.role)
        if role is EvidenceRole.HOLDOUT:
            raise ValueError("HOLDOUT cannot enter public S10 evidence")
        if self.is_production_failure is not False:
            raise ValueError("controlled canary must not claim a production failure")
        code = FemFailureCode(self.failure_code)
        _count(self.represented_failure_count, "represented_failure_count")
        if self.represented_failure_count != 1:
            raise ValueError(
                "failure visibility canary must represent exactly one case"
            )
        if self.message != f"failed:{code.value}":
            raise ValueError("failure visibility canary message must be sanitized")
        require_digest(self.snapshot_content_hash, "snapshot_content_hash")
        require_digest(self.store_content_hash, "store_content_hash")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "failure_code", code)


@dataclass(frozen=True, slots=True)
class S10CloseoutReport:
    """Compact public manifest for content-addressed S10 materialization."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    run_id: str
    fem_source_revision: str
    materializer_source_revision: str
    campaign_hash: str
    public_case_manifest_hash: str
    request_plan_hash: str
    execution_profile_hash: str
    policy: S10DatasetPolicy
    role_snapshots: tuple[RoleSnapshotEvidence, ...]
    cases: tuple[CaseMaterializationEvidence, ...]
    cross_mesh_comparisons: tuple[CrossMeshComparisonEvidence, ...]
    normalization: NormalizationEvidence
    failed_cases: tuple[FailedCaseEvidence, ...]
    failure_visibility_canary: FailureVisibilityCanaryEvidence | None
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.run_id, "run_id")
        require_revision(self.fem_source_revision, "fem_source_revision")
        require_revision(
            self.materializer_source_revision,
            "materializer_source_revision",
        )
        for name in (
            "campaign_hash",
            "public_case_manifest_hash",
            "request_plan_hash",
            "execution_profile_hash",
        ):
            require_digest(getattr(self, name), name)
        if type(self.policy) is not S10DatasetPolicy:
            raise TypeError("policy must be an S10DatasetPolicy")
        snapshots = tuple(sorted(self.role_snapshots, key=lambda item: item.role.value))
        if not snapshots or any(
            type(item) is not RoleSnapshotEvidence for item in snapshots
        ):
            raise TypeError("role_snapshots must contain evidence values")
        if len({item.role for item in snapshots}) != len(snapshots):
            raise ValueError("role snapshot evidence must have unique roles")
        cases = tuple(sorted(self.cases, key=lambda item: item.case_id))
        if not cases or any(
            type(item) is not CaseMaterializationEvidence for item in cases
        ):
            raise TypeError("cases must contain materialization evidence")
        if len({item.case_id for item in cases}) != len(cases):
            raise ValueError("materialized case IDs must be unique")
        comparisons = tuple(
            sorted(self.cross_mesh_comparisons, key=lambda item: item.compared_case_id)
        )
        if any(type(item) is not CrossMeshComparisonEvidence for item in comparisons):
            raise TypeError("cross_mesh_comparisons must contain evidence values")
        if len(comparisons) != max(0, len(cases) - 1):
            raise ValueError("cross-mesh population must compare every non-anchor case")
        failed = tuple(sorted(self.failed_cases, key=lambda item: item.case_id))
        if any(type(item) is not FailedCaseEvidence for item in failed):
            raise TypeError("failed_cases must contain evidence values")
        all_case_ids = (
            *tuple(item.case_id for item in cases),
            *tuple(item.case_id for item in failed),
        )
        if len(all_case_ids) != len(set(all_case_ids)):
            raise ValueError("successful and failed case IDs must be disjoint")
        snapshot_total = sum(item.case_count for item in snapshots)
        if snapshot_total != len(all_case_ids):
            raise ValueError("snapshot and case evidence populations do not reconcile")
        snapshot_successes = sum(item.successful_case_count for item in snapshots)
        snapshot_failures = sum(item.failed_case_count for item in snapshots)
        if snapshot_successes != len(cases) or snapshot_failures != len(failed):
            raise ValueError("snapshot outcome counts do not reconcile")
        if any(
            item.collocation_point_count != self.policy.collocation_points_per_case
            for item in cases
        ):
            raise ValueError("case collocation counts do not match the fixed policy")
        if type(self.normalization) is not NormalizationEvidence:
            raise TypeError("normalization must be NormalizationEvidence")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("unsupported S10 closeout report schema")
        object.__setattr__(self, "role_snapshots", snapshots)
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "cross_mesh_comparisons", comparisons)
        object.__setattr__(self, "failed_cases", failed)

    @property
    def content_hash(self) -> str:
        return hash_spec(self, "content")

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    @property
    def successful_case_count(self) -> int:
        return len(self.cases)

    @property
    def source_verified_extraction_count(self) -> int:
        return len(self.cases)

    @property
    def projection_field_count(self) -> int:
        return len(self.cases)

    @property
    def projection_error_artifact_count(self) -> int:
        return len(self.cases)

    @property
    def source_verified_projection_count(self) -> int:
        return sum(case.projection_source_verified for case in self.cases)

    @property
    def physical_integral_comparison_count(self) -> int:
        return len(self.cross_mesh_comparisons)

    @property
    def source_verified_integral_comparison_count(self) -> int:
        return sum(item.source_verified for item in self.cross_mesh_comparisons)

    @property
    def collocation_point_count(self) -> int:
        return sum(case.collocation_point_count for case in self.cases)

    @property
    def collocation_replay_count(self) -> int:
        return len(self.cases)

    @property
    def byte_identical_replay_count(self) -> int:
        return sum(case.collocation_replay_verified for case in self.cases)

    @property
    def collocation_request_binding_verified(self) -> bool:
        return all(case.collocation_replay_verified for case in self.cases)

    @property
    def actual_failed_case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.failed_cases)

    @property
    def store_content_hashes(self) -> tuple[str, ...]:
        values = {
            *(item.store_content_hash for item in self.role_snapshots),
            *(item.fem_result_store_content_hash for item in self.cases),
            *(item.extraction_store_content_hash for item in self.cases),
            *(item.stress_field_store_content_hash for item in self.cases),
            *(item.projection_store_content_hash for item in self.cases),
            *(item.collocation_store_content_hash for item in self.cases),
            *(item.store_content_hash for item in self.cross_mesh_comparisons),
            self.normalization.statistics_store_content_hash,
            *(
                item.store_content_hash
                for item in self.normalization.normalized_samples
            ),
        }
        if self.failure_visibility_canary is not None:
            values.add(self.failure_visibility_canary.store_content_hash)
        return tuple(sorted(values))


__all__ = (
    "DOGBONE_S10_DATASET_POLICY",
    "DOGBONE_S10_QUERY_POINTS",
    "S10_FAILURE_CANARY_ID",
    "CaseMaterializationEvidence",
    "CrossMeshComparisonEvidence",
    "FailedCaseEvidence",
    "FailureVisibilityCanaryEvidence",
    "NormalizationEvidence",
    "NormalizedSampleEvidence",
    "RoleSnapshotEvidence",
    "S10CloseoutReport",
    "S10DatasetPolicy",
    "S10MutationEvidence",
    "require_revision",
    "require_text",
)
