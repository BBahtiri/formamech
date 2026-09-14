"""Leakage-safe runner contracts for the S11 architecture contingency screen."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, cast

import torch

from campaigns.dogbone_v1.s10_production import (
    ProductionFemRunLedger,
    ProductionFemRunRecord,
)
from campaigns.dogbone_v1.s11_contingency import (
    ArchitectureScreenMetrics,
    ArchitectureScreenProtocol,
    ArchitectureSelection,
    build_architecture_screen_protocol,
    select_architecture_candidate,
)
from piginot.contracts import CanonicalMesh, canonical_json_bytes, hash_spec
from piginot.datasets.request_plan import (
    PublicFemRequestPlan,
    PublicFemRequestPlanEntry,
)
from piginot.doe import EvidenceRole, PublishedCaseManifest

TRAIN_FIT_PHASE_ID = "TRAIN_FIT"
TRAIN_SCREEN_PHASE_ID = "TRAIN_SCREEN"
DEVELOPMENT_PHASE_ID = "DEVELOPMENT_AFTER_TRAIN_SELECTION"
_CAMPAIGN_ROOT = (
    Path(__file__).resolve().parents[2] / "artifacts" / "campaigns" / "dogbone-v1"
)
_PROTOCOL_NAME = "S11-CONTINGENCY-PROTOCOL.json"
_MANIFEST_NAME = "public-case-manifest.json"
_BOUNDARY_CLOUD_TYPE = "piginot.campaigns.BoundaryCloudTensors"
_RUN_INDEX_TYPE = "piginot.campaigns.VerifiedScreenRunIndex"
_DECISION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class CandidateModelStateEvidence:
    """Initial and trained state identities for one screened candidate."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    candidate_id: str
    initial_state_hash: str
    final_state_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be non-empty text")
        _digest(self.initial_state_hash, "initial_state_hash")
        _digest(self.final_state_hash, "final_state_hash")


@dataclass(frozen=True, slots=True)
class TrainScreenDecision:
    """Integrity-bound TRAIN reduction consumed by the phase orchestrator."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    protocol_content_hash: str
    source_bindings_hash: str
    training_schedule_hash: str
    normalization_hash: str
    implementation_revision: str
    model_states: tuple[CandidateModelStateEvidence, ...]
    metrics: tuple[ArchitectureScreenMetrics, ...]
    selection: ArchitectureSelection
    _token: object = field(repr=False, compare=False, metadata={"canonical": False})

    def __post_init__(self) -> None:
        if self._token is not _DECISION_TOKEN:
            raise TypeError(
                "TrainScreenDecision must be created by complete_train_screen"
            )
        for value, label in (
            (self.protocol_content_hash, "protocol_content_hash"),
            (self.source_bindings_hash, "source_bindings_hash"),
            (self.training_schedule_hash, "training_schedule_hash"),
            (self.normalization_hash, "normalization_hash"),
        ):
            _digest(value, label)
        _revision(self.implementation_revision, "implementation_revision")
        if type(self.model_states) is not tuple or any(
            type(item) is not CandidateModelStateEvidence for item in self.model_states
        ):
            raise TypeError("model_states must contain exact state evidence")
        if type(self.metrics) is not tuple or any(
            type(metric) is not ArchitectureScreenMetrics for metric in self.metrics
        ):
            raise TypeError("metrics must contain exact ArchitectureScreenMetrics")
        if type(self.selection) is not ArchitectureSelection:
            raise TypeError("selection must be an exact ArchitectureSelection")

    def to_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self))

    @property
    def content_hash(self) -> str:
        return cast(str, hash_spec(self, "content"))


def complete_train_screen(
    protocol: ArchitectureScreenProtocol,
    metrics: tuple[ArchitectureScreenMetrics, ...],
    *,
    source_bindings_hash: str,
    training_schedule_hash: str,
    normalization_hash: str,
    implementation_revision: str,
    model_states: tuple[CandidateModelStateEvidence, ...],
) -> TrainScreenDecision:
    """Reduce exact TRAIN evidence without claiming that evaluation was performed."""

    if type(protocol) is not ArchitectureScreenProtocol:
        raise TypeError("protocol must be an exact ArchitectureScreenProtocol")
    if type(metrics) is not tuple or any(
        type(metric) is not ArchitectureScreenMetrics for metric in metrics
    ):
        raise TypeError("metrics must contain exact ArchitectureScreenMetrics")
    expected_ids = tuple(candidate.candidate_id for candidate in protocol.candidates)
    if tuple(metric.candidate_id for metric in metrics) != expected_ids:
        raise ValueError("metrics must follow the exact registered candidate order")
    if (
        type(model_states) is not tuple
        or tuple(
            item.candidate_id if type(item) is CandidateModelStateEvidence else None
            for item in model_states
        )
        != expected_ids
    ):
        raise ValueError(
            "model_states must follow the exact registered candidate order"
        )
    selection = select_architecture_candidate(protocol, metrics)
    return TrainScreenDecision(
        protocol.content_hash,
        _digest(source_bindings_hash, "source_bindings_hash"),
        _digest(training_schedule_hash, "training_schedule_hash"),
        _digest(normalization_hash, "normalization_hash"),
        _revision(implementation_revision, "implementation_revision"),
        model_states,
        metrics,
        selection,
        _DECISION_TOKEN,
    )


def validate_train_screen_decision(
    protocol: ArchitectureScreenProtocol,
    decision: TrainScreenDecision,
) -> TrainScreenDecision:
    """Recompute a decision at its consumption boundary to reject replaced fields."""

    if type(decision) is not TrainScreenDecision:
        raise TypeError("decision must be an exact TrainScreenDecision")
    if decision.protocol_content_hash != protocol.content_hash:
        raise ValueError("TRAIN decision is not bound to this protocol")
    expected = complete_train_screen(
        protocol,
        decision.metrics,
        source_bindings_hash=decision.source_bindings_hash,
        training_schedule_hash=decision.training_schedule_hash,
        normalization_hash=decision.normalization_hash,
        implementation_revision=decision.implementation_revision,
        model_states=decision.model_states,
    )
    if decision != expected:
        raise ValueError("TRAIN decision differs from its deterministic reduction")
    return decision


@dataclass(frozen=True, slots=True)
class BoundaryCloudTensors:
    """Fixed boundary point-cloud tensors used by every S11c candidate."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    xy: torch.Tensor
    features: torch.Tensor
    weights: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self) -> None:
        _validate_tensor(self.xy, "xy", ndim=2)
        _validate_tensor(self.features, "features", ndim=2)
        _validate_tensor(self.weights, "weights", ndim=1)
        if self.xy.shape[1] != 2:
            raise ValueError("xy must contain 2D coordinates")
        if self.features.shape[0] != self.xy.shape[0]:
            raise ValueError("features must align with xy")
        if self.weights.shape != self.xy.shape[:1]:
            raise ValueError("weights must align with xy")
        if self.mask.dtype is not torch.bool or self.mask.shape != self.xy.shape[:1]:
            raise ValueError("mask must be boolean and align with xy")
        if self.mask.device != self.xy.device:
            raise ValueError("mask must share the xy device")
        if bool((self.weights <= 0.0).any().item()):
            raise ValueError("boundary weights must be positive")


@dataclass(frozen=True, slots=True)
class VerifiedScreenRunIndex:
    """S10 run identities verified before any FEM label payload is opened."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    protocol_content_hash: str
    request_plan_hash: str
    ledger_content_hash: str
    train_fit_case_ids: tuple[str, ...]
    train_screen_case_ids: tuple[str, ...]
    development_case_ids: tuple[str, ...]
    development_opened: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "protocol_content_hash",
            "request_plan_hash",
            "ledger_content_hash",
        ):
            _digest(getattr(self, field_name), field_name)
        for field_name in (
            "train_fit_case_ids",
            "train_screen_case_ids",
            "development_case_ids",
        ):
            values = tuple(getattr(self, field_name))
            if not values or values != tuple(sorted(set(values))):
                raise ValueError(f"{field_name} must be non-empty sorted unique IDs")
            object.__setattr__(self, field_name, values)
        if type(self.development_opened) is not bool:
            raise ValueError("development_opened must be boolean")

    def to_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self))


def load_committed_protocol(path: Path | None = None) -> ArchitectureScreenProtocol:
    """Load only the exact preregistered protocol artifact committed before labels."""

    protocol_path = _CAMPAIGN_ROOT / _PROTOCOL_NAME if path is None else Path(path)
    manifest = PublishedCaseManifest.from_bytes(
        (_CAMPAIGN_ROOT / _MANIFEST_NAME).read_bytes()
    )
    protocol = build_architecture_screen_protocol(manifest)
    expected = protocol.to_bytes() + b"\n"
    if protocol_path.read_bytes() != expected:
        raise ValueError("committed protocol artifact does not match the source tree")
    return protocol


def case_ids_for_phase(
    protocol: ArchitectureScreenProtocol,
    phase_id: str,
    *,
    decision: TrainScreenDecision | None = None,
) -> tuple[str, ...]:
    """Return the only case IDs a runner may open for one leakage phase."""

    if type(protocol) is not ArchitectureScreenProtocol:
        raise TypeError("protocol must be an exact ArchitectureScreenProtocol")
    if phase_id == TRAIN_FIT_PHASE_ID:
        return protocol.fit_case_ids
    if phase_id == TRAIN_SCREEN_PHASE_ID:
        return protocol.screen_case_ids
    if phase_id == DEVELOPMENT_PHASE_ID:
        if decision is None:
            raise ValueError("DEVELOPMENT labels require a completed TRAIN decision")
        validate_train_screen_decision(protocol, decision)
        if decision.selection.fallback_to_baseline:
            raise ValueError("DEVELOPMENT labels remain closed on TRAIN fallback")
        return protocol.development_case_ids
    raise ValueError(f"unknown S11 contingency phase {phase_id!r}")


def verify_s10_screen_run_index(
    protocol: ArchitectureScreenProtocol,
    *,
    request_plan: PublicFemRequestPlan,
    ledger: ProductionFemRunLedger,
) -> VerifiedScreenRunIndex:
    """Validate public S10 plan/ledger identities before resolving store bytes."""

    if type(protocol) is not ArchitectureScreenProtocol:
        raise TypeError("protocol must be an exact ArchitectureScreenProtocol")
    if type(request_plan) is not PublicFemRequestPlan:
        raise TypeError("request_plan must be an exact PublicFemRequestPlan")
    if type(ledger) is not ProductionFemRunLedger:
        raise TypeError("ledger must be an exact ProductionFemRunLedger")
    if request_plan.public_case_manifest_hash != protocol.public_case_manifest_hash:
        raise ValueError("request plan is not bound to the S11c public manifest")
    if ledger.public_case_manifest_hash != protocol.public_case_manifest_hash:
        raise ValueError("ledger is not bound to the S11c public manifest")
    if ledger.request_plan_hash != request_plan.content_hash:
        raise ValueError("ledger is not bound to the supplied request plan")
    if ledger.execution_profile_hash != request_plan.execution_profile_hash:
        raise ValueError("ledger execution profile does not match request plan")
    if not ledger.is_complete:
        raise ValueError("S10 production ledger must be complete")

    entries = {entry.case_id: entry for entry in request_plan.entries}
    records = {record.case_id: record for record in ledger.records}
    if set(entries) != set(records):
        raise ValueError("request plan and ledger case populations differ")
    _require_role_and_success(
        protocol.fit_case_ids,
        entries=entries,
        records=records,
        role=EvidenceRole.TRAIN,
    )
    _require_role_and_success(
        protocol.screen_case_ids,
        entries=entries,
        records=records,
        role=EvidenceRole.TRAIN,
    )
    _require_role_and_success(
        protocol.development_case_ids,
        entries=entries,
        records=records,
        role=EvidenceRole.DEVELOPMENT,
    )
    return VerifiedScreenRunIndex(
        protocol_content_hash=protocol.content_hash,
        request_plan_hash=request_plan.content_hash,
        ledger_content_hash=hash_spec(ledger, "content"),
        train_fit_case_ids=protocol.fit_case_ids,
        train_screen_case_ids=protocol.screen_case_ids,
        development_case_ids=protocol.development_case_ids,
        development_opened=False,
    )


def qoi_vertex_indices(
    mesh: CanonicalMesh,
    *,
    semantic_tag: str,
    excluded_tags: tuple[str, ...],
) -> tuple[int, ...]:
    """Return stable vertex IDs for the smooth peak-stress boundary QoI."""

    if type(mesh) is not CanonicalMesh:
        raise TypeError("mesh must be an exact CanonicalMesh")
    if semantic_tag not in mesh.semantic_tags:
        raise ValueError(f"unknown QoI semantic tag {semantic_tag!r}")
    excluded: set[int] = set()
    for tag in excluded_tags:
        if tag not in mesh.semantic_tags:
            raise ValueError(f"unknown excluded semantic tag {tag!r}")
        excluded.update(_vertices_for_tag(mesh, tag))
    selected = _vertices_for_tag(mesh, semantic_tag) - excluded
    if not selected:
        raise ValueError("QoI vertex set is empty after exclusions")
    return tuple(sorted(selected))


def boundary_cloud_tensors(
    mesh: CanonicalMesh,
    *,
    tag_names: tuple[str, ...],
    sample_count: int = 320,
) -> BoundaryCloudTensors:
    """Sample a deterministic arclength boundary cloud with one-hot tag semantics."""

    if type(mesh) is not CanonicalMesh:
        raise TypeError("mesh must be an exact CanonicalMesh")
    if (
        type(tag_names) is not tuple
        or not tag_names
        or len(set(tag_names)) != len(tag_names)
        or any(
            type(name) is not str or name not in mesh.semantic_tags
            for name in tag_names
        )
    ):
        raise ValueError("tag_names must be unique semantic tags on the mesh")
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    facets = tuple(
        _facet_geometry(mesh, index, tag_names)
        for index in range(len(mesh.boundary_facets))
    )
    total_length = sum(facet.length for facet in facets)
    if not math.isfinite(total_length) or total_length <= 0.0:
        raise ValueError("boundary cloud requires positive total boundary measure")
    cumulative: list[float] = []
    running = 0.0
    for facet in facets:
        running += facet.length
        cumulative.append(running)

    points: list[tuple[float, float]] = []
    features: list[tuple[float, ...]] = []
    weights: list[float] = []
    for index in range(sample_count):
        arclength = (index + 0.5) * total_length / sample_count
        facet_index = min(bisect.bisect_right(cumulative, arclength), len(facets) - 1)
        start = 0.0 if facet_index == 0 else cumulative[facet_index - 1]
        facet = facets[facet_index]
        alpha = (arclength - start) / facet.length
        x = (1.0 - alpha) * facet.start[0] + alpha * facet.end[0]
        y = (1.0 - alpha) * facet.start[1] + alpha * facet.end[1]
        points.append((x, y))
        features.append((*facet.normal, *facet.one_hot))
        weights.append(total_length / sample_count)
    return BoundaryCloudTensors(
        xy=torch.tensor(points, dtype=torch.float64),
        features=torch.tensor(features, dtype=torch.float64),
        weights=torch.tensor(weights, dtype=torch.float64),
        mask=torch.ones(sample_count, dtype=torch.bool),
    )


@dataclass(frozen=True, slots=True)
class _FacetGeometry:
    start: tuple[float, float]
    end: tuple[float, float]
    length: float
    normal: tuple[float, float]
    one_hot: tuple[float, ...]


def _vertices_for_tag(mesh: CanonicalMesh, tag: str) -> set[int]:
    return {
        vertex
        for facet_index in mesh.semantic_tags[tag]
        for vertex in mesh.boundary_facets[facet_index]
    }


def _facet_geometry(
    mesh: CanonicalMesh,
    facet_index: int,
    tag_names: tuple[str, ...],
) -> _FacetGeometry:
    left_index, right_index = mesh.boundary_facets[facet_index]
    left = mesh.nodes[left_index]
    right = mesh.nodes[right_index]
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    length = math.hypot(dx, dy)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("boundary facet has non-positive length")
    normal = _outward_unit_normal(mesh, facet_index, dx, dy)
    one_hot = tuple(
        1.0 if facet_index in mesh.semantic_tags[tag] else 0.0 for tag in tag_names
    )
    if sum(one_hot) != 1.0:
        raise ValueError("each boundary facet must carry exactly one selected tag")
    return _FacetGeometry(left, right, length, normal, one_hot)


def _outward_unit_normal(
    mesh: CanonicalMesh,
    facet_index: int,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    left_index, right_index = mesh.boundary_facets[facet_index]
    midpoint = (
        0.5 * (mesh.nodes[left_index][0] + mesh.nodes[right_index][0]),
        0.5 * (mesh.nodes[left_index][1] + mesh.nodes[right_index][1]),
    )
    cell = next(
        (
            candidate
            for candidate in mesh.cells
            if left_index in candidate and right_index in candidate
        ),
        None,
    )
    if cell is None:
        raise ValueError("boundary facet is not incident to a cell")
    centroid = (
        sum(mesh.nodes[index][0] for index in cell) / len(cell),
        sum(mesh.nodes[index][1] for index in cell) / len(cell),
    )
    length = math.hypot(dx, dy)
    candidate = (dy / length, -dx / length)
    away = (midpoint[0] - centroid[0], midpoint[1] - centroid[1])
    if candidate[0] * away[0] + candidate[1] * away[1] < 0.0:
        candidate = (-candidate[0], -candidate[1])
    return candidate


def _require_role_and_success(
    case_ids: tuple[str, ...],
    *,
    entries: dict[str, PublicFemRequestPlanEntry],
    records: dict[str, ProductionFemRunRecord],
    role: EvidenceRole,
) -> None:
    for case_id in case_ids:
        entry = entries.get(case_id)
        if entry is None or entry.role is not role:
            raise ValueError(f"{case_id} must be a {role.value} S10 request")
        record = records.get(case_id)
        if record is None or not record.success:
            raise ValueError(f"{case_id} must have a successful S10 FEM record")


def _validate_tensor(value: torch.Tensor, label: str, *, ndim: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    if value.layout is not torch.strided or value.ndim != ndim:
        raise ValueError(f"{label} must be a dense {ndim}D tensor")
    if value.dtype is not torch.float64:
        raise ValueError(f"{label} must use torch.float64")
    if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{label} must be non-empty and finite")


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 256-bit lowercase digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a 256-bit lowercase digest") from exc
    if value.lower() != value:
        raise ValueError(f"{label} must be lowercase")
    return value


def _revision(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError(f"{label} must be a 160-bit lowercase Git revision")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a 160-bit lowercase Git revision") from exc
    if value.lower() != value:
        raise ValueError(f"{label} must be lowercase")
    return value


__all__ = (
    "DEVELOPMENT_PHASE_ID",
    "TRAIN_FIT_PHASE_ID",
    "TRAIN_SCREEN_PHASE_ID",
    "BoundaryCloudTensors",
    "CandidateModelStateEvidence",
    "TrainScreenDecision",
    "VerifiedScreenRunIndex",
    "boundary_cloud_tensors",
    "case_ids_for_phase",
    "complete_train_screen",
    "load_committed_protocol",
    "qoi_vertex_indices",
    "validate_train_screen_decision",
    "verify_s10_screen_run_index",
)
