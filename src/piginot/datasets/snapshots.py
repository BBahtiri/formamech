"""Immutable, public-only S10 dataset snapshot construction."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.datasets.request_plan import (
    PublicFemRequestPlan,
    PublicFemRequestPlanEntry,
)
from piginot.doe.checkpoint import CPEnvelope
from piginot.doe.publication import (
    PublishedCase,
    PublishedCaseManifest,
    PublishedCaseStatus,
)
from piginot.doe.roles import EvidenceRole
from piginot.fem import FemAttempt, FemFailureCode, FemRequest, FemResult
from piginot.store import ObjectStore, StoreError
from piginot.store.store import content_hash as store_content_hash

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_PUBLIC_ROLES = frozenset(
    {
        EvidenceRole.PILOT,
        EvidenceRole.TRAIN,
        EvidenceRole.DEVELOPMENT,
        EvidenceRole.LEGACY,
    }
)
_SCHEMA_VERSION = "1.0"


class DatasetSnapshotValidationError(ValueError):
    """A public case or FEM outcome cannot enter an S10 snapshot safely."""


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise DatasetSnapshotValidationError(
            f"{label} must be a lowercase BLAKE2b-256 digest"
        )
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetSnapshotValidationError(f"{label} must be non-empty text")
    return value


def _frozen_map(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(sorted(values.items())))


@dataclass(frozen=True, slots=True)
class SampleTrace:
    """Exclusive source identity for one materialized dataset sample."""

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    fem_result_hash: str | None = None
    sampler_stream_hash: str | None = None

    def __post_init__(self) -> None:
        identities = tuple(
            value
            for value in (self.fem_result_hash, self.sampler_stream_hash)
            if value is not None
        )
        if len(identities) != 1:
            raise DatasetSnapshotValidationError(
                "sample trace must contain exactly one FEM result hash or sampler stream hash"
            )
        if self.fem_result_hash is not None:
            object.__setattr__(
                self,
                "fem_result_hash",
                _digest(self.fem_result_hash, "fem_result_hash"),
            )
        if self.sampler_stream_hash is not None:
            object.__setattr__(
                self,
                "sampler_stream_hash",
                _digest(self.sampler_stream_hash, "sampler_stream_hash"),
            )


@dataclass(frozen=True, slots=True)
class DatasetCaseRecord:
    """One public active case and its visible S7 outcome."""

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    case_id: str
    leakage_group_id: str
    role: EvidenceRole
    primitive_coordinates: Mapping[str, float]
    derived_coordinates: Mapping[str, float]
    strata: Mapping[str, int]
    request_recipe_hash: str
    request_content_hash: str
    mesh_id: str
    mesh_content_hash: str
    success: bool
    sample_trace: SampleTrace | None
    store_content_hash: str | None
    failure_code: FemFailureCode | None
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(
            self,
            "leakage_group_id",
            _text(self.leakage_group_id, "leakage_group_id"),
        )
        role = EvidenceRole(self.role)
        if role not in _PUBLIC_ROLES:
            raise DatasetSnapshotValidationError(
                "HOLDOUT cases must never enter a public dataset snapshot"
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self,
            "primitive_coordinates",
            _frozen_map(self.primitive_coordinates),
        )
        object.__setattr__(
            self,
            "derived_coordinates",
            _frozen_map(self.derived_coordinates),
        )
        object.__setattr__(self, "strata", _frozen_map(self.strata))
        object.__setattr__(
            self,
            "request_recipe_hash",
            _digest(self.request_recipe_hash, "request_recipe_hash"),
        )
        object.__setattr__(
            self,
            "request_content_hash",
            _digest(self.request_content_hash, "request_content_hash"),
        )
        object.__setattr__(self, "mesh_id", _text(self.mesh_id, "mesh_id"))
        object.__setattr__(
            self,
            "mesh_content_hash",
            _digest(self.mesh_content_hash, "mesh_content_hash"),
        )
        object.__setattr__(self, "message", _text(self.message, "message"))
        if type(self.success) is not bool:
            raise DatasetSnapshotValidationError("success must be boolean")
        if self.success:
            if type(self.sample_trace) is not SampleTrace:
                raise DatasetSnapshotValidationError(
                    "successful case must have exactly one sample trace"
                )
            if self.sample_trace.fem_result_hash is None:
                raise DatasetSnapshotValidationError(
                    "successful FEM case must trace to fem_result_hash"
                )
            if self.failure_code is not None:
                raise DatasetSnapshotValidationError(
                    "successful case cannot carry a failure code"
                )
            if self.store_content_hash is None:
                raise DatasetSnapshotValidationError(
                    "successful case must reference materialized ObjectStore bytes"
                )
            if self.message != "succeeded":
                raise DatasetSnapshotValidationError(
                    "successful case message must use the sanitized status"
                )
        else:
            if self.sample_trace is not None or self.store_content_hash is not None:
                raise DatasetSnapshotValidationError(
                    "failed case cannot carry result material"
                )
            if not isinstance(self.failure_code, FemFailureCode):
                raise DatasetSnapshotValidationError(
                    "failed case must carry a FEM failure code"
                )
            if self.message != f"failed:{self.failure_code.value}":
                raise DatasetSnapshotValidationError(
                    "failed case message must use the sanitized failure status"
                )
        if self.store_content_hash is not None:
            object.__setattr__(
                self,
                "store_content_hash",
                _digest(self.store_content_hash, "store_content_hash"),
            )


@dataclass(frozen=True, slots=True)
class RoleDatasetSnapshot:
    """Content-addressed snapshot for exactly one immutable evidence role."""

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    campaign_hash: str
    public_case_manifest_hash: str
    request_plan_hash: str
    role: EvidenceRole
    cases: tuple[DatasetCaseRecord, ...]
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "campaign_hash", _digest(self.campaign_hash, "campaign_hash")
        )
        object.__setattr__(
            self,
            "public_case_manifest_hash",
            _digest(self.public_case_manifest_hash, "public_case_manifest_hash"),
        )
        object.__setattr__(
            self,
            "request_plan_hash",
            _digest(self.request_plan_hash, "request_plan_hash"),
        )
        role = EvidenceRole(self.role)
        if role not in _PUBLIC_ROLES:
            raise DatasetSnapshotValidationError(
                "HOLDOUT role cannot have a public dataset snapshot"
            )
        cases = tuple(sorted(self.cases, key=lambda case: case.case_id))
        if not cases:
            raise DatasetSnapshotValidationError("role snapshot must not be empty")
        if any(type(case) is not DatasetCaseRecord for case in cases):
            raise DatasetSnapshotValidationError(
                "role snapshot cases must be DatasetCaseRecord values"
            )
        if any(case.role is not role for case in cases):
            raise DatasetSnapshotValidationError(
                "every case must have the snapshot evidence role"
            )
        case_ids = tuple(case.case_id for case in cases)
        if len(case_ids) != len(set(case_ids)):
            raise DatasetSnapshotValidationError("snapshot case IDs must be unique")
        if self.schema_version != _SCHEMA_VERSION:
            raise DatasetSnapshotValidationError("unsupported snapshot schema version")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "cases", cases)

    @property
    def content_hash(self) -> str:
        """Return the canonical content identity of this complete snapshot."""

        return hash_spec(self, "content")


@dataclass(frozen=True, slots=True)
class StoredSnapshotReference:
    """The two distinct identities of one persisted canonical snapshot."""

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    snapshot_content_hash: str
    store_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "snapshot_content_hash",
            _digest(self.snapshot_content_hash, "snapshot_content_hash"),
        )
        object.__setattr__(
            self,
            "store_content_hash",
            _digest(self.store_content_hash, "store_content_hash"),
        )


def persist_role_snapshot(
    snapshot: RoleDatasetSnapshot,
    store: ObjectStore,
) -> StoredSnapshotReference:
    """Persist canonical snapshot bytes without using a filesystem path as identity."""

    if type(snapshot) is not RoleDatasetSnapshot:
        raise DatasetSnapshotValidationError("snapshot must be a RoleDatasetSnapshot")
    if type(store) is not ObjectStore:
        raise DatasetSnapshotValidationError("store must be an ObjectStore")
    persisted_hash = store.put(canonical_json_bytes(snapshot))
    return StoredSnapshotReference(
        snapshot_content_hash=snapshot.content_hash,
        store_content_hash=persisted_hash,
    )


def build_role_snapshots(
    checkpoint: CPEnvelope,
    manifest: PublishedCaseManifest,
    request_plan: PublicFemRequestPlan,
    attempts: Mapping[str, FemAttempt],
    store: ObjectStore,
) -> tuple[RoleDatasetSnapshot, ...]:
    """Build deterministic public snapshots after validating every S7 outcome."""

    if type(checkpoint) is not CPEnvelope:
        raise DatasetSnapshotValidationError("checkpoint must be a CPEnvelope")
    if type(manifest) is not PublishedCaseManifest:
        raise DatasetSnapshotValidationError("manifest must be a PublishedCaseManifest")
    if type(request_plan) is not PublicFemRequestPlan:
        raise DatasetSnapshotValidationError(
            "request_plan must be a PublicFemRequestPlan"
        )
    if type(store) is not ObjectStore:
        raise DatasetSnapshotValidationError("store must be an ObjectStore")
    if manifest.content_hash != checkpoint.public_case_manifest_hash:
        raise DatasetSnapshotValidationError(
            "public manifest hash does not match CP-ENVELOPE"
        )
    try:
        PublishedCaseManifest.from_bytes(manifest.to_bytes())
    except ValueError as exc:
        raise DatasetSnapshotValidationError(
            f"public manifest violates its canonical schema or leakage groups: {exc}"
        ) from exc
    if not isinstance(attempts, Mapping):
        raise DatasetSnapshotValidationError("attempts must be a case-ID mapping")
    if (
        request_plan.campaign_hash != checkpoint.campaign_hash
        or request_plan.public_case_manifest_hash != manifest.content_hash
    ):
        raise DatasetSnapshotValidationError(
            "public FEM request plan does not match CP-ENVELOPE and manifest"
        )
    try:
        PublicFemRequestPlan.from_bytes(request_plan.to_bytes())
    except ValueError as exc:
        raise DatasetSnapshotValidationError(
            f"public FEM request plan violates its canonical schema: {exc}"
        ) from exc

    active_cases: dict[str, PublishedCase] = {}
    for case in manifest.cases:
        if case.role is EvidenceRole.HOLDOUT:
            raise DatasetSnapshotValidationError(
                "HOLDOUT cases must never enter dataset construction"
            )
        if case.status is PublishedCaseStatus.ACTIVE:
            active_cases[case.case_id] = case

    attempt_ids = set(attempts)
    active_ids = set(active_cases)
    if attempt_ids != active_ids:
        missing = sorted(active_ids - attempt_ids)
        unexpected = sorted(attempt_ids - active_ids)
        raise DatasetSnapshotValidationError(
            f"attempt case IDs must equal active public cases; "
            f"missing={missing}, unexpected={unexpected}"
        )
    plan_entries = {entry.case_id: entry for entry in request_plan.entries}
    if set(plan_entries) != active_ids:
        missing = sorted(active_ids - set(plan_entries))
        unexpected = sorted(set(plan_entries) - active_ids)
        raise DatasetSnapshotValidationError(
            "request plan case IDs must equal active public cases; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for case_id, case in active_cases.items():
        entry = plan_entries[case_id]
        if (
            entry.leakage_group_id != case.leakage_group_id
            or entry.role is not case.role
        ):
            raise DatasetSnapshotValidationError(
                f"case {case_id!r} request plan provenance does not match manifest"
            )

    by_role: dict[EvidenceRole, list[DatasetCaseRecord]] = defaultdict(list)
    for case_id in sorted(active_cases):
        case = active_cases[case_id]
        attempt = attempts[case_id]
        if type(attempt) is not FemAttempt:
            raise DatasetSnapshotValidationError(
                f"attempt for {case_id!r} must be a FemAttempt"
            )
        by_role[case.role].append(
            _case_record(case, plan_entries[case_id], attempt, request_plan, store)
        )

    return tuple(
        RoleDatasetSnapshot(
            campaign_hash=checkpoint.campaign_hash,
            public_case_manifest_hash=manifest.content_hash,
            request_plan_hash=request_plan.content_hash,
            role=role,
            cases=tuple(by_role[role]),
        )
        for role in sorted(by_role, key=lambda item: item.value)
    )


def _case_record(
    case: PublishedCase,
    plan_entry: PublicFemRequestPlanEntry,
    attempt: FemAttempt,
    request_plan: PublicFemRequestPlan,
    store: ObjectStore,
) -> DatasetCaseRecord:
    request = attempt.request
    if request.request_id != case.case_id:
        raise DatasetSnapshotValidationError(
            f"case {case.case_id!r} attempt has a different request identity"
        )
    expected_recipe_hash = request.recipe_hash
    if attempt.recipe_hash != expected_recipe_hash:
        raise DatasetSnapshotValidationError(
            f"case {case.case_id!r} recipe_hash does not match its request"
        )
    _digest(attempt.recipe_hash, "recipe_hash")
    _text(attempt.message, "attempt message")
    if (
        attempt.recipe_hash != plan_entry.request_recipe_hash
        or hash_spec(request, "content") != plan_entry.request_content_hash
        or request.mesh.mesh_id != plan_entry.mesh_id
        or hash_spec(request.mesh, "content") != plan_entry.mesh_content_hash
        or hash_spec(request.execution_profile, "content")
        != request_plan.execution_profile_hash
    ):
        raise DatasetSnapshotValidationError(
            f"case {case.case_id!r} request does not match its public request plan"
        )

    if attempt.success:
        _, expected_result_hash = validate_successful_fem_attempt(attempt, store)
        trace = SampleTrace(fem_result_hash=expected_result_hash)
        failure_code = None
        message = "succeeded"
    else:
        if (
            attempt.result is not None
            or attempt.result_hash is not None
            or attempt.store_content_hash is not None
        ):
            raise DatasetSnapshotValidationError(
                f"failed attempt {case.case_id!r} cannot carry result material"
            )
        if not isinstance(attempt.failure_code, FemFailureCode):
            raise DatasetSnapshotValidationError(
                f"failed attempt {case.case_id!r} must carry a failure code"
            )
        trace = None
        failure_code = attempt.failure_code
        message = f"failed:{failure_code.value}"

    return DatasetCaseRecord(
        case_id=case.case_id,
        leakage_group_id=case.leakage_group_id,
        role=case.role,
        primitive_coordinates=case.primitive_coordinates,
        derived_coordinates=case.derived_coordinates,
        strata=case.strata,
        request_recipe_hash=attempt.recipe_hash,
        request_content_hash=plan_entry.request_content_hash,
        mesh_id=attempt.request.mesh.mesh_id,
        mesh_content_hash=plan_entry.mesh_content_hash,
        success=attempt.success,
        sample_trace=trace,
        store_content_hash=attempt.store_content_hash,
        failure_code=failure_code,
        message=message,
    )


def validate_successful_fem_attempt(
    attempt: FemAttempt,
    store: ObjectStore,
) -> tuple[FemResult, str]:
    """Return a successful result only after recomputing every available identity."""

    if type(attempt) is not FemAttempt or not attempt.success:
        raise DatasetSnapshotValidationError(
            "field extraction requires a successful FemAttempt"
        )
    if type(store) is not ObjectStore:
        raise DatasetSnapshotValidationError("store must be an ObjectStore")
    if attempt.recipe_hash != attempt.request.recipe_hash:
        raise DatasetSnapshotValidationError(
            "recipe_hash does not match the successful FEM request"
        )
    result = attempt.result
    if type(result) is not FemResult:
        raise DatasetSnapshotValidationError(
            "successful FemAttempt must carry a FemResult"
        )
    if attempt.failure_code is not None:
        raise DatasetSnapshotValidationError(
            "successful attempt cannot carry a failure code"
        )
    _validate_result(request=attempt.request, result=result)
    expected_result_hash = hash_spec(result, "content")
    if attempt.result_hash != expected_result_hash:
        raise DatasetSnapshotValidationError(
            "result_hash does not match the successful FemResult"
        )
    payload = canonical_json_bytes({"domain": "content", "payload": result})
    expected_store_hash = store_content_hash(payload)
    if attempt.store_content_hash != expected_store_hash:
        raise DatasetSnapshotValidationError(
            "store_content_hash does not match the successful FemResult bytes"
        )
    try:
        persisted = store.get(expected_store_hash)
    except StoreError as exc:
        raise DatasetSnapshotValidationError(
            "successful FemResult bytes are not resolvable from ObjectStore"
        ) from exc
    if persisted != payload:
        raise DatasetSnapshotValidationError(
            "ObjectStore bytes do not match the successful FemResult"
        )
    return result, expected_result_hash


def validate_snapshot_attempt(
    snapshot: RoleDatasetSnapshot,
    request_plan: PublicFemRequestPlan,
    attempt: FemAttempt,
    store: ObjectStore,
) -> tuple[DatasetCaseRecord, FemResult, str]:
    """Resolve one successful attempt through its snapshot, plan, and store."""

    if type(snapshot) is not RoleDatasetSnapshot:
        raise TypeError("snapshot must be an exact RoleDatasetSnapshot")
    if type(request_plan) is not PublicFemRequestPlan:
        raise TypeError("request_plan must be an exact PublicFemRequestPlan")
    if type(attempt) is not FemAttempt:
        raise TypeError("attempt must be an exact FemAttempt")
    if type(store) is not ObjectStore:
        raise TypeError("store must be an exact ObjectStore")
    if (
        snapshot.request_plan_hash != request_plan.content_hash
        or snapshot.campaign_hash != request_plan.campaign_hash
        or snapshot.public_case_manifest_hash != request_plan.public_case_manifest_hash
    ):
        raise DatasetSnapshotValidationError(
            "snapshot does not match the public FEM request plan"
        )

    case_id = attempt.request.request_id
    matches = tuple(case for case in snapshot.cases if case.case_id == case_id)
    if len(matches) != 1:
        raise DatasetSnapshotValidationError(
            "attempt case must identify exactly one snapshot record"
        )
    record = matches[0]
    try:
        entry = request_plan.entry_by_case_id(case_id)
    except KeyError as exc:
        raise DatasetSnapshotValidationError(
            "attempt case is absent from the public FEM request plan"
        ) from exc
    request = attempt.request
    if (
        record.role is not snapshot.role
        or record.role is not entry.role
        or record.leakage_group_id != entry.leakage_group_id
        or record.request_recipe_hash != entry.request_recipe_hash
        or record.request_content_hash != entry.request_content_hash
        or record.mesh_id != entry.mesh_id
        or record.mesh_content_hash != entry.mesh_content_hash
        or request.recipe_hash != entry.request_recipe_hash
        or hash_spec(request, "content") != entry.request_content_hash
        or request.mesh.mesh_id != entry.mesh_id
        or hash_spec(request.mesh, "content") != entry.mesh_content_hash
        or hash_spec(request.execution_profile, "content")
        != request_plan.execution_profile_hash
    ):
        raise DatasetSnapshotValidationError(
            "attempt does not match snapshot and public request plan provenance"
        )
    if not record.success or record.sample_trace is None:
        raise DatasetSnapshotValidationError(
            "field extraction requires a successful snapshot case"
        )
    result, result_hash = validate_successful_fem_attempt(attempt, store)
    if (
        record.sample_trace.fem_result_hash != result_hash
        or record.store_content_hash != attempt.store_content_hash
    ):
        raise DatasetSnapshotValidationError(
            "FEM attempt does not match the trusted dataset snapshot"
        )
    return record, result, result_hash


def _validate_result(*, request: FemRequest, result: FemResult) -> None:
    fem_request = request
    if (
        result.request_id != fem_request.request_id
        or result.mesh_id != fem_request.mesh.mesh_id
        or result.element_order != fem_request.config.element_order
    ):
        raise DatasetSnapshotValidationError(
            "FemResult request identity, mesh, and element order must match its request"
        )

    mesh = fem_request.mesh
    if mesh.cell_type != "triangle" or any(len(cell) != 3 for cell in mesh.cells):
        raise DatasetSnapshotValidationError("S10 FEM snapshots require triangle cells")
    expected_dofs = len(mesh.nodes)
    if result.element_order == 2:
        edges = {
            tuple(sorted((cell[index], cell[(index + 1) % 3])))
            for cell in mesh.cells
            for index in range(3)
        }
        expected_dofs += len(edges)
    if (
        len(result.displacement_dof_coordinates) != expected_dofs
        or len(result.displacement) != expected_dofs
        or len(result.recovered_first_piola_stress) != expected_dofs
    ):
        raise DatasetSnapshotValidationError(
            "displacement and recovered stress shapes must match mesh scalar DOFs"
        )
    _vectors(result.displacement_dof_coordinates, 2, "displacement coordinates")
    _vectors(result.displacement, 2, "displacement")
    _matrices(result.recovered_first_piola_stress, "recovered stress")

    cell_count = len(mesh.cells)
    fields: tuple[Sequence[object], ...] = (
        result.quadrature_points,
        result.quadrature_weights,
        result.first_piola_stress,
        result.deformation_gradient,
    )
    if any(len(field) != cell_count for field in fields):
        raise DatasetSnapshotValidationError(
            "quadrature shapes must have one entry per mesh cell"
        )
    for cell_index, cell in enumerate(mesh.cells):
        points = result.quadrature_points[cell_index]
        weights = result.quadrature_weights[cell_index]
        stresses = result.first_piola_stress[cell_index]
        gradients = result.deformation_gradient[cell_index]
        count = len(points)
        if count == 0 or not (len(weights) == len(stresses) == len(gradients) == count):
            raise DatasetSnapshotValidationError(
                "quadrature shapes must agree within every mesh cell"
            )
        _vectors(points, 2, "quadrature points")
        _matrices(stresses, "first Piola stress")
        _matrices(gradients, "deformation gradient")
        for weight in weights:
            if not _is_finite(weight) or float(weight) <= 0.0:
                raise DatasetSnapshotValidationError(
                    "quadrature weight must be finite and strictly positive"
                )
        triangle = (
            mesh.nodes[cell[0]],
            mesh.nodes[cell[1]],
            mesh.nodes[cell[2]],
        )
        if any(not _point_in_triangle(point, triangle) for point in points):
            raise DatasetSnapshotValidationError(
                "each quadrature point must lie inside its mesh cell"
            )
        area = (
            abs(
                (triangle[1][0] - triangle[0][0]) * (triangle[2][1] - triangle[0][1])
                - (triangle[1][1] - triangle[0][1]) * (triangle[2][0] - triangle[0][0])
            )
            * 0.5
        )
        if not math.isclose(
            math.fsum(float(weight) for weight in weights),
            area,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12 * max(1.0, area),
        ):
            raise DatasetSnapshotValidationError(
                "quadrature weights must integrate to physical cell area"
            )

    for value in (
        result.total_strain_energy,
        result.external_work,
        result.total_potential_energy,
        *result.rigid_mode_multipliers,
        *result.qois.values(),
    ):
        if not _is_finite(value):
            raise DatasetSnapshotValidationError(
                "FemResult scalar outputs must be finite"
            )
    expected_qois = {qoi.name for qoi in fem_request.qois}
    if set(result.qois) != expected_qois:
        raise DatasetSnapshotValidationError(
            "FemResult QoIs must exactly match the request"
        )
    for reaction in result.reactions.values():
        if len(reaction) != 2 or any(not _is_finite(value) for value in reaction):
            raise DatasetSnapshotValidationError(
                "FemResult reactions must be finite 2D vectors"
            )
    for entry in result.convergence_history:
        if (
            type(entry.load_step) is not int
            or entry.load_step <= 0
            or type(entry.iteration) is not int
            or entry.iteration <= 0
            or not _is_finite(entry.residual_norm)
            or not _is_finite(entry.increment_norm)
        ):
            raise DatasetSnapshotValidationError(
                "FemResult convergence history must be finite and indexed positively"
            )


def _vectors(values: Sequence[Sequence[object]], width: int, label: str) -> None:
    for value in values:
        if len(value) != width or any(not _is_finite(component) for component in value):
            raise DatasetSnapshotValidationError(
                f"{label} must contain finite vectors of width {width}"
            )


def _matrices(values: Sequence[Sequence[Sequence[object]]], label: str) -> None:
    for matrix in values:
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise DatasetSnapshotValidationError(f"{label} must contain 3x3 matrices")
        if any(not _is_finite(component) for row in matrix for component in row):
            raise DatasetSnapshotValidationError(f"{label} must be finite")


def _is_finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _point_in_triangle(
    point: Sequence[float],
    triangle: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> bool:
    x, y = point
    (x0, y0), (x1, y1), (x2, y2) = triangle
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if denominator == 0.0:
        return False
    first = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denominator
    second = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denominator
    third = 1.0 - first - second
    tolerance = 64.0 * math.ulp(1.0)
    return min(first, second, third) >= -tolerance


__all__ = (
    "DatasetCaseRecord",
    "DatasetSnapshotValidationError",
    "RoleDatasetSnapshot",
    "SampleTrace",
    "StoredSnapshotReference",
    "build_role_snapshots",
    "persist_role_snapshot",
    "validate_successful_fem_attempt",
    "validate_snapshot_attempt",
)
