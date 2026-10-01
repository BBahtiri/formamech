"""Selective, leakage-safe access to the ratified S10 DogBone FEM results."""

from __future__ import annotations

import itertools
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, cast

from campaigns.dogbone_v1.family import CompiledFamily, compiled_family, parameter_point
from campaigns.dogbone_v1.s10_closeout import _load_run_execution_profile
from campaigns.dogbone_v1.s10_evidence import DOGBONE_S10_DATASET_POLICY
from campaigns.dogbone_v1.s10_production import (
    ProductionFemRunLedger,
    _attempt_from_record,
)
from campaigns.dogbone_v1.s10_requests import (
    _boundary_conditions,
    _load_public_sources,
    _material,
    _number,
    _solver_config,
)
from campaigns.dogbone_v1.s11_contingency import (
    ArchitectureScreenProtocol,
)
from campaigns.dogbone_v1.s11_contingency_report import (
    S10_PRODUCTION_REPORT_CONTENT_HASH,
    S10ReportCaseIndex,
    S10ReportIndex,
    S10RoleSnapshotIndex,
    load_s10_report_index,
)
from campaigns.dogbone_v1.s11_contingency_run import (
    TrainScreenDecision,
    load_committed_protocol,
    validate_train_screen_decision,
    verify_s10_screen_run_index,
)
from piginot.contracts import (
    CanonicalMesh,
    ExecutionProfile,
    hash_spec,
    strict_json_loads,
)
from piginot.datasets import (
    CollocationBatch,
    DatasetCaseRecord,
    RoleDatasetSnapshot,
    SampleTrace,
    sample_request_collocation,
)
from piginot.datasets.request_plan import (
    PublicFemRequestPlan,
    PublicFemRequestPlanEntry,
)
from piginot.doe import EvidenceRole, PublishedCase, PublishedCaseManifest
from piginot.doe.publication import PublishedCaseStatus
from piginot.fem import FemRequest, FemResult
from piginot.geometry import GeometryValidationError
from piginot.geometry.canonicalize import canonicalize_mesh
from piginot.geometry.models import CellType, MeshPayload
from piginot.morphing import apply_morph
from piginot.store import ObjectStore, StoreError

_S10_COMPILED_FAMILY_HASH = (
    "a8143a75e62c614d40f52049cfc88c1cda74d38c9bd488cc3bed4c5678d973a4"
)
_S10_REFERENCE_MESH_ID = (
    "3bfc1043f5aa9e989469adacb0ee1d0b483825ffabcd5dfac4bf9caca580d82c"
)
_S10_VALIDITY_POLICY_HASH = (
    "b3c2df59eed939f1aa14a69d05a6fd2dc57be571ffba33c6355982713cc6931d"
)
_ACTIVE_CASE_COUNT = 344
_GEOMETRY_AXES = ("L_total", "W_grip", "W_gauge", "R_fillet")
_REQUIRED_PRIMITIVES = frozenset(
    (*_GEOMETRY_AXES, "elastic_modulus", "poisson_ratio", "nominal_axial_strain")
)
_SOURCE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class S10SourceBindings:
    """Immutable identities verified before opening any FEM result bytes."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    report_content_hash: str
    public_case_manifest_hash: str
    request_plan_hash: str
    execution_profile_hash: str
    ledger_content_hash: str
    active_case_count: int
    successful_case_count: int

    def __post_init__(self) -> None:
        for name in (
            "report_content_hash",
            "public_case_manifest_hash",
            "request_plan_hash",
            "execution_profile_hash",
            "ledger_content_hash",
        ):
            _digest(getattr(self, name), name)
        if self.active_case_count != _ACTIVE_CASE_COUNT:
            raise ValueError("active_case_count must remain 344")
        if self.successful_case_count != self.active_case_count:
            raise ValueError("every active S10 case must be successful")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_spec(self, "content"))


@dataclass(frozen=True, slots=True)
class VerifiedS10Case:
    """One fully rehydrated S10 case after all public identities agree."""

    case: PublishedCase
    request: FemRequest
    result: FemResult

    def __post_init__(self) -> None:
        if type(self.case) is not PublishedCase:
            raise TypeError("case must be an exact PublishedCase")
        if type(self.request) is not FemRequest or type(self.result) is not FemResult:
            raise TypeError("request and result must be exact FEM contracts")
        if self.case.case_id != self.request.request_id:
            raise ValueError("published case and FEM request IDs differ")
        if self.result.request_id != self.request.request_id:
            raise ValueError("FEM result and request IDs differ")


@dataclass(frozen=True, slots=True)
class VerifiedS10Request:
    """One public case and its rebuilt request, verified without result bytes."""

    case: PublishedCase
    request: FemRequest

    def __post_init__(self) -> None:
        if type(self.case) is not PublishedCase:
            raise TypeError("case must be an exact PublishedCase")
        if type(self.request) is not FemRequest:
            raise TypeError("request must be an exact FemRequest")
        if self.case.case_id != self.request.request_id:
            raise ValueError("published case and FEM request IDs differ")


class _SelectiveS10SourceCore:
    """Shared resolver mechanics behind distinct production and fixture sources."""

    __slots__ = (
        "_bindings",
        "_cases",
        "_development_opened",
        "_development_access_record",
        "_entries",
        "_family",
        "_ledger_records",
        "_ledger",
        "_manifest",
        "_profile",
        "_protocol",
        "_report",
        "_report_cases",
        "_request_plan",
        "_store",
    )

    def __init__(
        self,
        *,
        bindings: S10SourceBindings,
        protocol: ArchitectureScreenProtocol,
        manifest: PublishedCaseManifest,
        execution_profile: ExecutionProfile,
        request_plan: PublicFemRequestPlan,
        ledger: ProductionFemRunLedger,
        report: S10ReportIndex,
        store: ObjectStore,
        family: CompiledFamily,
        development_access_record: Path,
    ) -> None:
        if type(bindings) is not S10SourceBindings:
            raise TypeError("bindings must be exact S10SourceBindings")
        if not isinstance(development_access_record, Path):
            raise TypeError("development_access_record must be a pathlib.Path")
        if not development_access_record.is_absolute():
            raise ValueError("development_access_record must be absolute")
        if (
            development_access_record.parent.is_symlink()
            or not development_access_record.parent.is_dir()
        ):
            raise ValueError(
                "development access record parent must be a real directory"
            )
        self._bindings = bindings
        self._protocol = protocol
        self._profile = execution_profile
        self._store = store
        self._family = family
        self._manifest = manifest
        self._request_plan = request_plan
        self._ledger = ledger
        self._report = report
        self._cases = {case.case_id: case for case in manifest.cases}
        self._entries = {entry.case_id: entry for entry in request_plan.entries}
        self._ledger_records = {record.case_id: record for record in ledger.records}
        self._report_cases = {item.case_id: item for item in report.cases}
        self._development_access_record = development_access_record
        self._development_opened = False

    @property
    def bindings(self) -> S10SourceBindings:
        return self._bindings

    @property
    def development_opened(self) -> bool:
        return self._development_opened or self._development_access_record.exists()

    @property
    def fit_case_ids(self) -> tuple[str, ...]:
        return self._protocol.fit_case_ids

    @property
    def train_screen_case_ids(self) -> tuple[str, ...]:
        return self._protocol.screen_case_ids

    @property
    def train_snapshot(self) -> RoleDatasetSnapshot:
        """Rebuild the exact TRAIN snapshot without resolving FEM result bytes."""

        if self._manifest.content_hash != self._bindings.public_case_manifest_hash:
            raise ValueError("retained manifest differs from verified bindings")
        if self._request_plan.content_hash != self._bindings.request_plan_hash:
            raise ValueError("retained request plan differs from verified bindings")
        if hash_spec(self._ledger, "content") != self._bindings.ledger_content_hash:
            raise ValueError("retained ledger differs from verified bindings")
        if self._report.content_hash != self._bindings.report_content_hash:
            raise ValueError("retained report differs from verified bindings")
        snapshot = _build_train_snapshot_from_public_metadata(
            manifest=self._manifest,
            request_plan=self._request_plan,
            ledger=self._ledger,
            report=self._report,
        )
        indices = tuple(
            item
            for item in self._report.role_snapshots
            if item.role is EvidenceRole.TRAIN
        )
        if len(indices) != 1:
            raise ValueError("retained report lacks exactly one TRAIN snapshot index")
        index = indices[0]
        if (
            index.case_count != len(snapshot.cases)
            or index.successful_case_count != len(snapshot.cases)
            or index.failed_case_count != 0
            or index.snapshot_content_hash != snapshot.content_hash
            or self._report.training_snapshot_hash != snapshot.content_hash
            or self._protocol.training_snapshot_hash != snapshot.content_hash
        ):
            raise ValueError("retained TRAIN snapshot differs from verified evidence")
        return snapshot

    def replay_train_collocation(
        self,
        verified_case: VerifiedS10Case,
    ) -> CollocationBatch:
        """Recreate and verify one persisted TRAIN collocation batch exactly."""

        return self._replay_train_collocation(verified_case, self.train_snapshot)

    def _replay_train_collocation(
        self,
        verified_case: VerifiedS10Case,
        snapshot: RoleDatasetSnapshot,
    ) -> CollocationBatch:
        if type(verified_case) is not VerifiedS10Case:
            raise TypeError("verified_case must be an exact VerifiedS10Case")
        return self._replay_train_collocation_from_parts(
            verified_case.case,
            verified_case.request,
            snapshot,
        )

    def _replay_train_collocation_from_parts(
        self,
        case: PublishedCase,
        request: FemRequest,
        snapshot: RoleDatasetSnapshot,
    ) -> CollocationBatch:
        if type(snapshot) is not RoleDatasetSnapshot:
            raise TypeError("snapshot must be an exact RoleDatasetSnapshot")
        if snapshot.role is not EvidenceRole.TRAIN:
            raise ValueError("collocation replay requires the TRAIN snapshot")
        if (
            case.role is not EvidenceRole.TRAIN
            or case.status is not PublishedCaseStatus.ACTIVE
            or self._cases.get(case.case_id) != case
        ):
            raise ValueError("collocation replay requires an active TRAIN case")
        evidence = self._report_cases.get(case.case_id)
        if evidence is None or evidence.role is not EvidenceRole.TRAIN:
            raise ValueError("TRAIN case is absent from the S10 report")

        policy = DOGBONE_S10_DATASET_POLICY
        batch = sample_request_collocation(
            snapshot,
            case.case_id,
            request,
            stream_index=policy.stream_index,
            interior_count=policy.interior_count,
            boundary_count=policy.boundary_count,
            interface_count=policy.interface_count,
            qoi_quotas=policy.qoi_quotas,
        )
        if (
            not evidence.collocation_replay_verified
            or evidence.collocation_point_count != len(batch.points)
            or evidence.collocation_point_count != policy.collocation_points_per_case
            or evidence.collocation_stream_hash != batch.sampler_stream_hash
            or evidence.collocation_batch_content_hash != batch.content_hash
        ):
            raise ValueError(
                "S10 collocation identities differ from deterministic replay"
            )
        try:
            persisted = self._store.get(evidence.collocation_store_content_hash)
        except StoreError as exc:
            raise ValueError("S10 collocation store bytes are not resolvable") from exc
        if persisted != batch.to_bytes():
            raise ValueError("S10 collocation store bytes differ from replay")
        return batch

    def iter_train_geometry(
        self,
        case_ids: tuple[str, ...],
    ) -> Iterator[VerifiedS10Request]:
        """Rebuild verified TRAIN requests without resolving FEM result bytes.

        Every identity that ``_load_case`` checks before reading a result is
        checked here: role and status agree across manifest, request plan and
        report, and the rebuilt request reproduces the planned recipe, content
        and mesh hashes. Only registered TRAIN cases are accepted.
        """

        if type(case_ids) is not tuple:
            raise TypeError("case_ids must be a tuple")
        registered = set(self._protocol.fit_case_ids) | set(
            self._protocol.screen_case_ids
        )
        for case_id in case_ids:
            if case_id not in registered:
                raise PermissionError(f"{case_id} is not a registered TRAIN case")
            yield self._load_request(case_id, EvidenceRole.TRAIN)

    def replay_train_collocation_for_request(
        self,
        verified_request: VerifiedS10Request,
    ) -> CollocationBatch:
        """Replay one TRAIN collocation batch from a verified request only."""

        if type(verified_request) is not VerifiedS10Request:
            raise TypeError("verified_request must be an exact VerifiedS10Request")
        return self._replay_train_collocation_from_parts(
            verified_request.case,
            verified_request.request,
            self.train_snapshot,
        )

    def iter_fit(self) -> Iterator[VerifiedS10Case]:
        return self._iter_cases(self._protocol.fit_case_ids, EvidenceRole.TRAIN)

    def iter_fit_with_collocation(
        self,
    ) -> Iterator[tuple[VerifiedS10Case, CollocationBatch]]:
        """Resolve each fit case with one shared verified TRAIN snapshot."""

        snapshot = self.train_snapshot
        for verified_case in self.iter_fit():
            yield (
                verified_case,
                self._replay_train_collocation(
                    verified_case,
                    snapshot,
                ),
            )

    def iter_train_screen(self) -> Iterator[VerifiedS10Case]:
        return self._iter_cases(self._protocol.screen_case_ids, EvidenceRole.TRAIN)

    def _open_development(
        self,
        decision: TrainScreenDecision | None = None,
    ) -> Iterator[VerifiedS10Case]:
        if self._development_opened or self._development_access_record.exists():
            raise RuntimeError("DEVELOPMENT labels have already been opened")
        if decision is None:
            raise TypeError("DEVELOPMENT requires an exact TrainScreenDecision")
        validate_train_screen_decision(self._protocol, decision)
        if decision.source_bindings_hash != self._bindings.content_hash:
            raise ValueError("TRAIN decision is not bound to this S10 source")
        if decision.selection.fallback_to_baseline:
            raise ValueError("DEVELOPMENT remains closed after a TRAIN fallback")
        self._write_development_access_record(decision)
        self._development_opened = True
        return self._iter_cases(
            self._protocol.development_case_ids,
            EvidenceRole.DEVELOPMENT,
        )

    def _write_development_access_record(self, decision: TrainScreenDecision) -> None:
        payload = decision.to_bytes() + b"\n"
        try:
            descriptor = os.open(
                self._development_access_record,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                0o600,
            )
        except FileExistsError as exc:
            raise RuntimeError("DEVELOPMENT labels have already been opened") from exc
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def _iter_cases(
        self,
        case_ids: tuple[str, ...],
        expected_role: EvidenceRole,
    ) -> Iterator[VerifiedS10Case]:
        for case_id in case_ids:
            yield self._load_case(case_id, expected_role)

    def _load_request(
        self,
        case_id: str,
        expected_role: EvidenceRole,
    ) -> VerifiedS10Request:
        case = self._cases[case_id]
        entry = self._entries[case_id]
        evidence = self._report_cases[case_id]
        if (
            case.status is not PublishedCaseStatus.ACTIVE
            or case.role is not expected_role
            or entry.role is not expected_role
            or evidence.role is not expected_role
        ):
            raise ValueError(f"{case_id} role/status differs across S10 evidence")
        request = _rebuild_request(case, self._profile, self._family)
        if not _request_matches_plan(request, entry):
            request = self._portable_request(case_id, entry, request)
        if not _request_matches_plan(request, entry):
            raise ValueError(f"{case_id} runtime request differs from S10 request plan")
        return VerifiedS10Request(case, request)

    def _portable_request(
        self,
        case_id: str,
        entry: PublicFemRequestPlanEntry,
        request: FemRequest,
    ) -> FemRequest:
        """Recover platform-dependent float bits; exact plan hashes still decide.

        S10 requests were materialized on Windows. Elsewhere, libm may round a
        few morphed node coordinates or the ``acos``-derived mesh-quality angles
        differently in the last place. Node coordinates are taken from the stored
        FEM result of the same case when the geometric ``mesh_id`` differs, and
        the two angle diagnostics are searched within a few ulps. Nothing is
        accepted unless every request-plan hash then matches bit for bit.
        """

        mesh = request.mesh
        if mesh.mesh_id != entry.mesh_id:
            stored = self._stored_vertex_mesh(case_id, mesh)
            if stored is None or stored.mesh_id != entry.mesh_id:
                return request
            mesh = stored
        repaired = _repair_quality_angle_ulps(mesh, entry.mesh_content_hash)
        if repaired is None:
            return request
        return replace(request, mesh=repaired)

    def _stored_vertex_mesh(
        self,
        case_id: str,
        mesh: CanonicalMesh,
    ) -> CanonicalMesh | None:
        evidence = self._report_cases[case_id]
        try:
            document = strict_json_loads(self._store.get(evidence.store_content_hash))
        except (StoreError, ValueError):
            return None
        if not isinstance(document, dict):
            return None
        payload = document.get("payload")
        if not isinstance(payload, dict):
            return None
        coordinates = payload.get("displacement_dof_coordinates")
        if not isinstance(coordinates, list) or len(coordinates) < len(mesh.nodes):
            return None
        nodes = tuple(
            (float(point[0]), float(point[1]))
            for point in coordinates[: len(mesh.nodes)]
        )
        try:
            return canonicalize_mesh(
                MeshPayload(
                    nodes=nodes,
                    cells=mesh.cells,
                    cell_type=CellType(mesh.cell_type),
                    boundary_facets=mesh.boundary_facets,
                    boundary_tags=mesh.semantic_tags,
                    region_ids=mesh.region_ids,
                    unit=mesh.unit,
                    source_provenance=mesh.source_provenance,
                )
            )
        except (GeometryValidationError, TypeError, ValueError):
            return None

    def _load_case(
        self,
        case_id: str,
        expected_role: EvidenceRole,
    ) -> VerifiedS10Case:
        verified_request = self._load_request(case_id, expected_role)
        case = verified_request.case
        request = verified_request.request
        entry = self._entries[case_id]
        record = self._ledger_records[case_id]
        evidence = self._report_cases[case_id]
        if (
            record.request_recipe_hash != entry.request_recipe_hash
            or record.request_content_hash != entry.request_content_hash
            or record.result_hash != evidence.fem_result_hash
            or record.store_content_hash != evidence.store_content_hash
        ):
            raise ValueError(f"{case_id} ledger/report identities differ")
        attempt = _attempt_from_record(record, request, self._store)
        if not attempt.success or type(attempt.result) is not FemResult:
            raise ValueError(f"{case_id} did not resolve to a successful FEM result")
        return VerifiedS10Case(case, request, attempt.result)


class SelectiveS10Source(_SelectiveS10SourceCore):
    """Production source constructed only from pinned, strictly verified evidence."""

    __slots__ = ()

    def __init__(
        self,
        *,
        _token: object,
        bindings: S10SourceBindings,
        protocol: ArchitectureScreenProtocol,
        manifest: PublishedCaseManifest,
        execution_profile: ExecutionProfile,
        request_plan: PublicFemRequestPlan,
        ledger: ProductionFemRunLedger,
        report: S10ReportIndex,
        store: ObjectStore,
        family: CompiledFamily,
        development_access_record: Path,
    ) -> None:
        if _token is not _SOURCE_TOKEN:
            raise TypeError("SelectiveS10Source must be opened through its verifier")
        super().__init__(
            bindings=bindings,
            protocol=protocol,
            manifest=manifest,
            execution_profile=execution_profile,
            request_plan=request_plan,
            ledger=ledger,
            report=report,
            store=store,
            family=family,
            development_access_record=development_access_record,
        )


class _FixtureSelectiveS10Source(_SelectiveS10SourceCore):
    """Distinct test resolver that cannot satisfy production exact-type checks."""

    __slots__ = ()

    @classmethod
    def from_components(
        cls,
        *,
        protocol: ArchitectureScreenProtocol,
        manifest: PublishedCaseManifest,
        execution_profile: ExecutionProfile,
        request_plan: PublicFemRequestPlan,
        ledger: ProductionFemRunLedger,
        report: S10ReportIndex,
        store: ObjectStore,
        development_access_record: Path,
    ) -> _FixtureSelectiveS10Source:
        if type(store) is not ObjectStore:
            raise TypeError("store must be an exact ObjectStore")
        bindings = _verify_s10_source_bindings_core(
            protocol,
            manifest=manifest,
            execution_profile=execution_profile,
            request_plan=request_plan,
            ledger=ledger,
            report=report,
        )
        return cls(
            bindings=bindings,
            protocol=protocol,
            manifest=manifest,
            execution_profile=execution_profile,
            request_plan=request_plan,
            ledger=ledger,
            report=report,
            store=store,
            family=_verified_runtime_family(),
            development_access_record=development_access_record,
        )


def _build_train_snapshot_from_public_metadata(
    *,
    manifest: PublishedCaseManifest,
    request_plan: PublicFemRequestPlan,
    ledger: ProductionFemRunLedger,
    report: S10ReportIndex,
) -> RoleDatasetSnapshot:
    """Rehydrate S10's TRAIN snapshot from public identities, without result I/O."""

    if type(manifest) is not PublishedCaseManifest:
        raise TypeError("manifest must be an exact PublishedCaseManifest")
    if type(request_plan) is not PublicFemRequestPlan:
        raise TypeError("request_plan must be an exact PublicFemRequestPlan")
    if type(ledger) is not ProductionFemRunLedger:
        raise TypeError("ledger must be an exact ProductionFemRunLedger")
    if type(report) is not S10ReportIndex:
        raise TypeError("report must be an exact S10ReportIndex")
    if (
        request_plan.public_case_manifest_hash != manifest.content_hash
        or ledger.public_case_manifest_hash != manifest.content_hash
        or report.public_case_manifest_hash != manifest.content_hash
    ):
        raise ValueError("TRAIN snapshot sources disagree on the public manifest")
    if (
        ledger.campaign_hash != request_plan.campaign_hash
        or report.campaign_hash != request_plan.campaign_hash
        or ledger.request_plan_hash != request_plan.content_hash
        or report.request_plan_hash != request_plan.content_hash
    ):
        raise ValueError("TRAIN snapshot sources disagree on the request plan")

    train_cases = {
        case.case_id: case
        for case in manifest.cases
        if case.status is PublishedCaseStatus.ACTIVE and case.role is EvidenceRole.TRAIN
    }
    train_ids = tuple(sorted(train_cases))
    if train_ids != report.training_case_ids:
        raise ValueError("TRAIN snapshot population differs from the S10 report")
    entries = {entry.case_id: entry for entry in request_plan.entries}
    records = {record.case_id: record for record in ledger.records}
    evidence = {item.case_id: item for item in report.cases}

    case_records: list[DatasetCaseRecord] = []
    for case_id in train_ids:
        case = train_cases[case_id]
        try:
            entry = entries[case_id]
            record = records[case_id]
            item = evidence[case_id]
        except KeyError as exc:
            raise ValueError(
                f"TRAIN case {case_id!r} is absent from public S10 metadata"
            ) from exc
        if (
            entry.role is not EvidenceRole.TRAIN
            or item.role is not EvidenceRole.TRAIN
            or entry.leakage_group_id != case.leakage_group_id
        ):
            raise ValueError(f"TRAIN case {case_id!r} provenance differs")
        if (
            not record.success
            or record.result_hash is None
            or record.store_content_hash is None
            or record.failure_code is not None
            or record.message != "succeeded"
        ):
            raise ValueError(f"TRAIN case {case_id!r} is not a successful S10 result")
        if (
            record.request_recipe_hash != entry.request_recipe_hash
            or record.request_content_hash != entry.request_content_hash
            or item.fem_result_hash != record.result_hash
            or item.store_content_hash != record.store_content_hash
        ):
            raise ValueError(f"TRAIN case {case_id!r} public identities differ")
        case_records.append(
            DatasetCaseRecord(
                case_id=case.case_id,
                leakage_group_id=case.leakage_group_id,
                role=case.role,
                primitive_coordinates=case.primitive_coordinates,
                derived_coordinates=case.derived_coordinates,
                strata=case.strata,
                request_recipe_hash=entry.request_recipe_hash,
                request_content_hash=entry.request_content_hash,
                mesh_id=entry.mesh_id,
                mesh_content_hash=entry.mesh_content_hash,
                success=True,
                sample_trace=SampleTrace(fem_result_hash=record.result_hash),
                store_content_hash=record.store_content_hash,
                failure_code=None,
                message="succeeded",
            )
        )

    snapshot = RoleDatasetSnapshot(
        campaign_hash=request_plan.campaign_hash,
        public_case_manifest_hash=manifest.content_hash,
        request_plan_hash=request_plan.content_hash,
        role=EvidenceRole.TRAIN,
        cases=tuple(case_records),
    )
    return snapshot


def verify_s10_source_bindings(
    protocol: ArchitectureScreenProtocol,
    *,
    manifest: PublishedCaseManifest,
    execution_profile: ExecutionProfile,
    request_plan: PublicFemRequestPlan,
    ledger: ProductionFemRunLedger,
    report: S10ReportIndex,
) -> S10SourceBindings:
    """Bind the exact committed protocol and pinned public S10 report."""

    if protocol != load_committed_protocol():
        raise ValueError("protocol differs from the committed protocol")
    if report != load_s10_report_index():
        raise ValueError("report content hash/preimage differs from the pinned report")
    return _verify_s10_source_bindings_core(
        protocol,
        manifest=manifest,
        execution_profile=execution_profile,
        request_plan=request_plan,
        ledger=ledger,
        report=report,
    )


def _verify_s10_source_bindings_core(
    protocol: ArchitectureScreenProtocol,
    *,
    manifest: PublishedCaseManifest,
    execution_profile: ExecutionProfile,
    request_plan: PublicFemRequestPlan,
    ledger: ProductionFemRunLedger,
    report: S10ReportIndex,
) -> S10SourceBindings:
    """Validate coherent components; production additionally pins their preimages."""

    if type(protocol) is not ArchitectureScreenProtocol:
        raise TypeError("protocol must be an exact ArchitectureScreenProtocol")
    if type(manifest) is not PublishedCaseManifest:
        raise TypeError("manifest must be an exact PublishedCaseManifest")
    if type(execution_profile) is not ExecutionProfile:
        raise TypeError("execution_profile must be an exact ExecutionProfile")
    if type(report) is not S10ReportIndex:
        raise TypeError("report must be an exact S10ReportIndex")
    run_index = verify_s10_screen_run_index(
        protocol,
        request_plan=request_plan,
        ledger=ledger,
    )
    profile_hash = hash_spec(execution_profile, "content")
    if manifest.content_hash != protocol.public_case_manifest_hash:
        raise ValueError("manifest differs from the contingency protocol")
    if report.public_case_manifest_hash != manifest.content_hash:
        raise ValueError("S10 report differs from the public manifest")
    if (
        request_plan.campaign_hash != report.campaign_hash
        or ledger.campaign_hash != report.campaign_hash
    ):
        raise ValueError("S10 campaign hashes differ")
    if report.request_plan_hash != request_plan.content_hash:
        raise ValueError("S10 report differs from the request plan")
    if (
        report.execution_profile_hash != profile_hash
        or request_plan.execution_profile_hash != profile_hash
    ):
        raise ValueError("S10 execution profile hashes differ")
    if (
        ledger.run_id != report.run_id
        or ledger.source_revision != report.fem_source_revision
    ):
        raise ValueError("S10 run identity differs from the closeout report")
    if report.training_snapshot_hash != protocol.training_snapshot_hash:
        raise ValueError("S10 TRAIN snapshot differs from the contingency protocol")
    if report.development_snapshot_hash != protocol.development_snapshot_hash:
        raise ValueError(
            "S10 DEVELOPMENT snapshot differs from the contingency protocol"
        )
    if (
        report.failed_case_count != 0
        or report.normalization_non_train_source_count != 0
        or report.normalization_non_active_source_count != 0
        or not report.normalization_snapshot_binding_verified
    ):
        raise ValueError(
            "S10 report does not satisfy its public normalization/outcome boundary"
        )

    active = {
        case.case_id: case
        for case in manifest.cases
        if case.status is PublishedCaseStatus.ACTIVE
    }
    entries = {entry.case_id: entry for entry in request_plan.entries}
    records = {record.case_id: record for record in ledger.records}
    evidence = {item.case_id: item for item in report.cases}
    if len(active) != _ACTIVE_CASE_COUNT or not (
        set(active) == set(entries) == set(records) == set(evidence)
    ):
        raise ValueError("S10 sources must cover the exact 344-case active population")
    if ledger.success_count != _ACTIVE_CASE_COUNT or any(
        not record.success for record in records.values()
    ):
        raise ValueError("S10 source ledger must contain 344 successful outcomes")
    train_ids = tuple(
        sorted(
            case_id
            for case_id, case in active.items()
            if case.role is EvidenceRole.TRAIN
        )
    )
    if report.training_case_ids != train_ids:
        raise ValueError("S10 normalization population differs from active TRAIN cases")
    snapshots = {item.role: item for item in report.role_snapshots}
    expected_counts = {
        EvidenceRole.TRAIN: 256,
        EvidenceRole.DEVELOPMENT: 64,
        EvidenceRole.PILOT: 24,
    }
    for role, count in expected_counts.items():
        snapshot = snapshots.get(role)
        if snapshot is None or (
            snapshot.case_count,
            snapshot.successful_case_count,
            snapshot.failed_case_count,
        ) != (count, count, 0):
            raise ValueError(f"S10 {role.value} snapshot population differs")
    if snapshots[EvidenceRole.TRAIN].snapshot_content_hash != (
        protocol.training_snapshot_hash
    ):
        raise ValueError(
            "S10 TRAIN role snapshot differs from the contingency protocol"
        )
    if snapshots[EvidenceRole.DEVELOPMENT].snapshot_content_hash != (
        protocol.development_snapshot_hash
    ):
        raise ValueError(
            "S10 DEVELOPMENT role snapshot differs from the contingency protocol"
        )
    for case_id, case in active.items():
        entry = entries[case_id]
        record = records[case_id]
        item = evidence[case_id]
        if (
            entry.role is not case.role
            or entry.leakage_group_id != case.leakage_group_id
        ):
            raise ValueError(f"{case_id} request-plan provenance differs from manifest")
        if item.role is not case.role:
            raise ValueError(f"{case_id} report role differs from manifest")
        if (
            record.request_recipe_hash != entry.request_recipe_hash
            or record.request_content_hash != entry.request_content_hash
        ):
            raise ValueError(f"{case_id} ledger request identities differ from plan")
        if record.result_hash != item.fem_result_hash:
            raise ValueError(f"{case_id} report result hash differs from ledger")
        if record.store_content_hash != item.store_content_hash:
            raise ValueError(f"{case_id} report store hash differs from ledger")
    return S10SourceBindings(
        report_content_hash=report.content_hash,
        public_case_manifest_hash=manifest.content_hash,
        request_plan_hash=run_index.request_plan_hash,
        execution_profile_hash=profile_hash,
        ledger_content_hash=run_index.ledger_content_hash,
        active_case_count=len(active),
        successful_case_count=ledger.success_count,
    )


def open_selective_s10_source(
    run_root: Path,
    *,
    development_access_record: Path,
) -> SelectiveS10Source:
    """Open the external S10 run only after all tracked/public identities load."""

    if not isinstance(run_root, Path):
        raise TypeError("run_root must be a pathlib.Path")
    if run_root.is_symlink() or not run_root.is_dir():
        raise ValueError("run_root must be an existing non-symlink directory")
    if not isinstance(development_access_record, Path):
        raise TypeError("development_access_record must be a pathlib.Path")
    resolved_run_root = run_root.resolve(strict=True)
    resolved_record_parent = development_access_record.parent.resolve(strict=True)
    if (
        resolved_record_parent == resolved_run_root
        or resolved_record_parent.is_relative_to(resolved_run_root)
    ):
        raise ValueError(
            "development access record must remain outside the S10 run root"
        )
    protocol = load_committed_protocol()
    _, manifest = _load_public_sources()
    profile = _load_run_execution_profile(run_root)
    plan = PublicFemRequestPlan.from_bytes(
        (run_root / "request-plan.json").read_bytes()
    )
    ledger = ProductionFemRunLedger.from_bytes((run_root / "ledger.json").read_bytes())
    report = load_s10_report_index()
    store = ObjectStore.open(run_root / "object-store")
    bindings = verify_s10_source_bindings(
        protocol=protocol,
        manifest=manifest,
        execution_profile=profile,
        request_plan=plan,
        ledger=ledger,
        report=report,
    )
    return SelectiveS10Source(
        _token=_SOURCE_TOKEN,
        bindings=bindings,
        protocol=protocol,
        manifest=manifest,
        execution_profile=profile,
        request_plan=plan,
        ledger=ledger,
        report=report,
        store=store,
        family=_verified_runtime_family(),
        development_access_record=development_access_record,
    )


def _verified_runtime_family() -> CompiledFamily:
    family = compiled_family()
    if hash_spec(family, "content") != _S10_COMPILED_FAMILY_HASH:
        raise ValueError("runtime compiled family differs from S10")
    if family.reference_mesh.mesh_id != _S10_REFERENCE_MESH_ID:
        raise ValueError("runtime reference mesh differs from S10")
    if family.membership.validity_policy_hash != _S10_VALIDITY_POLICY_HASH:
        raise ValueError("runtime validity policy differs from S10")
    return family


def _rebuild_request(
    case: PublishedCase,
    profile: ExecutionProfile,
    family: CompiledFamily,
) -> FemRequest:
    if frozenset(case.primitive_coordinates) != _REQUIRED_PRIMITIVES:
        raise ValueError(f"{case.case_id} primitive axes differ from S10")
    theta = parameter_point(*(_number(case, name) for name in _GEOMETRY_AXES))
    morph = apply_morph(
        family.reference_mesh, family.morph_map, family.membership, theta
    )
    if not morph.accepted or morph.mesh is None:
        raise ValueError(f"{case.case_id} no longer satisfies ratified morphing")
    return FemRequest(
        request_id=case.case_id,
        mesh=morph.mesh,
        material=_material(case),
        boundary_conditions=_boundary_conditions(case),
        loads=(),
        qois=(),
        config=_solver_config(),
        execution_profile=profile,
    )


def _request_matches_plan(
    request: FemRequest,
    entry: PublicFemRequestPlanEntry,
) -> bool:
    return (
        request.recipe_hash == entry.request_recipe_hash
        and hash_spec(request, "content") == entry.request_content_hash
        and request.mesh.mesh_id == entry.mesh_id
        and hash_spec(request.mesh, "content") == entry.mesh_content_hash
    )


_PORTABLE_QUALITY_ANGLES = ("minimum_angle_degrees", "maximum_angle_degrees")
_PORTABLE_ULP_RADIUS = 4


def _ulp_neighbourhood(value: float, radius: int) -> tuple[float, ...]:
    values = [value]
    lower = upper = value
    for _ in range(radius):
        lower = math.nextafter(lower, -math.inf)
        upper = math.nextafter(upper, math.inf)
        values.extend((lower, upper))
    return tuple(values)


def _repair_quality_angle_ulps(
    mesh: CanonicalMesh,
    expected_content_hash: str,
) -> CanonicalMesh | None:
    """Return the mesh whose angle diagnostics reproduce the planned hash."""

    if hash_spec(mesh, "content") == expected_content_hash:
        return mesh
    metrics = dict(mesh.quality_metrics)
    if any(name not in metrics for name in _PORTABLE_QUALITY_ANGLES):
        return None
    for minimum, maximum in itertools.product(
        *(
            _ulp_neighbourhood(float(metrics[name]), _PORTABLE_ULP_RADIUS)
            for name in _PORTABLE_QUALITY_ANGLES
        )
    ):
        candidate = replace(
            mesh,
            quality_metrics={
                **metrics,
                _PORTABLE_QUALITY_ANGLES[0]: minimum,
                _PORTABLE_QUALITY_ANGLES[1]: maximum,
            },
        )
        if hash_spec(candidate, "content") == expected_content_hash:
            return candidate
    return None


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ValueError(f"{label} must be a lowercase 256-bit digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a lowercase 256-bit digest") from exc
    return value


__all__ = (
    "S10_PRODUCTION_REPORT_CONTENT_HASH",
    "S10ReportCaseIndex",
    "S10ReportIndex",
    "S10RoleSnapshotIndex",
    "S10SourceBindings",
    "SelectiveS10Source",
    "VerifiedS10Case",
    "VerifiedS10Request",
    "load_s10_report_index",
    "open_selective_s10_source",
    "verify_s10_source_bindings",
)
