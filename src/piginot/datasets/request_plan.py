"""Canonical CP-bound public FEM request plan for S10 ingestion."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, cast

from piginot.contracts import canonical_json_bytes, hash_spec, strict_json_loads
from piginot.doe.checkpoint import CPEnvelope
from piginot.doe.publication import (
    PublishedCase,
    PublishedCaseManifest,
    PublishedCaseStatus,
)
from piginot.doe.roles import EvidenceRole
from piginot.fem import FemRequest

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA_VERSION = "1.0"
_PLAN_TYPE = "piginot.datasets.PublicFemRequestPlan"
_ENTRY_TYPE = "piginot.datasets.PublicFemRequestPlanEntry"
_PUBLIC_ROLES = frozenset(
    {
        EvidenceRole.PILOT,
        EvidenceRole.TRAIN,
        EvidenceRole.DEVELOPMENT,
        EvidenceRole.LEGACY,
    }
)
_PLAN_FIELDS = frozenset(
    {
        "$type",
        "campaign_hash",
        "public_case_manifest_hash",
        "execution_profile_hash",
        "entries",
        "schema_version",
    }
)
_ENTRY_FIELDS = frozenset(
    {
        "$type",
        "case_id",
        "leakage_group_id",
        "role",
        "request_recipe_hash",
        "request_content_hash",
        "mesh_id",
        "mesh_content_hash",
    }
)


class RequestPlanValidationError(ValueError):
    """The public FEM request plan is incomplete or provenance-incoherent."""


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestPlanValidationError(f"{label} must be non-empty text")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise RequestPlanValidationError(
            f"{label} must be a lowercase BLAKE2b-256 digest"
        )
    return value


def _record(
    value: object,
    *,
    label: str,
    expected_type: str,
    fields: frozenset[str],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RequestPlanValidationError(f"{label} must be a JSON object")
    record = cast(dict[str, object], value)
    actual = frozenset(record)
    missing = fields - actual
    if missing:
        raise RequestPlanValidationError(
            f"{label} has missing fields: {', '.join(sorted(missing))}"
        )
    unexpected = actual - fields
    if unexpected:
        raise RequestPlanValidationError(
            f"{label} has unexpected fields: {', '.join(sorted(unexpected))}"
        )
    if record["$type"] != expected_type:
        raise RequestPlanValidationError(
            f"{label} canonical type must be {expected_type}"
        )
    return record


@dataclass(frozen=True, slots=True)
class PublicFemRequestPlanEntry:
    """Planned immutable S7 request identities for one active public case."""

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    case_id: str
    leakage_group_id: str
    role: EvidenceRole
    request_recipe_hash: str
    request_content_hash: str
    mesh_id: str
    mesh_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(
            self,
            "leakage_group_id",
            _text(self.leakage_group_id, "leakage_group_id"),
        )
        try:
            role = EvidenceRole(self.role)
        except (TypeError, ValueError) as exc:
            raise RequestPlanValidationError("role must be an evidence role") from exc
        if role not in _PUBLIC_ROLES:
            raise RequestPlanValidationError(
                "HOLDOUT cases must never enter a public FEM request plan"
            )
        object.__setattr__(self, "role", role)
        for field_name in (
            "request_recipe_hash",
            "request_content_hash",
            "mesh_content_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "mesh_id", _text(self.mesh_id, "mesh_id"))


@dataclass(frozen=True, slots=True)
class PublicFemRequestPlan:
    """Strict canonical execution plan for every active public S9 case."""

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    campaign_hash: str
    public_case_manifest_hash: str
    execution_profile_hash: str
    entries: tuple[PublicFemRequestPlanEntry, ...]
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "campaign_hash",
            "public_case_manifest_hash",
            "execution_profile_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        entries = tuple(sorted(self.entries, key=lambda entry: entry.case_id))
        if not entries:
            raise RequestPlanValidationError("request plan entries must not be empty")
        if any(type(entry) is not PublicFemRequestPlanEntry for entry in entries):
            raise RequestPlanValidationError(
                "request plan entries must be PublicFemRequestPlanEntry values"
            )
        case_ids = tuple(entry.case_id for entry in entries)
        if len(case_ids) != len(set(case_ids)):
            raise RequestPlanValidationError("request plan case IDs must be unique")
        group_roles: dict[str, EvidenceRole] = {}
        for entry in entries:
            prior = group_roles.setdefault(entry.leakage_group_id, entry.role)
            if prior is not entry.role:
                raise RequestPlanValidationError(
                    "each leakage group must have exactly one role"
                )
        if self.schema_version != _SCHEMA_VERSION:
            raise RequestPlanValidationError("unsupported request plan schema version")
        object.__setattr__(self, "entries", entries)

    @property
    def content_hash(self) -> str:
        """Return the deterministic content identity of the complete plan."""

        return hash_spec(self, "content")

    def entry_by_case_id(self, case_id: str) -> PublicFemRequestPlanEntry:
        """Return the planned entry for an exact public case ID."""

        for entry in self.entries:
            if entry.case_id == case_id:
                return entry
        raise KeyError(case_id)

    def to_bytes(self) -> bytes:
        """Return the RFC 8785 canonical public representation."""

        return canonical_json_bytes(self)

    @classmethod
    def from_bytes(cls, document: str | bytes) -> PublicFemRequestPlan:
        """Strictly parse an exact-schema canonically encoded request plan."""

        if isinstance(document, str):
            raw = document.encode("utf-8")
        elif isinstance(document, bytes):
            raw = document
        else:
            raise RequestPlanValidationError(
                "request plan document must be text or bytes"
            )
        try:
            parsed = strict_json_loads(raw)
        except ValueError as exc:
            raise RequestPlanValidationError(
                f"invalid request plan JSON: {exc}"
            ) from exc
        root = _record(
            parsed,
            label="request plan",
            expected_type=_PLAN_TYPE,
            fields=_PLAN_FIELDS,
        )
        raw_entries = root["entries"]
        if not isinstance(raw_entries, list):
            raise RequestPlanValidationError("request plan entries must be an array")
        entries: list[PublicFemRequestPlanEntry] = []
        for index, raw_entry in enumerate(raw_entries):
            entry = _record(
                raw_entry,
                label=f"request plan entry {index}",
                expected_type=_ENTRY_TYPE,
                fields=_ENTRY_FIELDS,
            )
            entries.append(
                PublicFemRequestPlanEntry(
                    case_id=entry["case_id"],  # type: ignore[arg-type]
                    leakage_group_id=entry["leakage_group_id"],  # type: ignore[arg-type]
                    role=entry["role"],  # type: ignore[arg-type]
                    request_recipe_hash=entry["request_recipe_hash"],  # type: ignore[arg-type]
                    request_content_hash=entry["request_content_hash"],  # type: ignore[arg-type]
                    mesh_id=entry["mesh_id"],  # type: ignore[arg-type]
                    mesh_content_hash=entry["mesh_content_hash"],  # type: ignore[arg-type]
                )
            )
        plan = cls(
            campaign_hash=root["campaign_hash"],  # type: ignore[arg-type]
            public_case_manifest_hash=root["public_case_manifest_hash"],  # type: ignore[arg-type]
            execution_profile_hash=root["execution_profile_hash"],  # type: ignore[arg-type]
            entries=tuple(entries),
            schema_version=root["schema_version"],  # type: ignore[arg-type]
        )
        if plan.to_bytes() != raw:
            raise RequestPlanValidationError(
                "request plan document must use canonical JSON encoding"
            )
        return plan


def build_public_fem_request_plan(
    checkpoint: CPEnvelope,
    manifest: PublishedCaseManifest,
    requests: Mapping[str, FemRequest],
) -> PublicFemRequestPlan:
    """Bind exact S7 recipes to every active public case in CP-ENVELOPE."""

    if type(checkpoint) is not CPEnvelope:
        raise RequestPlanValidationError("checkpoint must be a CPEnvelope")
    if type(manifest) is not PublishedCaseManifest:
        raise RequestPlanValidationError("manifest must be a PublishedCaseManifest")
    if manifest.content_hash != checkpoint.public_case_manifest_hash:
        raise RequestPlanValidationError(
            "public manifest hash does not match CP-ENVELOPE"
        )
    try:
        canonical_manifest = PublishedCaseManifest.from_bytes(manifest.to_bytes())
    except ValueError as exc:
        raise RequestPlanValidationError(
            f"public manifest violates its canonical schema or leakage groups: {exc}"
        ) from exc
    if not isinstance(requests, Mapping):
        raise RequestPlanValidationError("requests must be a case-ID mapping")

    active_cases: dict[str, PublishedCase] = {}
    for case in canonical_manifest.cases:
        if case.role is EvidenceRole.HOLDOUT:
            raise RequestPlanValidationError(
                "HOLDOUT cases must never enter public FEM planning"
            )
        if case.status is PublishedCaseStatus.ACTIVE:
            active_cases[case.case_id] = case
    request_ids = set(requests)
    active_ids = set(active_cases)
    if request_ids != active_ids:
        missing = sorted(active_ids - request_ids)
        unexpected = sorted(request_ids - active_ids)
        raise RequestPlanValidationError(
            "request case IDs must equal active public cases; "
            f"missing={missing}, unexpected={unexpected}"
        )

    entries: list[PublicFemRequestPlanEntry] = []
    execution_profile_hashes: set[str] = set()
    for case_id in sorted(active_cases):
        case = active_cases[case_id]
        request = requests[case_id]
        if type(request) is not FemRequest:
            raise RequestPlanValidationError(
                f"request for {case_id!r} must be a FemRequest"
            )
        if request.request_id != case_id:
            raise RequestPlanValidationError(
                f"case {case_id!r} request has a different request identity"
            )
        execution_profile_hashes.add(hash_spec(request.execution_profile, "content"))
        entries.append(
            PublicFemRequestPlanEntry(
                case_id=case.case_id,
                leakage_group_id=case.leakage_group_id,
                role=case.role,
                request_recipe_hash=request.recipe_hash,
                request_content_hash=hash_spec(request, "content"),
                mesh_id=request.mesh.mesh_id,
                mesh_content_hash=hash_spec(request.mesh, "content"),
            )
        )
    if len(execution_profile_hashes) != 1:
        raise RequestPlanValidationError(
            "all planned requests must use one exact execution profile"
        )
    plan = PublicFemRequestPlan(
        campaign_hash=checkpoint.campaign_hash,
        public_case_manifest_hash=manifest.content_hash,
        execution_profile_hash=execution_profile_hashes.pop(),
        entries=tuple(entries),
    )
    return PublicFemRequestPlan.from_bytes(plan.to_bytes())


__all__ = (
    "PublicFemRequestPlan",
    "PublicFemRequestPlanEntry",
    "RequestPlanValidationError",
    "build_public_fem_request_plan",
)
