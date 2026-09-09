"""Strict compact parser for the pinned public S10 closeout report."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import rfc8785

from piginot.contracts import strict_json_loads
from piginot.doe import EvidenceRole

S10_PRODUCTION_REPORT_CONTENT_HASH = (
    "ed9a7758a87ce9fddab1ec365fc3529372b49c181406a09723573a91a96c9ef2"
)
_ROOT = Path(__file__).resolve().parents[2]
_REPORT_PATH = (
    _ROOT / "artifacts" / "campaigns" / "dogbone-v1" / "s10-production-dataset.json"
)
_REPORT_TYPE = "piginot.campaigns.S10CloseoutReport"
_CASE_TYPE = "piginot.campaigns.CaseMaterializationEvidence"
_SNAPSHOT_TYPE = "piginot.campaigns.RoleSnapshotEvidence"
_REPORT_FIELDS = frozenset(
    {
        "$type",
        "campaign_hash",
        "cases",
        "cross_mesh_comparisons",
        "execution_profile_hash",
        "failed_cases",
        "failure_visibility_canary",
        "fem_source_revision",
        "materializer_source_revision",
        "normalization",
        "policy",
        "public_case_manifest_hash",
        "request_plan_hash",
        "role_snapshots",
        "run_id",
        "schema_version",
    }
)
_CASE_FIELDS = frozenset(
    {
        "$type",
        "case_id",
        "collocation_batch_content_hash",
        "collocation_point_count",
        "collocation_replay_verified",
        "collocation_store_content_hash",
        "collocation_stream_hash",
        "extraction_content_hash",
        "extraction_store_content_hash",
        "fem_result_hash",
        "fem_result_store_content_hash",
        "projection_content_hash",
        "projection_source_verified",
        "projection_store_content_hash",
        "role",
        "stress_field_content_hash",
        "stress_field_store_content_hash",
        "transfer_error_content_hash",
    }
)
_SNAPSHOT_FIELDS = frozenset(
    {
        "$type",
        "case_count",
        "failed_case_count",
        "role",
        "snapshot_content_hash",
        "store_content_hash",
        "successful_case_count",
    }
)


@dataclass(frozen=True, slots=True)
class S10ReportCaseIndex:
    """The S10 result identities needed before one case may be resolved."""

    case_id: str
    role: EvidenceRole
    fem_result_hash: str
    store_content_hash: str

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        role = EvidenceRole(self.role)
        if role is EvidenceRole.HOLDOUT:
            raise ValueError("HOLDOUT cannot enter the S10 public report")
        _digest(self.fem_result_hash, "fem_result_hash")
        _digest(self.store_content_hash, "store_content_hash")
        object.__setattr__(self, "role", role)


@dataclass(frozen=True, slots=True)
class S10RoleSnapshotIndex:
    role: EvidenceRole
    case_count: int
    successful_case_count: int
    failed_case_count: int
    snapshot_content_hash: str

    def __post_init__(self) -> None:
        role = EvidenceRole(self.role)
        if role is EvidenceRole.HOLDOUT:
            raise ValueError("HOLDOUT cannot enter an S10 public snapshot")
        for name in ("case_count", "successful_case_count", "failed_case_count"):
            _nonnegative_int(getattr(self, name), name)
        if self.successful_case_count + self.failed_case_count != self.case_count:
            raise ValueError("S10 role snapshot counts do not reconcile")
        _digest(self.snapshot_content_hash, "snapshot_content_hash")
        object.__setattr__(self, "role", role)


@dataclass(frozen=True, slots=True)
class S10ReportIndex:
    """Strict, compact index of the pinned public S10 closeout report."""

    content_hash: str
    run_id: str
    fem_source_revision: str
    campaign_hash: str
    public_case_manifest_hash: str
    request_plan_hash: str
    execution_profile_hash: str
    role_snapshots: tuple[S10RoleSnapshotIndex, ...]
    cases: tuple[S10ReportCaseIndex, ...]
    training_case_ids: tuple[str, ...]
    training_snapshot_hash: str
    development_snapshot_hash: str
    normalization_non_train_source_count: int
    normalization_non_active_source_count: int
    normalization_snapshot_binding_verified: bool
    failed_case_count: int

    def __post_init__(self) -> None:
        for name in (
            "content_hash",
            "campaign_hash",
            "public_case_manifest_hash",
            "request_plan_hash",
            "execution_profile_hash",
            "training_snapshot_hash",
            "development_snapshot_hash",
        ):
            _digest(getattr(self, name), name)
        _text(self.run_id, "run_id")
        _revision(self.fem_source_revision, "fem_source_revision")
        snapshots = tuple(sorted(self.role_snapshots, key=lambda item: item.role.value))
        if any(type(item) is not S10RoleSnapshotIndex for item in snapshots):
            raise TypeError("role_snapshots must contain exact index values")
        if len({item.role for item in snapshots}) != len(snapshots):
            raise ValueError("role_snapshots contain duplicate roles")
        cases = tuple(sorted(self.cases, key=lambda item: item.case_id))
        if not cases or any(type(item) is not S10ReportCaseIndex for item in cases):
            raise TypeError("cases must contain exact S10ReportCaseIndex values")
        if len({item.case_id for item in cases}) != len(cases):
            raise ValueError("S10 report case IDs must be unique")
        training = tuple(self.training_case_ids)
        if training != tuple(sorted(set(training))):
            raise ValueError("training_case_ids must be sorted and unique")
        for name in (
            "normalization_non_train_source_count",
            "normalization_non_active_source_count",
            "failed_case_count",
        ):
            _nonnegative_int(getattr(self, name), name)
        if type(self.normalization_snapshot_binding_verified) is not bool:
            raise TypeError("normalization_snapshot_binding_verified must be boolean")
        object.__setattr__(self, "role_snapshots", snapshots)
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "training_case_ids", training)

    @property
    def case_count(self) -> int:
        return len(self.cases)


def load_s10_report_index(path: Path | None = None) -> S10ReportIndex:
    """Parse and pin the tracked canonical S10 closeout report."""

    report_path = _REPORT_PATH if path is None else Path(path)
    raw = report_path.read_bytes()
    document = strict_json_loads(raw)
    root = _record(document, _REPORT_FIELDS, _REPORT_TYPE, "S10 report")
    if rfc8785.dumps(root) != raw:
        raise ValueError("S10 report must use exact canonical JSON")
    content_hash = hashlib.blake2b(
        rfc8785.dumps({"domain": "content", "payload": root}),
        digest_size=32,
    ).hexdigest()
    if content_hash != S10_PRODUCTION_REPORT_CONTENT_HASH:
        raise ValueError("S10 report content hash differs from the ratified artifact")

    snapshots = tuple(_snapshot_index(value) for value in _list(root["role_snapshots"]))
    snapshot_by_role = {item.role: item for item in snapshots}
    if set(snapshot_by_role) != {
        EvidenceRole.TRAIN,
        EvidenceRole.DEVELOPMENT,
        EvidenceRole.PILOT,
    }:
        raise ValueError("S10 report must contain exact TRAIN/DEVELOPMENT/PILOT snapshots")
    cases = tuple(_case_index(value) for value in _list(root["cases"]))
    failed_cases = _list(root["failed_cases"])
    normalization = _mapping(root["normalization"], "normalization")
    training_ids = tuple(
        sorted(
            _text(value, "normalization training case ID")
            for value in _list(normalization.get("training_case_ids"))
        )
    )
    return S10ReportIndex(
        content_hash=content_hash,
        run_id=_text(root["run_id"], "run_id"),
        fem_source_revision=_text(root["fem_source_revision"], "fem_source_revision"),
        campaign_hash=_text(root["campaign_hash"], "campaign_hash"),
        public_case_manifest_hash=_text(
            root["public_case_manifest_hash"], "public_case_manifest_hash"
        ),
        request_plan_hash=_text(root["request_plan_hash"], "request_plan_hash"),
        execution_profile_hash=_text(
            root["execution_profile_hash"], "execution_profile_hash"
        ),
        role_snapshots=snapshots,
        cases=cases,
        training_case_ids=training_ids,
        training_snapshot_hash=_text(
            normalization.get("training_snapshot_hash"), "training_snapshot_hash"
        ),
        development_snapshot_hash=snapshot_by_role[
            EvidenceRole.DEVELOPMENT
        ].snapshot_content_hash,
        normalization_non_train_source_count=_integer(
            normalization.get("non_train_source_count"), "non_train_source_count"
        ),
        normalization_non_active_source_count=_integer(
            normalization.get("non_active_source_count"), "non_active_source_count"
        ),
        normalization_snapshot_binding_verified=_boolean(
            normalization.get("snapshot_binding_verified"), "snapshot_binding_verified"
        ),
        failed_case_count=len(failed_cases),
    )


def _case_index(value: object) -> S10ReportCaseIndex:
    record = _record(value, _CASE_FIELDS, _CASE_TYPE, "S10 case evidence")
    return S10ReportCaseIndex(
        case_id=_text(record["case_id"], "case_id"),
        role=EvidenceRole(_text(record["role"], "role")),
        fem_result_hash=_text(record["fem_result_hash"], "fem_result_hash"),
        store_content_hash=_text(
            record["fem_result_store_content_hash"], "fem_result_store_content_hash"
        ),
    )


def _snapshot_index(value: object) -> S10RoleSnapshotIndex:
    record = _record(value, _SNAPSHOT_FIELDS, _SNAPSHOT_TYPE, "S10 role snapshot")
    return S10RoleSnapshotIndex(
        role=EvidenceRole(_text(record["role"], "role")),
        case_count=_integer(record["case_count"], "case_count"),
        successful_case_count=_integer(
            record["successful_case_count"], "successful_case_count"
        ),
        failed_case_count=_integer(record["failed_case_count"], "failed_case_count"),
        snapshot_content_hash=_text(
            record["snapshot_content_hash"], "snapshot_content_hash"
        ),
    )


def _record(
    value: object,
    fields: frozenset[str],
    expected_type: str,
    label: str,
) -> dict[str, object]:
    record = _mapping(value, label)
    if frozenset(record) != fields or record.get("$type") != expected_type:
        raise ValueError(f"{label} schema/type differs from S10")
    return record


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("S10 report collection must be a JSON array")
    return cast(list[object], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _digest(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or text.lower() != text:
        raise ValueError(f"{label} must be a lowercase 256-bit digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a lowercase 256-bit digest") from exc
    return text


def _revision(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 40 or text.lower() != text:
        raise ValueError(f"{label} must be a lowercase Git revision")
    try:
        int(text, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a lowercase Git revision") from exc
    return text


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    integer = _integer(value, label)
    if integer < 0:
        raise ValueError(f"{label} must be non-negative")
    return integer


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


__all__ = (
    "S10_PRODUCTION_REPORT_CONTENT_HASH",
    "S10ReportCaseIndex",
    "S10ReportIndex",
    "S10RoleSnapshotIndex",
    "load_s10_report_index",
)
