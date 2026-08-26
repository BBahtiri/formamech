"""Canonical CP-ENVELOPE checkpoint and campaign-identity contracts for S9."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar, cast

from piginot.contracts import (
    canonical_json_bytes,
    hash_spec,
    strict_json_loads,
)
from piginot.doe.commitment import HoldoutCommitment
from piginot.doe.roles import EvidenceRole

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")
_MAX_SAFE_INTEGER = 2**53 - 1


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _hash(value: object, label: str) -> str:
    text = _text(value, label)
    if _HASH.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase 64-character hex hash")
    return text


def _utc_timestamp(value: object, label: str) -> str:
    text = _text(value, label)
    if not text.endswith("Z"):
        raise ValueError(f"{label} must be an RFC 3339 UTC timestamp ending in Z")
    if _UTC_TIMESTAMP.fullmatch(text) is None:
        raise ValueError(f"{label} must be a valid RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid RFC 3339 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{label} must be UTC")
    return text


def _count(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > _MAX_SAFE_INTEGER
    ):
        raise ValueError(
            f"{label} must be a non-negative integer within the RFC 8785 safe range"
        )
    return value


def _pilot_evidence(value: object) -> tuple[EvidenceReference, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError("pilot evidence must contain EvidenceReference values")
    try:
        evidence: tuple[object, ...] = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            "pilot evidence must contain EvidenceReference values"
        ) from exc
    if not evidence:
        raise ValueError("pilot evidence must not be empty")
    if any(not isinstance(item, EvidenceReference) for item in evidence):
        raise ValueError("pilot evidence must contain EvidenceReference values")
    typed = cast(tuple[EvidenceReference, ...], evidence)
    if len({item.evidence_id for item in typed}) != len(typed):
        raise ValueError("pilot evidence IDs must be unique")
    if any(
        item.role not in {EvidenceRole.PILOT, EvidenceRole.LEGACY} for item in typed
    ):
        raise ValueError("pilot evidence must have role PILOT or LEGACY")
    return tuple(
        sorted(
            typed,
            key=lambda item: (item.evidence_id, item.content_hash, item.role.value),
        )
    )


def _axes(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError("derived axes must be a sequence of valid identifiers")
    try:
        axes: tuple[object, ...] = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            "derived axes must be a sequence of valid identifiers"
        ) from exc
    if not axes:
        raise ValueError("CP-ENVELOPE must declare derived stratification axes")
    if any(not isinstance(axis, str) or not axis.isidentifier() for axis in axes):
        raise ValueError("derived axes must be valid identifiers")
    typed = cast(tuple[str, ...], axes)
    if len(typed) != len(set(typed)):
        raise ValueError("derived axes must be unique")
    return typed


def _record(
    value: object,
    *,
    label: str,
    canonical_type: str,
    fields: frozenset[str],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    record = cast(dict[str, object], value)
    expected = fields | {"$type"}
    actual = frozenset(record)
    missing = expected - actual
    if missing:
        raise ValueError(f"{label} has missing fields: {', '.join(sorted(missing))}")
    unexpected = actual - expected
    if unexpected:
        raise ValueError(
            f"{label} has unexpected fields: {', '.join(sorted(unexpected))}"
        )
    if record["$type"] != canonical_type:
        raise ValueError(f"{label} canonical type must be {canonical_type}")
    return record


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """Content-addressed evidence consumed by the checkpoint."""

    __canonical_namespace__: ClassVar[str] = "piginot.doe"

    evidence_id: str
    content_hash: str
    role: EvidenceRole

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _text(self.evidence_id, "evidence_id"))
        object.__setattr__(
            self, "content_hash", _hash(self.content_hash, "content_hash")
        )
        if not isinstance(self.role, EvidenceRole):
            object.__setattr__(self, "role", EvidenceRole(self.role))


@dataclass(frozen=True, slots=True)
class CampaignBudget:
    """Fixed case budget recorded before production labels are generated."""

    __canonical_namespace__: ClassVar[str] = "piginot.doe"

    pilot: int
    train: int
    development: int
    holdout: int
    legacy: int
    reserve: int

    def __post_init__(self) -> None:
        for field_name in (
            "pilot",
            "train",
            "development",
            "holdout",
            "legacy",
            "reserve",
        ):
            object.__setattr__(
                self,
                field_name,
                _count(getattr(self, field_name), field_name),
            )

    @property
    def production_cases(self) -> int:
        return self.pilot + self.train + self.development + self.holdout


@dataclass(frozen=True, slots=True)
class CampaignBinding:
    """Canonical, secret-free preimage for the S9 campaign identity."""

    __canonical_namespace__: ClassVar[str] = "piginot.doe"

    compiled_family_hash: str
    pilot_evidence: tuple[EvidenceReference, ...]
    executable_envelope_hash: str
    private_doe_plan_hash: str
    role_ledger_hash: str
    replacement_policy_hash: str
    public_case_manifest_hash: str
    doe_seed_commitment: str
    budget: CampaignBudget
    derived_axes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "compiled_family_hash",
            "executable_envelope_hash",
            "private_doe_plan_hash",
            "role_ledger_hash",
            "replacement_policy_hash",
            "public_case_manifest_hash",
            "doe_seed_commitment",
        ):
            object.__setattr__(
                self, field_name, _hash(getattr(self, field_name), field_name)
            )
        object.__setattr__(self, "pilot_evidence", _pilot_evidence(self.pilot_evidence))
        if not isinstance(self.budget, CampaignBudget):
            raise ValueError("budget must be a CampaignBudget")
        object.__setattr__(self, "derived_axes", _axes(self.derived_axes))

    @property
    def doe_plan_hash(self) -> str:
        """Compatibility name for the private DOE-plan digest."""

        return self.private_doe_plan_hash

    @property
    def doe_seed_commitment_hash(self) -> str:
        """Make the digest semantics explicit for downstream consumers."""

        return self.doe_seed_commitment


CampaignCore = CampaignBinding


def compute_campaign_hash(binding: CampaignBinding) -> str:
    """Return the domain-separated canonical identity of one campaign binding."""

    if not isinstance(binding, CampaignBinding):
        raise ValueError("binding must be a CampaignBinding")
    return hash_spec(binding, "content")


@dataclass(frozen=True, slots=True)
class CPEnvelope:
    """Public checkpoint binding the executable S9 campaign without secrets."""

    __canonical_namespace__: ClassVar[str] = "piginot.doe"

    campaign_hash: str
    compiled_family_hash: str
    pilot_evidence: tuple[EvidenceReference, ...]
    executable_envelope_hash: str
    private_doe_plan_hash: str
    role_ledger_hash: str
    replacement_policy_hash: str
    public_case_manifest_hash: str
    doe_seed_commitment: str
    holdout_commitment: HoldoutCommitment
    budget: CampaignBudget
    derived_axes: tuple[str, ...]
    created_utc: str
    checkpoint_id: str = "CP-ENVELOPE"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "campaign_hash", _hash(self.campaign_hash, "campaign_hash")
        )
        binding = CampaignBinding(
            compiled_family_hash=self.compiled_family_hash,
            pilot_evidence=self.pilot_evidence,
            executable_envelope_hash=self.executable_envelope_hash,
            private_doe_plan_hash=self.private_doe_plan_hash,
            role_ledger_hash=self.role_ledger_hash,
            replacement_policy_hash=self.replacement_policy_hash,
            public_case_manifest_hash=self.public_case_manifest_hash,
            doe_seed_commitment=self.doe_seed_commitment,
            budget=self.budget,
            derived_axes=self.derived_axes,
        )
        for field_name in (
            "compiled_family_hash",
            "pilot_evidence",
            "executable_envelope_hash",
            "private_doe_plan_hash",
            "role_ledger_hash",
            "replacement_policy_hash",
            "public_case_manifest_hash",
            "doe_seed_commitment",
            "budget",
            "derived_axes",
        ):
            object.__setattr__(self, field_name, getattr(binding, field_name))
        expected_campaign_hash = compute_campaign_hash(binding)
        if self.campaign_hash != expected_campaign_hash:
            raise ValueError("campaign_hash must equal the recomputed campaign hash")
        if not isinstance(self.holdout_commitment, HoldoutCommitment):
            raise ValueError("holdout_commitment must be a HoldoutCommitment")
        if expected_campaign_hash != self.holdout_commitment.campaign_hash:
            raise ValueError(
                "holdout commitment must bind the recomputed campaign hash"
            )
        if self.budget.holdout != self.holdout_commitment.n_holdout_points:
            raise ValueError("holdout budget must match the committed holdout count")
        if self.checkpoint_id != "CP-ENVELOPE":
            raise ValueError("checkpoint_id must be CP-ENVELOPE")
        if self.schema_version != "1.0":
            raise ValueError("unsupported CP-ENVELOPE schema version")
        object.__setattr__(
            self, "created_utc", _utc_timestamp(self.created_utc, "created_utc")
        )

    @property
    def binding(self) -> CampaignBinding:
        """Reconstruct the exact preimage whose hash identifies this campaign."""

        return CampaignBinding(
            compiled_family_hash=self.compiled_family_hash,
            pilot_evidence=self.pilot_evidence,
            executable_envelope_hash=self.executable_envelope_hash,
            private_doe_plan_hash=self.private_doe_plan_hash,
            role_ledger_hash=self.role_ledger_hash,
            replacement_policy_hash=self.replacement_policy_hash,
            public_case_manifest_hash=self.public_case_manifest_hash,
            doe_seed_commitment=self.doe_seed_commitment,
            budget=self.budget,
            derived_axes=self.derived_axes,
        )

    @property
    def doe_plan_hash(self) -> str:
        """Compatibility name for the private DOE-plan digest."""

        return self.private_doe_plan_hash

    @property
    def doe_seed_commitment_hash(self) -> str:
        """Make the digest semantics explicit for downstream consumers."""

        return self.doe_seed_commitment

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    @classmethod
    def from_bytes(cls, document: str | bytes) -> CPEnvelope:
        """Read one exact-schema, canonically encoded public checkpoint."""

        if isinstance(document, str):
            raw = document.encode("utf-8")
        elif isinstance(document, bytes):
            raw = document
        else:
            raise ValueError("CP-ENVELOPE document must be text or bytes")
        payload = _record(
            strict_json_loads(raw),
            label="CP-ENVELOPE",
            canonical_type="piginot.doe.CPEnvelope",
            fields=frozenset(
                {
                    "campaign_hash",
                    "compiled_family_hash",
                    "pilot_evidence",
                    "executable_envelope_hash",
                    "private_doe_plan_hash",
                    "role_ledger_hash",
                    "replacement_policy_hash",
                    "public_case_manifest_hash",
                    "doe_seed_commitment",
                    "holdout_commitment",
                    "budget",
                    "derived_axes",
                    "created_utc",
                    "checkpoint_id",
                    "schema_version",
                }
            ),
        )
        pilot_payload = payload["pilot_evidence"]
        if not isinstance(pilot_payload, list):
            raise ValueError("pilot_evidence must be a JSON array")
        evidence: list[EvidenceReference] = []
        for index, raw_reference in enumerate(pilot_payload):
            reference = _record(
                raw_reference,
                label=f"pilot_evidence[{index}]",
                canonical_type="piginot.doe.EvidenceReference",
                fields=frozenset({"evidence_id", "content_hash", "role"}),
            )
            evidence.append(
                EvidenceReference(
                    evidence_id=reference["evidence_id"],  # type: ignore[arg-type]
                    content_hash=reference["content_hash"],  # type: ignore[arg-type]
                    role=reference["role"],  # type: ignore[arg-type]
                )
            )
        budget_payload = _record(
            payload["budget"],
            label="budget",
            canonical_type="piginot.doe.CampaignBudget",
            fields=frozenset(
                {"pilot", "train", "development", "holdout", "legacy", "reserve"}
            ),
        )
        budget = CampaignBudget(
            pilot=budget_payload["pilot"],  # type: ignore[arg-type]
            train=budget_payload["train"],  # type: ignore[arg-type]
            development=budget_payload["development"],  # type: ignore[arg-type]
            holdout=budget_payload["holdout"],  # type: ignore[arg-type]
            legacy=budget_payload["legacy"],  # type: ignore[arg-type]
            reserve=budget_payload["reserve"],  # type: ignore[arg-type]
        )
        commitment_payload = _record(
            payload["holdout_commitment"],
            label="holdout_commitment",
            canonical_type="piginot.doe.HoldoutCommitment",
            fields=frozenset(
                {
                    "scheme",
                    "campaign_hash",
                    "n_holdout_points",
                    "commitment",
                    "created_utc",
                }
            ),
        )
        commitment = HoldoutCommitment(
            scheme=commitment_payload["scheme"],  # type: ignore[arg-type]
            campaign_hash=commitment_payload["campaign_hash"],  # type: ignore[arg-type]
            n_holdout_points=commitment_payload["n_holdout_points"],  # type: ignore[arg-type]
            commitment=commitment_payload["commitment"],  # type: ignore[arg-type]
            created_utc=commitment_payload["created_utc"],  # type: ignore[arg-type]
        )
        axes_payload = payload["derived_axes"]
        if not isinstance(axes_payload, list):
            raise ValueError("derived_axes must be a JSON array")
        checkpoint = cls(
            campaign_hash=payload["campaign_hash"],  # type: ignore[arg-type]
            compiled_family_hash=payload["compiled_family_hash"],  # type: ignore[arg-type]
            pilot_evidence=tuple(evidence),
            executable_envelope_hash=payload["executable_envelope_hash"],  # type: ignore[arg-type]
            private_doe_plan_hash=payload["private_doe_plan_hash"],  # type: ignore[arg-type]
            role_ledger_hash=payload["role_ledger_hash"],  # type: ignore[arg-type]
            replacement_policy_hash=payload["replacement_policy_hash"],  # type: ignore[arg-type]
            public_case_manifest_hash=payload["public_case_manifest_hash"],  # type: ignore[arg-type]
            doe_seed_commitment=payload["doe_seed_commitment"],  # type: ignore[arg-type]
            holdout_commitment=commitment,
            budget=budget,
            derived_axes=_axes(axes_payload),
            created_utc=payload["created_utc"],  # type: ignore[arg-type]
            checkpoint_id=payload["checkpoint_id"],  # type: ignore[arg-type]
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )
        if checkpoint.to_bytes() != raw:
            raise ValueError("CP-ENVELOPE must use canonical JSON encoding")
        return checkpoint


__all__ = (
    "CPEnvelope",
    "CampaignBinding",
    "CampaignBudget",
    "CampaignCore",
    "EvidenceReference",
    "compute_campaign_hash",
)
