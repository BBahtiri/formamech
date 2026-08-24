"""Canonical CP-ENVELOPE checkpoint contract for S9."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar

from piginot.contracts import canonical_json_bytes
from piginot.doe.commitment import HoldoutCommitment
from piginot.doe.roles import EvidenceRole

_HASH = re.compile(r"[0-9a-f]{64}\Z")


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
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid RFC 3339 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{label} must be UTC")
    return text


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


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
class CPEnvelope:
    """Public checkpoint binding the executable S9 campaign without holdout secrets."""

    __canonical_namespace__: ClassVar[str] = "piginot.doe"

    campaign_hash: str
    compiled_family_hash: str
    pilot_evidence: tuple[EvidenceReference, ...]
    executable_envelope_hash: str
    doe_plan_hash: str
    role_ledger_hash: str
    replacement_policy_hash: str
    holdout_commitment: HoldoutCommitment
    budget: CampaignBudget
    derived_axes: tuple[str, ...]
    created_utc: str
    checkpoint_id: str = "CP-ENVELOPE"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        for field_name in (
            "campaign_hash",
            "compiled_family_hash",
            "executable_envelope_hash",
            "doe_plan_hash",
            "role_ledger_hash",
            "replacement_policy_hash",
        ):
            object.__setattr__(
                self, field_name, _hash(getattr(self, field_name), field_name)
            )
        evidence = tuple(self.pilot_evidence)
        if not evidence:
            raise ValueError("pilot evidence must not be empty")
        if any(not isinstance(item, EvidenceReference) for item in evidence):
            raise ValueError("pilot evidence must contain EvidenceReference values")
        if len({item.evidence_id for item in evidence}) != len(evidence):
            raise ValueError("pilot evidence IDs must be unique")
        if any(
            item.role not in {EvidenceRole.PILOT, EvidenceRole.LEGACY}
            for item in evidence
        ):
            raise ValueError("pilot evidence must have role PILOT or LEGACY")
        if not isinstance(self.holdout_commitment, HoldoutCommitment):
            raise ValueError("holdout_commitment must be a HoldoutCommitment")
        if self.campaign_hash != self.holdout_commitment.campaign_hash:
            raise ValueError("campaign hash must match the holdout commitment")
        if not isinstance(self.budget, CampaignBudget):
            raise ValueError("budget must be a CampaignBudget")
        if self.budget.holdout != self.holdout_commitment.n_holdout_points:
            raise ValueError("holdout budget must match the committed holdout count")
        axes = tuple(self.derived_axes)
        if not axes:
            raise ValueError("CP-ENVELOPE must declare derived stratification axes")
        if any(not isinstance(axis, str) or not axis.isidentifier() for axis in axes):
            raise ValueError("derived axes must be valid identifiers")
        if len(axes) != len(set(axes)):
            raise ValueError("derived axes must be unique")
        if self.checkpoint_id != "CP-ENVELOPE":
            raise ValueError("checkpoint_id must be CP-ENVELOPE")
        if self.schema_version != "1.0":
            raise ValueError("unsupported CP-ENVELOPE schema version")
        object.__setattr__(self, "pilot_evidence", evidence)
        object.__setattr__(self, "derived_axes", axes)
        object.__setattr__(
            self, "created_utc", _utc_timestamp(self.created_utc, "created_utc")
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self)
