from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from piginot.contracts import canonical_json_bytes
from piginot.doe.checkpoint import (
    CampaignBudget,
    CPEnvelope,
    EvidenceReference,
)
from piginot.doe.commitment import HoldoutCommitment
from piginot.doe.roles import EvidenceRole


def _commitment(*, campaign_hash: str = "a" * 64) -> HoldoutCommitment:
    return HoldoutCommitment(
        scheme="blake2b-256",
        campaign_hash=campaign_hash,
        n_holdout_points=64,
        commitment="b" * 64,
        created_utc="2026-08-24T12:00:00Z",
    )


def _checkpoint(**overrides: object) -> CPEnvelope:
    values: dict[str, object] = {
        "campaign_hash": "a" * 64,
        "compiled_family_hash": "c" * 64,
        "pilot_evidence": (
            EvidenceReference(
                evidence_id="s1-closeout",
                content_hash="d" * 64,
                role=EvidenceRole.LEGACY,
            ),
        ),
        "executable_envelope_hash": "e" * 64,
        "doe_plan_hash": "f" * 64,
        "role_ledger_hash": "1" * 64,
        "replacement_policy_hash": "2" * 64,
        "holdout_commitment": _commitment(),
        "budget": CampaignBudget(
            pilot=8,
            train=128,
            development=32,
            holdout=64,
            legacy=60,
            reserve=32,
        ),
        "derived_axes": ("net_section_ratio",),
        "created_utc": "2026-08-24T12:05:00Z",
    }
    values.update(overrides)
    return CPEnvelope(**values)  # type: ignore[arg-type]


def test_checkpoint_binds_every_preregistered_input_without_holdout_secrets() -> None:
    checkpoint = _checkpoint()

    payload = checkpoint.to_bytes()

    assert checkpoint.checkpoint_id == "CP-ENVELOPE"
    assert checkpoint.budget.holdout == 64
    assert checkpoint.budget.production_cases == 232
    assert b'"salt"' not in payload
    assert b'"coordinates"' not in payload
    assert payload == canonical_json_bytes(checkpoint)
    with pytest.raises(FrozenInstanceError):
        checkpoint.campaign_hash = "9" * 64  # type: ignore[misc]


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"derived_axes": ()}, "derived"),
        ({"campaign_hash": "9" * 64}, "campaign"),
        (
            {
                "budget": CampaignBudget(
                    pilot=8,
                    train=128,
                    development=32,
                    holdout=63,
                    legacy=60,
                    reserve=32,
                )
            },
            "holdout",
        ),
        ({"pilot_evidence": ()}, "pilot evidence"),
    ),
)
def test_checkpoint_fails_closed_when_required_bindings_are_inconsistent(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _checkpoint(**override)


def test_checkpoint_rejects_production_pilot_evidence_mislabelled_as_holdout() -> None:
    reference = EvidenceReference(
        evidence_id="wrong-role",
        content_hash="3" * 64,
        role=EvidenceRole.HOLDOUT,
    )

    with pytest.raises(ValueError, match="PILOT or LEGACY"):
        _checkpoint(pilot_evidence=(reference,))


def test_evidence_reference_normalizes_a_valid_role_value() -> None:
    reference = EvidenceReference(
        evidence_id="pilot-evidence",
        content_hash="3" * 64,
        role="PILOT",  # type: ignore[arg-type]
    )

    assert reference.role is EvidenceRole.PILOT


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"evidence_id": ""}, "evidence_id must be non-empty text"),
        ({"evidence_id": 1}, "evidence_id must be non-empty text"),
        ({"content_hash": "A" * 64}, "lowercase 64-character hex hash"),
        ({"content_hash": "a" * 63}, "lowercase 64-character hex hash"),
        ({"role": "UNKNOWN"}, "UNKNOWN"),
    ),
)
def test_evidence_reference_rejects_noncanonical_values(
    kwargs: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "evidence_id": "pilot-evidence",
        "content_hash": "3" * 64,
        "role": EvidenceRole.PILOT,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        EvidenceReference(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_count", (True, 1.5, -1))
def test_campaign_budget_rejects_invalid_counts(invalid_count: object) -> None:
    with pytest.raises(ValueError, match="pilot must be a non-negative integer"):
        CampaignBudget(
            pilot=invalid_count,  # type: ignore[arg-type]
            train=128,
            development=32,
            holdout=64,
            legacy=60,
            reserve=32,
        )


def test_checkpoint_accepts_both_allowed_pilot_evidence_roles() -> None:
    references = (
        EvidenceReference("pilot", "3" * 64, EvidenceRole.PILOT),
        EvidenceReference("legacy", "4" * 64, EvidenceRole.LEGACY),
    )

    checkpoint = _checkpoint(pilot_evidence=references)

    assert checkpoint.pilot_evidence == references


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"compiled_family_hash": "not-a-hash"}, "compiled_family_hash"),
        ({"pilot_evidence": ("not-a-reference",)}, "EvidenceReference"),
        (
            {
                "pilot_evidence": (
                    EvidenceReference("duplicate", "3" * 64, EvidenceRole.PILOT),
                    EvidenceReference("duplicate", "4" * 64, EvidenceRole.LEGACY),
                )
            },
            "IDs must be unique",
        ),
        ({"holdout_commitment": object()}, "HoldoutCommitment"),
        ({"budget": object()}, "CampaignBudget"),
        ({"derived_axes": ("not an identifier",)}, "valid identifiers"),
        ({"derived_axes": ("net_section_ratio", 1)}, "valid identifiers"),
        (
            {"derived_axes": ("net_section_ratio", "net_section_ratio")},
            "must be unique",
        ),
        ({"checkpoint_id": "CP-OTHER"}, "checkpoint_id"),
        ({"schema_version": "2.0"}, "schema version"),
        ({"created_utc": "2026-08-24T12:05:00+00:00"}, "ending in Z"),
        ({"created_utc": "not-a-timestampZ"}, "valid RFC 3339"),
    ),
)
def test_checkpoint_rejects_malformed_public_contracts(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _checkpoint(**override)
