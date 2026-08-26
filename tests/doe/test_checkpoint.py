from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from piginot.contracts import canonical_json_bytes
from piginot.contracts.errors import DuplicateKeyError
from piginot.doe.checkpoint import (
    CampaignBinding,
    CampaignBudget,
    CampaignCore,
    CPEnvelope,
    EvidenceReference,
    compute_campaign_hash,
)
from piginot.doe.commitment import HoldoutCommitment
from piginot.doe.roles import EvidenceRole
from piginot.doe.secret_material import CAMPAIGN_SECRET_GENERATOR_ID


def _budget(**overrides: object) -> CampaignBudget:
    values: dict[str, object] = {
        "pilot": 8,
        "train": 128,
        "development": 32,
        "holdout": 64,
        "legacy": 60,
        "reserve": 32,
    }
    values.update(overrides)
    return CampaignBudget(**values)  # type: ignore[arg-type]


def _binding(**overrides: object) -> CampaignBinding:
    values: dict[str, object] = {
        "compiled_family_hash": "c" * 64,
        "pilot_evidence": (
            EvidenceReference(
                evidence_id="s1-closeout",
                content_hash="d" * 64,
                role=EvidenceRole.LEGACY,
            ),
        ),
        "executable_envelope_hash": "e" * 64,
        "private_doe_plan_hash": "f" * 64,
        "role_ledger_hash": "1" * 64,
        "replacement_policy_hash": "2" * 64,
        "public_case_manifest_hash": "3" * 64,
        "doe_seed_commitment": "4" * 64,
        "secret_generator_id": CAMPAIGN_SECRET_GENERATOR_ID,
        "budget": _budget(),
        "derived_axes": ("net_section_ratio",),
    }
    values.update(overrides)
    return CampaignBinding(**values)  # type: ignore[arg-type]


def _commitment(*, campaign_hash: str) -> HoldoutCommitment:
    return HoldoutCommitment(
        scheme="blake2b-256",
        campaign_hash=campaign_hash,
        n_holdout_points=64,
        commitment="b" * 64,
        created_utc="2026-08-24T12:00:00Z",
    )


def _checkpoint(
    *, binding: CampaignBinding | None = None, **overrides: object
) -> CPEnvelope:
    bound = binding or _binding()
    campaign_hash = compute_campaign_hash(bound)
    values: dict[str, object] = {
        "campaign_hash": campaign_hash,
        "compiled_family_hash": bound.compiled_family_hash,
        "pilot_evidence": bound.pilot_evidence,
        "executable_envelope_hash": bound.executable_envelope_hash,
        "private_doe_plan_hash": bound.private_doe_plan_hash,
        "role_ledger_hash": bound.role_ledger_hash,
        "replacement_policy_hash": bound.replacement_policy_hash,
        "public_case_manifest_hash": bound.public_case_manifest_hash,
        "doe_seed_commitment": bound.doe_seed_commitment,
        "secret_generator_id": bound.secret_generator_id,
        "holdout_commitment": _commitment(campaign_hash=campaign_hash),
        "budget": bound.budget,
        "derived_axes": bound.derived_axes,
        "created_utc": "2026-08-24T12:05:00Z",
    }
    values.update(overrides)
    return CPEnvelope(**values)  # type: ignore[arg-type]


def test_checkpoint_binds_every_preregistered_input_without_holdout_secrets() -> None:
    checkpoint = _checkpoint()

    payload = checkpoint.to_bytes()

    assert checkpoint.checkpoint_id == "CP-ENVELOPE"
    assert checkpoint.schema_version == "1.1"
    assert checkpoint.budget.holdout == 64
    assert checkpoint.budget.production_cases == 232
    assert checkpoint.binding == _binding()
    assert checkpoint.doe_plan_hash == checkpoint.private_doe_plan_hash
    assert checkpoint.doe_seed_commitment_hash == checkpoint.doe_seed_commitment
    assert checkpoint.secret_generator_id == CAMPAIGN_SECRET_GENERATOR_ID
    assert checkpoint.campaign_hash == compute_campaign_hash(checkpoint.binding)
    assert CAMPAIGN_SECRET_GENERATOR_ID.encode() in payload
    assert b'"salt":' not in payload
    assert b'"seed":' not in payload
    assert b'"coordinates":' not in payload
    assert payload == canonical_json_bytes(checkpoint)
    with pytest.raises(FrozenInstanceError):
        checkpoint.campaign_hash = "9" * 64  # type: ignore[misc]


def test_campaign_core_is_compatibility_alias_for_canonical_binding() -> None:
    assert CampaignCore is CampaignBinding


@pytest.mark.parametrize(
    "changed_binding",
    (
        _binding(compiled_family_hash="5" * 64),
        _binding(
            pilot_evidence=(
                EvidenceReference("s1-closeout", "6" * 64, EvidenceRole.LEGACY),
            )
        ),
        _binding(executable_envelope_hash="7" * 64),
        _binding(private_doe_plan_hash="8" * 64),
        _binding(role_ledger_hash="9" * 64),
        _binding(replacement_policy_hash="a" * 64),
        _binding(public_case_manifest_hash="b" * 64),
        _binding(doe_seed_commitment="c" * 64),
        _binding(budget=_budget(train=129)),
        _binding(derived_axes=("net_section_ratio", "ligament_ratio")),
    ),
)
def test_every_bound_field_changes_campaign_identity(
    changed_binding: CampaignBinding,
) -> None:
    assert compute_campaign_hash(changed_binding) != compute_campaign_hash(_binding())


def test_checkpoint_rejects_arbitrary_hash_even_when_commitment_repeats_it() -> None:
    arbitrary_hash = "9" * 64

    with pytest.raises(ValueError, match="recomputed campaign hash"):
        _checkpoint(
            campaign_hash=arbitrary_hash,
            holdout_commitment=_commitment(campaign_hash=arbitrary_hash),
        )


def test_checkpoint_rejects_holdout_commitment_for_another_campaign() -> None:
    with pytest.raises(ValueError, match="holdout commitment"):
        _checkpoint(holdout_commitment=_commitment(campaign_hash="9" * 64))


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"derived_axes": ()}, "derived"),
        ({"pilot_evidence": ()}, "pilot evidence"),
    ),
)
def test_checkpoint_fails_closed_when_required_bindings_are_inconsistent(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _checkpoint(**override)  # type: ignore[arg-type]


def test_checkpoint_rejects_budget_that_disagrees_with_holdout_commitment() -> None:
    binding = _binding(budget=_budget(holdout=63))

    with pytest.raises(ValueError, match="holdout budget"):
        _checkpoint(binding=binding)


def test_campaign_binding_canonicalizes_pilot_evidence_order() -> None:
    legacy = EvidenceReference("legacy", "4" * 64, EvidenceRole.LEGACY)
    pilot = EvidenceReference("pilot", "3" * 64, EvidenceRole.PILOT)

    first = _binding(pilot_evidence=(pilot, legacy))
    second = _binding(pilot_evidence=(legacy, pilot))

    assert first.pilot_evidence == (legacy, pilot)
    assert first == second
    assert compute_campaign_hash(first) == compute_campaign_hash(second)


def test_campaign_binding_rejects_production_evidence_mislabelled_as_holdout() -> None:
    reference = EvidenceReference(
        evidence_id="wrong-role",
        content_hash="3" * 64,
        role=EvidenceRole.HOLDOUT,
    )

    with pytest.raises(ValueError, match="PILOT or LEGACY"):
        _binding(pilot_evidence=(reference,))


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
        _budget(pilot=invalid_count)


@pytest.mark.parametrize(
    "field_name",
    (
        "compiled_family_hash",
        "executable_envelope_hash",
        "private_doe_plan_hash",
        "role_ledger_hash",
        "replacement_policy_hash",
        "public_case_manifest_hash",
        "doe_seed_commitment",
    ),
)
def test_campaign_binding_rejects_malformed_hashes(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        _binding(**{field_name: "not-a-hash"})


@pytest.mark.parametrize(
    ("override", "message"),
    (
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
        ({"budget": object()}, "CampaignBudget"),
        ({"secret_generator_id": ""}, "secret_generator_id"),
        ({"secret_generator_id": "different-generator-v1"}, "secret_generator"),
        ({"derived_axes": ("not an identifier",)}, "valid identifiers"),
        ({"derived_axes": ("net_section_ratio", 1)}, "valid identifiers"),
        (
            {"derived_axes": ("net_section_ratio", "net_section_ratio")},
            "must be unique",
        ),
    ),
)
def test_campaign_binding_rejects_malformed_contracts(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _binding(**override)


def test_compute_campaign_hash_requires_a_campaign_binding() -> None:
    with pytest.raises(ValueError, match="CampaignBinding"):
        compute_campaign_hash(object())  # type: ignore[arg-type]


def test_checkpoint_from_bytes_round_trips_the_canonical_public_record() -> None:
    checkpoint = _checkpoint()

    restored = CPEnvelope.from_bytes(checkpoint.to_bytes())

    assert restored == checkpoint
    assert restored.to_bytes() == checkpoint.to_bytes()


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda value: {**value, "unexpected": True},
            "unexpected fields",
        ),
        (
            lambda value: {
                key: item
                for key, item in value.items()
                if key != "public_case_manifest_hash"
            },
            "missing fields",
        ),
        (
            lambda value: {**value, "$type": "piginot.doe.NotCPEnvelope"},
            "canonical type",
        ),
        (
            lambda value: {
                **value,
                "budget": {
                    **value["budget"],
                    "$type": "piginot.doe.NotCampaignBudget",
                },
            },
            "budget canonical type",
        ),
    ),
)
def test_checkpoint_from_bytes_rejects_schema_drift(
    mutate: object, message: str
) -> None:
    raw = json.loads(_checkpoint().to_bytes())
    changed = mutate(raw)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        CPEnvelope.from_bytes(json.dumps(changed).encode())


def test_checkpoint_from_bytes_rejects_noncanonical_json() -> None:
    payload = _checkpoint().to_bytes()

    with pytest.raises(ValueError, match="canonical JSON"):
        CPEnvelope.from_bytes(b" " + payload)


def test_checkpoint_from_bytes_rejects_duplicate_keys() -> None:
    payload = (
        _checkpoint()
        .to_bytes()
        .replace(
            b'{"$type":"piginot.doe.CPEnvelope",',
            b'{"$type":"piginot.doe.CPEnvelope","campaign_hash":"0",',
            1,
        )
    )

    with pytest.raises(DuplicateKeyError):
        CPEnvelope.from_bytes(payload)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"holdout_commitment": object()}, "HoldoutCommitment"),
        ({"checkpoint_id": "CP-OTHER"}, "checkpoint_id"),
        ({"schema_version": "1.0"}, "schema version"),
        ({"schema_version": "2.0"}, "schema version"),
        ({"created_utc": "2026-08-24T12:05:00+00:00"}, "ending in Z"),
        ({"created_utc": "not-a-timestampZ"}, "valid RFC 3339"),
    ),
)
def test_checkpoint_rejects_malformed_public_contracts(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _checkpoint(**override)  # type: ignore[arg-type]


def test_binding_and_checkpoint_are_immutable() -> None:
    binding = _binding()
    checkpoint = _checkpoint(binding=binding)

    with pytest.raises(FrozenInstanceError):
        binding.role_ledger_hash = "9" * 64  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        checkpoint.budget = replace(checkpoint.budget, train=129)  # type: ignore[misc]
