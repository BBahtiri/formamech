from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError

import pytest

from piginot import doe
from piginot.contracts import canonical_json_bytes
from piginot.contracts.errors import ContractValidationError
from piginot.doe.secret_material import (
    CAMPAIGN_SECRET_GENERATOR_ID,
    CampaignSecrets,
    SecretGenerationError,
    generate_campaign_secrets,
)

_DOE_SEED = bytes(range(32))
_ALLOCATION_SEED = bytes(range(64, 96))
_HOLDOUT_SALT = bytes(range(32, 64))


def _install_draws(
    monkeypatch: pytest.MonkeyPatch,
    draws: tuple[bytes, ...],
) -> list[int]:
    remaining = list(draws)
    requested: list[int] = []

    def token_bytes(byte_count: int) -> bytes:
        requested.append(byte_count)
        if not remaining:
            raise AssertionError("test CSPRNG draw sequence was exhausted")
        return remaining.pop(0)

    monkeypatch.setattr(
        "piginot.doe.secret_material.secrets.token_bytes",
        token_bytes,
    )
    return requested


def test_campaign_secrets_come_from_three_32_byte_csprng_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _install_draws(
        monkeypatch,
        (_DOE_SEED, _ALLOCATION_SEED, _HOLDOUT_SALT),
    )

    generated = generate_campaign_secrets()

    assert requested == [32, 32, 32]
    assert generated.doe_seed == _DOE_SEED
    assert generated.allocation_seed == _ALLOCATION_SEED
    assert generated.holdout_salt == _HOLDOUT_SALT
    assert generated.generator_id == CAMPAIGN_SECRET_GENERATOR_ID
    assert doe.generate_campaign_secrets is generate_campaign_secrets
    assert doe.CAMPAIGN_SECRET_GENERATOR_ID == CAMPAIGN_SECRET_GENERATOR_ID
    assert doe.CampaignSecrets is CampaignSecrets


def test_campaign_secret_bundle_is_factory_only_immutable_and_not_canonicalizable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_draws(monkeypatch, (_DOE_SEED, _ALLOCATION_SEED, _HOLDOUT_SALT))
    generated = generate_campaign_secrets()

    with pytest.raises(TypeError, match="generate_campaign_secrets"):
        CampaignSecrets()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="must not be subclassed"):
        type("_ForgedCampaignSecrets", (CampaignSecrets,), {})

    assert not hasattr(CampaignSecrets, "_create")
    with pytest.raises((AttributeError, FrozenInstanceError)):
        generated.doe_seed = b"replacement"  # type: ignore[misc]
    with pytest.raises(ContractValidationError, match="unsupported canonical"):
        canonical_json_bytes(generated)
    with pytest.raises(TypeError, match="must not be pickled"):
        pickle.dumps(generated)
    assert repr(generated) == (
        f"CampaignSecrets(generator_id={CAMPAIGN_SECRET_GENERATOR_ID!r})"
    )


def test_generator_discards_structurally_weak_and_duplicate_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _install_draws(
        monkeypatch,
        (
            bytes(32),
            _DOE_SEED,
            _DOE_SEED,
            _ALLOCATION_SEED,
            _HOLDOUT_SALT,
        ),
    )

    generated = generate_campaign_secrets()

    assert requested == [32, 32, 32, 32, 32]
    assert (
        len({generated.doe_seed, generated.allocation_seed, generated.holdout_salt})
        == 3
    )


def test_generator_fails_closed_when_csprng_never_returns_usable_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "piginot.doe.secret_material.secrets.token_bytes",
        lambda _byte_count: bytes(32),
    )

    with pytest.raises(SecretGenerationError, match="CSPRNG"):
        generate_campaign_secrets()
