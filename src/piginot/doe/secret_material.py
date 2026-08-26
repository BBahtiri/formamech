"""Factory-only CSPRNG material for one S9 campaign ceremony."""

from __future__ import annotations

import secrets
from typing import Final, NoReturn, SupportsIndex

CAMPAIGN_SECRET_GENERATOR_ID: Final = "python-secrets-token-bytes-32-v1"

_SECRET_BYTES = 32
_MINIMUM_DISTINCT_OCTETS = 16
_MAX_CSPRNG_DRAWS = 64


class SecretMaterialError(ValueError):
    """Secret bytes fail the S9 structural defense-in-depth checks."""


class SecretGenerationError(RuntimeError):
    """The operating-system CSPRNG did not yield usable campaign material."""


def validate_secret_bytes(value: object, label: str) -> bytes:
    """Reject malformed or obviously weak material without claiming entropy proof."""

    if type(value) is not bytes or len(value) < _SECRET_BYTES:
        raise SecretMaterialError(
            f"{label} must be bytes containing at least {_SECRET_BYTES} bytes"
        )
    if len(set(value)) < _MINIMUM_DISTINCT_OCTETS or _is_periodic(value):
        raise SecretMaterialError(f"{label} has obviously insufficient octet diversity")
    return value


def _is_periodic(value: bytes) -> bool:
    for period in range(1, len(value) // 2 + 1):
        if len(value) % period == 0 and value == value[:period] * (
            len(value) // period
        ):
            return True
    return False


class CampaignSecrets:
    """Non-serializable, immutable secret bundle created only by the CSPRNG factory."""

    _allocation_seed: bytes
    _doe_seed: bytes
    _generator_id: str
    _holdout_salt: bytes

    __slots__ = (
        "_allocation_seed",
        "_doe_seed",
        "_generator_id",
        "_holdout_salt",
    )

    def __new__(cls) -> CampaignSecrets:
        raise TypeError(
            "CampaignSecrets must be created by generate_campaign_secrets()"
        )

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("CampaignSecrets must not be subclassed")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("campaign secret material is immutable")

    def __repr__(self) -> str:
        return f"CampaignSecrets(generator_id={self.generator_id!r})"

    def __reduce_ex__(self, _protocol: SupportsIndex) -> NoReturn:
        raise TypeError("campaign secret material must not be pickled")

    @property
    def doe_seed(self) -> bytes:
        return self._doe_seed

    @property
    def allocation_seed(self) -> bytes:
        return self._allocation_seed

    @property
    def holdout_salt(self) -> bytes:
        return self._holdout_salt

    @property
    def generator_id(self) -> str:
        return self._generator_id


def _draw_secret(existing: frozenset[bytes]) -> bytes:
    for _attempt in range(_MAX_CSPRNG_DRAWS):
        candidate = secrets.token_bytes(_SECRET_BYTES)
        try:
            validated = validate_secret_bytes(candidate, "CSPRNG output")
        except SecretMaterialError:
            continue
        if validated not in existing:
            return validated
    raise SecretGenerationError(
        "operating-system CSPRNG did not yield distinct usable campaign material"
    )


def generate_campaign_secrets() -> CampaignSecrets:
    """Generate independent DOE, allocation, and holdout secrets from the OS CSPRNG."""

    doe_seed = _draw_secret(frozenset())
    allocation_seed = _draw_secret(frozenset({doe_seed}))
    holdout_salt = _draw_secret(frozenset({doe_seed, allocation_seed}))
    generated = object.__new__(CampaignSecrets)
    object.__setattr__(generated, "_doe_seed", doe_seed)
    object.__setattr__(generated, "_allocation_seed", allocation_seed)
    object.__setattr__(generated, "_holdout_salt", holdout_salt)
    object.__setattr__(
        generated,
        "_generator_id",
        CAMPAIGN_SECRET_GENERATOR_ID,
    )
    return generated


__all__ = (
    "CAMPAIGN_SECRET_GENERATOR_ID",
    "CampaignSecrets",
    "SecretGenerationError",
    "SecretMaterialError",
    "generate_campaign_secrets",
)
