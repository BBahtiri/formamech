"""Public holdout commitments whose secret reveal stays out of repository state."""

from __future__ import annotations

import hashlib
import hmac
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import ClassVar, TypeVar

from piginot.contracts import canonical_json_bytes

_SCHEME = "blake2b-256"
_PRODUCTION_HOLDOUT_COUNT = 64
_MINIMUM_SALT_BYTES = 32
_LOWER_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")


def _validate_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _LOWER_HEX_256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase 64-character hex")
    return value


def _validate_utc_timestamp(value: object) -> str:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise ValueError("created_utc must be a strict UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError("created_utc must be a valid strict UTC timestamp") from exc
    return value


def _validate_salt(salt: object) -> bytes:
    if not isinstance(salt, bytes) or len(salt) < _MINIMUM_SALT_BYTES:
        raise ValueError("salt must be bytes containing at least 32 bytes")
    return salt


def _coordinate_payload(
    coordinates: Iterable[Mapping[str, float]] | object,
    *,
    expected_count: int | None,
) -> tuple[Mapping[str, float], ...]:
    if isinstance(coordinates, (str, bytes, bytearray, Mapping)):
        raise ValueError("holdout coordinates must be an iterable of mappings")
    try:
        supplied: tuple[object, ...] = tuple(coordinates)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("holdout coordinates must be an iterable of mappings") from exc
    if expected_count is not None and len(supplied) != expected_count:
        raise ValueError(
            f"production holdout must contain exactly {expected_count} coordinates"
        )
    if not supplied:
        raise ValueError("holdout coordinates must not be empty")

    checked: list[Mapping[str, float]] = []
    schemas: set[tuple[str, ...]] = set()
    canonical_points: set[bytes] = set()
    for point_index, coordinate in enumerate(supplied):
        if not isinstance(coordinate, Mapping) or not coordinate:
            raise ValueError("each holdout coordinate must be a non-empty mapping")
        point: dict[str, float] = {}
        for name, value in coordinate.items():
            if not isinstance(name, str) or not name:
                raise ValueError("holdout coordinate names must be non-empty text")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"holdout coordinate {point_index}:{name} must be a finite number"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(
                    f"holdout coordinate {point_index}:{name} must be a finite number"
                )
            point[name] = number
        schemas.add(tuple(sorted(point)))
        frozen = MappingProxyType(dict(sorted(point.items())))
        canonical = canonical_json_bytes(frozen)
        if canonical in canonical_points:
            raise ValueError("duplicate holdout coordinates are forbidden")
        canonical_points.add(canonical)
        checked.append(frozen)

    if len(schemas) != 1:
        raise ValueError("holdout coordinates must all use the same schema")
    return tuple(sorted(checked, key=canonical_json_bytes))


@dataclass(frozen=True, slots=True)
class HoldoutCommitment:
    """Public exact-64 holdout commitment; no reveal material is retained."""

    __canonical_namespace__: ClassVar[str] = "piginot.doe"
    _required_count: ClassVar[int | None] = _PRODUCTION_HOLDOUT_COUNT

    scheme: str
    campaign_hash: str
    n_holdout_points: int
    commitment: str
    created_utc: str

    def __post_init__(self) -> None:
        if self.scheme != _SCHEME:
            raise ValueError(f"holdout commitment scheme must be {_SCHEME}")
        _validate_hash(self.campaign_hash, label="campaign_hash")
        _validate_hash(self.commitment, label="commitment")
        if isinstance(self.n_holdout_points, bool) or not isinstance(
            self.n_holdout_points, int
        ):
            raise ValueError("n_holdout_points must be an integer")
        required_count = type(self)._required_count
        if required_count is not None and self.n_holdout_points != required_count:
            raise ValueError(
                f"production holdout must contain exactly {required_count} coordinates"
            )
        if self.n_holdout_points <= 0:
            raise ValueError("n_holdout_points must be positive")
        _validate_utc_timestamp(self.created_utc)

    def public_record(self) -> Mapping[str, object]:
        """Return only the five public preregistration fields."""

        return MappingProxyType(
            {
                "scheme": self.scheme,
                "campaign_hash": self.campaign_hash,
                "n_holdout_points": self.n_holdout_points,
                "commitment": self.commitment,
                "created_utc": self.created_utc,
            }
        )

    def verify_reveal(
        self,
        *,
        salt: bytes,
        holdout_coordinates: Iterable[Mapping[str, float]] | object,
    ) -> bool:
        return verify_holdout_reveal(
            self,
            salt=salt,
            holdout_coordinates=holdout_coordinates,
        )


@dataclass(frozen=True, slots=True)
class _FixtureHoldoutCommitment(HoldoutCommitment):
    """Private small-population record for unit fixtures only."""

    _required_count: ClassVar[int | None] = None


_Commitment = TypeVar("_Commitment", bound=HoldoutCommitment)


def _compute(
    *,
    record_type: type[_Commitment],
    expected_count: int | None,
    campaign_hash: str,
    salt: bytes,
    holdout_coordinates: Iterable[Mapping[str, float]] | object,
    created_utc: str,
) -> _Commitment:
    validated_campaign = _validate_hash(campaign_hash, label="campaign_hash")
    validated_salt = _validate_salt(salt)
    validated_created = _validate_utc_timestamp(created_utc)
    coordinates = _coordinate_payload(
        holdout_coordinates,
        expected_count=expected_count,
    )
    digest = hashlib.blake2b(
        validated_salt + canonical_json_bytes(coordinates),
        digest_size=32,
    ).hexdigest()
    return record_type(
        scheme=_SCHEME,
        campaign_hash=validated_campaign,
        n_holdout_points=len(coordinates),
        commitment=digest,
        created_utc=validated_created,
    )


def compute_holdout_commitment(
    *,
    campaign_hash: str,
    salt: bytes,
    holdout_coordinates: Iterable[Mapping[str, float]] | object,
    created_utc: str,
) -> HoldoutCommitment:
    """Create a production commitment for exactly 64 unique coordinates."""

    return _compute(
        record_type=HoldoutCommitment,
        expected_count=_PRODUCTION_HOLDOUT_COUNT,
        campaign_hash=campaign_hash,
        salt=salt,
        holdout_coordinates=holdout_coordinates,
        created_utc=created_utc,
    )


def _compute_holdout_commitment_for_test(
    *,
    campaign_hash: str,
    salt: bytes,
    holdout_coordinates: Iterable[Mapping[str, float]] | object,
    created_utc: str,
) -> _FixtureHoldoutCommitment:
    """Create a deliberately non-production commitment for a small unit fixture."""

    return _compute(
        record_type=_FixtureHoldoutCommitment,
        expected_count=None,
        campaign_hash=campaign_hash,
        salt=salt,
        holdout_coordinates=holdout_coordinates,
        created_utc=created_utc,
    )


def verify_holdout_reveal(
    commitment: HoldoutCommitment,
    *,
    salt: bytes,
    holdout_coordinates: Iterable[Mapping[str, float]] | object,
) -> bool:
    """Verify a withheld reveal using constant-time digest comparison."""

    if not isinstance(commitment, HoldoutCommitment):
        raise ValueError("commitment must be a HoldoutCommitment")
    coordinates = _coordinate_payload(
        holdout_coordinates,
        expected_count=commitment.n_holdout_points,
    )
    digest = hashlib.blake2b(
        _validate_salt(salt) + canonical_json_bytes(coordinates),
        digest_size=32,
    ).hexdigest()
    return hmac.compare_digest(digest, commitment.commitment)


__all__ = (
    "HoldoutCommitment",
    "compute_holdout_commitment",
    "verify_holdout_reveal",
)
