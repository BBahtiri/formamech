from __future__ import annotations

import hashlib

import pytest
import rfc8785

import piginot.doe.commitment as commitment_module
from piginot.doe.commitment import (
    HoldoutCommitment,
    compute_holdout_commitment,
    verify_holdout_reveal,
)

_STRONG_SALT = bytes(range(32))
_ALTERNATE_STRONG_SALT = bytes(range(32, 64))


def _coordinates(count: int = 64) -> tuple[dict[str, float], ...]:
    return tuple(
        {"net_section_ratio": 1.0 + index / 100.0, "load_ratio": index / 63.0}
        for index in range(count)
    )


def _commitment(
    coordinates: object | None = None,
    *,
    salt: bytes = _STRONG_SALT,
    created_utc: str = "2026-08-24T10:00:00Z",
) -> HoldoutCommitment:
    return compute_holdout_commitment(
        campaign_hash="a" * 64,
        salt=salt,
        holdout_coordinates=(_coordinates() if coordinates is None else coordinates),
        created_utc=created_utc,
    )


def test_commitment_is_rfc8785_digest_of_salt_and_sorted_coordinates() -> None:
    coordinates = _coordinates()

    actual = _commitment(reversed(coordinates))
    sorted_payload = sorted(
        coordinates,
        key=lambda coordinate: rfc8785.dumps(coordinate),
    )
    expected = hashlib.blake2b(
        _STRONG_SALT + rfc8785.dumps(sorted_payload),
        digest_size=32,
    ).hexdigest()

    assert actual.commitment == expected
    assert actual.n_holdout_points == 64


def test_input_order_and_mapping_key_order_do_not_change_commitment() -> None:
    coordinates = _coordinates()
    reversed_keys = tuple(
        dict(reversed(tuple(coordinate.items()))) for coordinate in coordinates
    )

    first = _commitment(coordinates)
    second = _commitment(reversed(reversed_keys))

    assert first == second


@pytest.mark.parametrize(
    "coordinates, message",
    [
        (_coordinates(63), "exactly 64"),
        ((*_coordinates(63), _coordinates(63)[0]), "duplicate"),
        (
            (*_coordinates(63), {"net_section_ratio": 1.7, "other": 0.5}),
            "schema",
        ),
    ],
)
def test_production_commitment_rejects_wrong_count_duplicates_and_mixed_schemas(
    coordinates: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _commitment(coordinates)


def test_test_only_helper_is_explicitly_nonproduction() -> None:
    fixture = commitment_module._compute_holdout_commitment_for_test(
        campaign_hash="a" * 64,
        salt=_STRONG_SALT,
        holdout_coordinates=_coordinates(2),
        created_utc="2026-08-24T10:00:00Z",
    )

    assert fixture.n_holdout_points == 2
    assert type(fixture).__name__ == "_FixtureHoldoutCommitment"


@pytest.mark.parametrize(
    "salt",
    [b"", b"s" * 31, bytearray(b"s" * 32), "s" * 32],
)
def test_salt_must_be_at_least_32_bytes(salt: object) -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        _commitment(salt=salt)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "salt",
    (bytes(32), b"x" * 32, bytes(range(8)) * 4),
)
def test_exported_commitment_primitive_rejects_obviously_weak_salts(
    salt: bytes,
) -> None:
    with pytest.raises(ValueError, match="salt.*diversity"):
        _commitment(salt=salt)


def test_reveal_refuses_a_weak_salt_even_for_a_preexisting_record() -> None:
    coordinates = _coordinates()
    sorted_payload = sorted(
        coordinates,
        key=lambda coordinate: rfc8785.dumps(coordinate),
    )
    weak_salt = bytes(32)
    record = HoldoutCommitment(
        scheme="blake2b-256",
        campaign_hash="a" * 64,
        n_holdout_points=64,
        commitment=hashlib.blake2b(
            weak_salt + rfc8785.dumps(sorted_payload),
            digest_size=32,
        ).hexdigest(),
        created_utc="2026-08-24T10:00:00Z",
    )

    with pytest.raises(ValueError, match="salt.*diversity"):
        verify_holdout_reveal(
            record,
            salt=weak_salt,
            holdout_coordinates=coordinates,
        )


@pytest.mark.parametrize(
    "created_utc",
    [
        "",
        "2026-08-24T10:00:00",
        "2026-08-24T10:00:00+00:00",
        "2026-08-24T12:00:00+02:00",
        "2026-13-24T10:00:00Z",
    ],
)
def test_created_timestamp_must_be_strict_utc(created_utc: str) -> None:
    with pytest.raises(ValueError, match="UTC"):
        _commitment(created_utc=created_utc)


def test_direct_public_constructor_enforces_production_contract() -> None:
    with pytest.raises(ValueError, match="exactly 64"):
        HoldoutCommitment(
            scheme="blake2b-256",
            campaign_hash="a" * 64,
            n_holdout_points=2,
            commitment="b" * 64,
            created_utc="2026-08-24T10:00:00Z",
        )


def test_public_record_never_carries_salt_or_coordinates() -> None:
    commitment = _commitment()

    assert set(commitment.public_record()) == {
        "scheme",
        "campaign_hash",
        "n_holdout_points",
        "commitment",
        "created_utc",
    }
    assert not hasattr(commitment, "salt")
    assert not hasattr(commitment, "coordinates")


def test_reveal_uses_constant_time_digest_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commitment = _commitment()
    calls: list[tuple[str, str]] = []

    def recording_compare(left: str, right: str) -> bool:
        calls.append((left, right))
        return left == right

    monkeypatch.setattr(
        "piginot.doe.commitment.hmac.compare_digest",
        recording_compare,
    )

    assert verify_holdout_reveal(
        commitment,
        salt=_STRONG_SALT,
        holdout_coordinates=reversed(_coordinates()),
    )
    assert calls == [(commitment.commitment, commitment.commitment)]


def test_reveal_rejects_wrong_secret_and_malformed_coordinates() -> None:
    commitment = _commitment()

    assert not verify_holdout_reveal(
        commitment,
        salt=_ALTERNATE_STRONG_SALT,
        holdout_coordinates=_coordinates(),
    )
    with pytest.raises(ValueError, match="duplicate"):
        verify_holdout_reveal(
            commitment,
            salt=_STRONG_SALT,
            holdout_coordinates=(*_coordinates(63), _coordinates(63)[0]),
        )


def test_coordinates_must_be_finite_numeric_mappings() -> None:
    invalid: tuple[object, ...] = (*_coordinates(63), {"x": True, "y": 1.0})

    with pytest.raises(ValueError, match="finite number"):
        _commitment(invalid)


@pytest.mark.parametrize(
    "coordinates, message",
    [
        ({"x": 1.0}, "iterable of mappings"),
        (object(), "iterable of mappings"),
        ((*_coordinates(63), object()), "non-empty mapping"),
        ((*_coordinates(63), {}), "non-empty mapping"),
        ((*_coordinates(63), {1: 1.0}), "names"),
        ((*_coordinates(63), {"x": float("nan")}), "finite number"),
        ((*_coordinates(63), {"x": float("inf")}), "finite number"),
    ],
)
def test_malformed_coordinate_containers_fail_closed(
    coordinates: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _commitment(coordinates)


def test_empty_test_fixture_population_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        commitment_module._compute_holdout_commitment_for_test(
            campaign_hash="a" * 64,
            salt=_STRONG_SALT,
            holdout_coordinates=(),
            created_utc="2026-08-24T10:00:00Z",
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"scheme": "sha256"}, "scheme"),
        ({"campaign_hash": "A" * 64}, "lowercase"),
        ({"commitment": "not-a-digest"}, "lowercase"),
        ({"n_holdout_points": False}, "integer"),
    ],
)
def test_direct_constructor_rejects_invalid_public_metadata(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "scheme": "blake2b-256",
        "campaign_hash": "a" * 64,
        "n_holdout_points": 64,
        "commitment": "b" * 64,
        "created_utc": "2026-08-24T10:00:00.123456Z",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        HoldoutCommitment(**values)  # type: ignore[arg-type]


def test_fixture_constructor_still_requires_a_positive_population() -> None:
    with pytest.raises(ValueError, match="positive"):
        commitment_module._FixtureHoldoutCommitment(
            scheme="blake2b-256",
            campaign_hash="a" * 64,
            n_holdout_points=0,
            commitment="b" * 64,
            created_utc="2026-08-24T10:00:00Z",
        )


def test_commitment_instance_verifies_its_reveal() -> None:
    commitment = _commitment()

    assert commitment.verify_reveal(
        salt=_STRONG_SALT,
        holdout_coordinates=_coordinates(),
    )


def test_reveal_rejects_non_commitment_record() -> None:
    with pytest.raises(ValueError, match="HoldoutCommitment"):
        verify_holdout_reveal(
            object(),  # type: ignore[arg-type]
            salt=_STRONG_SALT,
            holdout_coordinates=_coordinates(),
        )
