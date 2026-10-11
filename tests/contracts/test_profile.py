from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

from piginot.contracts import ExecutionProfile
from piginot.contracts.errors import (
    ContractValidationError,
    NonFiniteNumberError,
    UnknownFieldError,
)


def _profile(**overrides: object) -> ExecutionProfile:
    values: dict[str, object] = {
        "python_version": "3.12.5",
        "library_versions": {"numpy": "2.3.2", "scipy": "1.16.1"},
        "blas_threads": 1,
        "device": "cpu",
        "dtype": "float64",
        "rng_algorithm": "PCG64",
        "seed": 314159,
        "container_digest": None,
    }
    values.update(overrides)
    return ExecutionProfile(**values)


def test_execution_profile_has_every_attribution_field() -> None:
    profile_fields = {field.name for field in fields(ExecutionProfile)}

    assert {
        "python_version",
        "library_versions",
        "blas_threads",
        "device",
        "dtype",
        "rng_algorithm",
        "seed",
        "container_digest",
    } <= profile_fields


def test_execution_profile_is_frozen_and_hashable() -> None:
    profile = _profile()

    assert is_dataclass(profile)
    assert profile.__dataclass_params__.frozen
    assert hash(profile)

    with pytest.raises(FrozenInstanceError):
        profile.seed = 7


def test_execution_profile_detaches_and_freezes_library_versions() -> None:
    versions = {"numpy": "2.3.2", "scipy": "1.16.1"}
    profile = _profile(library_versions=versions)

    versions["numpy"] = "0.0.0"
    versions["torch"] = "2.8.0"

    assert dict(profile.library_versions) == {
        "numpy": "2.3.2",
        "scipy": "1.16.1",
    }
    with pytest.raises(TypeError):
        profile.library_versions["numpy"] = "0.0.0"


def test_execution_profile_equality_ignores_mapping_insertion_order() -> None:
    first = _profile(library_versions={"numpy": "2.3.2", "scipy": "1.16.1"})
    second = _profile(library_versions={"scipy": "1.16.1", "numpy": "2.3.2"})

    assert first == second
    assert hash(first) == hash(second)


def test_execution_profile_allows_an_attributable_container_digest() -> None:
    profile = _profile(
        container_digest=(
            "sha256:9123d0810ca35b8e14d62af4f4887a61b1237df4cc333962131a9a814ac42b34"
        )
    )

    assert profile.container_digest is not None
    assert profile.container_digest.startswith("sha256:")


def test_execution_profile_from_mapping_rejects_unknown_fields() -> None:
    with pytest.raises(UnknownFieldError):
        ExecutionProfile.from_mapping(
            {
                "python_version": "3.12.5",
                "unexpected_field": "not permitted",
            }
        )


def test_execution_profile_from_mapping_constructs_and_detaches_profile() -> None:
    versions = {"numpy": "2.3.2"}
    profile = ExecutionProfile.from_mapping(
        {
            "python_version": "3.12.5",
            "library_versions": versions,
            "blas_threads": 1,
            "device": "cpu",
            "dtype": "float64",
            "rng_algorithm": "PCG64",
            "seed": 0,
            "container_digest": None,
        }
    )
    versions.clear()

    assert dict(profile.library_versions) == {"numpy": "2.3.2"}


def test_execution_profile_from_mapping_wraps_missing_required_fields() -> None:
    with pytest.raises(ContractValidationError):
        ExecutionProfile.from_mapping({"python_version": "3.12.5"})


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("python_version", ""),
        ("library_versions", {"numpy": ""}),
        ("blas_threads", 0),
        ("blas_threads", True),
        ("device", ""),
        ("dtype", ""),
        ("rng_algorithm", ""),
        ("seed", -1),
        ("container_digest", ""),
    ],
)
def test_execution_profile_rejects_invalid_attribution_values(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ContractValidationError):
        _profile(**{field_name: invalid_value})


def test_execution_profile_rejects_non_finite_numbers_with_typed_error() -> None:
    with pytest.raises(NonFiniteNumberError):
        _profile(blas_threads=float("inf"))


# --- run binding v2: platform fingerprint (S20, ADR 0101) -------------------

_V1 = {
    "python_version": "3.11.15",
    "library_versions": {"torch": "2.13.0"},
    "blas_threads": 2,
    "device": "cpu",
    "dtype": "float32",
    "rng_algorithm": "torch-cpu-mt19937",
    "seed": 7,
}
_FINGERPRINT = {"os": "Linux", "machine": "x86_64", "cpu_capability": "AVX2"}


def test_a_v1_profile_keeps_its_canonical_bytes_and_hash() -> None:
    from piginot.contracts import hash_spec
    from piginot.contracts.canonical import canonical_json_bytes

    profile = ExecutionProfile(**_V1)
    assert b"platform_fingerprint" not in canonical_json_bytes(profile)
    expected = (
        b'{"$type":"piginot.contracts.ExecutionProfile","blas_threads":2,'
        b'"container_digest":null,"device":"cpu","dtype":"float32",'
        b'"library_versions":{"torch":"2.13.0"},"python_version":"3.11.15",'
        b'"rng_algorithm":"torch-cpu-mt19937","seed":7}'
    )
    assert canonical_json_bytes(profile) == expected
    assert hash_spec(profile, "content") == hash_spec(
        ExecutionProfile.from_mapping(dict(_V1)), "content"
    )


def test_the_platform_fingerprint_enters_the_identity() -> None:
    from piginot.contracts import hash_spec

    v1 = ExecutionProfile(**_V1)
    avx2 = ExecutionProfile(**_V1, platform_fingerprint=_FINGERPRINT)
    avx512 = ExecutionProfile(
        **_V1, platform_fingerprint={**_FINGERPRINT, "cpu_capability": "AVX512"}
    )
    hashes = {hash_spec(p, "content") for p in (v1, avx2, avx512)}
    assert len(hashes) == 3
    assert dict(avx2.platform_fingerprint or {}) == _FINGERPRINT


@pytest.mark.parametrize(
    "fingerprint",
    [
        {"os": "Linux", "machine": "x86_64"},
        {**_FINGERPRINT, "gpu": "none"},
        {**_FINGERPRINT, "os": ""},
        {**_FINGERPRINT, "os": " Linux"},
        {**_FINGERPRINT, "cpu_capability": 2},
    ],
)
def test_a_malformed_fingerprint_is_refused(fingerprint: dict[str, object]) -> None:
    with pytest.raises(ContractValidationError):
        ExecutionProfile(**_V1, platform_fingerprint=fingerprint)  # type: ignore[arg-type]


def test_the_running_platform_fingerprint_is_complete() -> None:
    from piginot.training.platform_fingerprint import platform_fingerprint

    fingerprint = platform_fingerprint()
    assert set(fingerprint) == {"os", "machine", "cpu_capability"}
    ExecutionProfile(**_V1, platform_fingerprint=fingerprint)
