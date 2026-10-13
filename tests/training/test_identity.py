from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, fields

import pytest

from piginot.contracts import (
    ExecutionProfile,
    canonical_json_bytes,
    hash_spec,
)
from piginot.contracts.errors import ContractValidationError, UnknownFieldError
from piginot.training.identity import TrainingRunBinding

_HASH_A = "a" * 64
_HASH_B = "b" * 64


def _profile(**overrides: object) -> ExecutionProfile:
    values: dict[str, object] = {
        "python_version": "3.12.5",
        "library_versions": {
            "numpy": "2.2.6",
            "safetensors": "0.8.1",
            "torch": "2.7.1",
        },
        "blas_threads": 1,
        "device": "cpu",
        "dtype": "float64",
        "rng_algorithm": "MT19937",
        "seed": 1729,
        "container_digest": None,
    }
    values.update(overrides)
    return ExecutionProfile(**values)  # type: ignore[arg-type]


def _binding(**overrides: object) -> TrainingRunBinding:
    values: dict[str, object] = {
        "dataset_snapshot_hash": _HASH_A,
        "public_manifest_hash": "1" * 64,
        "normalization_hash": "2" * 64,
        "architecture_id": "pi-ginot-s11-boundary-cross-attention-v1",
        "architecture_module_digest": f"blake2b256:{'3' * 64}",
        "objective_registry_hash": "4" * 64,
        "trainer_config_hash": "5" * 64,
        "sampler_plan_hash": "6" * 64,
        "initial_weights_hash": "7" * 64,
        "implementation_revision": "8" * 40,
        "seed": 1729,
        "execution_profile": _profile(),
    }
    values.update(overrides)
    return TrainingRunBinding(**values)  # type: ignore[arg-type]


def test_training_run_binding_has_the_complete_canonical_preimage() -> None:
    binding = _binding()

    assert {field.name for field in fields(binding)} == {
        "dataset_snapshot_hash",
        "public_manifest_hash",
        "normalization_hash",
        "architecture_id",
        "architecture_module_digest",
        "objective_registry_hash",
        "trainer_config_hash",
        "sampler_plan_hash",
        "initial_weights_hash",
        "implementation_revision",
        "seed",
        "execution_profile",
        "problem_spec_hash",
    }
    assert b'"$type":"piginot.training.TrainingRunBinding"' in canonical_json_bytes(
        binding
    )


def test_run_hash_is_the_recipe_domain_hash_and_is_repeatable() -> None:
    first = _binding()
    second = _binding(
        execution_profile=_profile(
            library_versions={
                "torch": "2.7.1",
                "numpy": "2.2.6",
                "safetensors": "0.8.1",
            }
        )
    )

    assert first == second
    assert first.run_hash == hash_spec(first, "recipe")
    assert first.run_hash == second.run_hash
    assert first.execution_profile_hash == hash_spec(first.execution_profile, "content")


@pytest.mark.parametrize(
    "change",
    (
        {"dataset_snapshot_hash": _HASH_B},
        {"trainer_config_hash": _HASH_B},
        {"implementation_revision": "9" * 40},
        {"execution_profile": _profile(dtype="float32")},
    ),
    ids=("data", "config", "code", "profile"),
)
def test_every_run_identity_dimension_changes_the_run_hash(
    change: dict[str, object],
) -> None:
    assert _binding(**change).run_hash != _binding().run_hash


def test_training_run_binding_is_frozen_and_hashable() -> None:
    binding = _binding()

    assert hash(binding)
    with pytest.raises(FrozenInstanceError):
        binding.seed = 7  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_name",
    (
        "dataset_snapshot_hash",
        "public_manifest_hash",
        "normalization_hash",
        "objective_registry_hash",
        "trainer_config_hash",
        "sampler_plan_hash",
        "initial_weights_hash",
    ),
)
@pytest.mark.parametrize("invalid", ("", "A" * 64, "a" * 63, "g" * 64))
def test_content_hash_fields_require_lowercase_blake2b256_hex(
    field_name: str, invalid: str
) -> None:
    with pytest.raises(ContractValidationError, match=field_name):
        _binding(**{field_name: invalid})


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("architecture_id", ""),
        ("architecture_id", "  "),
        ("architecture_module_digest", "3" * 64),
        ("architecture_module_digest", f"blake2b256:{'A' * 64}"),
        ("architecture_module_digest", f"sha256:{'3' * 64}"),
        ("implementation_revision", "8" * 39),
        ("implementation_revision", "A" * 40),
        ("implementation_revision", "g" * 40),
        ("seed", True),
        ("seed", -1),
        ("seed", 2**53),
    ),
)
def test_non_hash_identity_fields_are_strictly_validated(
    field_name: str, invalid: object
) -> None:
    with pytest.raises(ContractValidationError, match=field_name):
        _binding(**{field_name: invalid})


def test_binding_requires_the_exact_profile_and_matching_seed() -> None:
    with pytest.raises(ContractValidationError, match="exact ExecutionProfile"):
        _binding(execution_profile=object())
    with pytest.raises(ContractValidationError, match="profile seed"):
        _binding(seed=1730)


def test_from_mapping_rejects_unknown_fields() -> None:
    with pytest.raises(UnknownFieldError):
        TrainingRunBinding.from_mapping(
            {
                **{
                    field.name: getattr(_binding(), field.name)
                    for field in fields(TrainingRunBinding)
                },
                "absolute_checkpoint_path": "must-not-enter-run-identity",
            }
        )


def test_a_binding_without_a_problem_spec_keeps_its_v1_canonical_form() -> None:
    binding = _binding()
    payload = canonical_json_bytes(binding)
    assert b"problem_spec_hash" not in payload
    v1_fields = {f.name for f in fields(binding)} - {"problem_spec_hash"}
    assert set(json.loads(payload)) == v1_fields | {"$type"}


def test_a_problem_spec_hash_is_bound_and_validated() -> None:
    bound = _binding(problem_spec_hash=_HASH_B)
    assert b'"problem_spec_hash":"' + _HASH_B.encode() in canonical_json_bytes(bound)
    assert bound.run_hash != _binding().run_hash
    assert bound.run_hash != _binding(problem_spec_hash=_HASH_A).run_hash
    assert (
        TrainingRunBinding.from_mapping(
            {f.name: getattr(bound, f.name) for f in fields(bound)}
        )
        == bound
    )
    with pytest.raises(ContractValidationError, match="problem_spec_hash"):
        _binding(problem_spec_hash="B" * 64)
