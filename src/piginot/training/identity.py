"""Canonical identity for one deterministic S13 training run."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar

from piginot.contracts import ExecutionProfile, hash_spec
from piginot.contracts.errors import ContractValidationError
from piginot.contracts.specs import SpecContract

_CONTENT_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MODULE_DIGEST = re.compile(r"blake2b256:[0-9a-f]{64}\Z")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_MAX_SAFE_INTEGER = 2**53 - 1


def _content_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH.fullmatch(value) is None:
        raise ContractValidationError(
            f"{label} must be a lowercase 64-character hex hash"
        )
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{label} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class TrainingRunBinding(SpecContract):
    """Complete canonical preimage identifying one training recipe."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    dataset_snapshot_hash: str
    public_manifest_hash: str
    normalization_hash: str
    architecture_id: str
    architecture_module_digest: str
    objective_registry_hash: str
    trainer_config_hash: str
    sampler_plan_hash: str
    initial_weights_hash: str
    implementation_revision: str
    seed: int
    execution_profile: ExecutionProfile
    problem_spec_hash: str | None = field(default=None, metadata={"omit_if_none": True})
    """The ProblemSpec a run was built from (binding v2, ADR 0106); absent in
    v1 bindings, whose canonical form and run hash it leaves unchanged."""

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_snapshot_hash",
            "public_manifest_hash",
            "normalization_hash",
            "objective_registry_hash",
            "trainer_config_hash",
            "sampler_plan_hash",
            "initial_weights_hash",
        ):
            _content_hash(getattr(self, field_name), field_name)
        _text(self.architecture_id, "architecture_id")
        if (
            not isinstance(self.architecture_module_digest, str)
            or _MODULE_DIGEST.fullmatch(self.architecture_module_digest) is None
        ):
            raise ContractValidationError(
                "architecture_module_digest must be a lowercase "
                "blake2b256:<64-hex> digest"
            )
        if (
            not isinstance(self.implementation_revision, str)
            or _GIT_REVISION.fullmatch(self.implementation_revision) is None
        ):
            raise ContractValidationError(
                "implementation_revision must be a lowercase 40-character git hash"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or self.seed > _MAX_SAFE_INTEGER
        ):
            raise ContractValidationError(
                "seed must be a non-negative integer within the RFC 8785 safe range"
            )
        if type(self.execution_profile) is not ExecutionProfile:
            raise ContractValidationError(
                "execution_profile must be an exact ExecutionProfile"
            )
        if self.execution_profile.seed != self.seed:
            raise ContractValidationError(
                "execution profile seed must match the training run seed"
            )
        if self.problem_spec_hash is not None:
            _content_hash(self.problem_spec_hash, "problem_spec_hash")

    @property
    def execution_profile_hash(self) -> str:
        """Return the content identity of the exact execution environment."""

        return hash_spec(self.execution_profile, "content")

    @property
    def run_hash(self) -> str:
        """Return the domain-separated identity of this complete run recipe."""

        return hash_spec(self, "recipe")


__all__ = ("TrainingRunBinding",)
