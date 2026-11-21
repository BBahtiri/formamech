"""Execution attribution contract."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from piginot.contracts.errors import ContractValidationError, NonFiniteNumberError
from piginot.contracts.specs import FrozenMap, SpecContract, _nonempty

_CONTAINER_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
PLATFORM_FINGERPRINT_KEYS = frozenset({"os", "machine", "cpu_capability"})
REPLAY_TIERS = frozenset({"cpu-deterministic-bitwise", "gpu-accelerated-tolerance"})
GPU_FINGERPRINT_KEYS = frozenset({"device_name", "cuda_version", "cublas_workspace"})


@dataclass(frozen=True, slots=True)
class ExecutionProfile(SpecContract):
    python_version: str
    library_versions: Mapping[str, str]
    blas_threads: int
    device: str
    dtype: str
    rng_algorithm: str
    seed: int
    container_digest: str | None = None
    # Run binding v2 (S20, ADR 0101): operating system, machine and CPU
    # instruction set, which change trained weights bit for bit. Omitted from
    # the canonical form when unset, so every v1 profile keeps its hash.
    platform_fingerprint: Mapping[str, str] | None = field(
        default=None, metadata={"omit_if_none": True}
    )
    # Dual execution tier (omitted when None so existing CPU profile hashes stay intact).
    replay_tier: str | None = field(default=None, metadata={"omit_if_none": True})
    gpu_fingerprint: Mapping[str, str] | None = field(
        default=None, metadata={"omit_if_none": True}
    )

    def __post_init__(self) -> None:
        _nonempty(self.python_version, "python_version")
        _nonempty(self.device, "device")
        _nonempty(self.dtype, "dtype")
        _nonempty(self.rng_algorithm, "rng_algorithm")
        if isinstance(self.blas_threads, float) and not math.isfinite(
            self.blas_threads
        ):
            raise NonFiniteNumberError("blas_threads must be finite")
        if isinstance(self.blas_threads, bool) or not isinstance(
            self.blas_threads, int
        ):
            raise ContractValidationError("blas_threads must be an integer")
        if self.blas_threads <= 0:
            raise ContractValidationError("blas_threads must be positive")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ContractValidationError("seed must be a non-negative integer")
        versions: FrozenMap[str] = FrozenMap.copy_of(self.library_versions)
        if any(
            not isinstance(version, str) or not version for version in versions.values()
        ):
            raise ContractValidationError("library versions must be non-empty strings")
        if self.container_digest is not None and not _CONTAINER_DIGEST.fullmatch(
            self.container_digest
        ):
            raise ContractValidationError(
                "container_digest must be a lowercase sha256 digest"
            )
        object.__setattr__(self, "library_versions", versions)
        if self.platform_fingerprint is not None:
            fingerprint: FrozenMap[str] = FrozenMap.copy_of(self.platform_fingerprint)
            if set(fingerprint) != PLATFORM_FINGERPRINT_KEYS:
                raise ContractValidationError(
                    "platform_fingerprint must have exactly the keys "
                    + ", ".join(sorted(PLATFORM_FINGERPRINT_KEYS))
                )
            if any(
                not isinstance(value, str) or not value or value.strip() != value
                for value in fingerprint.values()
            ):
                raise ContractValidationError(
                    "platform_fingerprint values must be non-empty canonical text"
                )
            object.__setattr__(self, "platform_fingerprint", fingerprint)
        if self.replay_tier is not None and self.replay_tier not in REPLAY_TIERS:
            raise ContractValidationError(
                f"replay_tier must be one of {sorted(REPLAY_TIERS)}, got {self.replay_tier!r}"
            )
        if self.gpu_fingerprint is not None:
            gpu_fp: FrozenMap[str] = FrozenMap.copy_of(self.gpu_fingerprint)
            if set(gpu_fp) != GPU_FINGERPRINT_KEYS:
                raise ContractValidationError(
                    "gpu_fingerprint must have exactly the keys "
                    + ", ".join(sorted(GPU_FINGERPRINT_KEYS))
                )
            if any(
                not isinstance(value, str) or not value or value.strip() != value
                for value in gpu_fp.values()
            ):
                raise ContractValidationError(
                    "gpu_fingerprint values must be non-empty canonical text"
                )
            object.__setattr__(self, "gpu_fingerprint", gpu_fp)
