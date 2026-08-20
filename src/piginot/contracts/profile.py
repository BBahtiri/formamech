"""Execution attribution contract."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from piginot.contracts.errors import ContractValidationError, NonFiniteNumberError
from piginot.contracts.specs import FrozenMap, SpecContract, _nonempty

_CONTAINER_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


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
