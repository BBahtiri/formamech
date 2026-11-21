"""Phase 1 tests: pluggable SolverBackend registry and dual CPU/GPU profiles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from piginot.compiler.execution_runtime import observe_runtime, profile_mismatches
from piginot.contracts import ExecutionProfile, canonical_json_bytes
from piginot.contracts.errors import ContractValidationError
from piginot.fem import (
    BackendCapabilities,
    ScikitFemBackend,
    SolverBackend,
    available_backends,
    register_backend,
    resolve_backend,
    unregister_backend,
)
from piginot.fem.models import FemAttempt, FemRequest


class _DummyBackend:
    backend_name = "dummy"
    backend_id = "dummy-test-v1"
    capabilities = BackendCapabilities(supports_gpu=True)

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        return ()

    def solve_from(
        self,
        request: FemRequest,
        initial: NDArray[np.float64],
        load_factors: Sequence[float] = (1.0,),
    ) -> FemAttempt:
        raise NotImplementedError

    def linearize(self, request: FemRequest) -> Any:
        raise NotImplementedError


def test_scikit_fem_satisfies_solver_backend_protocol_and_registry() -> None:
    backend = resolve_backend("scikit-fem")
    assert isinstance(backend, ScikitFemBackend)
    assert isinstance(backend, SolverBackend)
    assert backend.backend_id == "scikit-fem-2d-v1"
    assert "scikit-fem" in available_backends()
    caps = backend.capabilities.document()
    assert caps["spatial_dimensions"] == [2]
    assert caps["supports_warm_start"] is True

    register_backend("dummy-test", _DummyBackend)
    try:
        assert "dummy-test" in available_backends()
        resolved = resolve_backend("dummy-test")
        assert resolved.backend_id == "dummy-test-v1"
        assert resolved.capabilities.supports_gpu is True
    finally:
        unregister_backend("dummy-test")

    register_backend("bad-test", lambda: object())  # type: ignore[arg-type, return-value]
    try:
        with pytest.raises(TypeError, match="SolverBackend"):
            resolve_backend("bad-test")
    finally:
        unregister_backend("bad-test")

    with pytest.raises(LookupError, match="unknown FEM solver backend"):
        resolve_backend("nonexistent-backend-xyz")


def test_execution_profile_preserves_legacy_cpu_hash_and_validates_cuda_tier() -> None:
    base = ExecutionProfile(
        python_version="3.13.0",
        library_versions={"numpy": "2.0.0", "torch": "2.6.0"},
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="torch-manual-seed",
        seed=0,
    )
    raw = canonical_json_bytes(base)
    assert b"replay_tier" not in raw
    assert b"gpu_fingerprint" not in raw

    with pytest.raises(ContractValidationError, match="replay_tier"):
        ExecutionProfile(
            python_version="3.13.0",
            library_versions={"numpy": "2.0.0"},
            blas_threads=1,
            device="cuda:0",
            dtype="float64",
            rng_algorithm="torch-manual-seed",
            seed=0,
            replay_tier="invalid-tier",
        )

    gpu_profile_missing_tier = ExecutionProfile(
        python_version="3.13.0",
        library_versions={
            "numpy": "2.0.0",
            "scikit-fem": "11.0.0",
            "scipy": "1.14.0",
            "sympy": "1.13.0",
            "torch": "2.6.0",
            "safetensors": "0.8.0",
            "threadpoolctl": "3.5.0",
        },
        blas_threads=1,
        device="cuda:0",
        dtype="float64",
        rng_algorithm="torch-manual-seed",
        seed=0,
    )
    observed = observe_runtime(tuple(gpu_profile_missing_tier.library_versions))
    mismatches = profile_mismatches(gpu_profile_missing_tier, observed)
    assert any("replay_tier" in msg for msg in mismatches)
