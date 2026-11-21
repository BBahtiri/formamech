"""Observed runtime profiles and scoped one-thread numerical execution."""

from __future__ import annotations

import platform
from importlib import import_module, metadata
from typing import Any

import torch

from piginot.compiler.config_types import NUMERICAL_LIBRARIES
from piginot.compiler.execution_records import Document
from piginot.contracts import ExecutionProfile, hash_spec
from piginot.training.platform_fingerprint import platform_fingerprint


def threadpool_info() -> Any:
    return import_module("threadpoolctl").threadpool_info()


def threadpool_limits(*, limits: int) -> Any:
    return import_module("threadpoolctl").threadpool_limits(limits=limits)


def load_blas() -> None:
    """Load both declared numerical BLAS providers before limiting them."""
    import numpy  # noqa: F401
    import scipy.linalg  # type: ignore[import-untyped]  # noqa: F401


def observe_runtime(versions: tuple[str, ...]) -> Document:
    observed = {}
    for name in versions:
        try:
            observed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            observed[name] = "UNAVAILABLE"
    return {
        "python_version": platform.python_version(),
        "library_versions": observed,
        "platform_fingerprint": platform_fingerprint(),
        "torch_threads": torch.get_num_threads(),
        "threadpools": threadpool_info(),
    }


def profile_mismatches(profile: ExecutionProfile, observed: Document) -> list[str]:
    failures = []
    if not NUMERICAL_LIBRARIES <= profile.library_versions.keys():
        failures.append("execution profile omits a required numerical library version")
    for name in ("python_version", "platform_fingerprint"):
        expected = getattr(profile, name)
        if name == "platform_fingerprint" and expected is not None:
            expected = dict(expected)
        if expected != observed[name]:
            failures.append(f"{name} differs from the declared execution profile")
    for name, version in profile.library_versions.items():
        if observed["library_versions"].get(name) != version:
            failures.append(f"library {name} differs from the declared version")
    if profile.device.startswith("cuda"):
        if profile.replay_tier != "gpu-accelerated-tolerance":
            failures.append(
                "CUDA execution profiles must declare replay_tier="
                "'gpu-accelerated-tolerance'"
            )
        if not torch.cuda.is_available():
            failures.append("CUDA execution requested but torch.cuda is unavailable")
        elif profile.gpu_fingerprint is not None:
            observed_gpu = {
                "device_name": str(torch.cuda.get_device_name(0)),
                "cuda_version": str(torch.version.cuda or "unknown"),
                "cublas_workspace": str(
                    profile.gpu_fingerprint.get("cublas_workspace", "")
                ),
            }
            if observed_gpu != dict(profile.gpu_fingerprint):
                failures.append("gpu_fingerprint differs from the observed CUDA device")
    elif profile.device != "cpu" or profile.blas_threads != 1:
        failures.append("only CPU execution with one BLAS thread is supported")
    if profile.rng_algorithm != "torch-manual-seed":
        failures.append("unsupported RNG algorithm")
    if profile.container_digest is not None:
        failures.append(
            "container identity lacks an independently attested runtime digest"
        )
    if observed["torch_threads"] != 1 or any(
        pool["num_threads"] != 1 for pool in observed["threadpools"]
    ):
        failures.append("observed numerical thread counts differ from one")
    return failures


class RuntimeGuard:
    def __init__(self) -> None:
        self.limiter: Any = None
        self.previous_threads: int | None = None

    def probe(
        self, training: ExecutionProfile, reference: ExecutionProfile
    ) -> Document:
        load_blas()
        self.previous_threads = torch.get_num_threads()
        self.limiter = threadpool_limits(limits=1)
        torch.set_num_threads(1)
        libraries = tuple(
            sorted(set(training.library_versions) | set(reference.library_versions))
        )
        observed = observe_runtime(libraries)
        failures = {
            "training": profile_mismatches(training, observed),
            "reference": profile_mismatches(reference, observed),
        }
        return {
            "status": "FAIL" if any(failures.values()) else "PASS",
            "profile_sha256": {
                "training": hash_spec(training, "content"),
                "reference": hash_spec(reference, "content"),
            },
            "observed": observed,
            "failures": failures,
        }

    def close(self) -> None:
        if self.limiter is not None:
            self.limiter.restore_original_limits()
            self.limiter = None
        if self.previous_threads is not None:
            torch.set_num_threads(self.previous_threads)
            self.previous_threads = None


__all__ = ("RuntimeGuard", "profile_mismatches")
