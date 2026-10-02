"""Single-thread inference latency (``G14.latency``, ``G-LATENCY``)."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from piginot.verification.budget import linear_quantile


@dataclass(frozen=True, slots=True)
class LatencyResult:
    """Median and 99th-percentile wall time of repeated calls, in seconds."""

    repeats: int
    median_seconds: float
    p99_seconds: float
    threads: int


def measure_latency(
    call: Callable[[], object],
    *,
    repeats: int = 50,
    warmup: int = 3,
    threads: int = 1,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> LatencyResult:
    """Time ``call`` with ``threads`` intra-op threads; restore the setting after."""

    if not callable(call):
        raise TypeError("call must be callable")
    if repeats < 1 or warmup < 0 or threads < 1:
        raise ValueError("repeats and threads must be positive, warmup non-negative")
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        for _ in range(warmup):
            call()
        samples: list[float] = []
        for _ in range(repeats):
            start = clock()
            call()
            samples.append((clock() - start) * 1e-9)
    finally:
        torch.set_num_threads(previous)
    return LatencyResult(
        repeats=repeats,
        median_seconds=linear_quantile(tuple(samples), 0.5),
        p99_seconds=linear_quantile(tuple(samples), 0.99),
        threads=threads,
    )


def speedup(reference_seconds: float, candidate_seconds: float) -> float:
    """Reference time divided by candidate time."""

    for label, value in (
        ("reference_seconds", reference_seconds),
        ("candidate_seconds", candidate_seconds),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
    return reference_seconds / candidate_seconds


__all__ = ("LatencyResult", "measure_latency", "speedup")
