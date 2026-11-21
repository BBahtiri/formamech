"""Amortized break-even query analysis for surrogate-vs-FEM cost disclosure."""

from __future__ import annotations

import math
from dataclasses import dataclass

from piginot.verification.latency import speedup


@dataclass(frozen=True, slots=True)
class AmortizationSummary:
    """Amortized break-even cost accounting for a surrogate campaign.

    Computes ``N* = (T_train + N_cal * T_FEM) / (T_FEM - T_infer)``. When legacy
    artifacts did not record ``training_wall_seconds``, ``break_even_queries``
    stays ``None`` and ``calibration_only_break_even_queries`` reports the exact
    lower bound from calibration FEM solves alone.
    """

    calibration_cases: int
    fem_seconds: float
    inference_seconds: float
    training_wall_seconds: float | None
    calibration_only_break_even_queries: float
    break_even_queries: float | None
    online_speedup: float
    threads: int = 1
    precision: str = "float64"
    device: str = "cpu"
    dof_count: int | None = None

    def document(self) -> dict[str, object]:
        return {
            "calibration_cases": self.calibration_cases,
            "fem_seconds": self.fem_seconds,
            "inference_seconds": self.inference_seconds,
            "training_wall_seconds": self.training_wall_seconds,
            "calibration_only_break_even_queries": (
                self.calibration_only_break_even_queries
            ),
            "break_even_queries": self.break_even_queries,
            "online_speedup": self.online_speedup,
            "threads": self.threads,
            "precision": self.precision,
            "device": self.device,
            "dof_count": self.dof_count,
        }


def amortized_break_even_queries(
    *,
    training_wall_seconds: float | None,
    calibration_cases: int,
    fem_seconds: float,
    inference_seconds: float,
    threads: int = 1,
    precision: str = "float64",
    device: str = "cpu",
    dof_count: int | None = None,
) -> AmortizationSummary:
    """Compute amortized break-even query count ``N*`` with explicit hardware context."""

    if isinstance(calibration_cases, bool) or not isinstance(calibration_cases, int):
        raise TypeError("calibration_cases must be an integer")
    if calibration_cases < 0:
        raise ValueError("calibration_cases must be non-negative")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if dof_count is not None and (
        isinstance(dof_count, bool) or not isinstance(dof_count, int) or dof_count < 1
    ):
        raise ValueError("dof_count must be a positive integer when provided")
    for label, value in (
        ("fem_seconds", fem_seconds),
        ("inference_seconds", inference_seconds),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
    delta = fem_seconds - inference_seconds
    if delta <= 0.0:
        raise ValueError(
            "fem_seconds must strictly exceed inference_seconds to achieve break-even"
        )
    if training_wall_seconds is not None and (
        isinstance(training_wall_seconds, bool)
        or not math.isfinite(training_wall_seconds)
        or training_wall_seconds < 0.0
    ):
        raise ValueError("training_wall_seconds must be finite and non-negative")

    cal_only = (calibration_cases * fem_seconds) / delta
    full_n_star = (
        None
        if training_wall_seconds is None
        else (training_wall_seconds + calibration_cases * fem_seconds) / delta
    )
    return AmortizationSummary(
        calibration_cases=calibration_cases,
        fem_seconds=float(fem_seconds),
        inference_seconds=float(inference_seconds),
        training_wall_seconds=(
            None if training_wall_seconds is None else float(training_wall_seconds)
        ),
        calibration_only_break_even_queries=float(cal_only),
        break_even_queries=None if full_n_star is None else float(full_n_star),
        online_speedup=speedup(fem_seconds, inference_seconds),
        threads=threads,
        precision=precision,
        device=device,
        dof_count=dof_count,
    )


__all__ = (
    "AmortizationSummary",
    "amortized_break_even_queries",
)
