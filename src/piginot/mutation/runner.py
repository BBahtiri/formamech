"""Run every detector against the unmutated code and under each mutant.

A detector is the executable procedure of one gate check on a fixed,
CI-sized fixture. It *fires* when its status is not ``PASS``: a failed
threshold (``FAIL``) or an exception (``ERROR``). The unmutated control must
pass every detector before and after the mutants run, so a fired detector is
caused by the mutant and the mutants leave nothing behind.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from piginot.mutation.engine import Mutant, applied, clear_caches
from piginot.verification.status import CheckStatus


@dataclass(frozen=True, slots=True)
class DetectorOutcome:
    """What a detector measured and whether its threshold holds."""

    passed: bool
    measured: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Detector:
    """One gate check, run as a function of the code currently loaded."""

    check_id: str
    description: str
    run: Callable[[], DetectorOutcome] = field(repr=False)


@dataclass(frozen=True, slots=True)
class DetectorResult:
    check_id: str
    status: CheckStatus
    seconds: float
    measured: Mapping[str, Any]
    error: str | None = None

    @property
    def fired(self) -> bool:
        return self.status is not CheckStatus.PASS

    def document(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status.value,
            "seconds": round(self.seconds, 3),
            "measured": dict(self.measured),
            "error": self.error,
        }


def run_detector(detector: Detector) -> DetectorResult:
    start = time.perf_counter()
    try:
        outcome = detector.run()
    except Exception as exc:  # a crash under a mutant is a detection
        return DetectorResult(
            detector.check_id,
            CheckStatus.ERROR,
            time.perf_counter() - start,
            {},
            "".join(traceback.format_exception_only(type(exc), exc)).strip()[:500],
        )
    return DetectorResult(
        detector.check_id,
        CheckStatus.PASS if outcome.passed else CheckStatus.FAIL,
        time.perf_counter() - start,
        MappingProxyType(dict(outcome.measured)),
    )


@dataclass(frozen=True, slots=True)
class MutantRun:
    mutant: Mutant
    results: tuple[DetectorResult, ...]
    seconds: float

    @property
    def fired(self) -> tuple[str, ...]:
        return tuple(sorted(r.check_id for r in self.results if r.fired))

    @property
    def primary_fired(self) -> bool:
        return self.mutant.primary in self.fired

    def document(self) -> dict[str, Any]:
        return {
            **self.mutant.document(),
            "fired": list(self.fired),
            "primary_fired": self.primary_fired,
            "seconds": round(self.seconds, 3),
            "results": [result.document() for result in self.results],
        }


@dataclass(frozen=True, slots=True)
class SuiteReport:
    detectors: tuple[str, ...]
    control_before: tuple[DetectorResult, ...]
    runs: tuple[MutantRun, ...]
    control_after: tuple[DetectorResult, ...]
    seconds: float

    @property
    def control_passed(self) -> bool:
        return not any(
            result.fired for result in (*self.control_before, *self.control_after)
        )

    @property
    def primary_count(self) -> int:
        return sum(run.primary_fired for run in self.runs)

    def document(self) -> dict[str, Any]:
        return {
            "detectors": list(self.detectors),
            "control_passed": self.control_passed,
            "control_before": [r.document() for r in self.control_before],
            "control_after": [r.document() for r in self.control_after],
            "mutants": [run.document() for run in self.runs],
            "primary_fired": self.primary_count,
            "mutant_count": len(self.runs),
            "seconds": round(self.seconds, 3),
        }


def _run_all(
    detectors: Sequence[Detector], caches: Sequence[Callable[..., Any]]
) -> tuple[DetectorResult, ...]:
    clear_caches(caches)
    return tuple(run_detector(detector) for detector in detectors)


def run_suite(
    mutants: Sequence[Mutant],
    detectors: Sequence[Detector],
    *,
    caches: Sequence[Callable[..., Any]] = (),
    progress: Callable[[str], None] = lambda message: None,
) -> SuiteReport:
    """Control, every mutant, control again; each run starts with clear caches."""

    identifiers = tuple(detector.check_id for detector in detectors)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("detector check ids must be unique")
    names = [mutant.name for mutant in mutants]
    if len(set(names)) != len(names):
        raise ValueError("mutant names must be unique")
    unknown = sorted({m.primary for m in mutants} - set(identifiers))
    if unknown:
        raise ValueError(f"primary gates without a detector: {unknown}")
    start = time.perf_counter()
    control_before = _run_all(detectors, caches)
    progress(f"control: {sum(not r.fired for r in control_before)}/{len(detectors)}")
    runs: list[MutantRun] = []
    for mutant in mutants:
        began = time.perf_counter()
        with applied(mutant):
            results = _run_all(detectors, caches)
        run = MutantRun(mutant, results, time.perf_counter() - began)
        runs.append(run)
        progress(
            f"{mutant.name}: primary {'fired' if run.primary_fired else 'MISSED'}; "
            f"fired {len(run.fired)}"
        )
    control_after = _run_all(detectors, caches)
    return SuiteReport(
        identifiers,
        control_before,
        tuple(runs),
        control_after,
        time.perf_counter() - start,
    )


__all__ = (
    "Detector",
    "DetectorOutcome",
    "DetectorResult",
    "MutantRun",
    "SuiteReport",
    "run_detector",
    "run_suite",
)
