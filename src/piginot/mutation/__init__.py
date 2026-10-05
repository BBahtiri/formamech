"""Source-level mutation testing of the gate checks (S16)."""

from piginot.mutation.engine import (
    Mutant,
    MutationError,
    SourceEdit,
    applied,
    clear_caches,
    mutated_code,
    resolve_function,
)
from piginot.mutation.runner import (
    Detector,
    DetectorOutcome,
    DetectorResult,
    MutantRun,
    SuiteReport,
    run_detector,
    run_suite,
)
from piginot.mutation.signatures import (
    EXIT_CODES,
    SCHEMA,
    Comparison,
    ComparisonStatus,
    MutantComparison,
    compare_signatures,
    signature_document,
)

__all__ = (
    "EXIT_CODES",
    "SCHEMA",
    "Comparison",
    "ComparisonStatus",
    "Detector",
    "DetectorOutcome",
    "DetectorResult",
    "Mutant",
    "MutantComparison",
    "MutantRun",
    "MutationError",
    "SourceEdit",
    "SuiteReport",
    "applied",
    "clear_caches",
    "compare_signatures",
    "mutated_code",
    "resolve_function",
    "run_detector",
    "run_suite",
    "signature_document",
)
