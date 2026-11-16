"""Calibrated error bounds and the calibration envelope gate (S23, ADR 0104).

NumPy only: fast serving with ``formamech[runtime]`` can attach bounds without
the FEM stack.
"""

from piginot.bounds.confirmation import (
    ConfirmedCalibration,
    H1Confirmation,
    confirmation_path,
)
from piginot.bounds.conformal import (
    ConformalBound,
    NoFiniteBound,
    acceptance_count,
    beta_binomial_pmf,
    conformal_rank,
    split_conformal_bound,
)
from piginot.bounds.gate import GateError, MahalanobisGate
from piginot.bounds.record import (
    BOUND_KINDS,
    IN_CALIBRATION,
    OUT_OF_CALIBRATION,
    CalibrationError,
    CalibrationRecord,
    calibration_path,
    case_ids_sha256,
)

__all__ = (
    "BOUND_KINDS",
    "IN_CALIBRATION",
    "OUT_OF_CALIBRATION",
    "CalibrationError",
    "CalibrationRecord",
    "ConfirmedCalibration",
    "ConformalBound",
    "GateError",
    "H1Confirmation",
    "MahalanobisGate",
    "NoFiniteBound",
    "acceptance_count",
    "beta_binomial_pmf",
    "calibration_path",
    "case_ids_sha256",
    "confirmation_path",
    "conformal_rank",
    "split_conformal_bound",
)
