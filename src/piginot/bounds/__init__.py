"""Calibrated error bounds and the calibration envelope gate (S23, ADR 0104).

NumPy only: fast serving with ``formamech[runtime]`` can attach bounds without
the FEM stack.
"""

from piginot.bounds.cad_calibration import (
    CAD_CALIBRATION_MACHINE,
    CAD_CALIBRATION_SCHEMA,
    CadCalibrationGateState,
    CadCalibrationRecord,
    FunctionalBandCalibration,
    ScalarQoiCalibration,
    build_cad_calibration,
    calibrate_functional_band,
)
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
    "CAD_CALIBRATION_MACHINE",
    "CAD_CALIBRATION_SCHEMA",
    "IN_CALIBRATION",
    "OUT_OF_CALIBRATION",
    "CadCalibrationGateState",
    "CadCalibrationRecord",
    "CalibrationError",
    "CalibrationRecord",
    "ConfirmedCalibration",
    "ConformalBound",
    "FunctionalBandCalibration",
    "GateError",
    "H1Confirmation",
    "MahalanobisGate",
    "NoFiniteBound",
    "ScalarQoiCalibration",
    "acceptance_count",
    "beta_binomial_pmf",
    "build_cad_calibration",
    "calibrate_functional_band",
    "calibration_path",
    "case_ids_sha256",
    "confirmation_path",
    "conformal_rank",
    "split_conformal_bound",
)
