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
from piginot.bounds.conformal_3d import (
    AS6983_ASSESSMENT_SCHEMA,
    CONFORMAL_3D_SCHEMA,
    As6983ConservativeBound,
    As6983HoldoutCoverageReport,
    Conformal3DAssessment,
    Conformal3DCalibrationRecord,
    Conformal3DCalibrator,
    Conformal3DDevSample,
    Conformal3DQoiCalibration,
    PhysicalUnitErrorSummary,
    summarize_physical_errors,
    verify_as6983_holdout_coverage,
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
    "AS6983_ASSESSMENT_SCHEMA",
    "BOUND_KINDS",
    "CAD_CALIBRATION_MACHINE",
    "CAD_CALIBRATION_SCHEMA",
    "CONFORMAL_3D_SCHEMA",
    "IN_CALIBRATION",
    "OUT_OF_CALIBRATION",
    "As6983ConservativeBound",
    "As6983HoldoutCoverageReport",
    "CadCalibrationGateState",
    "CadCalibrationRecord",
    "CalibrationError",
    "CalibrationRecord",
    "ConfirmedCalibration",
    "Conformal3DAssessment",
    "Conformal3DCalibrationRecord",
    "Conformal3DCalibrator",
    "Conformal3DDevSample",
    "Conformal3DQoiCalibration",
    "ConformalBound",
    "FunctionalBandCalibration",
    "GateError",
    "H1Confirmation",
    "MahalanobisGate",
    "NoFiniteBound",
    "PhysicalUnitErrorSummary",
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
    "summarize_physical_errors",
    "verify_as6983_holdout_coverage",
)
