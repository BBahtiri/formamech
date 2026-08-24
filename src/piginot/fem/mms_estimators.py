"""Held-out calibration of S8 discretization-error estimators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from skfem import (  # type: ignore[import-untyped]
    Basis,
    ElementTriP1,
    ElementTriP2,
    MeshTri,
)

from piginot.fem.error_estimation import (
    EstimatorCalibration,
    calibrate_error_estimator,
    richardson_fine_grid_error,
)
from piginot.fem.field_providers import FloatArray
from piginot.fem.mms_convergence import run_all_mms_convergence_studies
from piginot.fem.mms_runner import MmsLevelResult

_CALIBRATION_MARGIN = 1.25


@dataclass(frozen=True, slots=True)
class EstimatorFamilyCalibration:
    """One estimator's declared factor plus calibration and held-out results."""

    safety_factor: float
    calibration: EstimatorCalibration
    validation: EstimatorCalibration


@dataclass(frozen=True, slots=True)
class MmsEstimatorCalibration:
    """Richardson displacement and recovery-stress calibration evidence."""

    richardson: EstimatorFamilyCalibration
    recovery: EstimatorFamilyCalibration

    @property
    def minimum_validation_bracketing_rate(self) -> float:
        return min(
            self.richardson.validation.bracketing_rate,
            self.recovery.validation.bracketing_rate,
        )

    @property
    def minimum_validation_conservative_rate(self) -> float:
        return min(
            self.richardson.validation.conservative_rate,
            self.recovery.validation.conservative_rate,
        )


def coarse_fine_displacement_l2_difference(
    coarse: MmsLevelResult,
    fine: MmsLevelResult,
) -> float:
    """Measure ``||u_h - u_h/2||_L2`` on the fine result quadrature."""

    if coarse.case_name != fine.case_name:
        raise ValueError("coarse and fine results must use the same MMS case")
    if coarse.element_order != fine.element_order:
        raise ValueError("coarse and fine results must use the same element order")
    if coarse.mesh_kind is not fine.mesh_kind:
        raise ValueError("coarse and fine results must use the same mesh family")
    if coarse.h_max_mm <= fine.h_max_mm:
        raise ValueError("coarse result must have a larger mesh size")

    fine_points = np.asarray(fine.result.quadrature_points, dtype=np.float64).reshape(
        -1, 2
    )
    weights = np.asarray(fine.result.quadrature_weights, dtype=np.float64).reshape(-1)
    coarse_values = _probe_displacement(coarse, fine_points)
    fine_values = _probe_displacement(fine, fine_points)
    squared = np.sum(np.square(coarse_values - fine_values), axis=1)
    value = float(np.sqrt(np.sum(weights * squared)))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "coarse/fine displacement difference must be finite and positive"
        )
    return value


@lru_cache(maxsize=1)
def calibrate_mms_estimators() -> MmsEstimatorCalibration:
    """Calibrate on coarse pairs/levels and evaluate only on held-out finest data."""

    richardson_calibration_estimates: list[float] = []
    richardson_calibration_truths: list[float] = []
    richardson_validation_estimates: list[float] = []
    richardson_validation_truths: list[float] = []
    recovery_calibration_estimates: list[float] = []
    recovery_calibration_truths: list[float] = []
    recovery_validation_estimates: list[float] = []
    recovery_validation_truths: list[float] = []

    for study in run_all_mms_convergence_studies():
        theoretical_order = float(study.element_order + 1)
        for pair_index in (0, 1, 2):
            coarse = study.levels[pair_index]
            fine = study.levels[pair_index + 1]
            difference = coarse_fine_displacement_l2_difference(coarse, fine)
            ratio = coarse.h_max_mm / fine.h_max_mm
            estimate = richardson_fine_grid_error(
                difference,
                ratio,
                theoretical_order=theoretical_order,
            ).fine_grid_error
            truth = fine.norms.displacement_l2_error
            if pair_index < 2:
                richardson_calibration_estimates.append(estimate)
                richardson_calibration_truths.append(truth)
            else:
                richardson_validation_estimates.append(estimate)
                richardson_validation_truths.append(truth)

        for level_index, level in enumerate(study.levels):
            estimate = level.norms.recovery_estimator_l2
            truth = level.norms.first_piola_l2_error
            if level_index < 3:
                recovery_calibration_estimates.append(estimate)
                recovery_calibration_truths.append(truth)
            else:
                recovery_validation_estimates.append(estimate)
                recovery_validation_truths.append(truth)

    richardson = _calibrate_family(
        richardson_calibration_estimates,
        richardson_calibration_truths,
        richardson_validation_estimates,
        richardson_validation_truths,
    )
    recovery = _calibrate_family(
        recovery_calibration_estimates,
        recovery_calibration_truths,
        recovery_validation_estimates,
        recovery_validation_truths,
    )
    return MmsEstimatorCalibration(richardson=richardson, recovery=recovery)


def _probe_displacement(
    level: MmsLevelResult,
    physical_points: FloatArray,
) -> FloatArray:
    request = level.request
    skmesh = MeshTri(
        np.asarray(request.mesh.nodes, dtype=np.float64).T,
        np.asarray(request.mesh.cells, dtype=np.int32).T,
        sort_t=False,
    )
    element = ElementTriP1() if level.element_order == 1 else ElementTriP2()
    basis = Basis(skmesh, element)
    values = np.asarray(level.result.displacement, dtype=np.float64)
    if values.shape != (basis.N, 2):
        raise ValueError("result displacement does not match the rebuilt scalar basis")
    probe = basis.probes(np.asarray(physical_points, dtype=np.float64).T)
    return np.column_stack((probe @ values[:, 0], probe @ values[:, 1]))


def _calibrate_family(
    calibration_estimates: list[float],
    calibration_truths: list[float],
    validation_estimates: list[float],
    validation_truths: list[float],
) -> EstimatorFamilyCalibration:
    ratios = tuple(
        max(estimate / truth, truth / estimate)
        for estimate, truth in zip(
            calibration_estimates,
            calibration_truths,
            strict=True,
        )
    )
    factor = math.ceil(max(ratios) * _CALIBRATION_MARGIN * 100.0) / 100.0
    calibration = calibrate_error_estimator(
        calibration_estimates,
        calibration_truths,
        safety_factor=factor,
    )
    validation = calibrate_error_estimator(
        validation_estimates,
        validation_truths,
        safety_factor=factor,
    )
    return EstimatorFamilyCalibration(
        safety_factor=factor,
        calibration=calibration,
        validation=validation,
    )


__all__ = (
    "EstimatorFamilyCalibration",
    "MmsEstimatorCalibration",
    "calibrate_mms_estimators",
    "coarse_fine_displacement_l2_difference",
)
