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
from piginot.fem.manufactured import MANUFACTURED_CASE_NAMES
from piginot.fem.mms_convergence import run_all_mms_convergence_studies
from piginot.fem.mms_runner import MmsLevelResult, solve_mms_level
from piginot.fem.verification_meshes import VerificationMeshKind

_CALIBRATION_MARGIN = 1.25


@dataclass(frozen=True, slots=True)
class EstimatorFamilyCalibration:
    """One estimator's factor plus results excluded from its selection."""

    safety_factor: float
    calibration: EstimatorCalibration
    validation: EstimatorCalibration


@dataclass(frozen=True, slots=True)
class MmsEstimatorCalibration:
    """Richardson displacement and recovery-stress calibration evidence."""

    richardson: EstimatorFamilyCalibration
    recovery: EstimatorFamilyCalibration
    calibration_mesh_kind: VerificationMeshKind
    validation_mesh_kinds: tuple[VerificationMeshKind, ...]
    validation_mesh_population_sizes: tuple[int, ...]

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
    """Calibrate on coarse data and evaluate on factor-selection holdouts."""

    richardson_calibration_estimates: list[float] = []
    richardson_calibration_truths: list[float] = []
    richardson_validation_estimates: list[float] = []
    richardson_validation_truths: list[float] = []
    recovery_calibration_estimates: list[float] = []
    recovery_calibration_truths: list[float] = []
    recovery_validation_estimates: list[float] = []
    recovery_validation_truths: list[float] = []

    for study in run_all_mms_convergence_studies():
        for pair_index in (0, 1, 2):
            coarse = study.levels[pair_index]
            fine = study.levels[pair_index + 1]
            estimate, truth = _richardson_sample(coarse, fine)
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

    (
        unstructured_richardson_estimates,
        unstructured_richardson_truths,
        unstructured_recovery_estimates,
        unstructured_recovery_truths,
    ) = _unstructured_validation_population()
    structured_validation_sizes = (
        len(richardson_validation_estimates),
        len(recovery_validation_estimates),
    )
    unstructured_validation_sizes = (
        len(unstructured_richardson_estimates),
        len(unstructured_recovery_estimates),
    )
    if (
        len(set(structured_validation_sizes)) != 1
        or len(set(unstructured_validation_sizes)) != 1
    ):
        raise RuntimeError("estimator families must use balanced mesh populations")
    richardson_validation_estimates.extend(unstructured_richardson_estimates)
    richardson_validation_truths.extend(unstructured_richardson_truths)
    recovery_validation_estimates.extend(unstructured_recovery_estimates)
    recovery_validation_truths.extend(unstructured_recovery_truths)

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
    return MmsEstimatorCalibration(
        richardson=richardson,
        recovery=recovery,
        calibration_mesh_kind=VerificationMeshKind.STRUCTURED,
        validation_mesh_kinds=(
            VerificationMeshKind.STRUCTURED,
            VerificationMeshKind.UNSTRUCTURED,
        ),
        validation_mesh_population_sizes=(
            structured_validation_sizes[0],
            unstructured_validation_sizes[0],
        ),
    )


def _unstructured_validation_population() -> tuple[
    tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]
]:
    richardson_estimates: list[float] = []
    richardson_truths: list[float] = []
    recovery_estimates: list[float] = []
    recovery_truths: list[float] = []
    for case_name in MANUFACTURED_CASE_NAMES:
        for element_order in (1, 2):
            coarse = solve_mms_level(
                case_name,
                level=3,
                element_order=element_order,
                mesh_kind=VerificationMeshKind.UNSTRUCTURED,
            )
            fine = solve_mms_level(
                case_name,
                level=4,
                element_order=element_order,
                mesh_kind=VerificationMeshKind.UNSTRUCTURED,
            )
            estimate, truth = _richardson_sample(coarse, fine)
            richardson_estimates.append(estimate)
            richardson_truths.append(truth)
            recovery_estimates.append(fine.norms.recovery_estimator_l2)
            recovery_truths.append(fine.norms.first_piola_l2_error)
    return (
        tuple(richardson_estimates),
        tuple(richardson_truths),
        tuple(recovery_estimates),
        tuple(recovery_truths),
    )


def _richardson_sample(
    coarse: MmsLevelResult,
    fine: MmsLevelResult,
) -> tuple[float, float]:
    difference = coarse_fine_displacement_l2_difference(coarse, fine)
    ratio = coarse.h_max_mm / fine.h_max_mm
    estimate = richardson_fine_grid_error(
        difference,
        ratio,
        theoretical_order=float(coarse.element_order + 1),
    ).fine_grid_error
    return estimate, fine.norms.displacement_l2_error


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
