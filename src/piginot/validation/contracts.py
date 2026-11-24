"""Zero-heavy-dependency ASME V&V 10 experimental validation contracts (V3-3).

Strictly orthogonal to the byte-frozen S14 numerical verification ``ClaimLevel``
in ``piginot.verification.status``: numerical verification (agreement with FEM)
is never conflated with experimental validation (agreement with physical tests).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from piginot.contracts.canonical import canonical_json_bytes
from piginot.contracts.state_machine import (
    StateMachineSpec,
    StateSnapshot,
    TransitionSpec,
    register_state_machine,
)

EXPERIMENTAL_DATASET_SCHEMA: Final = "formamech.experimental-dataset-v1"
VV10_VALIDATION_DOSSIER_SCHEMA: Final = "formamech.vv10-validation-dossier-v1"


class ExperimentalValidationTier(StrEnum):
    """ASME V&V 10 experimental validation tier (orthogonal to ``ClaimLevel``)."""

    UNVALIDATED = "UNVALIDATED"
    COUPON_CALIBRATED = "COUPON_CALIBRATED"
    HOLDOUT_EXPERIMENTALLY_VALIDATED = "HOLDOUT_EXPERIMENTALLY_VALIDATED"


class ValidationLifecycleState(StrEnum):
    """Explicit lifecycle state machine for physical coupon calibration & V&V 10 validation."""

    INGESTED = "INGESTED"
    CALIBRATED = "CALIBRATED"
    HOLDOUT_EVALUATED = "HOLDOUT_EVALUATED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"


VALIDATION_LIFECYCLE_MACHINE: Final[StateMachineSpec[ValidationLifecycleState]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.validation.vv10",
            description=(
                "ASME V&V 10 physical validation lifecycle separating calibration "
                "coupons from blind holdout validation coupons."
            ),
            states=tuple(ValidationLifecycleState),
            initial_states=(ValidationLifecycleState.INGESTED,),
            terminal_states=(
                ValidationLifecycleState.VALIDATED,
                ValidationLifecycleState.REJECTED,
            ),
            transitions=(
                TransitionSpec(
                    ValidationLifecycleState.INGESTED,
                    ValidationLifecycleState.CALIBRATED,
                    "calibrate_material",
                    "Inverse material identification converged on CALIBRATION coupons.",
                ),
                TransitionSpec(
                    ValidationLifecycleState.INGESTED,
                    ValidationLifecycleState.REJECTED,
                    "reject_leakage_or_invalid",
                    "Refused coupon overlap, missing uncertainties, or invalid fit.",
                ),
                TransitionSpec(
                    ValidationLifecycleState.CALIBRATED,
                    ValidationLifecycleState.HOLDOUT_EVALUATED,
                    "evaluate_holdout",
                    "Evaluated calibrated model on unseen VALIDATION_HOLDOUT coupons.",
                ),
                TransitionSpec(
                    ValidationLifecycleState.CALIBRATED,
                    ValidationLifecycleState.REJECTED,
                    "reject_no_holdout",
                    "Validation cannot complete without unseen holdout coupons.",
                ),
                TransitionSpec(
                    ValidationLifecycleState.HOLDOUT_EVALUATED,
                    ValidationLifecycleState.VALIDATED,
                    "accept_vv10",
                    "Holdout comparison error E satisfied ASME V&V 10 budget.",
                ),
                TransitionSpec(
                    ValidationLifecycleState.HOLDOUT_EVALUATED,
                    ValidationLifecycleState.REJECTED,
                    "reject_vv10",
                    "Holdout comparison error E exceeded validation threshold.",
                ),
            ),
        )
    )
)


class CouponRole(StrEnum):
    """Strict partition role preventing calibration/validation coupon leakage."""

    CALIBRATION = "CALIBRATION"
    VALIDATION_HOLDOUT = "VALIDATION_HOLDOUT"


class ExperimentalTestMode(StrEnum):
    """Supported physical mechanical testing protocols."""

    UNIAXIAL_TENSION = "uniaxial_tension"
    EQUIBIAXIAL_TENSION = "equibiaxial_tension"
    SIMPLE_SHEAR = "simple_shear"
    UNIAXIAL_CYCLIC_J2 = "uniaxial_cyclic_j2"


@dataclass(frozen=True, slots=True)
class ExperimentalCouponDataset:
    """Canonical physical coupon stress-stretch/strain dataset with measurement uncertainty."""

    specimen_id: str
    role: CouponRole
    test_mode: ExperimentalTestMode
    stretches_or_strains: tuple[float, ...]
    measured_stress_mpa: tuple[float, ...]
    measurement_std_mpa: float
    metadata: Mapping[str, str] = ()  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.specimen_id, str) or not self.specimen_id.strip():
            raise ValueError("specimen_id must be non-empty text")
        object.__setattr__(self, "role", CouponRole(self.role))
        object.__setattr__(self, "test_mode", ExperimentalTestMode(self.test_mode))
        if not self.stretches_or_strains or len(self.stretches_or_strains) != len(
            self.measured_stress_mpa
        ):
            raise ValueError(
                "stretches_or_strains and measured_stress_mpa must have equal non-empty length"
            )
        if any(not math.isfinite(float(v)) for v in self.stretches_or_strains):
            raise ValueError("stretches_or_strains must be finite")
        if any(not math.isfinite(float(v)) for v in self.measured_stress_mpa):
            raise ValueError("measured_stress_mpa must be finite")
        if (
            not math.isfinite(self.measurement_std_mpa)
            or self.measurement_std_mpa <= 0.0
        ):
            raise ValueError("measurement_std_mpa (u_exp) must be finite and positive")
        meta = (
            {str(k): str(v) for k, v in self.metadata.items()}
            if isinstance(self.metadata, Mapping)
            else {}
        )
        object.__setattr__(self, "metadata", meta)

    @property
    def dataset_hash(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()

    def document(self) -> dict[str, Any]:
        return {
            "schema": EXPERIMENTAL_DATASET_SCHEMA,
            "specimen_id": self.specimen_id,
            "role": self.role.value,
            "test_mode": self.test_mode.value,
            "stretches_or_strains": [float(v) for v in self.stretches_or_strains],
            "measured_stress_mpa": [float(v) for v in self.measured_stress_mpa],
            "measurement_std_mpa": float(self.measurement_std_mpa),
            "metadata": dict(sorted(self.metadata.items())),
        }


@dataclass(frozen=True, slots=True)
class CalibratedParameterEstimate:
    """One identified constitutive parameter with Laplace/Gauss-Newton uncertainty."""

    name: str
    value: float
    std_error: float
    unit: str

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": float(self.value),
            "std_error": float(self.std_error),
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class ExperimentalValidationDossier:
    """ASME V&V 10 physical validation dossier keeping numerical & experimental claims distinct."""

    model_family: str
    numerical_claim_level: str
    experimental_validation_tier: ExperimentalValidationTier
    state: ValidationLifecycleState
    calibrated_parameters: tuple[CalibratedParameterEstimate, ...]
    calibration_rmse_mpa: float
    holdout_comparison_error_e_mpa: float | None
    u_num_mpa: float
    u_input_mpa: float
    u_exp_mpa: float
    u_val_mpa: float
    validation_ratio: float | None  # E / (k * u_val)
    passed: bool
    calibration_specimen_ids: tuple[str, ...]
    holdout_specimen_ids: tuple[str, ...]
    reason: str

    def snapshot(self) -> StateSnapshot[ValidationLifecycleState]:
        return VALIDATION_LIFECYCLE_MACHINE.snapshot(
            self.state,
            blockers=() if self.passed else (self.reason,),
            metadata={
                "experimental_validation_tier": self.experimental_validation_tier.value,
                "numerical_claim_level": self.numerical_claim_level,
            },
        )

    @property
    def dossier_hash(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()

    def document(self) -> dict[str, Any]:
        return {
            "schema": VV10_VALIDATION_DOSSIER_SCHEMA,
            "model_family": self.model_family,
            "numerical_claim_level": self.numerical_claim_level,
            "experimental_validation_tier": self.experimental_validation_tier.value,
            "state": self.state.value,
            "calibrated_parameters": [p.document() for p in self.calibrated_parameters],
            "calibration_rmse_mpa": float(self.calibration_rmse_mpa),
            "holdout_comparison_error_e_mpa": (
                None
                if self.holdout_comparison_error_e_mpa is None
                else float(self.holdout_comparison_error_e_mpa)
            ),
            "u_num_mpa": float(self.u_num_mpa),
            "u_input_mpa": float(self.u_input_mpa),
            "u_exp_mpa": float(self.u_exp_mpa),
            "u_val_mpa": float(self.u_val_mpa),
            "validation_ratio": (
                None if self.validation_ratio is None else float(self.validation_ratio)
            ),
            "passed": self.passed,
            "calibration_specimen_ids": list(self.calibration_specimen_ids),
            "holdout_specimen_ids": list(self.holdout_specimen_ids),
            "reason": self.reason,
        }


__all__ = (
    "EXPERIMENTAL_DATASET_SCHEMA",
    "VALIDATION_LIFECYCLE_MACHINE",
    "VV10_VALIDATION_DOSSIER_SCHEMA",
    "CalibratedParameterEstimate",
    "CouponRole",
    "ExperimentalCouponDataset",
    "ExperimentalTestMode",
    "ExperimentalValidationDossier",
    "ExperimentalValidationTier",
    "ValidationLifecycleState",
)
