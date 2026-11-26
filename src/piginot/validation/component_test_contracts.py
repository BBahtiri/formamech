"""Zero-heavy-dependency component-level experimental grounding contracts (V5-2).

Extends FormaMech's ASME V&V 10 epistemic firewall from 1D analytical coupons
to full 2D/3D component-level sensor & field measurements (DIC displacement/strain,
rosette strain sensors, load-cell reactions, thermocouples, deflection profiles,
and run-to-failure component test matrices).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from piginot.contracts.canonical import canonical_json_bytes
from piginot.contracts.state_machine import (
    ALLOWED_CLAIM_LEVELS,
    StateMachineSpec,
    StateSnapshot,
    TransitionSpec,
    register_state_machine,
)
from piginot.validation.contracts import CouponRole, ExperimentalValidationTier

COMPONENT_EXPERIMENT_SCHEMA: Final = "formamech.component-experiment-v1"
HYBRID_GROUNDING_DOSSIER_SCHEMA: Final = "formamech.hybrid-grounding-dossier-v1"


class SensorObservationKind(StrEnum):
    """Supported physical laboratory & field metrology modalities."""

    DIC_DISPLACEMENT_FIELD = "dic_displacement_field"
    DIC_STRAIN_FIELD = "dic_strain_field"
    STRAIN_ROSETTE = "strain_rosette"
    LOAD_CELL_REACTION = "load_cell_reaction"
    THERMOCOUPLE_POINT = "thermocouple_point"
    DEFLECTION_PROFILE = "deflection_profile"
    DEGRADATION_METRIC = "degradation_metric"


class HybridGroundingState(StrEnum):
    """Explicit fail-closed lifecycle for hybrid sim + physical-test grounding."""

    INGESTED = "INGESTED"
    ALIGNED = "ALIGNED"
    GROUNDED = "GROUNDED"
    HOLDOUT_EVALUATED = "HOLDOUT_EVALUATED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"


HYBRID_GROUNDING_MACHINE: Final[StateMachineSpec[HybridGroundingState]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.validation.hybrid_grounding",
            description=(
                "Component-level hybrid simulation + physical-test grounding "
                "lifecycle enforcing blind physical holdout validation under ASME V&V 10."
            ),
            states=tuple(HybridGroundingState),
            initial_states=(HybridGroundingState.INGESTED,),
            terminal_states=(
                HybridGroundingState.VALIDATED,
                HybridGroundingState.REJECTED,
            ),
            transitions=(
                TransitionSpec(
                    HybridGroundingState.INGESTED,
                    HybridGroundingState.ALIGNED,
                    "align_sensors",
                    "Validated zero leakage and aligned sensor/prior channels.",
                ),
                TransitionSpec(
                    HybridGroundingState.INGESTED,
                    HybridGroundingState.REJECTED,
                    "reject_leakage_or_invalid",
                    "Detected calibration/holdout overlap or invalid physical records.",
                ),
                TransitionSpec(
                    HybridGroundingState.ALIGNED,
                    HybridGroundingState.GROUNDED,
                    "fit_discrepancy",
                    "Fitted physics-constrained multi-fidelity discrepancy model on CALIBRATION.",
                ),
                TransitionSpec(
                    HybridGroundingState.GROUNDED,
                    HybridGroundingState.HOLDOUT_EVALUATED,
                    "evaluate_holdout",
                    "Evaluated grounded hybrid model on blind VALIDATION_HOLDOUT specimens.",
                ),
                TransitionSpec(
                    HybridGroundingState.GROUNDED,
                    HybridGroundingState.REJECTED,
                    "reject_no_holdout",
                    "Cannot promote to experimental validation without blind holdouts.",
                ),
                TransitionSpec(
                    HybridGroundingState.HOLDOUT_EVALUATED,
                    HybridGroundingState.VALIDATED,
                    "accept_vv10_hybrid",
                    "Grounded holdout error E satisfied ASME V&V 10 uncertainty budget.",
                ),
                TransitionSpec(
                    HybridGroundingState.HOLDOUT_EVALUATED,
                    HybridGroundingState.REJECTED,
                    "reject_vv10_hybrid",
                    "Grounded holdout error E exceeded ASME V&V 10 threshold.",
                ),
            ),
        )
    )
)


@dataclass(frozen=True, slots=True)
class ComponentTestRecord:
    """Paired physical-test measurement & physics-simulation prior for one specimen."""

    specimen_id: str
    role: CouponRole
    sensor_kind: SensorObservationKind
    condition_vector: tuple[float, ...]
    observed_values: tuple[float, ...]
    simulated_prior_values: tuple[float, ...]
    measurement_std: float
    unit: str
    sensor_locations: tuple[tuple[float, ...], ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.specimen_id, str) or not self.specimen_id.strip():
            raise ValueError("specimen_id must be non-empty text")
        object.__setattr__(self, "role", CouponRole(self.role))
        object.__setattr__(self, "sensor_kind", SensorObservationKind(self.sensor_kind))
        if not self.condition_vector or any(
            not math.isfinite(float(v)) for v in self.condition_vector
        ):
            raise ValueError("condition_vector must be non-empty finite floats")
        if not self.observed_values or len(self.observed_values) != len(
            self.simulated_prior_values
        ):
            raise ValueError(
                "observed_values and simulated_prior_values must share equal non-empty length"
            )
        if any(not math.isfinite(float(v)) for v in self.observed_values):
            raise ValueError("observed_values must be finite")
        if any(not math.isfinite(float(v)) for v in self.simulated_prior_values):
            raise ValueError("simulated_prior_values must be finite")
        if not math.isfinite(self.measurement_std) or self.measurement_std <= 0.0:
            raise ValueError("measurement_std (u_exp) must be finite and positive")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("unit must be non-empty text")
        if self.sensor_locations:
            if len(self.sensor_locations) != len(self.observed_values):
                raise ValueError(
                    "sensor_locations length must match observed_values length"
                )
            loc_dim = len(self.sensor_locations[0])
            if loc_dim not in (1, 2, 3):
                raise ValueError("sensor_locations coordinates must be 1D, 2D, or 3D")
            for loc in self.sensor_locations:
                if len(loc) != loc_dim or any(not math.isfinite(float(v)) for v in loc):
                    raise ValueError(
                        "all sensor_locations must be finite with uniform dimension"
                    )
        meta = (
            {str(k): str(v) for k, v in self.metadata.items()}
            if isinstance(self.metadata, Mapping)
            else {}
        )
        object.__setattr__(self, "metadata", meta)

    @property
    def dataset_hash(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()

    @property
    def measurement_hash(self) -> str:
        """Role- and specimen-ID-agnostic hash preventing duplicate physical leakage."""
        payload = {
            "sensor_kind": self.sensor_kind.value,
            "condition_vector": [float(v) for v in self.condition_vector],
            "observed_values": [float(v) for v in self.observed_values],
            "measurement_std": float(self.measurement_std),
            "unit": self.unit,
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def document(self) -> dict[str, Any]:
        return {
            "schema": COMPONENT_EXPERIMENT_SCHEMA,
            "specimen_id": self.specimen_id,
            "role": self.role.value,
            "sensor_kind": self.sensor_kind.value,
            "condition_vector": [float(v) for v in self.condition_vector],
            "observed_values": [float(v) for v in self.observed_values],
            "simulated_prior_values": [float(v) for v in self.simulated_prior_values],
            "measurement_std": float(self.measurement_std),
            "unit": self.unit,
            "sensor_locations": [
                [float(x) for x in pt] for pt in self.sensor_locations
            ],
            "metadata": dict(sorted(self.metadata.items())),
        }


@dataclass(frozen=True, slots=True)
class HybridGroundingDossier:
    """ASME V&V 10 dossier contrasting raw simulation vs. test-grounded hybrid."""

    sensor_kind: SensorObservationKind
    unit: str
    numerical_claim_level: str
    experimental_validation_tier: ExperimentalValidationTier
    state: HybridGroundingState
    fitted_prior_scale_rho: float
    calibration_specimen_ids: tuple[str, ...]
    holdout_specimen_ids: tuple[str, ...]
    raw_sim_holdout_rmse: float | None
    grounded_holdout_rmse: float | None
    raw_sim_max_abs_error: float | None
    grounded_max_abs_error: float | None
    raw_sim_holdout_r2: float | None
    grounded_holdout_r2: float | None
    bias_reduction_fraction: float | None
    u_num: float
    u_input: float
    u_discrepancy: float
    u_exp: float
    u_val: float
    validation_ratio: float | None
    passed: bool
    reason: str

    def __post_init__(self) -> None:
        if self.numerical_claim_level not in ALLOWED_CLAIM_LEVELS:
            raise ValueError(
                f"numerical_claim_level must lie in {ALLOWED_CLAIM_LEVELS}, "
                f"got {self.numerical_claim_level!r}"
            )

    def snapshot(self) -> StateSnapshot[HybridGroundingState]:
        return HYBRID_GROUNDING_MACHINE.snapshot(
            self.state,
            blockers=() if self.passed else (self.reason,),
            metadata={
                "numerical_claim_level": self.numerical_claim_level,
                "experimental_validation_tier": self.experimental_validation_tier.value,
            },
        )

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()

    def document(self) -> dict[str, Any]:
        return {
            "schema": HYBRID_GROUNDING_DOSSIER_SCHEMA,
            "sensor_kind": self.sensor_kind.value,
            "unit": self.unit,
            "numerical_claim_level": self.numerical_claim_level,
            "experimental_validation_tier": self.experimental_validation_tier.value,
            "state": self.state.value,
            "fitted_prior_scale_rho": float(self.fitted_prior_scale_rho),
            "calibration_specimen_ids": list(self.calibration_specimen_ids),
            "holdout_specimen_ids": list(self.holdout_specimen_ids),
            "raw_sim_holdout_rmse": self.raw_sim_holdout_rmse,
            "grounded_holdout_rmse": self.grounded_holdout_rmse,
            "raw_sim_max_abs_error": self.raw_sim_max_abs_error,
            "grounded_max_abs_error": self.grounded_max_abs_error,
            "raw_sim_holdout_r2": self.raw_sim_holdout_r2,
            "grounded_holdout_r2": self.grounded_holdout_r2,
            "bias_reduction_fraction": self.bias_reduction_fraction,
            "u_num": float(self.u_num),
            "u_input": float(self.u_input),
            "u_discrepancy": float(self.u_discrepancy),
            "u_exp": float(self.u_exp),
            "u_val": float(self.u_val),
            "validation_ratio": self.validation_ratio,
            "passed": self.passed,
            "reason": self.reason,
        }


__all__ = (
    "COMPONENT_EXPERIMENT_SCHEMA",
    "HYBRID_GROUNDING_DOSSIER_SCHEMA",
    "HYBRID_GROUNDING_MACHINE",
    "ComponentTestRecord",
    "HybridGroundingDossier",
    "HybridGroundingState",
    "SensorObservationKind",
)
