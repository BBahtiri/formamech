"""ASME V&V 10 experimental validation, inverse calibration & hybrid grounding (V3-3 / V5-2)."""

from piginot.validation.component_test_contracts import (
    COMPONENT_EXPERIMENT_SCHEMA,
    HYBRID_GROUNDING_DOSSIER_SCHEMA,
    HYBRID_GROUNDING_MACHINE,
    ComponentTestRecord,
    HybridGroundingDossier,
    HybridGroundingState,
    SensorObservationKind,
)
from piginot.validation.contracts import (
    EXPERIMENTAL_DATASET_SCHEMA,
    VALIDATION_LIFECYCLE_MACHINE,
    VV10_VALIDATION_DOSSIER_SCHEMA,
    CalibratedParameterEstimate,
    CouponRole,
    ExperimentalCouponDataset,
    ExperimentalTestMode,
    ExperimentalValidationDossier,
    ExperimentalValidationTier,
    ValidationLifecycleState,
)
from piginot.validation.hybrid_grounding import (
    HybridDiscrepancyModel,
    HybridGroundingConfig,
    SensorMeshAlignmentReport,
    align_sensor_observations_to_mesh,
    ground_simulation_with_experiments,
)
from piginot.validation.inverse import (
    CalibrationModelFamily,
    InverseCalibrationConfig,
    calibrate_material_from_experiments,
    predict_coupon_stress_torch,
)

__all__ = (
    "COMPONENT_EXPERIMENT_SCHEMA",
    "EXPERIMENTAL_DATASET_SCHEMA",
    "HYBRID_GROUNDING_DOSSIER_SCHEMA",
    "HYBRID_GROUNDING_MACHINE",
    "VALIDATION_LIFECYCLE_MACHINE",
    "VV10_VALIDATION_DOSSIER_SCHEMA",
    "CalibratedParameterEstimate",
    "CalibrationModelFamily",
    "ComponentTestRecord",
    "CouponRole",
    "ExperimentalCouponDataset",
    "ExperimentalTestMode",
    "ExperimentalValidationDossier",
    "ExperimentalValidationTier",
    "HybridDiscrepancyModel",
    "HybridGroundingConfig",
    "HybridGroundingDossier",
    "HybridGroundingState",
    "InverseCalibrationConfig",
    "SensorMeshAlignmentReport",
    "SensorObservationKind",
    "ValidationLifecycleState",
    "align_sensor_observations_to_mesh",
    "calibrate_material_from_experiments",
    "ground_simulation_with_experiments",
    "predict_coupon_stress_torch",
)
