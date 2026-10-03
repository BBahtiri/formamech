"""Self-contained, hash-closed bundles and their inference closure (S15)."""

from piginot.bundle.build import build_bundle
from piginot.bundle.inference import (
    BUNDLE_FILES,
    Bundle,
    PointFields,
    Prediction,
    PredictionReasonCode,
    QoiValue,
    Verdict,
    inspect,
    load_bundle,
    predict,
)
from piginot.bundle.limitations import REQUIRED_LIMITATION_PARAGRAPH
from piginot.bundle.manifest import (
    BUNDLE_FORMAT,
    MANIFEST_NAME,
    BundleIntegrityError,
    BundleManifest,
    BundleReasonCode,
    verify_bundle,
)
from piginot.bundle.serving import CONSTITUTIVE_PLANE_STRESS, ServingError
from piginot.bundle.specs import (
    ConditioningInput,
    ConfidenceSpec,
    ModelSpec,
    QoiSpec,
    ScaledParameter,
    SpecDecodeError,
)

__all__ = (
    "BUNDLE_FILES",
    "BUNDLE_FORMAT",
    "CONSTITUTIVE_PLANE_STRESS",
    "MANIFEST_NAME",
    "REQUIRED_LIMITATION_PARAGRAPH",
    "Bundle",
    "BundleIntegrityError",
    "BundleManifest",
    "BundleReasonCode",
    "ConditioningInput",
    "ConfidenceSpec",
    "ModelSpec",
    "PointFields",
    "Prediction",
    "PredictionReasonCode",
    "QoiSpec",
    "QoiValue",
    "ScaledParameter",
    "ServingError",
    "SpecDecodeError",
    "Verdict",
    "build_bundle",
    "inspect",
    "load_bundle",
    "predict",
    "verify_bundle",
)
