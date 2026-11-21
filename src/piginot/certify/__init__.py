"""Per-prediction physics certificates (S21) and CAD numerical trust reports (Feature D)."""

from __future__ import annotations

from typing import Any

from piginot.certify.certificate import Certificate, CertificateVerdict
from piginot.certify.receipt import (
    TRUST_RECEIPT_SCHEMA,
    ArtifactDigests,
    TrustReceipt,
    VerificationTier,
    epistemic_scope,
    verify_receipt_document,
)
from piginot.certify.trust_report import (
    CAD_TRUST_REPORT_SCHEMA,
    SUPPORTED_QOI_KINDS,
    CADTrustReport,
    DiscretizationErrorRecord,
    GeometryErrorRecord,
    QoITrustResult,
    StatisticalCalibrationRecord,
    SurrogateErrorRecord,
    TrustStatus,
    build_cad_trust_report,
    reduce_qoi_trust_status,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AdmissibilityThresholds": (
        "piginot.certify.admissibility",
        "AdmissibilityThresholds",
    ),
    "CoerciveInertia": ("piginot.certify.guards", "CoerciveInertia"),
    "DifferentiableQoi": ("piginot.certify.adjoint", "DifferentiableQoi"),
    "DwrEstimate": ("piginot.certify.adjoint", "DwrEstimate"),
    "Inertia": ("piginot.certify.guards", "Inertia"),
    "ReactionForceQoi": ("piginot.certify.adjoint", "ReactionForceQoi"),
    "RegionalAverageStressQoi": (
        "piginot.certify.adjoint",
        "RegionalAverageStressQoi",
    ),
    "SmoothedPeakVonMisesQoi": (
        "piginot.certify.adjoint",
        "SmoothedPeakVonMisesQoi",
    ),
    "Tier0AdmissibilityResult": (
        "piginot.certify.admissibility",
        "Tier0AdmissibilityResult",
    ),
    "TotalStrainEnergyQoi": ("piginot.certify.adjoint", "TotalStrainEnergyQoi"),
    "certify": ("piginot.certify.estimator", "certify"),
    "certify_with_correction": ("piginot.certify.estimator", "certify_with_correction"),
    "coercive_inertia": ("piginot.certify.guards", "coercive_inertia"),
    "estimate_qoi_dwr": ("piginot.certify.adjoint", "estimate_qoi_dwr"),
    "evaluate_cad_trust": ("piginot.certify.cad_trust", "evaluate_cad_trust"),
    "evaluate_tier0_admissibility": (
        "piginot.certify.admissibility",
        "evaluate_tier0_admissibility",
    ),
    "field_difference_l2": ("piginot.certify.estimator", "field_difference_l2"),
    "field_l2": ("piginot.certify.estimator", "field_l2"),
    "symmetric_inertia": ("piginot.certify.guards", "symmetric_inertia"),
}


def __getattr__(name: str) -> Any:
    """Lazily load FEM-backed certification exports so runtime-only imports stay light."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = (
    "CAD_TRUST_REPORT_SCHEMA",
    "SUPPORTED_QOI_KINDS",
    "TRUST_RECEIPT_SCHEMA",
    "AdmissibilityThresholds",
    "ArtifactDigests",
    "CADTrustReport",
    "Certificate",
    "CertificateVerdict",
    "CoerciveInertia",
    "DifferentiableQoi",
    "DiscretizationErrorRecord",
    "DwrEstimate",
    "GeometryErrorRecord",
    "Inertia",
    "QoITrustResult",
    "ReactionForceQoi",
    "RegionalAverageStressQoi",
    "SmoothedPeakVonMisesQoi",
    "StatisticalCalibrationRecord",
    "SurrogateErrorRecord",
    "Tier0AdmissibilityResult",
    "TotalStrainEnergyQoi",
    "TrustReceipt",
    "TrustStatus",
    "VerificationTier",
    "build_cad_trust_report",
    "certify",
    "certify_with_correction",
    "coercive_inertia",
    "epistemic_scope",
    "estimate_qoi_dwr",
    "evaluate_cad_trust",
    "evaluate_tier0_admissibility",
    "field_difference_l2",
    "field_l2",
    "reduce_qoi_trust_status",
    "symmetric_inertia",
    "verify_receipt_document",
)
