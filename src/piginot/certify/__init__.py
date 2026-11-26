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
    "DWR_3D_CERTIFICATION_SCHEMA": (
        "piginot.certify.adjoint_3d",
        "DWR_3D_CERTIFICATION_SCHEMA",
    ),
    "ComplianceQoi3D": ("piginot.certify.adjoint_3d", "ComplianceQoi3D"),
    "DifferentiableQoi3D": ("piginot.certify.adjoint_3d", "DifferentiableQoi3D"),
    "Dwr3DCertificationResult": (
        "piginot.certify.adjoint_3d",
        "Dwr3DCertificationResult",
    ),
    "DwrEstimate3D": ("piginot.certify.adjoint_3d", "DwrEstimate3D"),
    "Linearized3DState": ("piginot.certify.adjoint_3d", "Linearized3DState"),
    "ReactionForceQoi3D": ("piginot.certify.adjoint_3d", "ReactionForceQoi3D"),
    "SmoothedPeakVonMisesQoi3D": (
        "piginot.certify.adjoint_3d",
        "SmoothedPeakVonMisesQoi3D",
    ),
    "TotalStrainEnergyQoi3D": (
        "piginot.certify.adjoint_3d",
        "TotalStrainEnergyQoi3D",
    ),
    "estimate_qoi_dwr_3d": ("piginot.certify.adjoint_3d", "estimate_qoi_dwr_3d"),
    "linearize_3d_state": ("piginot.certify.adjoint_3d", "linearize_3d_state"),
    "BC_LAYOUT_OOD_SCHEMA": (
        "piginot.certify.regime_guards",
        "BC_LAYOUT_OOD_SCHEMA",
    ),
    "REGIME_GUARD_3D_SCHEMA": (
        "piginot.certify.regime_guards",
        "REGIME_GUARD_3D_SCHEMA",
    ),
    "BcLayoutDescriptor3D": (
        "piginot.certify.regime_guards",
        "BcLayoutDescriptor3D",
    ),
    "BcLayoutOODDetector3D": (
        "piginot.certify.regime_guards",
        "BcLayoutOODDetector3D",
    ),
    "ContactGapPairSpec": (
        "piginot.certify.regime_guards",
        "ContactGapPairSpec",
    ),
    "PhysicsRegimeGuard3D": (
        "piginot.certify.regime_guards",
        "PhysicsRegimeGuard3D",
    ),
    "PhysicsRegimeGuardConfig3D": (
        "piginot.certify.regime_guards",
        "PhysicsRegimeGuardConfig3D",
    ),
    "RegimeFlag": ("piginot.certify.regime_guards", "RegimeFlag"),
    "RegimeGuardVerdict": ("piginot.certify.regime_guards", "RegimeGuardVerdict"),
    "compute_bc_layout_descriptor_3d": (
        "piginot.certify.regime_guards",
        "compute_bc_layout_descriptor_3d",
    ),
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
    "BC_LAYOUT_OOD_SCHEMA",
    "DWR_3D_CERTIFICATION_SCHEMA",
    "REGIME_GUARD_3D_SCHEMA",
    "SUPPORTED_QOI_KINDS",
    "TRUST_RECEIPT_SCHEMA",
    "AdmissibilityThresholds",
    "ArtifactDigests",
    "BcLayoutDescriptor3D",
    "BcLayoutOODDetector3D",
    "CADTrustReport",
    "Certificate",
    "CertificateVerdict",
    "CoerciveInertia",
    "ComplianceQoi3D",
    "ContactGapPairSpec",
    "DifferentiableQoi",
    "DifferentiableQoi3D",
    "DiscretizationErrorRecord",
    "Dwr3DCertificationResult",
    "DwrEstimate",
    "DwrEstimate3D",
    "GeometryErrorRecord",
    "Inertia",
    "Linearized3DState",
    "PhysicsRegimeGuard3D",
    "PhysicsRegimeGuardConfig3D",
    "QoITrustResult",
    "ReactionForceQoi",
    "ReactionForceQoi3D",
    "RegimeFlag",
    "RegimeGuardVerdict",
    "RegionalAverageStressQoi",
    "SmoothedPeakVonMisesQoi",
    "SmoothedPeakVonMisesQoi3D",
    "StatisticalCalibrationRecord",
    "SurrogateErrorRecord",
    "Tier0AdmissibilityResult",
    "TotalStrainEnergyQoi",
    "TotalStrainEnergyQoi3D",
    "TrustReceipt",
    "TrustStatus",
    "VerificationTier",
    "build_cad_trust_report",
    "certify",
    "certify_with_correction",
    "coercive_inertia",
    "compute_bc_layout_descriptor_3d",
    "epistemic_scope",
    "estimate_qoi_dwr",
    "estimate_qoi_dwr_3d",
    "evaluate_cad_trust",
    "evaluate_tier0_admissibility",
    "field_difference_l2",
    "field_l2",
    "linearize_3d_state",
    "reduce_qoi_trust_status",
    "symmetric_inertia",
    "verify_receipt_document",
)
