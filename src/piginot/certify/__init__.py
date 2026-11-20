"""Per-prediction physics certificates (S21) and CAD numerical trust reports (Feature D)."""

from __future__ import annotations

from typing import Any

from piginot.certify.certificate import Certificate, CertificateVerdict
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
    "DifferentiableQoi": ("piginot.certify.adjoint", "DifferentiableQoi"),
    "DwrEstimate": ("piginot.certify.adjoint", "DwrEstimate"),
    "Inertia": ("piginot.certify.guards", "Inertia"),
    "ReactionForceQoi": ("piginot.certify.adjoint", "ReactionForceQoi"),
    "RegionalAverageStressQoi": (
        "piginot.certify.adjoint",
        "RegionalAverageStressQoi",
    ),
    "certify": ("piginot.certify.estimator", "certify"),
    "certify_with_correction": ("piginot.certify.estimator", "certify_with_correction"),
    "estimate_qoi_dwr": ("piginot.certify.adjoint", "estimate_qoi_dwr"),
    "evaluate_cad_trust": ("piginot.certify.cad_trust", "evaluate_cad_trust"),
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
    "CADTrustReport",
    "Certificate",
    "CertificateVerdict",
    "DifferentiableQoi",
    "DiscretizationErrorRecord",
    "DwrEstimate",
    "GeometryErrorRecord",
    "Inertia",
    "QoITrustResult",
    "ReactionForceQoi",
    "RegionalAverageStressQoi",
    "StatisticalCalibrationRecord",
    "SurrogateErrorRecord",
    "TrustStatus",
    "build_cad_trust_report",
    "certify",
    "certify_with_correction",
    "estimate_qoi_dwr",
    "evaluate_cad_trust",
    "field_difference_l2",
    "field_l2",
    "reduce_qoi_trust_status",
    "symmetric_inertia",
)
