"""Deterministic S6 morphing and family-membership contracts."""

from piginot.morphing.ledger import apply_morph, run_morph_sweep
from piginot.morphing.maps import FFDMap, RBFMap
from piginot.morphing.membership import (
    DerivedBound,
    FamilyMembershipPredicate,
    PrimitiveBound,
)
from piginot.morphing.models import (
    FFDControlPoint,
    Matrix2,
    MembershipDecision,
    MorphAttempt,
    MorphingError,
    MorphLedger,
    MorphReasonCode,
    MorphRecord,
    MorphResult,
    MorphStatus,
    MorphValidityCriteria,
    MorphValidityReport,
    Point,
    RBFControlPoint,
)

__all__ = (
    "DerivedBound",
    "FFDControlPoint",
    "FFDMap",
    "FamilyMembershipPredicate",
    "Matrix2",
    "MembershipDecision",
    "MorphAttempt",
    "MorphLedger",
    "MorphReasonCode",
    "MorphRecord",
    "MorphResult",
    "MorphStatus",
    "MorphValidityCriteria",
    "MorphValidityReport",
    "MorphingError",
    "Point",
    "PrimitiveBound",
    "RBFControlPoint",
    "RBFMap",
    "apply_morph",
    "run_morph_sweep",
)
