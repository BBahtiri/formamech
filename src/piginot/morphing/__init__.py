"""Deterministic S6 morphing and family-membership contracts."""

from piginot.morphing.blended_notch import (
    BlendedNotchMap,
    BlendedNotchMeshSpec,
    BlendedNotchParameters,
    BoundBlendedNotchDeformation,
    build_blended_notch_reference_mesh,
)
from piginot.morphing.circular_transition import (
    BoundCircularTransitionDeformation,
    CircularTransitionMap,
    CircularTransitionMeshSpec,
    CircularTransitionParameters,
    build_circular_transition_reference_mesh,
)
from piginot.morphing.ledger import apply_morph, run_morph_sweep
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


def __getattr__(name: str) -> object:
    if name == "FFDMap":
        from piginot.morphing.maps import FFDMap

        return FFDMap
    if name == "RBFMap":
        from piginot.morphing.maps import RBFMap

        return RBFMap
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "BlendedNotchMap",
    "BlendedNotchMeshSpec",
    "BlendedNotchParameters",
    "BoundBlendedNotchDeformation",
    "BoundCircularTransitionDeformation",
    "CircularTransitionMap",
    "CircularTransitionMeshSpec",
    "CircularTransitionParameters",
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
    "build_blended_notch_reference_mesh",
    "RBFMap",
    "apply_morph",
    "build_circular_transition_reference_mesh",
    "run_morph_sweep",
)
