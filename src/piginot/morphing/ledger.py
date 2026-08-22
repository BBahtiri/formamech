"""S6 orchestration that preserves one auditable record per supplied attempt."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from piginot.contracts import CanonicalMesh, CanonicalQuantity
from piginot.morphing.models import (
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
)


class _BoundDeformation(Protocol):
    @property
    def kind(self) -> str: ...

    @property
    def parameter_hash(self) -> str: ...

    def map_point(self, point: Point) -> Point: ...

    def jacobian(self, point: Point) -> Matrix2: ...


class _MorphMap(Protocol):
    @property
    def parameter_names(self) -> tuple[str, ...]: ...

    def bind(self, theta: Mapping[str, CanonicalQuantity]) -> _BoundDeformation: ...


class _Predicate(Protocol):
    @property
    def reference_mesh_id(self) -> str: ...

    @property
    def validity_policy_hash(self) -> str: ...

    @property
    def parameter_names(self) -> tuple[str, ...]: ...

    @property
    def validity(self) -> MorphValidityCriteria: ...

    def evaluate(self, theta: Mapping[str, object]) -> MembershipDecision: ...


def apply_morph(
    reference: CanonicalMesh,
    morph_map: _MorphMap,
    predicate: _Predicate,
    theta: Mapping[str, object],
) -> MorphResult:
    """Evaluate membership, bind one map, and validate the resulting mesh."""

    context_reason = _context_reason(reference, morph_map, predicate)
    if context_reason is not None:
        membership = MembershipDecision(False, (context_reason,))
        report = _empty_report(reference, predicate, (context_reason,))
        return MorphResult(None, report, membership, (context_reason,))

    membership = predicate.evaluate(theta)
    if not membership.accepted:
        report = _empty_report(reference, predicate, membership.reason_codes)
        return MorphResult(None, report, membership, membership.reason_codes)

    canonical_theta = {
        name: value
        for name, value in theta.items()
        if isinstance(value, CanonicalQuantity)
    }
    try:
        deformation = morph_map.bind(canonical_theta)
    except MorphingError as exc:
        bind_reasons = (exc.reason_code,)
        report = _empty_report(reference, predicate, bind_reasons)
        return MorphResult(None, report, membership, bind_reasons)
    except (ArithmeticError, TypeError, ValueError):
        bind_reasons = (MorphReasonCode.INVALID_MAP_CONFIGURATION,)
        report = _empty_report(reference, predicate, bind_reasons)
        return MorphResult(None, report, membership, bind_reasons)

    from piginot.morphing.validation import validate_deformation

    mesh, report = validate_deformation(reference, deformation, predicate.validity)
    reasons = report.reason_codes
    return MorphResult(mesh if not reasons else None, report, membership, reasons)


def run_morph_sweep(
    reference: CanonicalMesh,
    morph_map: _MorphMap,
    predicate: _Predicate,
    attempts: Sequence[MorphAttempt],
) -> MorphLedger:
    """Return one accepted or reason-bearing rejected record per attempt."""

    frozen_attempts = tuple(attempts)
    attempt_ids = tuple(attempt.attempt_id for attempt in frozen_attempts)
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("sweep attempt IDs must be unique")

    records = []
    for attempt in frozen_attempts:
        result = apply_morph(reference, morph_map, predicate, attempt.theta)
        records.append(
            MorphRecord(
                attempt_id=attempt.attempt_id,
                status=(
                    MorphStatus.ACCEPTED if result.accepted else MorphStatus.REJECTED
                ),
                reason_codes=result.reason_codes,
                mesh=result.mesh,
                report=result.report,
            )
        )
    ledger = MorphLedger(tuple(records))
    if ledger.accepted_count + ledger.rejected_count != ledger.attempted_count:
        raise AssertionError("morph ledger accounting invariant failed")
    return ledger


def _context_reason(
    reference: CanonicalMesh, morph_map: _MorphMap, predicate: _Predicate
) -> MorphReasonCode | None:
    if reference.mesh_id != predicate.reference_mesh_id:
        return MorphReasonCode.VALIDATION_CONTEXT_MISMATCH
    if set(morph_map.parameter_names) != set(predicate.parameter_names):
        return MorphReasonCode.INVALID_MAP_CONFIGURATION
    return None


def _empty_report(
    reference: CanonicalMesh,
    predicate: _Predicate,
    reasons: tuple[MorphReasonCode, ...],
) -> MorphValidityReport:
    return MorphValidityReport(
        reference_mesh_id=reference.mesh_id,
        validity_policy_hash=predicate.validity_policy_hash,
        quadrature_rule="triangle-degree-2-v1",
        quadrature_points_checked=0,
        minimum_jacobian_determinant=None,
        topology_signature=None,
        semantic_tag_counts={},
        tags_preserved=False,
        minimum_feature_size_mm=None,
        minimum_angle_degrees=None,
        maximum_aspect_ratio=None,
        reason_codes=reasons,
    )
