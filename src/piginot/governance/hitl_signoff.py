"""Cryptographic Veteran-Engineer Human-in-the-Loop (HITL) Signoff FSM (V5-4).

Enforces explicit senior-engineer review and hand-calculation sanity checks
before high-risk PLM ECO release or continual surrogate promotion:

    DRAFT -> PENDING_HUMAN_REVIEW -> APPROVED | REJECTED_WITH_FEEDBACK

Zero ML/FEM runtime dependencies (pure stdlib + RFC 8785 canonical JSON).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final

from piginot.contracts.canonical import canonical_json_bytes
from piginot.contracts.state_machine import (
    InvalidTransitionError,
    StateMachineSpec,
    StateSnapshot,
    TransitionSpec,
    register_state_machine,
)
from piginot.serving.audit_ledger import AuditEventKind, SignedAuditLedger

HITL_SIGNOFF_SCHEMA: Final = "formamech.hitl-signoff-v1"


def _sha256_doc(doc: object) -> str:
    return hashlib.sha256(canonical_json_bytes(doc)).hexdigest()


class HitlSignoffState(StrEnum):
    """Explicit fail-closed lifecycle for senior-engineer HITL release signoff."""

    DRAFT = "DRAFT"
    PENDING_HUMAN_REVIEW = "PENDING_HUMAN_REVIEW"
    APPROVED = "APPROVED"
    REJECTED_WITH_FEEDBACK = "REJECTED_WITH_FEEDBACK"


HITL_SIGNOFF_MACHINE: Final[StateMachineSpec[HitlSignoffState]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.governance.hitl_signoff",
            description=(
                "Veteran-engineer human-in-the-loop signoff gate: "
                "DRAFT -> PENDING_HUMAN_REVIEW -> APPROVED | REJECTED_WITH_FEEDBACK."
            ),
            states=tuple(HitlSignoffState),
            initial_states=(HitlSignoffState.DRAFT,),
            terminal_states=(
                HitlSignoffState.APPROVED,
                HitlSignoffState.REJECTED_WITH_FEEDBACK,
            ),
            transitions=(
                TransitionSpec(
                    HitlSignoffState.DRAFT,
                    HitlSignoffState.PENDING_HUMAN_REVIEW,
                    "submit_for_review",
                    "Submit verified artifact + dossier for senior-engineer review.",
                ),
                TransitionSpec(
                    HitlSignoffState.PENDING_HUMAN_REVIEW,
                    HitlSignoffState.APPROVED,
                    "approve_release",
                    "Senior engineer verified hand-calc sanity check and co-signed release.",
                ),
                TransitionSpec(
                    HitlSignoffState.PENDING_HUMAN_REVIEW,
                    HitlSignoffState.REJECTED_WITH_FEEDBACK,
                    "reject_with_feedback",
                    "Senior engineer rejected artifact with engineering corrective feedback.",
                ),
            ),
        )
    )
)


class ReviewerRole(StrEnum):
    """Authorized engineering oversight roles for HITL signoff."""

    LEAD_STRESS_ENGINEER = "LEAD_STRESS_ENGINEER"
    VALIDITY_OWNER = "VALIDITY_OWNER"
    CHIEF_ENGINEER = "CHIEF_ENGINEER"


@dataclass(frozen=True, slots=True)
class HandCalculationSanityCheck:
    """Classical mechanics sanity check recorded by the reviewing engineer."""

    hand_calc_note: str
    hand_calc_value: float
    model_predicted_value: float
    unit: str
    max_allowed_ratio: float = 1.50

    def __post_init__(self) -> None:
        if not self.hand_calc_note.strip():
            raise ValueError("hand_calc_note must be non-empty text")
        if (
            not math.isfinite(self.hand_calc_value)
            or abs(self.hand_calc_value) <= 1.0e-12
        ):
            raise ValueError("hand_calc_value must be finite and non-zero")
        if not math.isfinite(self.model_predicted_value):
            raise ValueError("model_predicted_value must be finite")
        if not self.unit.strip():
            raise ValueError("unit must be non-empty text")
        if not math.isfinite(self.max_allowed_ratio) or self.max_allowed_ratio < 1.0:
            raise ValueError("max_allowed_ratio must be >= 1.0")

    @property
    def sanity_check_ratio(self) -> float:
        return abs(self.model_predicted_value / self.hand_calc_value)

    @property
    def within_tolerance(self) -> bool:
        r = self.sanity_check_ratio
        return (1.0 / self.max_allowed_ratio) <= r <= self.max_allowed_ratio

    def document(self) -> dict[str, Any]:
        return {
            "hand_calc_note": self.hand_calc_note,
            "hand_calc_value": float(self.hand_calc_value),
            "model_predicted_value": float(self.model_predicted_value),
            "unit": self.unit,
            "sanity_check_ratio": float(self.sanity_check_ratio),
            "max_allowed_ratio": float(self.max_allowed_ratio),
            "within_tolerance": self.within_tolerance,
        }


@dataclass(frozen=True, slots=True)
class HitlSignoffRecord:
    """Cryptographically bound senior-engineer HITL signoff record."""

    artifact_id: str
    artifact_sha256: str
    state: HitlSignoffState
    reviewer_id: str
    reviewer_role: ReviewerRole
    decision_note: str
    hand_calc: HandCalculationSanityCheck | None
    timestamp_utc: str
    signature_sha256: str

    def unsigned_document(self) -> dict[str, Any]:
        return {
            "schema": HITL_SIGNOFF_SCHEMA,
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "state": self.state.value,
            "reviewer_id": self.reviewer_id,
            "reviewer_role": self.reviewer_role.value,
            "decision_note": self.decision_note,
            "hand_calc": None if self.hand_calc is None else self.hand_calc.document(),
            "timestamp_utc": self.timestamp_utc,
        }

    def document(self) -> dict[str, Any]:
        doc = self.unsigned_document()
        doc["signature_sha256"] = self.signature_sha256
        return doc

    def verify_signature(self) -> bool:
        """Return True iff the RFC 8785 SHA-256 signature matches the exact payload."""
        return _sha256_doc(self.unsigned_document()) == self.signature_sha256

    def snapshot(self) -> StateSnapshot[HitlSignoffState]:
        return HITL_SIGNOFF_MACHINE.snapshot(
            self.state,
            metadata={
                "artifact_id": self.artifact_id,
                "reviewer_id": self.reviewer_id,
                "reviewer_role": self.reviewer_role.value,
                "signature_valid": self.verify_signature(),
            },
        )


def create_hitl_signoff(
    *,
    artifact_id: str,
    artifact_sha256: str,
    reviewer_id: str,
    reviewer_role: ReviewerRole | str,
    approved: bool,
    decision_note: str,
    hand_calc: HandCalculationSanityCheck | None = None,
    audit_ledger: SignedAuditLedger | None = None,
    timestamp_utc: str | None = None,
) -> HitlSignoffRecord:
    """Execute the HITL FSM (DRAFT -> PENDING_HUMAN_REVIEW -> APPROVED | REJECTED) and sign."""
    if not artifact_id.strip() or not artifact_sha256.strip():
        raise ValueError("artifact_id and artifact_sha256 must be non-empty")
    if not reviewer_id.strip() or not decision_note.strip():
        raise ValueError("reviewer_id and decision_note must be non-empty")
    role = ReviewerRole(reviewer_role)

    state = HitlSignoffState.DRAFT
    state = HITL_SIGNOFF_MACHINE.require_transition(
        state, HitlSignoffState.PENDING_HUMAN_REVIEW, trigger="submit_for_review"
    ).to_state

    if approved:
        if hand_calc is None:
            raise ValueError(
                "HITL approval requires a HandCalculationSanityCheck from the reviewing engineer"
            )
        if not hand_calc.within_tolerance:
            raise ValueError(
                f"hand-calculation sanity check ratio {hand_calc.sanity_check_ratio:.3f} "
                f"exceeds allowed tolerance {hand_calc.max_allowed_ratio:.2f}"
            )
        state = HITL_SIGNOFF_MACHINE.require_transition(
            state, HitlSignoffState.APPROVED, trigger="approve_release"
        ).to_state
    else:
        state = HITL_SIGNOFF_MACHINE.require_transition(
            state,
            HitlSignoffState.REJECTED_WITH_FEEDBACK,
            trigger="reject_with_feedback",
        ).to_state

    ts = (
        datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        if timestamp_utc is None
        else timestamp_utc
    )
    unsigned = {
        "schema": HITL_SIGNOFF_SCHEMA,
        "artifact_id": artifact_id,
        "artifact_sha256": artifact_sha256,
        "state": state.value,
        "reviewer_id": reviewer_id,
        "reviewer_role": role.value,
        "decision_note": decision_note,
        "hand_calc": None if hand_calc is None else hand_calc.document(),
        "timestamp_utc": ts,
    }
    sig = _sha256_doc(unsigned)
    record = HitlSignoffRecord(
        artifact_id=artifact_id,
        artifact_sha256=artifact_sha256,
        state=state,
        reviewer_id=reviewer_id,
        reviewer_role=role,
        decision_note=decision_note,
        hand_calc=hand_calc,
        timestamp_utc=ts,
        signature_sha256=sig,
    )
    if audit_ledger is not None:
        audit_ledger.append(
            event_kind=AuditEventKind.VERIFY_RECEIPT,
            bundle_name=artifact_id,
            mode="hitl-signoff",
            verdict=state.value,
            inputs={
                "artifact_sha256": artifact_sha256,
                "reviewer_id": reviewer_id,
                "reviewer_role": role.value,
            },
            trust_receipt_sha256=sig,
            metadata={"decision_note": decision_note},
        )
    return record


def require_valid_hitl_signoff(
    signoff: HitlSignoffRecord | None,
    *,
    expected_artifact_sha256: str | None = None,
) -> HitlSignoffRecord:
    """Fail-closed release gate raising InvalidTransitionError unless validly APPROVED."""
    if signoff is None:
        raise InvalidTransitionError(
            machine_id=HITL_SIGNOFF_MACHINE.machine_id,
            current_state=HitlSignoffState.PENDING_HUMAN_REVIEW.value,
            target_state=HitlSignoffState.APPROVED.value,
            allowed_targets=(
                HitlSignoffState.APPROVED.value,
                HitlSignoffState.REJECTED_WITH_FEEDBACK.value,
            ),
            trigger="release_gate",
            reason=(
                "HITL signoff required: release paused in PENDING_HUMAN_REVIEW "
                "(no senior-engineer signoff record provided)"
            ),
        )
    if signoff.state is not HitlSignoffState.APPROVED:
        raise InvalidTransitionError(
            machine_id=HITL_SIGNOFF_MACHINE.machine_id,
            current_state=signoff.state.value,
            target_state=HitlSignoffState.APPROVED.value,
            allowed_targets=(),
            trigger="release_gate",
            reason=f"HITL signoff not approved: current state is {signoff.state.value!r}",
        )
    if not signoff.verify_signature():
        raise InvalidTransitionError(
            machine_id=HITL_SIGNOFF_MACHINE.machine_id,
            current_state=signoff.state.value,
            target_state=HitlSignoffState.APPROVED.value,
            allowed_targets=(),
            trigger="verify_signature",
            reason="HITL signoff signature verification failed (tampered signoff record)",
        )
    if (
        expected_artifact_sha256 is not None
        and signoff.artifact_sha256 != expected_artifact_sha256
    ):
        raise InvalidTransitionError(
            machine_id=HITL_SIGNOFF_MACHINE.machine_id,
            current_state=signoff.state.value,
            target_state=HitlSignoffState.APPROVED.value,
            allowed_targets=(),
            trigger="verify_artifact_binding",
            reason="HITL signoff artifact_sha256 mismatch: signature was bound to a different artifact",
        )
    return signoff


__all__ = (
    "HITL_SIGNOFF_MACHINE",
    "HITL_SIGNOFF_SCHEMA",
    "HandCalculationSanityCheck",
    "HitlSignoffRecord",
    "HitlSignoffState",
    "ReviewerRole",
    "create_hitl_signoff",
    "require_valid_hitl_signoff",
)
