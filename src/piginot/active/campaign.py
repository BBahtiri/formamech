"""Auditable multi-round active-FEM campaign orchestration (Feature C / P3).

Runs closed-loop TRAIN-only candidate selection, charges every FEM attempt into
an immutable :class:`FEMCostLedger`, enforces explicit multi-fidelity disclosure
whenever acquired FEM targets supervise training, and persists canonical JSON
evidence artifacts (``plan.json``, ``selection-evidence.json``,
``fem-cost-ledger.json``, and ``active-summary.json``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from piginot.active.acquisition import (
    AcquisitionCandidate,
    AcquisitionPlan,
    FEMCostLedger,
    FEMCostLedgerEntry,
    SelectionEvidence,
    select_acquisition_batch,
)
from piginot.contracts import canonical_json_bytes

ACTIVE_CAMPAIGN_SUMMARY_SCHEMA: Final = "piginot.active-fem-campaign-v1"


@dataclass(frozen=True, slots=True)
class ActiveFemSolveOutcome:
    """Result returned by a user/compiler FEM oracle for one TRAIN candidate."""

    succeeded: bool
    wall_seconds: float
    dof_count: int
    result_hash: str | None = None
    failure_reason: str = ""
    payload: Any = None


@dataclass(frozen=True, slots=True)
class ActiveFemCampaignResult:
    """Immutable provenance summary for one completed active-FEM campaign."""

    plan: AcquisitionPlan
    selections: tuple[SelectionEvidence, ...]
    ledger: FEMCostLedger
    acquired_case_ids: tuple[str, ...]
    training_mode_disclosure: str
    claim_level: str = "NOT_VERIFIED"

    def document(self) -> dict[str, Any]:
        return {
            "schema": ACTIVE_CAMPAIGN_SUMMARY_SCHEMA,
            "plan_hash": self.plan.plan_hash,
            "arm_kind": self.plan.arm_kind,
            "training_mode": self.plan.training_mode,
            "training_mode_disclosure": self.training_mode_disclosure,
            "claim_level": self.claim_level,
            "rounds_completed": len(self.selections),
            "acquired_case_ids": list(self.acquired_case_ids),
            "selection_hashes": [s.selection_hash for s in self.selections],
            "ledger_hash": self.ledger.ledger_hash,
            "fem_attempted": self.ledger.attempted_count,
            "fem_succeeded": self.ledger.succeeded_count,
            "fem_failed": self.ledger.failed_count,
            "total_fem_wall_seconds": self.ledger.total_wall_seconds,
        }

    @property
    def summary_hash(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()


def run_active_fem_campaign(
    *,
    plan: AcquisitionPlan,
    score_pool: Callable[[int, Sequence[str]], Sequence[AcquisitionCandidate]],
    solve_train_case: Callable[[AcquisitionCandidate], ActiveFemSolveOutcome],
    on_round_complete: (
        Callable[
            [int, Sequence[tuple[AcquisitionCandidate, ActiveFemSolveOutcome]]], None
        ]
        | None
    ) = None,
    output_dir: Path | None = None,
) -> ActiveFemCampaignResult:
    """Execute a budgeted, deterministic, TRAIN-only active-FEM campaign."""
    ledger = FEMCostLedger(plan_hash=plan.plan_hash, budget_solves=plan.budget_solves)
    selections: list[SelectionEvidence] = []
    acquired_ids: list[str] = []
    acquired_features: list[tuple[float, ...]] = []

    disclosure = (
        "multi-fidelity-label-efficient-not-label-free"
        if plan.supervises_training
        else "label-free-audit-only"
    )

    round_index = 0
    while ledger.attempted_count < plan.budget_solves:
        pool = list(score_pool(round_index, tuple(acquired_ids)))
        by_id = {cand.case_id: cand for cand in pool}
        evidence = select_acquisition_batch(
            pool,
            already_acquired_ids=acquired_ids,
            plan=plan,
            round_index=round_index,
            already_acquired_features=acquired_features,
        )
        if not evidence.selected_case_ids:
            break
        selections.append(evidence)

        round_outcomes: list[tuple[AcquisitionCandidate, ActiveFemSolveOutcome]] = []
        for cid in evidence.selected_case_ids:
            cand = by_id[cid]
            if cand.partition != "TRAIN":
                raise ValueError("active campaign refuses non-TRAIN FEM acquisition")
            outcome = solve_train_case(cand)
            ledger = ledger.append(
                FEMCostLedgerEntry(
                    case_id=cand.case_id,
                    partition="TRAIN",
                    attempt_index=ledger.attempted_count + 1,
                    succeeded=outcome.succeeded,
                    wall_seconds=outcome.wall_seconds,
                    dof_count=outcome.dof_count,
                    result_hash=outcome.result_hash,
                    failure_reason=outcome.failure_reason,
                )
            )
            acquired_ids.append(cand.case_id)
            if plan.arm_kind == "physics_dwr_diversity" and cand.geometry_token_vector:
                acquired_features.append(
                    (*cand.feature_vector, *cand.geometry_token_vector)
                )
            else:
                acquired_features.append(cand.feature_vector)
            round_outcomes.append((cand, outcome))

        if on_round_complete is not None:
            on_round_complete(round_index, round_outcomes)
        round_index += 1

    result = ActiveFemCampaignResult(
        plan=plan,
        selections=tuple(selections),
        ledger=ledger,
        acquired_case_ids=tuple(acquired_ids),
        training_mode_disclosure=disclosure,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plan.json").write_bytes(
            canonical_json_bytes(plan.document()) + b"\n"
        )
        (output_dir / "selection-evidence.json").write_bytes(
            canonical_json_bytes([s.document() for s in selections]) + b"\n"
        )
        (output_dir / "fem-cost-ledger.json").write_bytes(
            canonical_json_bytes(ledger.document()) + b"\n"
        )
        (output_dir / "active-summary.json").write_bytes(
            canonical_json_bytes(
                {**result.document(), "summary_hash": result.summary_hash}
            )
            + b"\n"
        )

    return result


__all__ = (
    "ACTIVE_CAMPAIGN_SUMMARY_SCHEMA",
    "ActiveFemCampaignResult",
    "ActiveFemSolveOutcome",
    "run_active_fem_campaign",
)
