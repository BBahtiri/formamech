"""Fail-closed Continual Qualification Loop (CLPM) with atomic rollback (V4-5).

Closes the production feedback loop safely without catastrophic forgetting:

    OBSERVING -> COLLECTING_FALLBACKS -> REPLAY_CURATED -> ADAPTING
      -> SHADOW_QUALIFYING -> PROMOTED | ROLLED_BACK

Every cycle:

1. Verifies the SHA-256 hash chain of :class:`~piginot.serving.audit_ledger.SignedAuditLedger`,
2. Ingests live OOD / high-residual fallback cases and appends verified FEM
   solutions to :class:`~piginot.store.dataset_3d.ShardedMeshDataset3D` under
   ``DatasetSplit.TRAIN`` (enforcing zero leakage against frozen ``HOLDOUT``),
3. Curates a replay-protected adaptation batch mixing prior ``TRAIN`` + new
   fallback cases,
4. Produces an isolated shadow candidate checkpoint (never mutating the active
   production checkpoint in place),
5. Evaluates BOTH active and shadow candidates on the frozen blind ``HOLDOUT``
   suite AND the new fallback distribution,
6. Atomically promotes ONLY when zero holdout regression occurs; otherwise
   rolls back deterministically with an RFC 8785 audit receipt.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from piginot.contracts import canonical_json_bytes
from piginot.contracts.state_machine import (
    StateMachineSpec,
    StateSnapshot,
    TransitionSpec,
    register_state_machine,
)
from piginot.serving.audit_ledger import AuditEventKind, SignedAuditLedger
from piginot.store.dataset_3d import (
    DatasetSplit,
    ShardedMeshDataset3D,
    ShardedRecord3D,
)

CONTINUAL_LOOP_SCHEMA: Final = "formamech.continual-clpm-v1"


class ContinualLoopState(StrEnum):
    """Explicit fail-closed lifecycle state machine for continual qualification."""

    OBSERVING = "OBSERVING"
    COLLECTING_FALLBACKS = "COLLECTING_FALLBACKS"
    REPLAY_CURATED = "REPLAY_CURATED"
    ADAPTING = "ADAPTING"
    SHADOW_QUALIFYING = "SHADOW_QUALIFYING"
    PROMOTED = "PROMOTED"
    ROLLED_BACK = "ROLLED_BACK"


CONTINUAL_LOOP_MACHINE: Final[StateMachineSpec[ContinualLoopState]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.training.continual_clpm",
            description=(
                "Fail-closed continual qualification loop (CLPM): "
                "OBSERVING -> COLLECTING_FALLBACKS -> REPLAY_CURATED -> ADAPTING -> "
                "SHADOW_QUALIFYING -> PROMOTED | ROLLED_BACK."
            ),
            states=tuple(ContinualLoopState),
            initial_states=(ContinualLoopState.OBSERVING,),
            terminal_states=(
                ContinualLoopState.PROMOTED,
                ContinualLoopState.ROLLED_BACK,
            ),
            transitions=(
                TransitionSpec(
                    ContinualLoopState.OBSERVING,
                    ContinualLoopState.COLLECTING_FALLBACKS,
                    "collect_fallbacks",
                    "Ingest verified fallback/OOD events from SignedAuditLedger.",
                ),
                TransitionSpec(
                    ContinualLoopState.COLLECTING_FALLBACKS,
                    ContinualLoopState.REPLAY_CURATED,
                    "curate_replay",
                    "Merge solved fallbacks with replay-protected TRAIN buffer.",
                ),
                TransitionSpec(
                    ContinualLoopState.COLLECTING_FALLBACKS,
                    ContinualLoopState.ROLLED_BACK,
                    "abort_no_fallbacks",
                    "No eligible fallback cases or ledger integrity failed.",
                ),
                TransitionSpec(
                    ContinualLoopState.REPLAY_CURATED,
                    ContinualLoopState.ADAPTING,
                    "start_shadow_adapt",
                    "Adapt isolated shadow checkpoint on curated replay batch.",
                ),
                TransitionSpec(
                    ContinualLoopState.ADAPTING,
                    ContinualLoopState.SHADOW_QUALIFYING,
                    "evaluate_shadow",
                    "Evaluate shadow candidate on frozen blind HOLDOUT + fallback cases.",
                ),
                TransitionSpec(
                    ContinualLoopState.SHADOW_QUALIFYING,
                    ContinualLoopState.PROMOTED,
                    "promote_shadow",
                    "Zero holdout regression confirmed; atomically promote shadow checkpoint.",
                ),
                TransitionSpec(
                    ContinualLoopState.SHADOW_QUALIFYING,
                    ContinualLoopState.ROLLED_BACK,
                    "rollback_regression",
                    "Holdout regression or insufficient improvement detected; roll back.",
                ),
            ),
        )
    )
)


@dataclass(frozen=True, slots=True)
class ContinualQualificationConfig:
    """Policy gates governing shadow adaptation and atomic promotion."""

    min_fallbacks_to_trigger: int = 1
    max_holdout_regression_tol: float = 1.0e-6
    max_allowed_holdout_error: float = 0.25
    require_fallback_improvement: bool = True
    replay_max_records: int = 64

    def __post_init__(self) -> None:
        if self.min_fallbacks_to_trigger < 1:
            raise ValueError("min_fallbacks_to_trigger must be >= 1")
        if self.max_holdout_regression_tol < 0.0:
            raise ValueError("max_holdout_regression_tol must be >= 0")


@dataclass(frozen=True, slots=True)
class ShadowEvaluationMetrics:
    """Side-by-side active vs shadow metrics on frozen blind HOLDOUT & new fallbacks."""

    holdout_count: int
    fallback_count: int
    holdout_error_active: float
    holdout_error_shadow: float
    fallback_error_active: float
    fallback_error_shadow: float
    holdout_regressed: bool
    improved_on_fallbacks: bool

    def document(self) -> dict[str, Any]:
        return {
            "holdout_count": self.holdout_count,
            "fallback_count": self.fallback_count,
            "holdout_error_active": float(self.holdout_error_active),
            "holdout_error_shadow": float(self.holdout_error_shadow),
            "fallback_error_active": float(self.fallback_error_active),
            "fallback_error_shadow": float(self.fallback_error_shadow),
            "holdout_regressed": self.holdout_regressed,
            "improved_on_fallbacks": self.improved_on_fallbacks,
        }


@dataclass(frozen=True, slots=True)
class ContinualCycleResult:
    """Auditable outcome of one continual qualification cycle."""

    cycle_id: str
    state: ContinualLoopState
    state_history: tuple[StateSnapshot[ContinualLoopState], ...]
    promoted: bool
    ingested_fallback_count: int
    replay_record_count: int
    shadow_metrics: ShadowEvaluationMetrics | None
    active_checkpoint_dir: str
    shadow_checkpoint_dir: str | None
    receipt_sha256: str
    reason: str
    wall_time_s: float

    def document(self) -> dict[str, Any]:
        return {
            "schema": CONTINUAL_LOOP_SCHEMA,
            "cycle_id": self.cycle_id,
            "state": self.state.value,
            "promoted": self.promoted,
            "ingested_fallback_count": self.ingested_fallback_count,
            "replay_record_count": self.replay_record_count,
            "shadow_metrics": (
                None if self.shadow_metrics is None else self.shadow_metrics.document()
            ),
            "active_checkpoint_dir": self.active_checkpoint_dir,
            "shadow_checkpoint_dir": self.shadow_checkpoint_dir,
            "receipt_sha256": self.receipt_sha256,
            "reason": self.reason,
            "wall_time_s": float(self.wall_time_s),
        }


ShadowAdapterFn = Callable[[Path, Path, Sequence[ShardedRecord3D]], Mapping[str, Any]]
ShadowEvaluatorFn = Callable[[Path, Sequence[ShardedRecord3D]], float]


def _default_copy_adapter(
    active_dir: Path, shadow_dir: Path, replay_records: Sequence[ShardedRecord3D]
) -> dict[str, Any]:
    """Safe default shadow builder cloning active checkpoint and writing adaptation metadata."""
    if shadow_dir.exists():
        shutil.rmtree(shadow_dir)
    shutil.copytree(active_dir, shadow_dir)
    meta = {
        "adapted_from": str(active_dir),
        "replay_record_ids": [r.record_id for r in replay_records],
        "replay_count": len(replay_records),
    }
    (shadow_dir / "continual-adapt.json").write_bytes(canonical_json_bytes(meta))
    return meta


class ContinualQualificationLoop:
    """Fail-closed continual learning orchestrator with frozen holdout gating."""

    def __init__(
        self,
        *,
        active_checkpoint_dir: Path | str,
        audit_ledger: SignedAuditLedger,
        dataset: ShardedMeshDataset3D,
        workdir: Path | str,
        config: ContinualQualificationConfig | None = None,
    ) -> None:
        self.active_checkpoint_dir = Path(active_checkpoint_dir).resolve()
        if not self.active_checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"active checkpoint dir does not exist: {self.active_checkpoint_dir}"
            )
        self.audit_ledger = audit_ledger
        self.dataset = dataset
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.pointer_path = self.workdir / "active-checkpoint.json"
        self.config = ContinualQualificationConfig() if config is None else config
        if not self.pointer_path.is_file():
            self._write_pointer(self.active_checkpoint_dir, cycle_id="bootstrap")

    def _write_pointer(self, ckpt_dir: Path, *, cycle_id: str) -> None:
        tmp = self.pointer_path.with_suffix(".json.tmp")
        tmp.write_bytes(
            canonical_json_bytes(
                {
                    "schema": CONTINUAL_LOOP_SCHEMA,
                    "active_checkpoint_dir": str(ckpt_dir.resolve()),
                    "cycle_id": cycle_id,
                }
            )
        )
        tmp.replace(self.pointer_path)

    def current_active_checkpoint(self) -> Path:
        doc = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        return Path(str(doc["active_checkpoint_dir"])).resolve()

    def status_summary(self) -> dict[str, Any]:
        ledger_valid, _ = self.audit_ledger.verify_chain()
        return {
            "schema": CONTINUAL_LOOP_SCHEMA,
            "active_checkpoint_dir": str(self.current_active_checkpoint()),
            "dataset": self.dataset.summary(),
            "ledger_chain_valid": ledger_valid,
            "ledger_fallback_count": len(self.audit_ledger.query_fallbacks()),
        }

    def run_cycle(
        self,
        cycle_id: str,
        *,
        solved_fallbacks: Sequence[ShardedRecord3D],
        evaluator: ShadowEvaluatorFn,
        adapter: ShadowAdapterFn | None = None,
    ) -> ContinualCycleResult:
        """Execute one full OBSERVING -> ... -> PROMOTED | ROLLED_BACK cycle."""
        t0 = time.perf_counter()
        state = ContinualLoopState.OBSERVING
        hist: list[StateSnapshot[ContinualLoopState]] = [
            CONTINUAL_LOOP_MACHINE.snapshot(state)
        ]

        # 1) Verify SignedAuditLedger integrity and inspect fallback backlog
        CONTINUAL_LOOP_MACHINE.require_transition(
            state, ContinualLoopState.COLLECTING_FALLBACKS
        )
        state = ContinualLoopState.COLLECTING_FALLBACKS
        hist.append(CONTINUAL_LOOP_MACHINE.snapshot(state))

        active_dir = self.current_active_checkpoint()
        ledger_valid, _ = self.audit_ledger.verify_chain()
        ledger_fallbacks = self.audit_ledger.query_fallbacks()

        if not ledger_valid:
            return self._rollback(
                cycle_id=cycle_id,
                state=state,
                hist=hist,
                active_dir=active_dir,
                shadow_dir=None,
                ingested=0,
                replay_count=0,
                metrics=None,
                reason="SignedAuditLedger chain integrity verification failed",
                t0=t0,
            )

        if (
            len(ledger_fallbacks) < self.config.min_fallbacks_to_trigger
            or len(solved_fallbacks) < self.config.min_fallbacks_to_trigger
        ):
            return self._rollback(
                cycle_id=cycle_id,
                state=state,
                hist=hist,
                active_dir=active_dir,
                shadow_dir=None,
                ingested=0,
                replay_count=0,
                metrics=None,
                reason=(
                    f"insufficient fallback events ({len(ledger_fallbacks)} logged, "
                    f"{len(solved_fallbacks)} solved; min={self.config.min_fallbacks_to_trigger})"
                ),
                t0=t0,
            )

        # Ensure all ingested fallback records are strictly marked TRAIN (never HOLDOUT)
        for rec in solved_fallbacks:
            if rec.split is not DatasetSplit.TRAIN:
                raise ValueError(
                    "solved fallback records ingested by continual loop must belong to TRAIN"
                )

        # 2) Append solved fallbacks into ShardedMeshDataset3D (enforces zero leakage)
        self.dataset.append_records(solved_fallbacks)
        self.dataset.verify_integrity()

        prior_train = list(self.dataset.iter_records(split=DatasetSplit.TRAIN))
        holdout_records = list(self.dataset.iter_records(split=DatasetSplit.HOLDOUT))
        if not holdout_records:
            return self._rollback(
                cycle_id=cycle_id,
                state=state,
                hist=hist,
                active_dir=active_dir,
                shadow_dir=None,
                ingested=len(solved_fallbacks),
                replay_count=0,
                metrics=None,
                reason="cannot qualify shadow without frozen blind HOLDOUT records",
                t0=t0,
            )

        replay_batch = prior_train[-self.config.replay_max_records :]
        CONTINUAL_LOOP_MACHINE.require_transition(
            state, ContinualLoopState.REPLAY_CURATED
        )
        state = ContinualLoopState.REPLAY_CURATED
        hist.append(CONTINUAL_LOOP_MACHINE.snapshot(state))

        # 3) Adapt isolated shadow checkpoint (active checkpoint stays untouched!)
        CONTINUAL_LOOP_MACHINE.require_transition(state, ContinualLoopState.ADAPTING)
        state = ContinualLoopState.ADAPTING
        hist.append(CONTINUAL_LOOP_MACHINE.snapshot(state))

        shadow_dir = self.workdir / f"shadow-{cycle_id}"
        adapt_fn = _default_copy_adapter if adapter is None else adapter
        adapt_fn(active_dir, shadow_dir, replay_batch)

        # 4) Shadow qualification on frozen blind HOLDOUT + new fallbacks
        CONTINUAL_LOOP_MACHINE.require_transition(
            state, ContinualLoopState.SHADOW_QUALIFYING
        )
        state = ContinualLoopState.SHADOW_QUALIFYING
        hist.append(CONTINUAL_LOOP_MACHINE.snapshot(state))

        h_active = float(evaluator(active_dir, holdout_records))
        h_shadow = float(evaluator(shadow_dir, holdout_records))
        f_active = float(evaluator(active_dir, solved_fallbacks))
        f_shadow = float(evaluator(shadow_dir, solved_fallbacks))

        holdout_regressed = (
            h_shadow > h_active + self.config.max_holdout_regression_tol
            or h_shadow > self.config.max_allowed_holdout_error
        )
        improved_fb = f_shadow < f_active - 1.0e-12
        metrics = ShadowEvaluationMetrics(
            holdout_count=len(holdout_records),
            fallback_count=len(solved_fallbacks),
            holdout_error_active=h_active,
            holdout_error_shadow=h_shadow,
            fallback_error_active=f_active,
            fallback_error_shadow=f_shadow,
            holdout_regressed=holdout_regressed,
            improved_on_fallbacks=improved_fb,
        )

        # 5) Fail-closed atomic promotion OR rollback
        if holdout_regressed:
            return self._rollback(
                cycle_id=cycle_id,
                state=state,
                hist=hist,
                active_dir=active_dir,
                shadow_dir=shadow_dir,
                ingested=len(solved_fallbacks),
                replay_count=len(replay_batch),
                metrics=metrics,
                reason=(
                    f"ROLLED BACK: frozen HOLDOUT regression detected "
                    f"(shadow={h_shadow:.4g} > active={h_active:.4g} + tol)"
                ),
                t0=t0,
            )

        if self.config.require_fallback_improvement and not improved_fb:
            return self._rollback(
                cycle_id=cycle_id,
                state=state,
                hist=hist,
                active_dir=active_dir,
                shadow_dir=shadow_dir,
                ingested=len(solved_fallbacks),
                replay_count=len(replay_batch),
                metrics=metrics,
                reason=(
                    f"ROLLED BACK: shadow did not improve on new fallbacks "
                    f"(shadow={f_shadow:.4g} >= active={f_active:.4g})"
                ),
                t0=t0,
            )

        # Atomic promotion!
        CONTINUAL_LOOP_MACHINE.require_transition(state, ContinualLoopState.PROMOTED)
        state = ContinualLoopState.PROMOTED
        hist.append(CONTINUAL_LOOP_MACHINE.snapshot(state))
        self._write_pointer(shadow_dir, cycle_id=cycle_id)

        reason = (
            f"PROMOTED shadow {shadow_dir.name}: fallback error {f_active:.4g} -> {f_shadow:.4g}, "
            f"frozen HOLDOUT {h_active:.4g} -> {h_shadow:.4g}"
        )
        receipt = hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema": CONTINUAL_LOOP_SCHEMA,
                    "cycle_id": cycle_id,
                    "state": state.value,
                    "promoted": True,
                    "active_checkpoint_dir": str(shadow_dir),
                    "metrics": metrics.document(),
                }
            )
        ).hexdigest()
        self.audit_ledger.append(
            event_kind=AuditEventKind.VERIFY_RECEIPT,
            bundle_name=shadow_dir.name,
            mode="continual-clpm",
            verdict="PROMOTED",
            inputs={"cycle_id": cycle_id},
            trust_receipt_sha256=receipt,
            metadata={"shadow_checkpoint": str(shadow_dir)},
        )
        return ContinualCycleResult(
            cycle_id=cycle_id,
            state=state,
            state_history=tuple(hist),
            promoted=True,
            ingested_fallback_count=len(solved_fallbacks),
            replay_record_count=len(replay_batch),
            shadow_metrics=metrics,
            active_checkpoint_dir=str(shadow_dir),
            shadow_checkpoint_dir=str(shadow_dir),
            receipt_sha256=receipt,
            reason=reason,
            wall_time_s=time.perf_counter() - t0,
        )

    def _rollback(
        self,
        *,
        cycle_id: str,
        state: ContinualLoopState,
        hist: list[StateSnapshot[ContinualLoopState]],
        active_dir: Path,
        shadow_dir: Path | None,
        ingested: int,
        replay_count: int,
        metrics: ShadowEvaluationMetrics | None,
        reason: str,
        t0: float,
    ) -> ContinualCycleResult:
        CONTINUAL_LOOP_MACHINE.require_transition(state, ContinualLoopState.ROLLED_BACK)
        state = ContinualLoopState.ROLLED_BACK
        hist.append(CONTINUAL_LOOP_MACHINE.snapshot(state))
        receipt = hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema": CONTINUAL_LOOP_SCHEMA,
                    "cycle_id": cycle_id,
                    "state": state.value,
                    "promoted": False,
                    "active_checkpoint_dir": str(active_dir),
                    "reason": reason,
                }
            )
        ).hexdigest()
        self.audit_ledger.append(
            event_kind=AuditEventKind.OOD_REFUSED,
            bundle_name=active_dir.name,
            mode="continual-clpm",
            verdict="ROLLED_BACK",
            inputs={"cycle_id": cycle_id},
            trust_receipt_sha256=receipt,
            metadata={"reason": reason},
        )
        return ContinualCycleResult(
            cycle_id=cycle_id,
            state=state,
            state_history=tuple(hist),
            promoted=False,
            ingested_fallback_count=ingested,
            replay_record_count=replay_count,
            shadow_metrics=metrics,
            active_checkpoint_dir=str(active_dir),
            shadow_checkpoint_dir=None if shadow_dir is None else str(shadow_dir),
            receipt_sha256=receipt,
            reason=reason,
            wall_time_s=time.perf_counter() - t0,
        )


__all__ = (
    "CONTINUAL_LOOP_MACHINE",
    "CONTINUAL_LOOP_SCHEMA",
    "ContinualCycleResult",
    "ContinualLoopState",
    "ContinualQualificationConfig",
    "ContinualQualificationLoop",
    "ShadowEvaluationMetrics",
)
