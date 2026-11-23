"""Immutable public models and explicit lifecycle state machines for content retention."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from piginot.contracts.state_machine import (
    StateMachineSpec,
    TransitionSpec,
    register_state_machine,
)


class RetentionTier(StrEnum):
    """Retention policy for stored object bytes."""

    PERMANENT = "PERMANENT"
    RETAINED = "RETAINED"
    PRUNABLE = "PRUNABLE"


class ContentLifecycleState(StrEnum):
    """Explicit valid states of a content-addressed object (no paradoxical states)."""

    UNCOMMITTED = "UNCOMMITTED"
    PRUNABLE_MATERIALIZED = "PRUNABLE_MATERIALIZED"
    RETAINED_MATERIALIZED = "RETAINED_MATERIALIZED"
    PERMANENT_MATERIALIZED = "PERMANENT_MATERIALIZED"
    PRUNABLE_PRUNED = "PRUNABLE_PRUNED"


class RecipeState(StrEnum):
    """Explicit 3-state monotonic divergence lattice for recipe resolutions."""

    UNRECORDED = "UNRECORDED"
    RESOLVED = "RESOLVED"
    DIVERGENT = "DIVERGENT"


STORE_CONTENT_MACHINE: Final[StateMachineSpec[ContentLifecycleState]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.store.content",
            description=(
                "Content-addressed ObjectStore object lifecycle enforcing "
                "PERMANENT irreversibility and forbidding unmaterialized "
                "RETAINED/PERMANENT paradoxical states."
            ),
            states=tuple(ContentLifecycleState),
            initial_states=(ContentLifecycleState.UNCOMMITTED,),
            terminal_states=(),
            transitions=(
                TransitionSpec(
                    ContentLifecycleState.UNCOMMITTED,
                    ContentLifecycleState.PRUNABLE_MATERIALIZED,
                    "put_prunable",
                    "Commit new bytes with PRUNABLE retention.",
                ),
                TransitionSpec(
                    ContentLifecycleState.UNCOMMITTED,
                    ContentLifecycleState.RETAINED_MATERIALIZED,
                    "put_retained",
                    "Commit new bytes with RETAINED retention.",
                ),
                TransitionSpec(
                    ContentLifecycleState.UNCOMMITTED,
                    ContentLifecycleState.PERMANENT_MATERIALIZED,
                    "put_permanent",
                    "Commit new bytes with PERMANENT retention.",
                ),
                TransitionSpec(
                    ContentLifecycleState.PRUNABLE_MATERIALIZED,
                    ContentLifecycleState.PRUNABLE_MATERIALIZED,
                    "set_retention_prunable",
                    "Idempotent retention confirmation.",
                ),
                TransitionSpec(
                    ContentLifecycleState.PRUNABLE_MATERIALIZED,
                    ContentLifecycleState.RETAINED_MATERIALIZED,
                    "promote_retained",
                    "Promote materialized object to RETAINED.",
                ),
                TransitionSpec(
                    ContentLifecycleState.PRUNABLE_MATERIALIZED,
                    ContentLifecycleState.PERMANENT_MATERIALIZED,
                    "promote_permanent",
                    "Irreversibly promote materialized object to PERMANENT.",
                ),
                TransitionSpec(
                    ContentLifecycleState.PRUNABLE_MATERIALIZED,
                    ContentLifecycleState.PRUNABLE_PRUNED,
                    "prune",
                    "Delete PRUNABLE object bytes while retaining identity metadata.",
                ),
                TransitionSpec(
                    ContentLifecycleState.RETAINED_MATERIALIZED,
                    ContentLifecycleState.RETAINED_MATERIALIZED,
                    "set_retention_retained",
                    "Idempotent retention confirmation.",
                ),
                TransitionSpec(
                    ContentLifecycleState.RETAINED_MATERIALIZED,
                    ContentLifecycleState.PRUNABLE_MATERIALIZED,
                    "downgrade_prunable",
                    "Downgrade RETAINED object to PRUNABLE.",
                ),
                TransitionSpec(
                    ContentLifecycleState.RETAINED_MATERIALIZED,
                    ContentLifecycleState.PERMANENT_MATERIALIZED,
                    "promote_permanent",
                    "Irreversibly promote RETAINED object to PERMANENT.",
                ),
                TransitionSpec(
                    ContentLifecycleState.PERMANENT_MATERIALIZED,
                    ContentLifecycleState.PERMANENT_MATERIALIZED,
                    "set_retention_permanent",
                    "Idempotent PERMANENT confirmation (downgrade forbidden).",
                ),
                TransitionSpec(
                    ContentLifecycleState.PRUNABLE_PRUNED,
                    ContentLifecycleState.PRUNABLE_PRUNED,
                    "set_retention_prunable",
                    "Idempotent PRUNABLE confirmation while pruned.",
                ),
                TransitionSpec(
                    ContentLifecycleState.PRUNABLE_PRUNED,
                    ContentLifecycleState.PRUNABLE_MATERIALIZED,
                    "rehydrate_prunable",
                    "Re-commit identical bytes to rematerialize as PRUNABLE.",
                ),
                TransitionSpec(
                    ContentLifecycleState.PRUNABLE_PRUNED,
                    ContentLifecycleState.RETAINED_MATERIALIZED,
                    "rehydrate_retained",
                    "Re-commit identical bytes to rematerialize and promote to RETAINED.",
                ),
                TransitionSpec(
                    ContentLifecycleState.PRUNABLE_PRUNED,
                    ContentLifecycleState.PERMANENT_MATERIALIZED,
                    "rehydrate_permanent",
                    "Re-commit identical bytes to rematerialize and promote to PERMANENT.",
                ),
            ),
        )
    )
)


STORE_RECIPE_MACHINE: Final[StateMachineSpec[RecipeState]] = register_state_machine(
    StateMachineSpec(
        machine_id="formamech.store.recipe",
        description=(
            "Monotonic recipe-to-content divergence lattice: UNRECORDED -> "
            "RESOLVED (singleton) -> DIVERGENT (absorbing evidence state)."
        ),
        states=tuple(RecipeState),
        initial_states=(RecipeState.UNRECORDED,),
        terminal_states=(),
        transitions=(
            TransitionSpec(
                RecipeState.UNRECORDED,
                RecipeState.RESOLVED,
                "record_first_result",
                "Bind recipe hash to its first committed content hash.",
            ),
            TransitionSpec(
                RecipeState.RESOLVED,
                RecipeState.RESOLVED,
                "record_identical_result",
                "Idempotently re-record the exact same content hash.",
            ),
            TransitionSpec(
                RecipeState.RESOLVED,
                RecipeState.DIVERGENT,
                "record_conflicting_result",
                "Persist conflicting content hash and mark recipe DIVERGENT.",
            ),
            TransitionSpec(
                RecipeState.DIVERGENT,
                RecipeState.DIVERGENT,
                "record_additional_result",
                "Append additional divergence evidence (never resolves back).",
            ),
        ),
    )
)


@dataclass(frozen=True, slots=True)
class ContentInfo:
    """Committed identity and materialization state for one content hash."""

    content_hash: str
    size_bytes: int
    retention: RetentionTier
    materialized: bool

    def __post_init__(self) -> None:
        if not self.materialized and self.retention is not RetentionTier.PRUNABLE:
            raise ValueError(
                f"invalid unmaterialized retention state "
                f"({self.retention.value}, materialized=False); only PRUNABLE "
                "objects may be unmaterialized"
            )

    @property
    def lifecycle_state(self) -> ContentLifecycleState:
        """Return the explicit :class:`ContentLifecycleState` of this object."""
        if not self.materialized:
            return ContentLifecycleState.PRUNABLE_PRUNED
        if self.retention is RetentionTier.PERMANENT:
            return ContentLifecycleState.PERMANENT_MATERIALIZED
        if self.retention is RetentionTier.RETAINED:
            return ContentLifecycleState.RETAINED_MATERIALIZED
        return ContentLifecycleState.PRUNABLE_MATERIALIZED


@dataclass(frozen=True, slots=True)
class PruneResult:
    """Deterministic summary of one prune operation."""

    pruned_hashes: tuple[str, ...]
    pruned_bytes: int
    dry_run: bool

    @property
    def object_count(self) -> int:
        """Return the number of selected content objects."""

        return len(self.pruned_hashes)
