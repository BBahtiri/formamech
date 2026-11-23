"""Zero-dependency, self-describing finite state machine contracts for FormaMech.

This module provides a single typed foundation for explicit state machines across
UI workflows, background jobs, serving funnels, export lifecycles, object stores,
qualification gates, and agent/MCP protocols:

- **DX (Developer Experience):** Declarative ``StateMachineSpec`` transition tables,
  immutable terminal states, exhaustive ``StrEnum`` vocabularies, and structured
  ``InvalidTransitionError`` diagnostics checked under ``mypy --strict``.
- **UX (User Experience):** Live ``StateSnapshot`` objects carrying current state,
  blockers, stale artifacts, allowed next transitions, and recovery hints.
- **AX (Agent Experience):** Deterministic machine-readable ``describe()`` schemas,
  Mermaid ``stateDiagram-v2`` rendering, and a central introspection registry.

Uses only the Python standard library so metadata-only, ``.[runtime]``, training,
and verification packages share one identity with zero heavy dependency leakage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final, Generic, TypeVar

STATE_MACHINE_SCHEMA: Final = "formamech.state-machine-spec-v1"
STATE_SNAPSHOT_SCHEMA: Final = "formamech.state-snapshot-v1"

StateT = TypeVar("StateT", bound=StrEnum)


class CheckStatus(StrEnum):
    """Outcome of one verification check.

    ``NOT_RUN`` is never a pass, and a ``SKIPPED`` required check is not a pass
    either; only ``PASS`` counts towards a claim.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    NOT_RUN = "NOT_RUN"
    SKIPPED = "SKIPPED"


class ClaimLevel(StrEnum):
    """Claim levels this code can emit, weakest first.

    ``HOLDOUT_VERIFIED`` is the ceiling: no stronger level exists here.
    """

    NOT_VERIFIED = "NOT_VERIFIED"
    DEVELOPMENT_VERIFIED = "DEVELOPMENT_VERIFIED"
    HOLDOUT_VERIFIED = "HOLDOUT_VERIFIED"

    @property
    def rank(self) -> int:
        """Monotonic ordinal rank (0 = NOT_VERIFIED, 1 = DEVELOPMENT, 2 = HOLDOUT)."""
        return _CLAIM_LEVEL_RANK[self]

    def satisfies(self, minimum: StrEnum) -> bool:
        """Return True iff this claim level is at least as strong as ``minimum``."""
        if not isinstance(minimum, StrEnum) or minimum.value not in _CLAIM_LEVEL_VALUES:
            raise TypeError(f"minimum must be a ClaimLevel, got {type(minimum)!r}")
        return self.rank >= _CLAIM_LEVEL_RANK[ClaimLevel(minimum.value)]


_CLAIM_LEVEL_RANK: Final[dict[ClaimLevel, int]] = {
    ClaimLevel.NOT_VERIFIED: 0,
    ClaimLevel.DEVELOPMENT_VERIFIED: 1,
    ClaimLevel.HOLDOUT_VERIFIED: 2,
}

ALLOWED_CLAIM_LEVELS: Final[tuple[str, ...]] = tuple(
    level.value for level in ClaimLevel
)
_CLAIM_LEVEL_VALUES: Final[frozenset[str]] = frozenset(ALLOWED_CLAIM_LEVELS)


def claim_level_rank(level: StrEnum) -> int:
    """Monotonic ordinal rank for any ``ClaimLevel`` enum member."""
    if not isinstance(level, StrEnum) or level.value not in _CLAIM_LEVEL_VALUES:
        raise TypeError(f"level must be a ClaimLevel, got {type(level)!r}")
    return _CLAIM_LEVEL_RANK[ClaimLevel(level.value)]


class InvalidTransitionError(ValueError):
    """Raised when a state machine rejects an illegal state transition."""

    def __init__(
        self,
        *,
        machine_id: str,
        current_state: str,
        target_state: str,
        allowed_targets: Sequence[str],
        trigger: str = "",
        reason: str = "",
    ) -> None:
        self.machine_id = str(machine_id)
        self.current_state = str(current_state)
        self.target_state = str(target_state)
        self.trigger = str(trigger)
        self.allowed_targets = tuple(str(t) for t in allowed_targets)
        self.reason = str(reason)
        allowed_text = (
            ", ".join(self.allowed_targets)
            if self.allowed_targets
            else "<none (terminal)>"
        )
        trigger_part = f" via {self.trigger!r}" if self.trigger else ""
        reason_part = f": {self.reason}" if self.reason else ""
        super().__init__(
            f"[{self.machine_id}] invalid transition "
            f"{self.current_state!r} -> {self.target_state!r}{trigger_part} "
            f"(allowed targets: {allowed_text}){reason_part}"
        )

    def document(self) -> dict[str, Any]:
        """Structured JSON-safe payload for CLI, UI, and MCP error reporting."""
        return {
            "machine_id": self.machine_id,
            "current_state": self.current_state,
            "target_state": self.target_state,
            "trigger": self.trigger,
            "allowed_targets": list(self.allowed_targets),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TransitionSpec(Generic[StateT]):
    """One legal directed transition in a finite state machine."""

    source: StateT
    target: StateT
    trigger: str
    description: str = ""
    guard_summary: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source, StrEnum):
            raise TypeError("source must be a StrEnum state")
        if not isinstance(self.target, StrEnum):
            raise TypeError("target must be a StrEnum state")
        if type(self.source) is not type(self.target):
            raise TypeError("source and target must belong to the same StrEnum")
        if not isinstance(self.trigger, str) or not self.trigger.strip():
            raise ValueError("trigger must be non-empty text")

    def document(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "target": self.target.value,
            "trigger": self.trigger,
            "description": self.description,
            "guard_summary": self.guard_summary,
        }


@dataclass(frozen=True, slots=True)
class TransitionRecord(Generic[StateT]):
    """Immutable audit record produced when a legal transition is validated."""

    machine_id: str
    from_state: StateT
    to_state: StateT
    trigger: str
    reason: str = ""

    def document(self) -> dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "trigger": self.trigger,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class StateSnapshot(Generic[StateT]):
    """Self-describing runtime state snapshot for UX, DX, and AX consumers."""

    machine_id: str
    state: StateT
    terminal: bool
    allowed_transitions: tuple[TransitionSpec[StateT], ...]
    blockers: tuple[str, ...] = ()
    stale_items: tuple[str, ...] = ()
    recommended_trigger: str | None = None
    recommended_hint: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @property
    def next_triggers(self) -> tuple[str, ...]:
        """Distinct legal trigger names from the current state in declaration order."""
        seen: dict[str, None] = {}
        for tr in self.allowed_transitions:
            seen[tr.trigger] = None
        return tuple(seen)

    def document(self) -> dict[str, Any]:
        return {
            "schema": STATE_SNAPSHOT_SCHEMA,
            "machine_id": self.machine_id,
            "state": self.state.value,
            "terminal": self.terminal,
            "allowed_transitions": [tr.document() for tr in self.allowed_transitions],
            "next_triggers": list(self.next_triggers),
            "blockers": list(self.blockers),
            "stale_items": list(self.stale_items),
            "recommended_trigger": self.recommended_trigger,
            "recommended_hint": self.recommended_hint,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class StateMachineSpec(Generic[StateT]):
    """Declarative, validated finite state machine specification."""

    machine_id: str
    states: tuple[StateT, ...]
    initial_states: tuple[StateT, ...]
    terminal_states: tuple[StateT, ...]
    transitions: tuple[TransitionSpec[StateT], ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.machine_id, str) or not self.machine_id.strip():
            raise ValueError("machine_id must be non-empty text")
        if not self.states:
            raise ValueError("states must be non-empty")
        state_type = type(self.states[0])
        if not issubclass(state_type, StrEnum):
            raise TypeError("states must be StrEnum members")
        if len(set(self.states)) != len(self.states):
            raise ValueError("states must be unique")
        state_set = frozenset(self.states)
        if not self.initial_states:
            raise ValueError("initial_states must be non-empty")
        for s in self.initial_states:
            if s not in state_set:
                raise ValueError(f"initial state {s!r} is not in declared states")
        terminal_set = frozenset(self.terminal_states)
        for s in self.terminal_states:
            if s not in state_set:
                raise ValueError(f"terminal state {s!r} is not in declared states")
        seen_edges: set[tuple[StateT, StateT, str]] = set()
        for tr in self.transitions:
            if tr.source not in state_set or tr.target not in state_set:
                raise ValueError(
                    f"transition {tr.source!r} -> {tr.target!r} references undeclared state"
                )
            if tr.source in terminal_set:
                raise ValueError(
                    f"terminal state {tr.source.value!r} cannot have outgoing transition {tr.trigger!r}"
                )
            edge = (tr.source, tr.target, tr.trigger)
            if edge in seen_edges:
                raise ValueError(f"duplicate transition edge {edge!r}")
            seen_edges.add(edge)

    def is_terminal(self, state: StateT) -> bool:
        """Return True iff ``state`` is a declared terminal state."""
        return state in self.terminal_states

    def allowed_transitions(
        self, current: StateT
    ) -> tuple[TransitionSpec[StateT], ...]:
        """Return all legal outgoing transitions from ``current``."""
        if current not in self.states:
            raise ValueError(f"unknown state {current!r} for {self.machine_id!r}")
        return tuple(tr for tr in self.transitions if tr.source is current)

    def allowed_targets(self, current: StateT) -> tuple[StateT, ...]:
        """Return distinct legal target states reachable in one step from ``current``."""
        seen: dict[StateT, None] = {}
        for tr in self.allowed_transitions(current):
            seen[tr.target] = None
        return tuple(seen)

    def can_transition(
        self,
        current: StateT,
        target: StateT,
        *,
        trigger: str | None = None,
    ) -> bool:
        """Return True iff transitioning ``current -> target`` (optionally via ``trigger``) is legal."""
        for tr in self.allowed_transitions(current):
            if tr.target is target and (trigger is None or tr.trigger == trigger):
                return True
        return False

    def require_transition(
        self,
        current: StateT,
        target: StateT,
        *,
        trigger: str = "",
        reason: str = "",
    ) -> TransitionRecord[StateT]:
        """Validate ``current -> target`` and return an audit record or raise."""
        matching = [
            tr
            for tr in self.allowed_transitions(current)
            if tr.target is target and (not trigger or tr.trigger == trigger)
        ]
        if not matching:
            raise InvalidTransitionError(
                machine_id=self.machine_id,
                current_state=current.value,
                target_state=target.value,
                allowed_targets=[s.value for s in self.allowed_targets(current)],
                trigger=trigger,
                reason=reason,
            )
        chosen = matching[0]
        return TransitionRecord(
            machine_id=self.machine_id,
            from_state=current,
            to_state=target,
            trigger=trigger or chosen.trigger,
            reason=reason,
        )

    def snapshot(
        self,
        current: StateT,
        *,
        blockers: Sequence[str] = (),
        stale_items: Sequence[str] = (),
        recommended_trigger: str | None = None,
        recommended_hint: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> StateSnapshot[StateT]:
        """Build a live, self-describing state snapshot for UX/DX/AX consumers."""
        allowed = self.allowed_transitions(current)
        rec_trigger = recommended_trigger
        if rec_trigger is None and allowed and not blockers:
            rec_trigger = allowed[0].trigger
        return StateSnapshot(
            machine_id=self.machine_id,
            state=current,
            terminal=self.is_terminal(current),
            allowed_transitions=allowed,
            blockers=tuple(str(b) for b in blockers),
            stale_items=tuple(str(item) for item in stale_items),
            recommended_trigger=rec_trigger,
            recommended_hint=str(recommended_hint),
            metadata={} if metadata is None else dict(metadata),
        )

    def describe(self) -> dict[str, Any]:
        """Machine-readable schema describing the full FSM topology."""
        return {
            "schema": STATE_MACHINE_SCHEMA,
            "machine_id": self.machine_id,
            "description": self.description,
            "states": [s.value for s in self.states],
            "initial_states": [s.value for s in self.initial_states],
            "terminal_states": [s.value for s in self.terminal_states],
            "transitions": [tr.document() for tr in self.transitions],
        }

    def to_mermaid(self) -> str:
        """Render a deterministic Mermaid ``stateDiagram-v2`` diagram."""
        lines = ["stateDiagram-v2"]
        for init in self.initial_states:
            lines.append(f"    [*] --> {init.value}")
        for tr in self.transitions:
            label = tr.trigger
            lines.append(f"    {tr.source.value} --> {tr.target.value}: {label}")
        for term in self.terminal_states:
            lines.append(f"    {term.value} --> [*]")
        return "\n".join(lines)


_REGISTRY: dict[str, StateMachineSpec[Any]] = {}


def register_state_machine(spec: StateMachineSpec[StateT]) -> StateMachineSpec[StateT]:
    """Register ``spec`` in the process-wide introspection registry (idempotent)."""
    existing = _REGISTRY.get(spec.machine_id)
    if existing is not None and existing != spec:
        raise ValueError(
            f"conflicting state machine already registered for {spec.machine_id!r}"
        )
    _REGISTRY[spec.machine_id] = spec
    return spec


def get_state_machine(machine_id: str) -> StateMachineSpec[Any]:
    """Retrieve a registered :class:`StateMachineSpec` by ``machine_id``."""
    try:
        return _REGISTRY[machine_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown state machine {machine_id!r}; registered: {sorted(_REGISTRY)}"
        ) from exc


def registered_state_machines() -> Mapping[str, StateMachineSpec[Any]]:
    """Return a read-only view of all currently registered state machine specs."""
    return MappingProxyType(dict(sorted(_REGISTRY.items())))


__all__ = (
    "ALLOWED_CLAIM_LEVELS",
    "STATE_MACHINE_SCHEMA",
    "STATE_SNAPSHOT_SCHEMA",
    "CheckStatus",
    "ClaimLevel",
    "InvalidTransitionError",
    "StateMachineSpec",
    "StateSnapshot",
    "TransitionRecord",
    "TransitionSpec",
    "claim_level_rank",
    "get_state_machine",
    "register_state_machine",
    "registered_state_machines",
)
