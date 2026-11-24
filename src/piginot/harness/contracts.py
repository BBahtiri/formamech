"""Zero-heavy-dependency contracts for the FormaMech Universal AI-Agent Harness (V4-0).

Defines capability descriptors, progressive-discovery metadata, budget ceilings,
explicit harness session state machines, and structured self-healing response
envelopes shared across MCP, CLI, HTTP, and multi-provider LLM tool adapters.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from piginot.contracts.canonical import canonical_json_bytes
from piginot.contracts.state_machine import (
    StateMachineSpec,
    StateSnapshot,
    TransitionSpec,
    register_state_machine,
)

HARNESS_CAPABILITY_SCHEMA: Final = "formamech.harness-capability-v1"
HARNESS_ENVELOPE_SCHEMA: Final = "formamech.harness-envelope-v1"
HARNESS_LEDGER_ENTRY_SCHEMA: Final = "formamech.harness-ledger-entry-v1"
HARNESS_SESSION_SCHEMA: Final = "formamech.harness-session-v1"


class CapabilityDomain(StrEnum):
    """Logical capability domains for progressive AI-agent discovery."""

    DISCOVER = "discover"
    CAD = "cad"
    FEM = "fem"
    MULTIPHYSICS = "multiphysics"
    SURROGATE = "surrogate"
    OPTIMIZE = "optimize"
    INGEST_VALIDATE = "ingest_validate"
    EXPORT = "export"
    GOVERNANCE = "governance"


class CostClass(StrEnum):
    """Computational cost tier guiding autonomous agent planning and budgets."""

    ZERO_SOLVE = "ZERO_SOLVE"
    SURROGATE = "SURROGATE"
    SOLVER_ORACLE = "SOLVER_ORACLE"
    TRAINING_JOB = "TRAINING_JOB"


class HarnessSessionState(StrEnum):
    """Explicit lifecycle states for a persistent AI-agent engineering session."""

    INITIALIZED = "INITIALIZED"
    ACTIVE = "ACTIVE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    SUSPENDED = "SUSPENDED"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"


HARNESS_SESSION_MACHINE: Final[StateMachineSpec[HarnessSessionState]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.harness.session",
            description=(
                "Lifecycle for persistent FormaMech AI-agent engineering sessions "
                "with budget enforcement, suspension/resumption, and clean completion."
            ),
            states=tuple(HarnessSessionState),
            initial_states=(HarnessSessionState.INITIALIZED,),
            terminal_states=(
                HarnessSessionState.COMPLETED,
                HarnessSessionState.ABORTED,
            ),
            transitions=(
                TransitionSpec(
                    HarnessSessionState.INITIALIZED,
                    HarnessSessionState.ACTIVE,
                    "invoke",
                    "Start active engineering session on first capability invocation.",
                ),
                TransitionSpec(
                    HarnessSessionState.ACTIVE,
                    HarnessSessionState.ACTIVE,
                    "invoke",
                    "Continue executing capabilities within configured resource budgets.",
                ),
                TransitionSpec(
                    HarnessSessionState.ACTIVE,
                    HarnessSessionState.BUDGET_EXHAUSTED,
                    "exhaust_budget",
                    "Refuse further paid solves/training once hard budget ceiling is reached.",
                ),
                TransitionSpec(
                    HarnessSessionState.BUDGET_EXHAUSTED,
                    HarnessSessionState.BUDGET_EXHAUSTED,
                    "invoke_free",
                    "Allow ZERO_SOLVE inspection/governance calls after compute budget exhaustion.",
                ),
                TransitionSpec(
                    HarnessSessionState.ACTIVE,
                    HarnessSessionState.SUSPENDED,
                    "suspend",
                    "Persist session state cleanly for later cross-process resumption.",
                ),
                TransitionSpec(
                    HarnessSessionState.SUSPENDED,
                    HarnessSessionState.ACTIVE,
                    "resume",
                    "Resume a previously persisted agent session after ledger verification.",
                ),
                TransitionSpec(
                    HarnessSessionState.INITIALIZED,
                    HarnessSessionState.COMPLETED,
                    "complete",
                    "Close an empty session cleanly.",
                ),
                TransitionSpec(
                    HarnessSessionState.ACTIVE,
                    HarnessSessionState.COMPLETED,
                    "complete",
                    "Mark engineering session completed with verified receipts.",
                ),
                TransitionSpec(
                    HarnessSessionState.BUDGET_EXHAUSTED,
                    HarnessSessionState.COMPLETED,
                    "complete",
                    "Close session after exhausting budget.",
                ),
                TransitionSpec(
                    HarnessSessionState.SUSPENDED,
                    HarnessSessionState.COMPLETED,
                    "complete",
                    "Close suspended session.",
                ),
                *(
                    TransitionSpec(
                        st,
                        HarnessSessionState.ABORTED,
                        "abort",
                        "Abort session on unrecoverable failure or explicit user cancellation.",
                    )
                    for st in (
                        HarnessSessionState.INITIALIZED,
                        HarnessSessionState.ACTIVE,
                        HarnessSessionState.BUDGET_EXHAUSTED,
                        HarnessSessionState.SUSPENDED,
                    )
                ),
            ),
        )
    )
)


@dataclass(frozen=True, slots=True)
class HarnessBudget:
    """Hard resource limits enforced across an entire AI-agent session."""

    max_fem_solves: int = 32
    max_dofs: int = 150_000
    max_training_epochs: int = 400
    max_wall_time_s: float = 1_800.0
    require_qualified_for_export: bool = False

    def __post_init__(self) -> None:
        if type(self.max_fem_solves) is not int or self.max_fem_solves < 0:
            raise ValueError("max_fem_solves must be an integer >= 0")
        if type(self.max_dofs) is not int or self.max_dofs < 2:
            raise ValueError("max_dofs must be an integer >= 2")
        if type(self.max_training_epochs) is not int or self.max_training_epochs < 0:
            raise ValueError("max_training_epochs must be an integer >= 0")
        if not math.isfinite(self.max_wall_time_s) or self.max_wall_time_s <= 0.0:
            raise ValueError("max_wall_time_s must be positive and finite")

    def document(self) -> dict[str, Any]:
        return {
            "max_fem_solves": self.max_fem_solves,
            "max_dofs": self.max_dofs,
            "max_training_epochs": self.max_training_epochs,
            "max_wall_time_s": float(self.max_wall_time_s),
            "require_qualified_for_export": self.require_qualified_for_export,
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> HarnessBudget:
        return cls(
            max_fem_solves=int(doc.get("max_fem_solves", 32)),
            max_dofs=int(doc.get("max_dofs", 150_000)),
            max_training_epochs=int(doc.get("max_training_epochs", 400)),
            max_wall_time_s=float(doc.get("max_wall_time_s", 1_800.0)),
            require_qualified_for_export=bool(
                doc.get("require_qualified_for_export", False)
            ),
        )


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """Declarative machine-readable specification for one FormaMech agent capability."""

    name: str
    domain: CapabilityDomain
    summary: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] = field(default_factory=lambda: {"type": "object"})
    cost_class: CostClass = CostClass.ZERO_SOLVE
    default_fem_solves: int = 0
    idempotent: bool = True
    epistemic_scope: str = (
        "Numerical verification against declared discrete FEM only unless "
        "an experimental ASME V&V 10 validation dossier is attached."
    )
    tags: tuple[str, ...] = ()
    version: str = "1.0"

    def compact_descriptor(self) -> dict[str, Any]:
        """Minimal token-efficient listing entry for progressive discovery."""
        return {
            "name": self.name,
            "domain": self.domain.value,
            "summary": self.summary,
            "cost_class": self.cost_class.value,
            "default_fem_solves": self.default_fem_solves,
            "tags": list(self.tags),
        }

    def full_descriptor(self) -> dict[str, Any]:
        """Complete JSON-Schema descriptor returned by ``describe_capability``."""
        return {
            "schema": HARNESS_CAPABILITY_SCHEMA,
            "name": self.name,
            "version": self.version,
            "domain": self.domain.value,
            "summary": self.summary,
            "description": self.description,
            "cost_class": self.cost_class.value,
            "default_fem_solves": self.default_fem_solves,
            "idempotent": self.idempotent,
            "epistemic_scope": self.epistemic_scope,
            "tags": list(self.tags),
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
        }


@dataclass(frozen=True, slots=True)
class HarnessEnvelope:
    """Uniform structured result or self-healing refusal returned by every capability."""

    ok: bool
    capability: str
    session_id: str
    invocation_index: int
    dry_run: bool
    result: Mapping[str, Any] | None
    error_code: str | None
    error_message: str | None
    recovery_hint: str
    allowed_next_actions: tuple[str, ...]
    state_snapshot: StateSnapshot[HarnessSessionState]
    budget: Mapping[str, Any]
    receipt_sha256: str | None = None
    artifact_paths: tuple[str, ...] = ()
    wall_time_s: float = 0.0
    ledger_entry_sha256: str | None = None

    def document(self) -> dict[str, Any]:
        return {
            "schema": HARNESS_ENVELOPE_SCHEMA,
            "ok": self.ok,
            "capability": self.capability,
            "session_id": self.session_id,
            "invocation_index": self.invocation_index,
            "dry_run": self.dry_run,
            "result": None if self.result is None else dict(self.result),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "recovery_hint": self.recovery_hint,
            "allowed_next_actions": list(self.allowed_next_actions),
            "state_snapshot": self.state_snapshot.document(),
            "budget": dict(self.budget),
            "receipt_sha256": self.receipt_sha256,
            "artifact_paths": list(self.artifact_paths),
            "wall_time_s": float(self.wall_time_s),
            "ledger_entry_sha256": self.ledger_entry_sha256,
        }

    @property
    def envelope_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()


__all__ = (
    "HARNESS_CAPABILITY_SCHEMA",
    "HARNESS_ENVELOPE_SCHEMA",
    "HARNESS_LEDGER_ENTRY_SCHEMA",
    "HARNESS_SESSION_MACHINE",
    "HARNESS_SESSION_SCHEMA",
    "CapabilityDomain",
    "CapabilitySpec",
    "CostClass",
    "HarnessBudget",
    "HarnessEnvelope",
    "HarnessSessionState",
)
