"""Guarded MCP Agentic Engineering Copilot FSM & budget governance (V3-5).

Defines :class:`AgentWorkflowState`, :data:`AGENT_WORKFLOW_MACHINE`, and
:class:`AgentWorkflowTracker` enforcing fail-closed CAD -> Physics -> Compile ->
Qualify -> Export lifecycles and explicit FEM-solve/DOF budget ceilings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from piginot.contracts.state_machine import (
    StateMachineSpec,
    StateSnapshot,
    TransitionRecord,
    TransitionSpec,
    register_state_machine,
)


class AgentWorkflowState(StrEnum):
    """Explicit engineering copilot lifecycle states for MCP agent workflows."""

    IDLE = "IDLE"
    CAD_IMPORTED = "CAD_IMPORTED"
    SETUP_PROPOSED = "SETUP_PROPOSED"
    PLAN_REVIEWED = "PLAN_REVIEWED"
    COMPILED = "COMPILED"
    QUALIFIED = "QUALIFIED"
    EXPORTED = "EXPORTED"
    REFUSED = "REFUSED"
    CANCELLED = "CANCELLED"


AGENT_WORKFLOW_MACHINE: Final[StateMachineSpec[AgentWorkflowState]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.mcp.agent_workflow",
            description=(
                "Fail-closed engineering copilot lifecycle enforcing CAD import, "
                "physics setup, budgeted compilation, numerical qualification, and "
                "gated artifact export."
            ),
            states=tuple(AgentWorkflowState),
            initial_states=(AgentWorkflowState.IDLE,),
            terminal_states=(AgentWorkflowState.CANCELLED,),
            transitions=(
                TransitionSpec(
                    AgentWorkflowState.IDLE,
                    AgentWorkflowState.CAD_IMPORTED,
                    "import_cad_part",
                    "Import and mesh a 2D/3D CAD part inside the workspace.",
                ),
                TransitionSpec(
                    AgentWorkflowState.CAD_IMPORTED,
                    AgentWorkflowState.CAD_IMPORTED,
                    "import_cad_part",
                    "Re-import or replace CAD geometry before physics setup.",
                ),
                TransitionSpec(
                    AgentWorkflowState.QUALIFIED,
                    AgentWorkflowState.CAD_IMPORTED,
                    "import_cad_part",
                    "Start a new part after completing a qualified workflow.",
                ),
                TransitionSpec(
                    AgentWorkflowState.EXPORTED,
                    AgentWorkflowState.CAD_IMPORTED,
                    "import_cad_part",
                    "Start a new part after exporting a verified artifact.",
                ),
                TransitionSpec(
                    AgentWorkflowState.REFUSED,
                    AgentWorkflowState.CAD_IMPORTED,
                    "import_cad_part",
                    "Recover from a refused run by importing a valid CAD part.",
                ),
                TransitionSpec(
                    AgentWorkflowState.CAD_IMPORTED,
                    AgentWorkflowState.SETUP_PROPOSED,
                    "propose_physics_setup",
                    "Propose and validate boundary conditions and material bounds.",
                ),
                TransitionSpec(
                    AgentWorkflowState.SETUP_PROPOSED,
                    AgentWorkflowState.SETUP_PROPOSED,
                    "propose_physics_setup",
                    "Refine proposed physics setup prior to compilation.",
                ),
                TransitionSpec(
                    AgentWorkflowState.SETUP_PROPOSED,
                    AgentWorkflowState.PLAN_REVIEWED,
                    "review_plan",
                    "Record explicit review of deterministic campaign plan.",
                ),
                TransitionSpec(
                    AgentWorkflowState.SETUP_PROPOSED,
                    AgentWorkflowState.COMPILED,
                    "compile_and_qualify",
                    "Compile candidate model directly from validated setup.",
                ),
                TransitionSpec(
                    AgentWorkflowState.PLAN_REVIEWED,
                    AgentWorkflowState.COMPILED,
                    "compile_and_qualify",
                    "Compile candidate model from reviewed plan.",
                ),
                TransitionSpec(
                    AgentWorkflowState.COMPILED,
                    AgentWorkflowState.QUALIFIED,
                    "qualify_complete",
                    "Numerical qualification passed and emitted qualification evidence.",
                ),
                TransitionSpec(
                    AgentWorkflowState.COMPILED,
                    AgentWorkflowState.COMPILED,
                    "run_active_fem_round",
                    "Run a budgeted active FEM refinement round.",
                ),
                TransitionSpec(
                    AgentWorkflowState.QUALIFIED,
                    AgentWorkflowState.QUALIFIED,
                    "run_active_fem_round",
                    "Run additional budgeted FEM verification checks.",
                ),
                TransitionSpec(
                    AgentWorkflowState.QUALIFIED,
                    AgentWorkflowState.EXPORTED,
                    "export_verified_artifact",
                    "Export verified VTU/GLB/FMU artifact after qualification.",
                ),
                TransitionSpec(
                    AgentWorkflowState.EXPORTED,
                    AgentWorkflowState.EXPORTED,
                    "export_verified_artifact",
                    "Export additional artifact format from the same qualified state.",
                ),
                *(
                    TransitionSpec(
                        st,
                        AgentWorkflowState.REFUSED,
                        "refuse",
                        "Fail-closed refusal due to budget, OOD, or qualification failure.",
                    )
                    for st in (
                        AgentWorkflowState.IDLE,
                        AgentWorkflowState.CAD_IMPORTED,
                        AgentWorkflowState.SETUP_PROPOSED,
                        AgentWorkflowState.PLAN_REVIEWED,
                        AgentWorkflowState.COMPILED,
                        AgentWorkflowState.QUALIFIED,
                    )
                ),
                *(
                    TransitionSpec(
                        st,
                        AgentWorkflowState.CANCELLED,
                        "cancel",
                        "Explicitly cancel engineering copilot session.",
                    )
                    for st in AgentWorkflowState
                    if st is not AgentWorkflowState.CANCELLED
                ),
            ),
        )
    )
)


@dataclass(frozen=True, slots=True)
class AgentBudgetSpec:
    """Hard resource limits enforced on every MCP engineering copilot session."""

    max_fem_solves: int = 16
    max_dofs: int = 100_000
    require_qualified_for_export: bool = True
    require_hitl_signoff: bool = False

    def __post_init__(self) -> None:
        if self.max_fem_solves < 1:
            raise ValueError("max_fem_solves must be >= 1")
        if self.max_dofs < 2:
            raise ValueError("max_dofs must be >= 2")


class AgentWorkflowTracker:
    """Stateful FSM & budget controller embedded in :class:`FormaMechMcpServer`."""

    def __init__(self, budget: AgentBudgetSpec) -> None:
        self.budget = budget
        self.state: AgentWorkflowState = AgentWorkflowState.IDLE
        self.history: list[TransitionRecord[AgentWorkflowState]] = []
        self.fem_solves_used: int = 0
        self.active_workspace_rel: str | None = None
        self.last_qualification_sha256: str | None = None
        self.last_receipt_sha256: str | None = None
        self.last_fem_result: Any = None

    @property
    def fem_solves_remaining(self) -> int:
        return max(0, self.budget.max_fem_solves - self.fem_solves_used)

    def transition(
        self,
        target: AgentWorkflowState,
        *,
        trigger: str,
        reason: str = "",
    ) -> TransitionRecord[AgentWorkflowState]:
        rec = AGENT_WORKFLOW_MACHINE.require_transition(
            self.state, target, trigger=trigger, reason=reason
        )
        self.state = target
        self.history.append(rec)
        return rec

    def reserve_fem_solves(self, count: int) -> None:
        """Reserve ``count`` FEM solves against the hard session budget."""
        if count < 0:
            raise ValueError("FEM solve count must be nonnegative")
        if self.fem_solves_used + count > self.budget.max_fem_solves:
            from piginot.mcp_server import ResourceLimitError

            raise ResourceLimitError(
                f"FEM solve budget exhausted: requested {count}, "
                f"used {self.fem_solves_used}/{self.budget.max_fem_solves}"
            )
        self.fem_solves_used += count

    def check_dofs(self, n_dofs: int) -> None:
        if n_dofs > self.budget.max_dofs:
            from piginot.mcp_server import ResourceLimitError

            raise ResourceLimitError(
                f"mesh requires {n_dofs} DOFs, exceeding max_dof_limit={self.budget.max_dofs}"
            )

    def snapshot(self) -> StateSnapshot[AgentWorkflowState]:
        hints = {
            AgentWorkflowState.IDLE: "Call import_cad_part to load a 2D or 3D CAD model.",
            AgentWorkflowState.CAD_IMPORTED: "Call propose_physics_setup to assign BCs and material.",
            AgentWorkflowState.SETUP_PROPOSED: "Call compile_and_qualify to run budgeted verification.",
            AgentWorkflowState.PLAN_REVIEWED: "Call compile_and_qualify to execute the reviewed plan.",
            AgentWorkflowState.COMPILED: "Complete qualification or run_active_fem_round.",
            AgentWorkflowState.QUALIFIED: "Call export_verified_artifact or run_experimental_validation.",
            AgentWorkflowState.EXPORTED: "Verified artifact exported; inspect or start a new CAD part.",
            AgentWorkflowState.REFUSED: "Workflow refused; import a valid CAD part to retry.",
            AgentWorkflowState.CANCELLED: "Workflow cancelled.",
        }
        return AGENT_WORKFLOW_MACHINE.snapshot(
            self.state,
            recommended_hint=hints[self.state],
            metadata={
                "fem_solves_used": self.fem_solves_used,
                "fem_solves_remaining": self.fem_solves_remaining,
                "max_fem_solves": self.budget.max_fem_solves,
                "max_dofs": self.budget.max_dofs,
                "active_workspace": self.active_workspace_rel,
                "last_qualification_sha256": self.last_qualification_sha256,
                "last_receipt_sha256": self.last_receipt_sha256,
            },
        )


__all__ = (
    "AGENT_WORKFLOW_MACHINE",
    "AgentBudgetSpec",
    "AgentWorkflowState",
    "AgentWorkflowTracker",
)
