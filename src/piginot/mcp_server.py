"""Model Context Protocol (MCP) server for verified FormaMech mechanics workflows.

Exposes FormaMech's envelope-bounded inference, CAD inspection, guarded design
optimization, and RFC 8785 Trust Receipt verification as structured JSON-RPC
2.0 / MCP tools with strict workspace path sandboxing, DOF ceiling guards,
an explicit session finite state machine, and runtime JSON Schema validation.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, TextIO, cast

from piginot import __version__
from piginot.certify import verify_receipt_document
from piginot.contracts.state_machine import (
    InvalidTransitionError,
    StateMachineSpec,
    StateSnapshot,
    TransitionRecord,
    TransitionSpec,
    register_state_machine,
)
from piginot.mcp_agent_workflows import (
    AGENT_WORKFLOW_MACHINE,
    AgentBudgetSpec,
    AgentWorkflowState,
    AgentWorkflowTracker,
)

MCP_PROTOCOL_VERSION: Final = "2024-11-05"
SERVER_NAME: Final = "formamech-mcp"
DEFAULT_MAX_DOF_LIMIT: Final = 100_000


@contextmanager
def _protocol_stdio_guard(stdout: TextIO | None) -> Iterator[TextIO]:
    """Keep native C/C++ stdout (e.g. OpenCASCADE) out of the JSON-RPC stdio stream."""
    if stdout is not None:
        yield stdout
        return
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield sys.stdout
        return
    sys.stdout.flush()
    sys.stderr.flush()
    proto_fd = os.dup(stdout_fd)
    proto_stream = os.fdopen(proto_fd, "w", encoding="utf-8", buffering=1)
    saved_py_stdout = sys.stdout
    try:
        os.dup2(stderr_fd, stdout_fd)
        sys.stdout = sys.stderr
        yield proto_stream
    finally:
        try:
            sys.stdout.flush()
            proto_stream.flush()
        finally:
            os.dup2(proto_fd, stdout_fd)
            sys.stdout = saved_py_stdout
            proto_stream.close()


@contextmanager
def redirect_native_stdout_to_stderr() -> Iterator[None]:
    """Temporarily route OS-level FD 1 (C/C++ stdout) and ``sys.stdout`` to ``stderr``."""
    with _protocol_stdio_guard(None):
        yield


class McpSessionState(StrEnum):
    """Explicit JSON-RPC 2.0 / MCP protocol handshake lifecycle states."""

    UNINITIALIZED = "UNINITIALIZED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    CLOSED = "CLOSED"


MCP_SESSION_MACHINE: Final[StateMachineSpec[McpSessionState]] = register_state_machine(
    StateMachineSpec(
        machine_id="formamech.mcp.session",
        description=(
            "MCP JSON-RPC 2.0 protocol lifecycle from UNINITIALIZED through "
            "initialize handshake to READY and terminal CLOSED."
        ),
        states=tuple(McpSessionState),
        initial_states=(McpSessionState.UNINITIALIZED,),
        terminal_states=(McpSessionState.CLOSED,),
        transitions=(
            TransitionSpec(
                McpSessionState.UNINITIALIZED,
                McpSessionState.INITIALIZING,
                "initialize",
                "Client sent JSON-RPC initialize request.",
            ),
            TransitionSpec(
                McpSessionState.INITIALIZING,
                McpSessionState.INITIALIZING,
                "initialize",
                "Idempotent re-initialization before notifications/initialized.",
            ),
            TransitionSpec(
                McpSessionState.INITIALIZING,
                McpSessionState.READY,
                "notifications/initialized",
                "Client confirmed initialization complete; tools ready.",
            ),
            TransitionSpec(
                McpSessionState.READY,
                McpSessionState.READY,
                "serve_request",
                "Process ping, tools/list, or tools/call while session is READY.",
            ),
            TransitionSpec(
                McpSessionState.UNINITIALIZED,
                McpSessionState.CLOSED,
                "close",
                "Close session prior to initialization.",
            ),
            TransitionSpec(
                McpSessionState.INITIALIZING,
                McpSessionState.CLOSED,
                "close",
                "Close session during initialization.",
            ),
            TransitionSpec(
                McpSessionState.READY,
                McpSessionState.CLOSED,
                "close",
                "Gracefully terminate active MCP session.",
            ),
        ),
    )
)


class SandboxViolationError(PermissionError):
    """Raised when an MCP tool request attempts to escape the workspace root."""


class ResourceLimitError(ValueError):
    """Raised when an MCP tool request exceeds the configured DOF or budget ceiling."""


def _validate_tool_arguments(
    tool_name: str, schema: Mapping[str, Any], args: Mapping[str, Any]
) -> None:
    """Validate ``args`` against a tool's declarative JSON Schema."""
    props: Mapping[str, Any] = schema.get("properties", {})
    required: Sequence[str] = schema.get("required", ())
    for key in required:
        if key not in args or args[key] is None:
            raise ValueError(
                f"tool {tool_name!r} missing required argument {key!r} "
                f"(required: {list(required)})"
            )
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(args) - set(props))
        if unknown:
            raise ValueError(
                f"tool {tool_name!r} received unexpected argument(s) {unknown}; "
                f"allowed arguments: {sorted(props)}"
            )
    for key, value in args.items():
        if value is None or key not in props:
            continue
        prop_schema = props[key]
        expected_type = prop_schema.get("type")
        if expected_type == "string":
            if not isinstance(value, str):
                raise TypeError(f"argument {key!r} must be a string")
        elif expected_type == "object":
            if not isinstance(value, Mapping):
                raise TypeError(f"argument {key!r} must be a JSON object")
        elif expected_type == "boolean":
            if not isinstance(value, bool):
                raise TypeError(f"argument {key!r} must be a boolean")
        elif expected_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"argument {key!r} must be an integer")
        elif expected_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"argument {key!r} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"argument {key!r} must be finite")
        enum_vals = prop_schema.get("enum")
        if enum_vals is not None and value not in enum_vals:
            raise ValueError(
                f"argument {key!r} must be one of {list(enum_vals)}, got {value!r}"
            )
        if "minimum" in prop_schema and float(value) < float(prop_schema["minimum"]):
            raise ValueError(
                f"argument {key!r} must be >= {prop_schema['minimum']}, got {value!r}"
            )
        if "maximum" in prop_schema and float(value) > float(prop_schema["maximum"]):
            raise ValueError(
                f"argument {key!r} must be <= {prop_schema['maximum']}, got {value!r}"
            )


def _error_category_and_hint(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, SandboxViolationError):
        return (
            "sandbox_violation",
            "Use paths located inside the configured workspace root.",
        )
    if isinstance(exc, ResourceLimitError):
        return (
            "resource_limit",
            "Reduce mesh DOF count or start the MCP server with a higher max_dof_limit.",
        )
    if isinstance(exc, InvalidTransitionError):
        return (
            "invalid_state_transition",
            "Inspect allowed_targets and satisfy protocol preconditions before retrying.",
        )
    if type(exc).__name__ == "DeploymentGateError":
        return (
            "deployment_gate_refused",
            "Provide a passing qualification_manifest_path or run verify_qualification first.",
        )
    if isinstance(exc, (TypeError, ValueError)):
        return (
            "invalid_arguments",
            "Check tool inputSchema from tools/list and verify all argument names and bounds.",
        )
    return (
        "execution_error",
        "Inspect the error message and underlying bundle/project inputs.",
    )


@dataclass(frozen=True, slots=True)
class McpToolSpec:
    """Declarative MCP tool contract (name, description, JSON Schema, handler)."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[Mapping[str, Any]], dict[str, Any]]

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class FormaMechMcpServer:
    """Sandboxed MCP JSON-RPC 2.0 server wrapping FormaMech's verified workflows."""

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        max_dof_limit: int = DEFAULT_MAX_DOF_LIMIT,
        max_fem_solves: int = 16,
        require_qualified: bool = False,
        strict_lifecycle: bool = False,
    ) -> None:
        root = Path(workspace_root).resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"workspace_root must be an existing directory: {root}")
        if type(max_dof_limit) is not int or max_dof_limit < 2:
            raise ValueError("max_dof_limit must be an integer >= 2")
        if type(max_fem_solves) is not int or max_fem_solves < 1:
            raise ValueError("max_fem_solves must be an integer >= 1")
        if not isinstance(require_qualified, bool):
            raise TypeError("require_qualified must be boolean")
        if not isinstance(strict_lifecycle, bool):
            raise TypeError("strict_lifecycle must be boolean")
        self.workspace_root = root
        self.max_dof_limit = max_dof_limit
        self.max_fem_solves = max_fem_solves
        self.require_qualified = require_qualified
        self.strict_lifecycle = strict_lifecycle
        self._session_state: McpSessionState = McpSessionState.UNINITIALIZED
        self._transition_history: list[TransitionRecord[McpSessionState]] = []
        self._workflow = AgentWorkflowTracker(
            AgentBudgetSpec(
                max_fem_solves=max_fem_solves,
                max_dofs=max_dof_limit,
                require_qualified_for_export=True,
            )
        )
        self._harness: Any = None
        self._tools: dict[str, McpToolSpec] = {
            spec.name: spec for spec in self._build_tools()
        }

    @property
    def harness(self) -> Any:
        """Lazily initialized full-capability :class:`FormaMechAgentHarness`."""
        if self._harness is None:
            from piginot.harness import FormaMechAgentHarness, HarnessBudget

            self._harness = FormaMechAgentHarness(
                self.workspace_root,
                budget=HarnessBudget(
                    max_fem_solves=self.max_fem_solves,
                    max_dofs=self.max_dof_limit,
                ),
            )
        return self._harness

    @property
    def workflow_state(self) -> AgentWorkflowState:
        """Current guarded engineering copilot workflow FSM state."""
        return self._workflow.state

    def workflow_snapshot(self) -> StateSnapshot[AgentWorkflowState]:
        """Return the current agent workflow StateSnapshot."""
        return self._workflow.snapshot()

    @property
    def session_state(self) -> McpSessionState:
        """Current explicit MCP protocol session state."""
        return self._session_state

    @property
    def transition_history(self) -> tuple[TransitionRecord[McpSessionState], ...]:
        """Audit trail of session state transitions."""
        return tuple(self._transition_history)

    def _transition(
        self,
        target: McpSessionState,
        *,
        trigger: str,
        reason: str = "",
    ) -> TransitionRecord[McpSessionState]:
        record = MCP_SESSION_MACHINE.require_transition(
            self._session_state, target, trigger=trigger, reason=reason
        )
        self._session_state = target
        self._transition_history.append(record)
        return record

    def close(self) -> TransitionRecord[McpSessionState]:
        """Transition the MCP session to terminal ``CLOSED`` state."""
        return self._transition(McpSessionState.CLOSED, trigger="close")

    def session_snapshot(self) -> StateSnapshot[McpSessionState]:
        """Return a self-describing session state snapshot for agents and diagnostics."""
        hints = {
            McpSessionState.UNINITIALIZED: "Send JSON-RPC 'initialize' to begin the MCP handshake.",
            McpSessionState.INITIALIZING: "Send 'notifications/initialized' to enter READY state.",
            McpSessionState.READY: "Session is ready for 'tools/list', 'tools/call', and 'ping'.",
            McpSessionState.CLOSED: "Session is closed; create a new server instance to reconnect.",
        }
        return MCP_SESSION_MACHINE.snapshot(
            self._session_state,
            recommended_hint=hints[self._session_state],
            metadata={
                "server_name": SERVER_NAME,
                "protocol_version": MCP_PROTOCOL_VERSION,
                "strict_lifecycle": self.strict_lifecycle,
                "require_qualified": self.require_qualified,
                "tool_count": len(self._tools),
            },
        )

    def resolve_path(self, raw_path: str | Path) -> Path:
        """Resolve a user-supplied path strictly within ``workspace_root``."""
        if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
            raise SandboxViolationError("path argument must be non-empty")
        candidate = Path(raw_path)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.workspace_root / candidate).resolve()
        )
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise SandboxViolationError(
                f"path {str(raw_path)!r} escapes workspace root {str(self.workspace_root)!r}"
            ) from exc
        return resolved

    def list_tools(self) -> list[dict[str, Any]]:
        return [spec.descriptor() for spec in self._tools.values()]

    def call_tool(
        self, name: str, arguments: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(f"unknown MCP tool {name!r}")
        args = {} if arguments is None else arguments
        if not isinstance(args, Mapping):
            raise TypeError("tool arguments must be a JSON object")
        _validate_tool_arguments(spec.name, spec.input_schema, args)
        return spec.handler(args)

    def handle_message(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        """Handle one JSON-RPC 2.0 MCP message and return the response (or None for notifications)."""
        has_id = "id" in message
        req_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}
        if not isinstance(method, str):
            if not has_id:
                return None
            return self._error(req_id, -32600, "invalid request: missing method")

        if self._session_state is McpSessionState.CLOSED:
            if not has_id:
                return None
            return self._error(
                req_id, -32000, "MCP session is CLOSED and cannot process requests"
            )

        if method == "notifications/initialized":
            if self._session_state is McpSessionState.INITIALIZING:
                self._transition(
                    McpSessionState.READY, trigger="notifications/initialized"
                )
            return None

        # JSON-RPC 2.0: never reply to notifications (messages without an 'id').
        if not has_id and method.startswith("notifications/"):
            return None

        if method == "ping":
            return self._result(req_id, {})

        if method == "initialize":
            if self._session_state in (
                McpSessionState.UNINITIALIZED,
                McpSessionState.INITIALIZING,
            ):
                self._transition(McpSessionState.INITIALIZING, trigger="initialize")
            elif self.strict_lifecycle:
                return self._error(
                    req_id,
                    -32600,
                    f"initialize not allowed in session state {self._session_state.value}",
                )
            return self._result(
                req_id,
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"listChanged": False},
                        "prompts": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": __version__,
                    },
                },
            )

        if method in (
            "tools/list",
            "tools/call",
            "resources/list",
            "resources/templates/list",
            "resources/read",
            "prompts/list",
            "prompts/get",
        ):
            if (
                self.strict_lifecycle
                and self._session_state is not McpSessionState.READY
            ):
                return self._error(
                    req_id,
                    -32002,
                    "MCP session is not READY; send 'initialize' then "
                    "'notifications/initialized' first",
                )
            if self._session_state is McpSessionState.READY:
                self._transition(McpSessionState.READY, trigger="serve_request")

        if method == "resources/list":
            return self._result(req_id, {"resources": self.list_resources()})

        if method == "resources/templates/list":
            return self._result(
                req_id, {"resourceTemplates": self.list_resource_templates()}
            )

        if method == "resources/read":
            if not isinstance(params, Mapping) or not isinstance(
                params.get("uri"), str
            ):
                return self._error(
                    req_id, -32602, "resources/read requires string 'uri'"
                )
            try:
                contents = self.read_resource(str(params["uri"]))
            except Exception as exc:
                return self._error(req_id, -32002, str(exc))
            return self._result(req_id, {"contents": contents})

        if method == "prompts/list":
            return self._result(req_id, {"prompts": self.list_prompts()})

        if method == "prompts/get":
            if not isinstance(params, Mapping) or not isinstance(
                params.get("name"), str
            ):
                return self._error(req_id, -32602, "prompts/get requires string 'name'")
            try:
                prompt_doc = self.get_prompt(
                    str(params["name"]),
                    params.get("arguments")
                    if isinstance(params.get("arguments"), Mapping)
                    else {},
                )
            except KeyError as exc:
                return self._error(req_id, -32601, str(exc))
            return self._result(req_id, prompt_doc)

        if method == "tools/list":
            return self._result(req_id, {"tools": self.list_tools()})

        if method == "tools/call":
            if not isinstance(params, Mapping):
                return self._error(req_id, -32602, "params must be an object")
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            if not isinstance(tool_name, str) or not tool_name:
                return self._error(req_id, -32602, "tools/call requires string 'name'")
            try:
                payload = self.call_tool(tool_name, tool_args)
            except KeyError as exc:
                return self._error(req_id, -32601, str(exc))
            except Exception as exc:
                category, hint = _error_category_and_hint(exc)
                err_payload: dict[str, Any] = {
                    "error": type(exc).__name__,
                    "category": category,
                    "message": str(exc),
                    "hint": hint,
                }
                if isinstance(exc, InvalidTransitionError):
                    err_payload["transition"] = exc.document()
                return self._result(
                    req_id,
                    {
                        "isError": True,
                        "structuredContent": err_payload,
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    err_payload, sort_keys=True, allow_nan=False
                                ),
                            }
                        ],
                    },
                )
            return self._result(
                req_id,
                {
                    "isError": False,
                    "structuredContent": payload,
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                payload, sort_keys=True, allow_nan=False
                            ),
                        }
                    ],
                },
            )
        if not has_id:
            return None
        return self._error(req_id, -32601, f"unsupported method {method!r}")

    def serve_stdio(
        self,
        *,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        """Serve newline-delimited JSON-RPC 2.0 MCP frames over stdio.

        When serving on real process stdio (``stdout is None``), native C/C++
        stdout output (such as OpenCASCADE STEP transfer statistics) is routed
        to ``stderr`` at the OS file-descriptor level so the MCP JSON-RPC 2.0
        protocol stream on FD 1 remains strictly uncorrupted.
        """
        in_stream = sys.stdin if stdin is None else stdin
        with _protocol_stdio_guard(stdout) as out_stream:
            for raw_line in in_stream:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as exc:
                    resp: dict[str, Any] | None = self._error(
                        None, -32700, f"parse error: {exc}"
                    )
                else:
                    resp = (
                        self.handle_message(msg)
                        if isinstance(msg, Mapping)
                        else self._error(
                            None, -32600, "JSON-RPC message must be an object"
                        )
                    )
                if resp is not None:
                    out_stream.write(
                        json.dumps(resp, sort_keys=True, allow_nan=False) + "\n"
                    )
                    out_stream.flush()

    @staticmethod
    def _result(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }

    # ------------------------------------------------------------------ tools

    def _build_tools(self) -> tuple[McpToolSpec, ...]:
        return (
            McpToolSpec(
                name="inspect_bundle",
                description=(
                    "Inspect a verified FormaMech bundle's manifest digests, "
                    "claim level, parameter envelope bounds, QoIs, and limitations."
                ),
                input_schema={
                    "type": "object",
                    "required": ["bundle_path"],
                    "properties": {
                        "bundle_path": {
                            "type": "string",
                            "description": "Path to bundle directory inside the workspace.",
                        }
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_inspect_bundle,
            ),
            McpToolSpec(
                name="inspect_cad_project",
                description=(
                    "Inspect a prepared FormaMech fixed-CAD project directory, "
                    "mesh topology, boundary tags, and geometry digests."
                ),
                input_schema={
                    "type": "object",
                    "required": ["project_path"],
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to prepared CAD project directory.",
                        }
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_inspect_cad_project,
            ),
            McpToolSpec(
                name="predict_verified",
                description=(
                    "Serve an envelope-checked solid mechanics prediction in fast, "
                    "certified, or exact mode and emit an RFC 8785 Trust Receipt."
                ),
                input_schema={
                    "type": "object",
                    "required": ["bundle_path", "inputs"],
                    "properties": {
                        "bundle_path": {"type": "string"},
                        "inputs": {
                            "type": "object",
                            "description": "Mapping of input name -> {value, unit}.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["fast", "certified", "exact"],
                            "default": "certified",
                        },
                        "recipe_path": {"type": "string"},
                        "cad_calibration_path": {"type": "string"},
                        "qualification_manifest_path": {"type": "string"},
                        "backend": {"type": "string"},
                        "escalate": {"type": "boolean", "default": False},
                        "tolerance": {"type": "number"},
                        "spectral_repair": {"type": "boolean", "default": True},
                        "trust_report": {"type": "boolean", "default": False},
                        "tta_steps": {"type": "integer", "minimum": 0, "maximum": 200},
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_predict_verified,
            ),
            McpToolSpec(
                name="run_design_optimization",
                description=(
                    "Run FEM-guarded differentiable design optimization (P5) on a "
                    "verified bundle and return the content-hashed optimization report."
                ),
                input_schema={
                    "type": "object",
                    "required": ["bundle_path", "problem_path"],
                    "properties": {
                        "bundle_path": {"type": "string"},
                        "problem_path": {"type": "string"},
                        "recipe_path": {"type": "string"},
                        "qualification_manifest_path": {"type": "string"},
                        "refuse_unsupported": {"type": "boolean", "default": False},
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_run_design_optimization,
            ),
            McpToolSpec(
                name="verify_receipt",
                description=(
                    "Independently verify the RFC 8785 JCS SHA-256 digest, tier "
                    "consistency, and epistemic scope of a FormaMech Trust Receipt."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "receipt": {
                            "type": "object",
                            "description": "Inline Trust Receipt JSON object.",
                        },
                        "receipt_path": {
                            "type": "string",
                            "description": "Path to Trust Receipt JSON file in workspace.",
                        },
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_verify_receipt,
            ),
            McpToolSpec(
                name="verify_qualification",
                description=(
                    "Verify a bundle's numerical qualification manifest and enforce "
                    "the fail-closed deployment gate."
                ),
                input_schema={
                    "type": "object",
                    "required": ["bundle_path"],
                    "properties": {
                        "bundle_path": {"type": "string"},
                        "manifest_path": {"type": "string"},
                        "policy_path": {"type": "string"},
                        "min_claim_level": {
                            "type": "string",
                            "enum": ["DEVELOPMENT_VERIFIED", "HOLDOUT_VERIFIED"],
                            "default": "DEVELOPMENT_VERIFIED",
                        },
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_verify_qualification,
            ),
            McpToolSpec(
                name="import_cad_part",
                description=(
                    "Import and mesh a 3D STEP part (or generate a sample 3D solid) "
                    "inside a sandboxed workspace directory, advancing the copilot FSM to CAD_IMPORTED."
                ),
                input_schema={
                    "type": "object",
                    "required": ["workspace_dir"],
                    "properties": {
                        "workspace_dir": {"type": "string"},
                        "step_path": {"type": "string"},
                        "sample_3d_box": {"type": "boolean", "default": False},
                        "mesh_size_mm": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_import_cad_part,
            ),
            McpToolSpec(
                name="propose_physics_setup",
                description=(
                    "Validate and persist 3D boundary conditions and material settings "
                    "on an imported CAD workspace, advancing the copilot FSM to SETUP_PROPOSED."
                ),
                input_schema={
                    "type": "object",
                    "required": ["workspace_dir"],
                    "properties": {
                        "workspace_dir": {"type": "string"},
                        "youngs_modulus_mpa": {"type": "number"},
                        "poisson_ratio": {"type": "number"},
                        "load_ux_mm": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_propose_physics_setup,
            ),
            McpToolSpec(
                name="compile_and_qualify",
                description=(
                    "Run a budgeted 3D FEM verification & qualification check on the "
                    "proposed setup, advancing SETUP_PROPOSED -> COMPILED -> QUALIFIED."
                ),
                input_schema={
                    "type": "object",
                    "required": ["workspace_dir"],
                    "properties": {
                        "workspace_dir": {"type": "string"},
                        "load_steps": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_compile_and_qualify,
            ),
            McpToolSpec(
                name="run_active_fem_round",
                description=(
                    "Run a budgeted active 3D FEM evaluation round at specified moduli/loads "
                    "enforcing hard FEM solve and DOF ceilings."
                ),
                input_schema={
                    "type": "object",
                    "required": ["workspace_dir", "n_samples"],
                    "properties": {
                        "workspace_dir": {"type": "string"},
                        "n_samples": {"type": "integer", "minimum": 1, "maximum": 64},
                        "youngs_modulus_mpa": {"type": "number"},
                        "poisson_ratio": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_run_active_fem_round,
            ),
            McpToolSpec(
                name="ingest_cae_archive",
                description=(
                    "Screen a directory of solver-agnostic CAE archives (.vtu/.xdmf/.inp/.frd/.bdf) "
                    "with deduplication, admissibility checks, MAD outlier screening, and leak-free splits."
                ),
                input_schema={
                    "type": "object",
                    "required": ["archive_dir"],
                    "properties": {
                        "archive_dir": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_ingest_cae_archive,
            ),
            McpToolSpec(
                name="run_experimental_validation",
                description=(
                    "Calibrate constitutive parameters from experimental coupon data and compute "
                    "an ASME V&V 10 validation budget and orthogonal ExperimentalValidationTier."
                ),
                input_schema={
                    "type": "object",
                    "required": ["dataset_path"],
                    "properties": {
                        "dataset_path": {"type": "string"},
                        "u_num_mpa": {
                            "type": "number",
                            "description": "Numerical uncertainty u_num in MPa (default 0.05).",
                        },
                        "u_num_rel": {
                            "type": "number",
                            "description": "Deprecated alias for u_num_mpa (in MPa).",
                        },
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_run_experimental_validation,
            ),
            McpToolSpec(
                name="export_verified_artifact",
                description=(
                    "Export a verified ParaView VTU or glTF GLB artifact strictly after the "
                    "engineering workflow reaches QUALIFIED state."
                ),
                input_schema={
                    "type": "object",
                    "required": ["workspace_dir", "output_path"],
                    "properties": {
                        "workspace_dir": {"type": "string"},
                        "output_path": {"type": "string"},
                        "format": {
                            "type": "string",
                            "enum": ["vtu", "glb"],
                            "default": "vtu",
                        },
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_export_verified_artifact,
            ),
            McpToolSpec(
                name="discover_capabilities",
                description=(
                    "Progressively discover all registered FormaMech engineering capabilities "
                    "across CAD, FEM, multiphysics, surrogates, optimization, ingest/V&V10, export, and governance."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string"},
                        "cost_class": {"type": "string"},
                        "search": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_discover_capabilities,
            ),
            McpToolSpec(
                name="describe_capability",
                description=(
                    "Return the complete JSON Schema, cost class, and epistemic scope for one "
                    "registered FormaMech harness capability."
                ),
                input_schema={
                    "type": "object",
                    "required": ["name"],
                    "properties": {"name": {"type": "string"}},
                    "additionalProperties": False,
                },
                handler=self._tool_describe_capability,
            ),
            McpToolSpec(
                name="invoke_capability",
                description=(
                    "Invoke (or dry-run) any registered FormaMech capability through the "
                    "sandboxed, budget-governed, hash-chained Agent Harness."
                ),
                input_schema={
                    "type": "object",
                    "required": ["capability"],
                    "properties": {
                        "capability": {"type": "string"},
                        "arguments": {"type": "object"},
                        "dry_run": {"type": "boolean", "default": False},
                    },
                    "additionalProperties": False,
                },
                handler=self._tool_invoke_capability,
            ),
            McpToolSpec(
                name="session_status",
                description=(
                    "Inspect the active FormaMech Agent Harness session state, remaining "
                    "FEM/epoch budgets, recorded artifacts, and ledger verification status."
                ),
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=self._tool_session_status,
            ),
            McpToolSpec(
                name="describe_state_machines",
                description=(
                    "Return all formal FormaMech finite state machine specifications "
                    "(states, allowed transitions, terminal states)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=self._tool_describe_state_machines,
            ),
        )

    def _tool_inspect_bundle(self, args: Mapping[str, Any]) -> dict[str, Any]:
        from piginot.bundle import inspect as inspect_bundle
        from piginot.bundle import load_bundle

        bpath = self.resolve_path(str(args["bundle_path"]))
        bundle = load_bundle(bpath)
        return dict(inspect_bundle(bundle))

    def _tool_inspect_cad_project(self, args: Mapping[str, Any]) -> dict[str, Any]:
        from piginot.cad.project import inspect_geometry

        ppath = self.resolve_path(str(args["project_path"]))
        return dict(inspect_geometry(ppath))

    def _tool_predict_verified(self, args: Mapping[str, Any]) -> dict[str, Any]:
        from piginot.bounds import CadCalibrationRecord
        from piginot.bundle import load_bundle
        from piginot.bundle.problem import PortableProblem
        from piginot.cli import parse_request
        from piginot.serving import ServingMode, load_recipe, serve
        from piginot.serving.test_time_lora import TestTimeLoraConfig

        bpath = self.resolve_path(str(args["bundle_path"]))
        if self.require_qualified or args.get("qualification_manifest_path"):
            from piginot.qualify import enforce_deployment_gate

            mpath = (
                self.resolve_path(str(args["qualification_manifest_path"]))
                if args.get("qualification_manifest_path")
                else None
            )
            enforce_deployment_gate(bpath, mpath)
        bundle = load_bundle(bpath)
        dof_estimate = len(bundle.reference_mesh.nodes) * 2
        if dof_estimate > self.max_dof_limit:
            raise ResourceLimitError(
                f"bundle mesh requires ~{dof_estimate} DOFs, exceeding "
                f"max_dof_limit={self.max_dof_limit}"
            )
        raw_inputs = args["inputs"]
        if not isinstance(raw_inputs, Mapping):
            raise TypeError("'inputs' must be a JSON object")
        parsed_inputs, parsed_queries = parse_request({"inputs": dict(raw_inputs)})
        mode = ServingMode(str(args.get("mode", "certified")))
        escalate = bool(args.get("escalate", False))
        tolerance = None if args.get("tolerance") is None else float(args["tolerance"])
        tta_steps = int(args.get("tta_steps", 0) or 0)

        if tolerance is not None and (not math.isfinite(tolerance) or tolerance <= 0.0):
            raise ValueError("tolerance must be a finite positive number")
        if mode is ServingMode.FAST:
            if escalate:
                raise ValueError("escalate=True is not supported when mode='fast'")
            if tolerance is not None:
                raise ValueError("tolerance is not supported when mode='fast'")
            if tta_steps > 0:
                raise ValueError("tta_steps > 0 is not supported when mode='fast'")
        elif mode is ServingMode.EXACT and tolerance is not None:
            raise ValueError("tolerance is not supported when mode='exact'")

        is_portable = isinstance(
            getattr(bundle.model, "problem", None), PortableProblem
        )
        recipe = None
        if mode is not ServingMode.FAST:
            if args.get("recipe_path"):
                recipe = load_recipe(
                    bundle, bpath, self.resolve_path(str(args["recipe_path"]))
                )
            elif not is_portable:
                recipe = load_recipe(bundle, bpath)
        cad_cal = None
        if args.get("cad_calibration_path"):
            cad_cal = CadCalibrationRecord.load(
                self.resolve_path(str(args["cad_calibration_path"])),
                expected_manifest_sha256=bundle.manifest.manifest_sha256,
            )
        tta_cfg = TestTimeLoraConfig(steps=tta_steps) if tta_steps > 0 else None
        served = serve(
            bundle,
            parsed_inputs,
            cast("Sequence[Sequence[float]] | None", parsed_queries),
            mode=mode,
            recipe=recipe,
            corrected=mode is not ServingMode.FAST,
            escalate=escalate,
            tolerance=tolerance,
            backend=args.get("backend"),
            cad_calibration=cad_cal,
            trust_report=bool(args.get("trust_report", False)),
            tier0_check=True,
            spectral_repair=bool(args.get("spectral_repair", True)),
            tta_config=tta_cfg,
            emit_receipt=True,
        )
        doc = served.document()
        return {
            "mode": doc["mode"],
            "serving_path": doc["path"],
            "verdict": doc["prediction"]["verdict"],
            "qois": doc["prediction"].get("qois", []),
            "tier0_admissibility": doc.get("tier0_admissibility"),
            "certificate": doc.get("certificate"),
            "bounds": doc.get("bounds"),
            "trust_report": doc.get("trust_report"),
            "trust_receipt": doc.get("trust_receipt"),
            "trust_receipt_sha256": doc.get("trust_receipt_sha256"),
            "reason": doc.get("reason", ""),
        }

    def _tool_run_design_optimization(self, args: Mapping[str, Any]) -> dict[str, Any]:
        from piginot.bundle import load_bundle
        from piginot.optimization import (
            BundleDesignBridge,
            parse_bundle_design_document,
        )
        from piginot.serving import load_recipe

        bpath = self.resolve_path(str(args["bundle_path"]))
        if self.require_qualified or args.get("qualification_manifest_path"):
            from piginot.qualify import enforce_deployment_gate

            mpath = (
                self.resolve_path(str(args["qualification_manifest_path"]))
                if args.get("qualification_manifest_path")
                else None
            )
            enforce_deployment_gate(bpath, mpath)
        ppath = self.resolve_path(str(args["problem_path"]))
        bundle = load_bundle(bpath)
        dof_estimate = len(bundle.reference_mesh.nodes) * 2
        if dof_estimate > self.max_dof_limit:
            raise ResourceLimitError(
                f"bundle mesh requires ~{dof_estimate} DOFs, exceeding "
                f"max_dof_limit={self.max_dof_limit}"
            )
        rpath = (
            self.resolve_path(str(args["recipe_path"]))
            if args.get("recipe_path")
            else None
        )
        recipe = load_recipe(bundle, bpath, rpath)
        prob_doc = json.loads(ppath.read_text(encoding="utf-8"))
        spec = parse_bundle_design_document(prob_doc, bundle=bundle)
        bridge = BundleDesignBridge(bundle, recipe, spec)
        report = bridge.run(
            geometry_supported=not bool(args.get("refuse_unsupported", False))
        )
        return {**report.document(), "report_hash": report.report_hash}

    def _tool_verify_receipt(self, args: Mapping[str, Any]) -> dict[str, Any]:
        receipt_obj = args.get("receipt")
        receipt_path = args.get("receipt_path")
        if receipt_obj is None and receipt_path is None:
            raise ValueError(
                "verify_receipt requires either 'receipt' or 'receipt_path'"
            )
        if receipt_obj is not None and receipt_path is not None:
            raise ValueError(
                "verify_receipt accepts either 'receipt' or 'receipt_path', not both"
            )
        if receipt_obj is None:
            rpath = self.resolve_path(str(receipt_path))
            receipt_obj = json.loads(rpath.read_text(encoding="utf-8"))
        if not isinstance(receipt_obj, Mapping):
            raise TypeError("receipt must be a JSON object")
        ok, errors = verify_receipt_document(receipt_obj)
        return {
            "verified": ok,
            "errors": list(errors),
            "verification_tier": receipt_obj.get("verification_tier"),
            "receipt_sha256": receipt_obj.get("receipt_sha256"),
        }

    def _tool_verify_qualification(self, args: Mapping[str, Any]) -> dict[str, Any]:
        from piginot.contracts.state_machine import ClaimLevel
        from piginot.qualify import QualificationPolicy, check_deployment_gate

        bpath = self.resolve_path(str(args["bundle_path"]))
        mpath = (
            self.resolve_path(str(args["manifest_path"]))
            if args.get("manifest_path")
            else None
        )
        policy = (
            QualificationPolicy.load(self.resolve_path(str(args["policy_path"])))
            if args.get("policy_path")
            else None
        )
        min_claim = ClaimLevel(
            str(args.get("min_claim_level", ClaimLevel.DEVELOPMENT_VERIFIED.value))
        )
        result = check_deployment_gate(
            bpath,
            mpath,
            policy=policy,
            min_claim_level=min_claim,
        )
        return result.document()

    def _require_active_workspace(self, ws_path: Path, *, tool_name: str) -> str:
        ws_rel = str(ws_path.relative_to(self.workspace_root).as_posix())
        active = self._workflow.active_workspace_rel
        if active is not None and ws_rel != active:
            raise ValueError(
                f"{tool_name} workspace {ws_rel!r} does not match active workflow "
                f"workspace {active!r}; call import_cad_part to start a new study"
            )
        return ws_rel

    def _clear_cached_qualification(self) -> None:
        self._workflow.last_fem_result = None
        self._workflow.last_qualification_sha256 = None
        self._workflow.last_receipt_sha256 = None

    def _tool_import_cad_part(self, args: Mapping[str, Any]) -> dict[str, Any]:
        from piginot.geometry.mesh3d import CellType3D
        from piginot.ui.workbench_3d.state_3d import Workbench3DWorkspace

        # Enforce FSM precondition BEFORE mutating any workspace files on disk
        AGENT_WORKFLOW_MACHINE.require_transition(
            self._workflow.state,
            AgentWorkflowState.CAD_IMPORTED,
            trigger="import_cad_part",
        )
        mesh_size = float(args.get("mesh_size_mm", 6.0))
        if not math.isfinite(mesh_size) or mesh_size <= 0.0:
            raise ValueError("mesh_size_mm must be finite and positive")

        ws_path = self.resolve_path(str(args["workspace_dir"]))
        ws = Workbench3DWorkspace(ws_path)
        if bool(args.get("sample_3d_box", False)):
            preview = ws.import_sample_box(
                mesh_size_mm=mesh_size, element_type=CellType3D.TET10
            )
        else:
            raw_step = args.get("step_path")
            if not raw_step:
                raise ValueError(
                    "import_cad_part requires 'step_path' or sample_3d_box=True"
                )
            step_path = self.resolve_path(str(raw_step))
            preview = ws.import_step_path(
                step_path, mesh_size_mm=mesh_size, element_type=CellType3D.TET10
            )

        n_dofs = int(preview["node_count"]) * 3
        self._workflow.check_dofs(n_dofs)
        self._clear_cached_qualification()
        self._workflow.transition(
            AgentWorkflowState.CAD_IMPORTED,
            trigger="import_cad_part",
            reason=f"Imported 3D mesh with {n_dofs} DOFs.",
        )
        self._workflow.active_workspace_rel = str(
            ws_path.relative_to(self.workspace_root).as_posix()
        )
        return {
            "geometry_id": preview["geometry_id"],
            "node_count": preview["node_count"],
            "cell_count": preview["cell_count"],
            "dof_count": n_dofs,
            "face_tags": [fg["face_tag"] for fg in preview["face_groups"]],
            "workflow_state": self._workflow.snapshot().document(),
        }

    def _tool_propose_physics_setup(self, args: Mapping[str, Any]) -> dict[str, Any]:
        from piginot.contracts import canonical_json_bytes
        from piginot.ui.workbench_3d.state_3d import (
            FaceAssignment3D,
            Workbench3DWorkspace,
            validate_face_assignments_3d,
        )

        # Enforce FSM precondition before mutating files
        AGENT_WORKFLOW_MACHINE.require_transition(
            self._workflow.state,
            AgentWorkflowState.SETUP_PROPOSED,
            trigger="propose_physics_setup",
        )
        ws_path = self.resolve_path(str(args["workspace_dir"]))
        self._require_active_workspace(ws_path, tool_name="propose_physics_setup")
        ws = Workbench3DWorkspace(ws_path)
        preview = ws.load_preview()
        state = ws.load_state()
        if preview is None or state is None:
            raise ValueError("workspace has no imported 3D CAD geometry")

        e_mpa = float(args.get("youngs_modulus_mpa", 100.0))
        nu = float(args.get("poisson_ratio", 0.30))
        load_ux = float(args.get("load_ux_mm", 0.25))
        if not math.isfinite(e_mpa) or not (1.0 <= e_mpa <= 10_000.0):
            raise ValueError("youngs_modulus_mpa must lie in [1.0, 10000.0]")
        if not math.isfinite(nu) or not (0.0 <= nu <= 0.49):
            raise ValueError("poisson_ratio must lie in [0.0, 0.49]")
        if not math.isfinite(load_ux):
            raise ValueError("load_ux_mm must be finite")

        # Update loaded face displacement if specified
        assignments = [FaceAssignment3D.from_document(a) for a in state["assignments"]]
        updated = []
        for a in assignments:
            if a.kind == "dirichlet":
                updated.append(
                    FaceAssignment3D(
                        face_tag=a.face_tag,
                        name=a.name,
                        kind="dirichlet",
                        components=("x",),
                        values_mm=(load_ux,),
                    )
                )
            else:
                updated.append(a)
        ok, errs = validate_face_assignments_3d(
            [fg["face_tag"] for fg in preview["face_groups"]], updated
        )
        if not ok:
            raise ValueError("invalid 3D physics setup: " + "; ".join(errs))

        setup_doc = {
            "geometry_id": preview["geometry_id"],
            "youngs_modulus_mpa": e_mpa,
            "poisson_ratio": nu,
            "load_ux_mm": load_ux,
            "assignments": [a.document() for a in updated],
        }
        (ws_path / "physics_3d_setup.json").write_bytes(canonical_json_bytes(setup_doc))
        ws._save_state(
            geometry_id=str(preview["geometry_id"]),
            assignments=updated,
            last_event_id="mcp-propose",
            last_result={"event_id": "mcp-propose", "status": "accepted"},
        )
        self._clear_cached_qualification()
        self._workflow.transition(
            AgentWorkflowState.SETUP_PROPOSED,
            trigger="propose_physics_setup",
            reason="Validated 3D BCs and material bounds.",
        )
        return {
            "setup": setup_doc,
            "workflow_state": self._workflow.snapshot().document(),
        }

    def _tool_compile_and_qualify(self, args: Mapping[str, Any]) -> dict[str, Any]:
        import hashlib

        from piginot.contracts import canonical_json_bytes
        from piginot.ui.workbench_3d.state_3d import Workbench3DWorkspace

        AGENT_WORKFLOW_MACHINE.require_transition(
            self._workflow.state,
            AgentWorkflowState.COMPILED,
            trigger="compile_and_qualify",
        )
        ws_path = self.resolve_path(str(args["workspace_dir"]))
        self._require_active_workspace(ws_path, tool_name="compile_and_qualify")
        ws = Workbench3DWorkspace(ws_path)
        mesh = ws.load_mesh()
        if mesh is None:
            raise ValueError("workspace has no imported 3D mesh")
        self._workflow.check_dofs(len(mesh.nodes) * 3)

        setup_path = ws_path / "physics_3d_setup.json"
        setup = (
            json.loads(setup_path.read_text(encoding="utf-8"))
            if setup_path.is_file()
            else {"youngs_modulus_mpa": 100.0, "poisson_ratio": 0.30}
        )
        e_mpa = float(setup["youngs_modulus_mpa"])
        nu = float(setup["poisson_ratio"])
        if not math.isfinite(e_mpa) or not (1.0 <= e_mpa <= 10_000.0):
            raise ValueError("youngs_modulus_mpa must lie in [1.0, 10000.0]")
        if not math.isfinite(nu) or not (0.0 <= nu <= 0.49):
            raise ValueError("poisson_ratio must lie in [0.0, 0.49]")

        self._workflow.reserve_fem_solves(1)
        res = ws.solve_preview_fem(
            youngs_modulus_mpa=e_mpa,
            poisson_ratio=nu,
            load_steps=int(args.get("load_steps", 2)),
        )
        self._workflow.transition(
            AgentWorkflowState.COMPILED,
            trigger="compile_and_qualify",
            reason="Executed authoritative 3D FEM verification solve.",
        )
        if not res.converged or res.min_det_f <= 0.0:
            self._clear_cached_qualification()
            self._workflow.transition(
                AgentWorkflowState.REFUSED,
                trigger="refuse",
                reason=f"3D FEM solve failed admissibility: {res.reason}",
            )
            return {
                "qualified": False,
                "reason": res.reason,
                "workflow_state": self._workflow.snapshot().document(),
            }

        qual_doc = {
            "schema": "formamech.mcp-3d-qualification-v1",
            "mesh_id": mesh.mesh_id,
            "result_hash": res.result_hash,
            "converged": res.converged,
            "min_det_f": float(res.min_det_f),
            "max_displacement_mm": float(res.max_displacement_mm),
            "peak_von_mises_mpa": float(res.peak_von_mises_mpa),
        }
        qual_bytes = canonical_json_bytes(qual_doc)
        qual_sha = hashlib.sha256(qual_bytes).hexdigest()
        (ws_path / "results_3d").mkdir(parents=True, exist_ok=True)
        (ws_path / "results_3d" / "qualification_3d.json").write_bytes(qual_bytes)

        self._workflow.last_fem_result = res
        self._workflow.last_qualification_sha256 = qual_sha
        self._workflow.last_receipt_sha256 = res.result_hash
        self._workflow.transition(
            AgentWorkflowState.QUALIFIED,
            trigger="qualify_complete",
            reason="3D FEM equilibrium and det(F) > 0 admissibility verified.",
        )
        return {
            "qualified": True,
            "qualification_sha256": qual_sha,
            "result_hash": res.result_hash,
            "max_displacement_mm": float(res.max_displacement_mm),
            "peak_von_mises_mpa": float(res.peak_von_mises_mpa),
            "min_det_f": float(res.min_det_f),
            "workflow_state": self._workflow.snapshot().document(),
        }

    def _tool_run_active_fem_round(self, args: Mapping[str, Any]) -> dict[str, Any]:
        from piginot.ui.workbench_3d.state_3d import Workbench3DWorkspace

        AGENT_WORKFLOW_MACHINE.require_transition(
            self._workflow.state,
            self._workflow.state,
            trigger="run_active_fem_round",
        )
        n_samples = int(args["n_samples"])
        ws_path = self.resolve_path(str(args["workspace_dir"]))
        self._require_active_workspace(ws_path, tool_name="run_active_fem_round")
        base_e = float(args.get("youngs_modulus_mpa", 100.0))
        nu = float(args.get("poisson_ratio", 0.30))
        if not math.isfinite(base_e) or not (1.0 <= base_e <= 10_000.0):
            raise ValueError("youngs_modulus_mpa must lie in [1.0, 10000.0]")
        if not math.isfinite(nu) or not (0.0 <= nu <= 0.49):
            raise ValueError("poisson_ratio must lie in [0.0, 0.49]")

        ws = Workbench3DWorkspace(ws_path)
        mesh = ws.load_mesh()
        if mesh is None:
            raise ValueError("workspace has no imported 3D mesh")
        self._workflow.check_dofs(len(mesh.nodes) * 3)
        self._workflow.reserve_fem_solves(n_samples)

        summaries: list[dict[str, Any]] = []
        for idx in range(n_samples):
            e_val = min(10_000.0, max(1.0, base_e * (1.0 + 0.05 * idx)))
            res = ws.solve_preview_fem(
                youngs_modulus_mpa=e_val, poisson_ratio=nu, load_steps=1
            )
            summaries.append(
                {
                    "sample_index": idx,
                    "youngs_modulus_mpa": e_val,
                    "converged": res.converged,
                    "peak_von_mises_mpa": float(res.peak_von_mises_mpa),
                    "result_hash": res.result_hash,
                }
            )
        self._workflow.transition(
            self._workflow.state,
            trigger="run_active_fem_round",
            reason=f"Completed {n_samples} budgeted FEM solves.",
        )
        return {
            "samples": summaries,
            "workflow_state": self._workflow.snapshot().document(),
        }

    def _tool_ingest_cae_archive(self, args: Mapping[str, Any]) -> dict[str, Any]:
        from piginot.ui.workbench_3d.vv10_panel import run_cae_ingest_summary

        arch_dir = self.resolve_path(str(args["archive_dir"]))
        return {
            "screening_report": run_cae_ingest_summary(arch_dir),
            "workflow_state": self._workflow.snapshot().document(),
        }

    def _tool_run_experimental_validation(
        self, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.ui.workbench_3d.vv10_panel import run_vv10_calibration_summary

        ds_path = self.resolve_path(str(args["dataset_path"]))
        if "u_num_mpa" in args and "u_num_rel" in args:
            raise ValueError("pass either 'u_num_mpa' or 'u_num_rel', not both")
        raw_u_num = (
            args["u_num_mpa"] if "u_num_mpa" in args else args.get("u_num_rel", 0.05)
        )
        u_num = float(raw_u_num)
        if not math.isfinite(u_num) or u_num < 0.0:
            raise ValueError("u_num_mpa must be finite and nonnegative")
        return {
            "validation_dossier": run_vv10_calibration_summary(
                ds_path, u_num_mpa=u_num
            ),
            "workflow_state": self._workflow.snapshot().document(),
        }

    def _tool_export_verified_artifact(self, args: Mapping[str, Any]) -> dict[str, Any]:
        import hashlib

        from piginot.export.vtu_export import export_deformed_glb, export_fields_to_vtu
        from piginot.ui.workbench_3d.state_3d import Workbench3DWorkspace

        # Fail-closed FSM enforcement: ONLY allowed from QUALIFIED (or repeat from EXPORTED)
        AGENT_WORKFLOW_MACHINE.require_transition(
            self._workflow.state,
            AgentWorkflowState.EXPORTED,
            trigger="export_verified_artifact",
        )
        ws_path = self.resolve_path(str(args["workspace_dir"]))
        self._require_active_workspace(ws_path, tool_name="export_verified_artifact")
        out_path = self.resolve_path(str(args["output_path"]))
        fmt = str(args.get("format", "vtu"))
        ws = Workbench3DWorkspace(ws_path)
        mesh = ws.load_mesh()
        res = self._workflow.last_fem_result
        if mesh is None or res is None:
            raise ValueError("no qualified 3D FEM solution is cached for export")
        if len(mesh.nodes) != len(res.nodal_displacement):
            raise ValueError(
                f"workspace mesh ({len(mesh.nodes)} nodes) does not match qualified "
                f"FEM solution ({len(res.nodal_displacement)} nodes)"
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "glb":
            export_deformed_glb(
                mesh,
                out_path,
                displacement=res.nodal_displacement,
                scalar_field=res.nodal_von_mises_mpa,
            )
        else:
            export_fields_to_vtu(
                mesh,
                out_path,
                point_fields={
                    "displacement_mm": res.nodal_displacement,
                    "von_mises_mpa": res.nodal_von_mises_mpa,
                },
            )
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        self._workflow.transition(
            AgentWorkflowState.EXPORTED,
            trigger="export_verified_artifact",
            reason=f"Exported verified {fmt.upper()} artifact.",
        )
        return {
            "format": fmt,
            "output_path": str(out_path.relative_to(self.workspace_root).as_posix()),
            "artifact_sha256": digest,
            "workflow_state": self._workflow.snapshot().document(),
        }

    # ------------------------------------------------- V4-0 universal harness

    def _tool_discover_capabilities(self, args: Mapping[str, Any]) -> dict[str, Any]:
        from piginot.harness.catalog import build_default_registry

        reg = (
            self._harness.registry
            if self._harness is not None
            else build_default_registry()
        )
        items = reg.list_capabilities(
            domain=args.get("domain"),
            cost_class=args.get("cost_class"),
            search=args.get("search"),
        )
        return {
            "count": len(items),
            "domains": reg.domains_summary(),
            "capabilities": items,
        }

    def _tool_describe_capability(self, args: Mapping[str, Any]) -> dict[str, Any]:
        from piginot.harness.catalog import build_default_registry

        reg = (
            self._harness.registry
            if self._harness is not None
            else build_default_registry()
        )
        return reg.describe(str(args["name"]))

    def _tool_invoke_capability(self, args: Mapping[str, Any]) -> dict[str, Any]:
        cap = str(args["capability"])
        cap_args = args.get("arguments", {})
        if not isinstance(cap_args, Mapping):
            raise TypeError("'arguments' must be a JSON object")
        dry_run = bool(args.get("dry_run", False))
        env = self.harness.invoke(cap, cap_args, dry_run=dry_run)
        return dict(env.document())

    def _tool_session_status(self, _args: Mapping[str, Any]) -> dict[str, Any]:
        ok, errs = self.harness.verify_ledger()
        return {
            "session_id": self.harness.session_id,
            "snapshot": self.harness.snapshot().document(),
            "budget": self.harness.budget_status(),
            "ledger_verified": ok,
            "ledger_errors": list(errs),
        }

    def _tool_describe_state_machines(self, _args: Mapping[str, Any]) -> dict[str, Any]:
        from piginot.contracts.state_machine import registered_state_machines

        machines = [spec.describe() for spec in registered_state_machines().values()]
        return {"count": len(machines), "machines": machines}

    # ---------------------------------------------- MCP resources & prompts

    def list_resources(self) -> list[dict[str, Any]]:
        """Declare MCP 2024-11-05 read-only resources for autonomous agents."""
        return [
            {
                "uri": "formamech://capabilities",
                "name": "FormaMech Capability Catalog",
                "description": "Compact index of all registered capabilities across 9 domains.",
                "mimeType": "application/json",
            },
            {
                "uri": "formamech://state-machines",
                "name": "FormaMech Formal State Machines",
                "description": "All registered explicit FSM specs and transition tables.",
                "mimeType": "application/json",
            },
            {
                "uri": "formamech://session/status",
                "name": "Active Agent Session Status",
                "description": "Current session state, budget remaining, and ledger verification.",
                "mimeType": "application/json",
            },
            {
                "uri": "formamech://session/ledger",
                "name": "Hash-Chained Agent Session Ledger",
                "description": "Append-only SHA-256 hash-chained log of all tool invocations.",
                "mimeType": "application/json",
            },
            {
                "uri": "formamech://workspace/tree",
                "name": "Sandboxed Workspace File Tree",
                "description": "Relative file listing inside the active workspace_root.",
                "mimeType": "application/json",
            },
        ]

    def list_resource_templates(self) -> list[dict[str, Any]]:
        """Declare parameterized MCP 2024-11-05 resource URI templates."""
        return [
            {
                "uriTemplate": "formamech://artifact/{relpath}",
                "name": "Workspace Artifact File",
                "description": "Read a text/JSON artifact strictly inside the active workspace_root.",
                "mimeType": "text/plain",
            }
        ]

    def read_resource(self, uri: str) -> list[dict[str, Any]]:
        """Resolve one ``formamech://`` resource URI safely within the workspace."""
        if uri == "formamech://capabilities":
            payload: Any = self._tool_discover_capabilities({})
        elif uri == "formamech://state-machines":
            payload = self._tool_describe_state_machines({})
        elif uri == "formamech://session/status":
            payload = self._tool_session_status({})
        elif uri == "formamech://session/ledger":
            payload = {"entries": self.harness.read_ledger()}
        elif uri == "formamech://workspace/tree":
            files: list[dict[str, Any]] = []
            for p in sorted(self.workspace_root.rglob("*")):
                if len(files) >= 200:
                    break
                if ".git" in p.parts or "__pycache__" in p.parts:
                    continue
                if p.is_file():
                    files.append(
                        {
                            "path": p.relative_to(self.workspace_root).as_posix(),
                            "bytes": p.stat().st_size,
                        }
                    )
            payload = {"workspace_root": str(self.workspace_root), "files": files}
        elif uri.startswith("formamech://artifact/"):
            rel = uri.removeprefix("formamech://artifact/")
            target = self.resolve_path(rel)
            if not target.is_file():
                raise FileNotFoundError(f"artifact not found: {rel}")
            text = target.read_text(encoding="utf-8", errors="replace")
            return [{"uri": uri, "mimeType": "text/plain", "text": text}]
        else:
            raise KeyError(f"unknown MCP resource URI {uri!r}")
        return [
            {
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(payload, sort_keys=True, allow_nan=False),
            }
        ]

    def list_prompts(self) -> list[dict[str, Any]]:
        """Declare turnkey engineering workflow prompts for MCP clients."""
        return [
            {
                "name": "cad-to-verified-surrogate",
                "description": "End-to-end 3D CAD import, FEM solve, parametric operator training, qualification, and VTU/GLB export.",
                "arguments": [
                    {"name": "workspace_dir", "required": False},
                    {"name": "template", "required": False},
                ],
            },
            {
                "name": "multiphysics-3d-audit",
                "description": "Run coupled 3D thermo-mechanical, modal eigenfrequency, and cyclic J2 plasticity checks with receipts.",
                "arguments": [{"name": "workspace_dir", "required": False}],
            },
            {
                "name": "experimental-vv10-calibration",
                "description": "Calibrate constitutive parameters from coupon data and emit an orthogonal ASME V&V 10 dossier.",
                "arguments": [{"name": "dataset_path", "required": True}],
            },
            {
                "name": "generative-lightweighting",
                "description": "Explore 3D parametric or generative topology/shape lightweighting certified by exact 3D FEM.",
                "arguments": [{"name": "target_volume_fraction", "required": False}],
            },
            {
                "name": "diagnose-refusal-or-ood",
                "description": "Audit a failed envelope, OOD refusal, or invalid FSM transition and propose safe recovery steps.",
                "arguments": [{"name": "error_code", "required": False}],
            },
        ]

    def get_prompt(
        self, name: str, arguments: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        args = {} if arguments is None else dict(arguments)
        ws = str(args.get("workspace_dir", "study_3d"))
        templates: dict[str, tuple[str, str]] = {
            "cad-to-verified-surrogate": (
                "Guide a verified 3D CAD-to-surrogate workflow.",
                f"Use FormaMech harness tools in workspace '{ws}': "
                "1) call discover_capabilities, 2) invoke cad.generate_sample_3d or cad.import_3d, "
                "3) solve reference FEM via fem.solve_3d_solid, 4) export VTU/GLB via export.vtu_or_glb_3d, "
                "5) verify session ledger via governance.verify_session_ledger.",
            ),
            "multiphysics-3d-audit": (
                "Run a 3D multiphysics verification audit.",
                f"In workspace '{ws}', run multiphysics.solve_thermo_mechanical_3d, "
                "multiphysics.solve_modal_3d, and multiphysics.solve_stateful_j2_3d, "
                "reporting min_det_f, dissipation non-negativity, and natural frequencies.",
            ),
            "experimental-vv10-calibration": (
                "Execute ASME V&V 10 physical coupon calibration.",
                f"Invoke validate.experimental_vv10 on '{args.get('dataset_path', 'coupons.json')}' "
                "and report orthogonal ExperimentalValidationTier, |E|, and u_val without conflating with numerical ClaimLevel.",
            ),
            "generative-lightweighting": (
                "Perform FEM-certified lightweighting.",
                f"Target volume fraction {args.get('target_volume_fraction', '0.50')}: "
                "discover optimize capabilities, verify every design candidate with authoritative 3D FEM, "
                "and attach RFC 8785 receipts.",
            ),
            "diagnose-refusal-or-ood": (
                "Diagnose fail-closed refusal or OOD event.",
                f"Inspect error_code={args.get('error_code', 'UNKNOWN')}, read formamech://session/status "
                "and formamech://state-machines, then follow allowed_next_actions.",
            ),
        }
        if name not in templates:
            raise KeyError(f"unknown MCP prompt {name!r}")
        desc, body = templates[name]
        return {
            "description": desc,
            "messages": [{"role": "user", "content": {"type": "text", "text": body}}],
        }


__all__ = (
    "AGENT_WORKFLOW_MACHINE",
    "DEFAULT_MAX_DOF_LIMIT",
    "MCP_PROTOCOL_VERSION",
    "MCP_SESSION_MACHINE",
    "SERVER_NAME",
    "AgentBudgetSpec",
    "AgentWorkflowState",
    "AgentWorkflowTracker",
    "FormaMechMcpServer",
    "McpSessionState",
    "McpToolSpec",
    "ResourceLimitError",
    "SandboxViolationError",
    "redirect_native_stdout_to_stderr",
)
