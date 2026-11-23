"""Model Context Protocol (MCP) server for verified FormaMech mechanics workflows.

Exposes FormaMech's envelope-bounded inference, CAD inspection, guarded design
optimization, and RFC 8785 Trust Receipt verification as structured JSON-RPC
2.0 / MCP tools with strict workspace path sandboxing, DOF ceiling guards,
an explicit session finite state machine, and runtime JSON Schema validation.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
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

MCP_PROTOCOL_VERSION: Final = "2024-11-05"
SERVER_NAME: Final = "formamech-mcp"
DEFAULT_MAX_DOF_LIMIT: Final = 100_000


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
        require_qualified: bool = False,
        strict_lifecycle: bool = False,
    ) -> None:
        root = Path(workspace_root).resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"workspace_root must be an existing directory: {root}")
        if type(max_dof_limit) is not int or max_dof_limit < 2:
            raise ValueError("max_dof_limit must be an integer >= 2")
        if not isinstance(require_qualified, bool):
            raise TypeError("require_qualified must be boolean")
        if not isinstance(strict_lifecycle, bool):
            raise TypeError("strict_lifecycle must be boolean")
        self.workspace_root = root
        self.max_dof_limit = max_dof_limit
        self.require_qualified = require_qualified
        self.strict_lifecycle = strict_lifecycle
        self._session_state: McpSessionState = McpSessionState.UNINITIALIZED
        self._transition_history: list[TransitionRecord[McpSessionState]] = []
        self._tools: dict[str, McpToolSpec] = {
            spec.name: spec for spec in self._build_tools()
        }

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
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": __version__,
                    },
                },
            )

        if method in ("tools/list", "tools/call"):
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
        """Serve newline-delimited JSON-RPC 2.0 MCP frames over stdio."""
        in_stream = sys.stdin if stdin is None else stdin
        out_stream = sys.stdout if stdout is None else stdout
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
                    else self._error(None, -32600, "JSON-RPC message must be an object")
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


__all__ = (
    "DEFAULT_MAX_DOF_LIMIT",
    "MCP_PROTOCOL_VERSION",
    "MCP_SESSION_MACHINE",
    "SERVER_NAME",
    "FormaMechMcpServer",
    "McpSessionState",
    "McpToolSpec",
    "ResourceLimitError",
    "SandboxViolationError",
)
