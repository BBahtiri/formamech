"""Model Context Protocol (MCP) server for verified FormaMech mechanics workflows.

Exposes FormaMech's envelope-bounded inference, CAD inspection, guarded design
optimization, and RFC 8785 Trust Receipt verification as structured JSON-RPC
2.0 / MCP tools with strict workspace path sandboxing and DOF ceiling guards.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TextIO, cast

from piginot import __version__
from piginot.certify import verify_receipt_document

MCP_PROTOCOL_VERSION: Final = "2024-11-05"
SERVER_NAME: Final = "formamech-mcp"
DEFAULT_MAX_DOF_LIMIT: Final = 100_000


class SandboxViolationError(PermissionError):
    """Raised when an MCP tool request attempts to escape the workspace root."""


class ResourceLimitError(ValueError):
    """Raised when an MCP tool request exceeds the configured DOF or budget ceiling."""


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
    ) -> None:
        root = Path(workspace_root).resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"workspace_root must be an existing directory: {root}")
        if type(max_dof_limit) is not int or max_dof_limit < 2:
            raise ValueError("max_dof_limit must be an integer >= 2")
        if not isinstance(require_qualified, bool):
            raise TypeError("require_qualified must be boolean")
        self.workspace_root = root
        self.max_dof_limit = max_dof_limit
        self.require_qualified = require_qualified
        self._tools: dict[str, McpToolSpec] = {
            spec.name: spec for spec in self._build_tools()
        }

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
        return spec.handler(args)

    def handle_message(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        """Handle one JSON-RPC 2.0 MCP message and return the response (or None for notifications)."""
        req_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}
        if not isinstance(method, str):
            return self._error(req_id, -32600, "invalid request: missing method")

        if method == "notifications/initialized":
            return None
        if method == "ping":
            return self._result(req_id, {})
        if method == "initialize":
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
            except (SandboxViolationError, ResourceLimitError) as exc:
                return self._result(
                    req_id,
                    {
                        "isError": True,
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "error": type(exc).__name__,
                                        "message": str(exc),
                                    },
                                    sort_keys=True,
                                ),
                            }
                        ],
                    },
                )
            except Exception as exc:
                return self._result(
                    req_id,
                    {
                        "isError": True,
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "error": type(exc).__name__,
                                        "message": str(exc),
                                    },
                                    sort_keys=True,
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
                            "text": json.dumps(payload, sort_keys=True),
                        }
                    ],
                },
            )
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
                out_stream.write(json.dumps(resp, sort_keys=True) + "\n")
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
        tta_steps = int(args.get("tta_steps", 0) or 0)
        tta_cfg = TestTimeLoraConfig(steps=tta_steps) if tta_steps > 0 else None
        served = serve(
            bundle,
            parsed_inputs,
            cast("Sequence[Sequence[float]] | None", parsed_queries),
            mode=mode,
            recipe=recipe,
            corrected=mode is not ServingMode.FAST,
            escalate=bool(args.get("escalate", False)),
            tolerance=(
                None if args.get("tolerance") is None else float(args["tolerance"])
            ),
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
        from piginot.qualify import QualificationPolicy, check_deployment_gate
        from piginot.verification.status import ClaimLevel

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
    "SERVER_NAME",
    "FormaMechMcpServer",
    "McpToolSpec",
    "ResourceLimitError",
    "SandboxViolationError",
)
