"""Multi-provider LLM tool-schema exporters & dispatchers for AI agents (V4-0).

Exports FormaMech capabilities into native tool definitions for OpenAI,
Anthropic, Google Gemini, and MCP without requiring any provider SDKs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from piginot.harness.contracts import CapabilityDomain
from piginot.harness.registry import CapabilityRegistry
from piginot.harness.session import FormaMechAgentHarness

ProviderKind = Literal["openai", "anthropic", "gemini", "mcp"]
ExportMode = Literal["progressive", "full"]


def _progressive_meta_tools() -> list[dict[str, Any]]:
    """Five compact meta-tools giving any LLM full access to all 40+ capabilities."""
    return [
        {
            "name": "formamech_discover_capabilities",
            "description": (
                "List registered FormaMech engineering capabilities by domain, "
                "cost class, or keyword. Start here to keep context compact."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "enum": [d.value for d in CapabilityDomain],
                    },
                    "search": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "formamech_describe_capability",
            "description": "Fetch exact input/output JSON Schema and preconditions for one capability.",
            "parameters": {
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            },
        },
        {
            "name": "formamech_invoke_capability",
            "description": (
                "Execute (or dry-run) any registered FormaMech capability inside the "
                "sandboxed, budget-governed, hash-chained session harness."
            ),
            "parameters": {
                "type": "object",
                "required": ["capability"],
                "properties": {
                    "capability": {"type": "string"},
                    "arguments": {"type": "object"},
                    "dry_run": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "formamech_session_status",
            "description": "Inspect current session state, remaining FEM/epoch budget, and receipts.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "formamech_describe_state_machines",
            "description": "Return all registered formal FormaMech finite state machines.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    ]


def _sanitize_tool_name(cap_name: str) -> str:
    return "fm_" + cap_name.replace(".", "_").replace("-", "_")


def export_tool_schemas(
    registry: CapabilityRegistry,
    *,
    provider: ProviderKind = "openai",
    mode: ExportMode = "progressive",
    domain: CapabilityDomain | str | None = None,
) -> list[dict[str, Any]]:
    """Export FormaMech harness tools in native OpenAI, Anthropic, Gemini, or MCP schema format."""
    if mode == "progressive":
        raw_tools = _progressive_meta_tools()
    else:
        raw_tools = []
        for entry in registry.list_capabilities(domain=domain):
            full = registry.describe(entry["name"])
            raw_tools.append(
                {
                    "name": _sanitize_tool_name(full["name"]),
                    "capability_name": full["name"],
                    "description": f"[{full['domain']}/{full['cost_class']}] {full['summary']}",
                    "parameters": full["input_schema"],
                }
            )

    if provider == "openai":
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in raw_tools
        ]
    if provider == "anthropic":
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in raw_tools
        ]
    if provider == "gemini":
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            }
            for t in raw_tools
        ]
    if provider == "mcp":
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["parameters"],
            }
            for t in raw_tools
        ]
    raise ValueError(f"unsupported tool schema provider {provider!r}")


def dispatch_tool_call(
    harness: FormaMechAgentHarness,
    tool_name: str,
    arguments: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Dispatch a progressive meta-tool or flat ``fm_*`` tool call onto ``harness``."""
    parsed: Mapping[str, Any]
    if arguments is None:
        parsed = {}
    elif isinstance(arguments, str):
        parsed = json.loads(arguments) if arguments.strip() else {}
    else:
        parsed = arguments

    if tool_name == "formamech_discover_capabilities":
        return harness.invoke("discover.capabilities", parsed).document()
    if tool_name == "formamech_describe_capability":
        return harness.invoke("discover.describe", parsed).document()
    if tool_name == "formamech_describe_state_machines":
        return harness.invoke("discover.state_machines", {}).document()
    if tool_name == "formamech_session_status":
        return {
            "session_id": harness.session_id,
            "snapshot": harness.snapshot().document(),
            "budget": harness.budget_status(),
        }
    if tool_name == "formamech_invoke_capability":
        cap = str(parsed["capability"])
        cap_args = parsed.get("arguments", {})
        dry_run = bool(parsed.get("dry_run", False))
        return harness.invoke(cap, cap_args, dry_run=dry_run).document()

    # Map sanitized flat names (fm_cad_import_3d -> cad.import_3d)
    for entry in harness.registry.list_capabilities():
        if (
            _sanitize_tool_name(entry["name"]) == tool_name
            or entry["name"] == tool_name
        ):
            return harness.invoke(entry["name"], parsed).document()

    return harness.invoke(tool_name, parsed).document()


def build_agent_system_prompt(harness: FormaMechAgentHarness) -> str:
    """Generate a concise, grounded system prompt for autonomous FormaMech agents."""
    domains = harness.registry.domains_summary()
    dom_lines = "\n".join(f"- `{k}` ({v} capabilities)" for k, v in domains.items())
    budget = harness.budget_status()
    return (
        "You are operating the FormaMech Verified Solid Mechanics Agent Harness.\n\n"
        "## Rules & Epistemic Invariants\n"
        "1. Always discover capabilities progressively (`discover.capabilities` -> `discover.describe` -> `invoke`).\n"
        "2. Respect workspace sandboxing: all paths must remain relative inside the active workspace.\n"
        "3. Never claim physical experimental validation from numerical FEM verification alone "
        "(ASME V&V 10 separation: `ClaimLevel` <= `HOLDOUT_VERIFIED`; experimental claims require `validate.experimental_vv10`).\n"
        "4. On any error envelope (`ok: false`), inspect `error_code`, `recovery_hint`, and `allowed_next_actions` before retrying.\n"
        "5. Verify receipts (`governance.verify_receipt`) and session ledger (`governance.verify_session_ledger`) before delivering final artifacts.\n\n"
        f"## Active Session Budget (`{harness.session_id}`)\n"
        f"- FEM solves remaining: {budget['fem_solves_remaining']}/{budget['max_fem_solves']}\n"
        f"- Max DOFs: {budget['max_dofs']}\n"
        f"- Training epochs remaining: {budget['training_epochs_remaining']}/{budget['max_training_epochs']}\n\n"
        "## Available Capability Domains\n"
        f"{dom_lines}\n"
    )


__all__ = (
    "ExportMode",
    "ProviderKind",
    "build_agent_system_prompt",
    "dispatch_tool_call",
    "export_tool_schemas",
)
