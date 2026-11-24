"""End-to-end autonomous AI-agent test driving all FormaMech v0.4 capabilities (V4-0..V4-5).

Simulates an autonomous LLM agent operating FormaMech **exclusively** through:

1. MCP 2024-11-05 JSON-RPC (``initialize``, ``tools/list``, ``resources/list``,
   ``resources/read``, ``prompts/list``, ``prompts/get``, ``tools/call``),
2. Progressive discovery (``discover_capabilities`` -> ``describe_capability`` ->
   ``dry_run`` -> ``invoke_capability``) with self-healing error recovery,
3. Every v0.4 engineering pillar: live CAD connector, 3D solid FEM, DKT shells,
   Timoshenko beams, 3D eigen-buckling, transient dynamics, ASTM E1049 rainflow
   fatigue, AT2 phase-field fracture, shipped ``struct3d-base-v1`` foundation
   LoRA transfer, verified 3D SIMP topology optimization, Bernstein FFD shape
   optimization, parallel fleet execution, continual qualification status,
4. Live HTTP/JSON + Prometheus serving API with hash-chained ``SignedAuditLedger``,
5. Native tool schema exports for OpenAI, Anthropic, Gemini, and MCP.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from piginot.harness import (
    build_default_registry,
    export_tool_schemas,
)
from piginot.mcp_server import FormaMechMcpServer
from piginot.serving.api_server import FormaMechApiApplication, FormaMechHttpServer
from piginot.training.foundation_pretrain import default_foundation_checkpoint_dir


def _mcp_tool_json(
    server: FormaMechMcpServer, req_id: int, tool_name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
    )
    assert resp is not None and "result" in resp, f"MCP error: {resp}"
    text = resp["result"]["content"][0]["text"]
    return dict(json.loads(text))


def test_autonomous_ai_agent_full_v4_e2e_workflow(tmp_path: Path) -> None:
    """Autonomous agent discovers, dry-runs, self-heals, executes V4-0..V4-5 & verifies ledgers."""
    ws = tmp_path / "autonomous_study"
    ws.mkdir(parents=True)
    server = FormaMechMcpServer(ws, max_dof_limit=150_000, max_fem_solves=32)

    # --- Turn 1: MCP 2024-11-05 handshake, tools, resources, and engineering prompts ---
    init_resp = server.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    assert init_resp is not None
    caps = init_resp["result"]["capabilities"]
    assert "tools" in caps and "resources" in caps and "prompts" in caps

    tools_resp = server.handle_message(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    )
    assert tools_resp is not None
    tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
    assert {
        "discover_capabilities",
        "describe_capability",
        "invoke_capability",
        "session_status",
        "describe_state_machines",
    }.issubset(tool_names)

    res_list = server.handle_message(
        {"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}}
    )
    assert res_list is not None
    uris = {r["uri"] for r in res_list["result"]["resources"]}
    assert "formamech://capabilities" in uris
    assert "formamech://state-machines" in uris

    cap_read = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/read",
            "params": {"uri": "formamech://capabilities"},
        }
    )
    assert cap_read is not None
    cap_doc = json.loads(cap_read["result"]["contents"][0]["text"])
    assert cap_doc["count"] >= 20

    prompt_get = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "prompts/get",
            "params": {"name": "generative-lightweighting", "arguments": {}},
        }
    )
    assert prompt_get is not None
    assert prompt_get["result"]["messages"]

    # --- Turn 2: Progressive Discovery & Self-Healing Error Recovery ---
    disc = _mcp_tool_json(server, 10, "discover_capabilities", {"domain": "optimize"})
    opt_names = {c["name"] for c in disc["capabilities"]}
    assert {"optimize.topology_simp_3d", "optimize.ffd_shape_3d"}.issubset(opt_names)

    desc = _mcp_tool_json(
        server, 11, "describe_capability", {"name": "fem.solve_shell_plate_3d"}
    )
    assert desc["name"] == "fem.solve_shell_plate_3d"
    assert "thickness_mm" in desc["input_schema"]["properties"]

    # Deliberately pass invalid thickness_mm (-5.0) -> fail-closed structured recovery hint
    bad = _mcp_tool_json(
        server,
        12,
        "invoke_capability",
        {
            "capability": "fem.solve_shell_plate_3d",
            "arguments": {"thickness_mm": -5.0},
        },
    )
    assert bad["ok"] is False
    assert bad["error_code"] == "INVALID_ARGUMENTS"
    assert bad["recovery_hint"]

    # Dry-run pre-flight check consumes zero FEM solve budget
    dry = _mcp_tool_json(
        server,
        13,
        "invoke_capability",
        {
            "capability": "fem.solve_shell_plate_3d",
            "arguments": {
                "lx_mm": 80.0,
                "ly_mm": 80.0,
                "thickness_mm": 2.0,
                "nx": 6,
                "ny": 6,
            },
            "dry_run": True,
        },
    )
    assert dry["ok"] is True and dry["dry_run"] is True

    # --- Turn 3: Autonomous Execution across V4-1..V4-5 via MCP invoke_capability ---
    # 3a) V4-1 CAD connector + backend inventory
    backends = _mcp_tool_json(
        server,
        20,
        "invoke_capability",
        {"capability": "discover.backends", "arguments": {}},
    )
    assert backends["ok"] is True
    assert "external_oracles" in backends["result"]
    assert "cad_connectors" in backends["result"]

    cad_gen = _mcp_tool_json(
        server,
        21,
        "invoke_capability",
        {
            "capability": "cad.connector_generate",
            "arguments": {
                "connector": "opencascade-builtin",
                "model_ref": "parametric-holed-block",
                "parameters": {
                    "lx": 28.0,
                    "ly": 14.0,
                    "lz": 10.0,
                    "hole_radius": 3.0,
                },
                "output_step": "agent_holed.step",
            },
        },
    )
    assert cad_gen["ok"] is True

    cad_imp = _mcp_tool_json(
        server,
        211,
        "invoke_capability",
        {
            "capability": "cad.import_3d",
            "arguments": {
                "step_path": "agent_holed.step",
                "workspace_dir": "study_3d",
                "mesh_size_mm": 9.0,
                "element_type": "tet4",
            },
        },
    )
    assert cad_imp["ok"] is True

    r_solid = _mcp_tool_json(
        server,
        212,
        "invoke_capability",
        {
            "capability": "fem.solve_3d_solid",
            "arguments": {
                "workspace_dir": "study_3d",
                "youngs_modulus_mpa": 100.0,
                "poisson_ratio": 0.30,
                "load_steps": 1,
            },
        },
    )
    assert r_solid["ok"] is True and r_solid["result"]["converged"] is True

    # 3b) V4-3 Production Physics Pack (Shell, Beam, Buckling, Dynamics, Fatigue, Fracture)
    r_shell = _mcp_tool_json(
        server,
        22,
        "invoke_capability",
        {
            "capability": "fem.solve_shell_plate_3d",
            "arguments": {
                "lx_mm": 80.0,
                "ly_mm": 80.0,
                "thickness_mm": 2.0,
                "nx": 6,
                "ny": 6,
            },
        },
    )
    assert r_shell["ok"] is True and r_shell["result"]["max_deflection_z_mm"] > 0.0

    r_beam = _mcp_tool_json(
        server,
        23,
        "invoke_capability",
        {
            "capability": "fem.solve_timoshenko_beam_3d",
            "arguments": {"length_mm": 60.0, "tip_load_y_n": 250.0},
        },
    )
    assert r_beam["ok"] is True and r_beam["result"]["max_displacement_mm"] > 0.0

    r_buck = _mcp_tool_json(
        server,
        24,
        "invoke_capability",
        {
            "capability": "fem.solve_linear_buckling_3d",
            "arguments": {"length_mm": 40.0, "nx": 4, "num_modes": 1},
        },
    )
    assert r_buck["ok"] is True and r_buck["result"]["critical_load_factor"] > 0.0

    r_dyn = _mcp_tool_json(
        server,
        25,
        "invoke_capability",
        {
            "capability": "multiphysics.solve_transient_dynamics_3d",
            "arguments": {"scheme": "newmark-beta", "num_steps": 5},
        },
    )
    assert r_dyn["ok"] is True and r_dyn["result"]["num_steps"] == 5

    r_fat = _mcp_tool_json(
        server,
        26,
        "invoke_capability",
        {
            "capability": "multiphysics.evaluate_fatigue_rainflow",
            "arguments": {"stress_history_mpa": [0.0, 240.0, -240.0, 240.0, 0.0]},
        },
    )
    assert r_fat["ok"] is True and r_fat["result"]["miner_damage"] > 0.0

    r_pf = _mcp_tool_json(
        server,
        27,
        "invoke_capability",
        {
            "capability": "multiphysics.solve_phase_field_fracture_3d",
            "arguments": {"end_displacement_x_mm": 0.08, "load_steps": 4},
        },
    )
    assert r_pf["ok"] is True and r_pf["result"]["monotone_irreversible"] is True

    # 3c) V4-2 Shipped Foundation Checkpoint 3D LoRA Transfer
    ckpt = default_foundation_checkpoint_dir()
    assert (ckpt / "foundation.json").is_file()
    r_lora = _mcp_tool_json(
        server,
        28,
        "invoke_capability",
        {
            "capability": "surrogate.adapt_foundation_lora_3d",
            "arguments": {"workspace_dir": "study_3d", "rank": 4, "steps": 4},
        },
    )
    assert r_lora["ok"] is True and r_lora["result"]["state"] == "COMPLETED"

    # 3d) V4-4 Verified 3D SIMP Topology Opt & Bernstein FFD Shape Opt
    r_simp = _mcp_tool_json(
        server,
        29,
        "invoke_capability",
        {
            "capability": "optimize.topology_simp_3d",
            "arguments": {
                "nx": 3,
                "ny": 2,
                "nz": 1,
                "max_iterations": 6,
                "volume_fraction": 0.6,
                "threshold_density": 0.3,
            },
        },
    )
    assert r_simp["ok"] is True
    assert r_simp["result"]["state"] == "CERTIFIED"
    assert r_simp["result"]["claim_level"] == "SINGLE_POINT_VERIFIED"

    r_ffd = _mcp_tool_json(
        server,
        30,
        "invoke_capability",
        {
            "capability": "optimize.ffd_shape_3d",
            "arguments": {"iterations": 3, "max_displacement_limit_mm": 0.3},
        },
    )
    assert r_ffd["ok"] is True and r_ffd["result"]["min_det_j"] > 0.0

    # 3e) V4-5 Distributed Fleet Batch & Continual Status
    r_fleet = _mcp_tool_json(
        server,
        31,
        "invoke_capability",
        {
            "capability": "governance.fleet_submit_3d",
            "arguments": {"backend": "local-threads", "job_id": "e2e-batch"},
        },
    )
    assert r_fleet["ok"] is True and r_fleet["result"]["state"] == "VERIFIED"

    r_cont = _mcp_tool_json(
        server,
        32,
        "invoke_capability",
        {"capability": "governance.continual_status", "arguments": {}},
    )
    assert r_cont["ok"] is True and r_cont["result"]["ledger_valid"] is True

    # --- Turn 4: Verify Cryptographic Session Hash Chain via MCP Resources ---
    status = _mcp_tool_json(server, 40, "session_status", {})
    assert status["ledger_verified"] is True
    assert status["ledger_errors"] == []

    # --- Turn 5: Live Production Serving API + SignedAuditLedger Verification ---
    api_ws = tmp_path / "api_ws"
    api_ws.mkdir()
    app = FormaMechApiApplication(api_ws)
    httpd = FormaMechHttpServer(app, host="127.0.0.1", port=0)
    httpd.start_background()
    try:
        base = httpd.base_url
        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            assert json.loads(r.read().decode("utf-8"))["status"] == "ok"

        req_body = json.dumps(
            {
                "capability": "multiphysics.evaluate_fatigue_rainflow",
                "arguments": {"stress_history_mpa": [0.0, 200.0, -200.0, 0.0]},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/v1/harness/invoke",
            data=req_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            inv_doc = json.loads(r.read().decode("utf-8"))
            assert inv_doc["ok"] is True

        with urllib.request.urlopen(f"{base}/v1/ledger/verify", timeout=5) as r:
            lv = json.loads(r.read().decode("utf-8"))
            assert lv["verified"] is True

        with urllib.request.urlopen(f"{base}/metrics", timeout=5) as r:
            prom = r.read().decode("utf-8")
            assert "formamech_requests_total" in prom
    finally:
        httpd.shutdown()

    # --- Turn 6: Multi-Provider Native LLM Tool Schema Exports ---
    reg = build_default_registry()
    for provider in ("openai", "anthropic", "gemini", "mcp"):
        prog_tools = export_tool_schemas(reg, provider=provider, mode="progressive")
        full_tools = export_tool_schemas(reg, provider=provider, mode="full")
        assert len(prog_tools) == 5
        assert len(full_tools) >= 20
