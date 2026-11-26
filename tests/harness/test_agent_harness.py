"""Verification tests for Phase 0 (V4-0) AI-Ready Universal Agent Harness."""

from __future__ import annotations

import json
from pathlib import Path

from piginot.cli import main as cli_main
from piginot.harness import (
    FormaMechAgentHarness,
    HarnessBudget,
    HarnessSessionState,
    build_agent_system_prompt,
    build_default_registry,
    dispatch_tool_call,
    export_tool_schemas,
)
from piginot.mcp_server import FormaMechMcpServer


def test_harness_discovery_dry_run_execution_resume_and_ledger_tamper(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "agent_ws"
    harness = FormaMechAgentHarness(
        ws,
        budget=HarnessBudget(max_fem_solves=3, max_dofs=50_000, max_training_epochs=20),
    )
    assert harness.state is HarnessSessionState.INITIALIZED

    # 1) Progressive discovery & schema inspection
    disc = harness.invoke("discover.capabilities", {"domain": "cad"})
    assert disc.ok is True
    assert disc.result is not None
    assert disc.result["count"] >= 4
    assert harness.state is HarnessSessionState.ACTIVE

    desc = harness.invoke("discover.describe", {"name": "fem.solve_3d_solid"})
    assert desc.ok is True
    assert desc.result is not None
    assert desc.result["cost_class"] == "SOLVER_ORACLE"

    # 2) Dry-run validates args & does not consume FEM budget
    dry = harness.invoke(
        "fem.solve_3d_solid", {"youngs_modulus_mpa": 120.0}, dry_run=True
    )
    assert dry.ok is True
    assert dry.dry_run is True
    assert harness.fem_solves_used == 0

    # 3) Real CAD generation -> 3D FEM solve -> VTU + GLB export
    cad = harness.invoke(
        "cad.generate_sample_3d",
        {"workspace_dir": "box_study", "template": "box", "mesh_size_mm": 8.0},
    )
    assert cad.ok is True
    assert cad.artifact_paths

    fem = harness.invoke(
        "fem.solve_3d_solid",
        {"workspace_dir": "box_study", "youngs_modulus_mpa": 100.0, "load_steps": 2},
    )
    assert fem.ok is True
    assert fem.receipt_sha256 is not None
    assert harness.fem_solves_used == 1

    exp_vtu = harness.invoke(
        "export.vtu_or_glb_3d",
        {"workspace_dir": "box_study", "output_path": "out/part.vtu", "format": "vtu"},
    )
    assert exp_vtu.ok is True
    assert (ws / "out" / "part.vtu").is_file()

    # 4) Fail-closed sandbox refusal & recovery guidance
    bad_path = harness.invoke("cad.import_3d", {"step_path": "../../outside.step"})
    assert bad_path.ok is False
    assert bad_path.error_code == "SANDBOX_VIOLATION"
    assert bad_path.recovery_hint

    # 5) Suspend & cross-process resume preserves budget + ledger integrity
    harness.suspend()
    assert harness.state is HarnessSessionState.SUSPENDED
    resumed = FormaMechAgentHarness.resume(ws)
    assert resumed.state is HarnessSessionState.ACTIVE
    assert resumed.fem_solves_used == 1
    assert resumed.session_id == harness.session_id
    ok_led, errs_led = resumed.verify_ledger()
    assert ok_led is True and errs_led == ()

    # 6) Deterministic replay into fresh workspace reproduces artifacts
    replay_ws = tmp_path / "replay_ws"
    replayed = resumed.replay_into(replay_ws)
    assert all(e.ok for e in replayed)
    assert (replay_ws / "out" / "part.vtu").is_file()

    # 7) Tampering with a single byte of ledger.jsonl is detected
    raw_lines = resumed.ledger_file.read_text(encoding="utf-8").splitlines()
    mutated = raw_lines[0].replace("discover.capabilities", "discover.tampered")
    resumed.ledger_file.write_text(
        "\n".join([mutated, *raw_lines[1:]]) + "\n", encoding="utf-8"
    )
    ok_bad, errs_bad = resumed.verify_ledger()
    assert ok_bad is False
    assert errs_bad


def test_harness_budget_exhaustion_provider_exporters_mcp_and_cli(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "budget_ws"
    harness = FormaMechAgentHarness(
        ws, budget=HarnessBudget(max_fem_solves=1, max_dofs=50_000)
    )
    assert harness.invoke(
        "cad.generate_sample_3d",
        {"workspace_dir": "s1", "template": "box", "mesh_size_mm": 9.0},
    ).ok
    assert harness.invoke(
        "fem.solve_3d_solid", {"workspace_dir": "s1", "load_steps": 1}
    ).ok
    # Second FEM solve exceeds max_fem_solves=1 -> BUDGET_EXHAUSTED
    over = harness.invoke(
        "fem.solve_3d_solid", {"workspace_dir": "s1", "load_steps": 1}
    )
    assert over.ok is False
    assert over.error_code == "BUDGET_OR_RESOURCE_LIMIT"
    assert harness.state is HarnessSessionState.BUDGET_EXHAUSTED
    # ZERO_SOLVE governance/discovery calls remain allowed after budget exhaustion
    chk = harness.invoke("governance.verify_session_ledger", {})
    assert chk.ok is True

    # Multi-provider schema exports + dispatcher + system prompt
    reg = build_default_registry()
    for provider in ("openai", "anthropic", "gemini", "mcp"):
        prog = export_tool_schemas(reg, provider=provider, mode="progressive")
        full = export_tool_schemas(reg, provider=provider, mode="full")
        assert len(prog) == 5
        assert len(full) == len(reg)

    dispatched = dispatch_tool_call(
        harness, "formamech_discover_capabilities", '{"domain": "governance"}'
    )
    assert dispatched["ok"] is True
    prompt = build_agent_system_prompt(harness)
    assert "ASME V&V 10" in prompt

    # MCP 2024-11-05 tools + resources + prompts round-trip
    mcp_ws = tmp_path / "mcp_ws"
    mcp_ws.mkdir(exist_ok=True)
    mcp = FormaMechMcpServer(mcp_ws)
    init = mcp.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    assert init is not None
    assert "resources" in init["result"]["capabilities"]
    assert "prompts" in init["result"]["capabilities"]
    mcp.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"})

    res_list = mcp.handle_message(
        {"jsonrpc": "2.0", "id": 2, "method": "resources/list"}
    )
    assert res_list is not None
    uris = {r["uri"] for r in res_list["result"]["resources"]}
    assert {"formamech://capabilities", "formamech://state-machines"}.issubset(uris)

    res_read = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/read",
            "params": {"uri": "formamech://capabilities"},
        }
    )
    assert res_read is not None
    cap_doc = json.loads(res_read["result"]["contents"][0]["text"])
    assert cap_doc["count"] >= 15

    pr_list = mcp.handle_message({"jsonrpc": "2.0", "id": 4, "method": "prompts/list"})
    assert pr_list is not None
    assert len(pr_list["result"]["prompts"]) == 5

    pr_get = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "prompts/get",
            "params": {"name": "cad-to-verified-surrogate"},
        }
    )
    assert pr_get is not None
    assert pr_get["result"]["messages"]

    # CLI `formamech agent` commands
    caps_out = tmp_path / "caps.json"
    assert cli_main(["agent", "capabilities", "--output", str(caps_out)]) == 0
    assert json.loads(caps_out.read_text(encoding="utf-8"))["count"] >= 15

    tools_out = tmp_path / "openai_tools.json"
    assert (
        cli_main(
            [
                "agent",
                "export-tools",
                "--provider",
                "openai",
                "--output",
                str(tools_out),
            ]
        )
        == 0
    )
    assert json.loads(tools_out.read_text(encoding="utf-8"))["count"] == 5

    cli_ws = tmp_path / "cli_agent_ws"
    inv_out = tmp_path / "inv.json"
    assert (
        cli_main(
            [
                "agent",
                "invoke",
                "discover.backends",
                "--workspace",
                str(cli_ws),
                "--output",
                str(inv_out),
            ]
        )
        == 0
    )
    assert json.loads(inv_out.read_text(encoding="utf-8"))["ok"] is True
    assert cli_main(["agent", "verify-ledger", "--workspace", str(cli_ws)]) == 0


def test_harness_v4_hardening_mcp_stdio_isolation_budget_self_heal_and_tamper_preservation(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys

    # 1) Real `formamech mcp serve` subprocess over stdio stays 100% pure JSON
    #    even when OpenCASCADE writes C-level STEP transfer statistics!
    mcp_ws = tmp_path / "mcp_stdio_ws"
    mcp_ws.mkdir()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "formamech",
            "mcp",
            "serve",
            "--workspace",
            str(mcp_ws),
            "--max-dofs",
            "150000",
            "--max-fem-solves",
            "16",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    reqs = (
        "\n".join(
            [
                json.dumps(
                    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
                ),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "resources/templates/list",
                        "params": {},
                    }
                ),
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {
                            "name": "invoke_capability",
                            "arguments": {
                                "capability": "cad.connector_generate",
                                "arguments": {
                                    "connector": "opencascade-builtin",
                                    "model_ref": "parametric-holed-block",
                                    "parameters": {
                                        "lx": 24.0,
                                        "ly": 12.0,
                                        "lz": 8.0,
                                        "hole_radius": 2.5,
                                    },
                                    "output_step": "cad/part.step",
                                },
                            },
                        },
                    }
                ),
            ]
        )
        + "\n"
    )
    out, _ = proc.communicate(reqs, timeout=30)
    assert proc.returncode == 0
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 3, (
        f"Unexpected stdout lines contaminating MCP stdio: {lines!r}"
    )
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0]["id"] == 1
    assert (
        parsed[1]["result"]["resourceTemplates"][0]["uriTemplate"]
        == "formamech://artifact/{relpath}"
    )
    step_env = json.loads(parsed[2]["result"]["content"][0]["text"])
    assert step_env["ok"] is True

    # 2) Oversized budget request when remaining > 0 refuses WITHOUT permanently locking session
    ws = tmp_path / "self_heal_budget"
    h = FormaMechAgentHarness(
        ws, budget=HarnessBudget(max_fem_solves=2, max_dofs=50_000)
    )
    over = h.invoke(
        "multiphysics.solve_stateful_j2_3d",
        {"output_dir": "cyc_over", "ux_sequence_mm": [0.05, 0.10, 0.15, 0.20]},
    )
    assert over.ok is False
    assert over.error_code == "BUDGET_OR_RESOURCE_LIMIT"
    assert h.state is HarnessSessionState.ACTIVE
    assert h.fem_solves_used == 0
    # Self-healed smaller retry succeeds and consumes budget accurately
    ok_retry = h.invoke(
        "fem.solve_timoshenko_beam_3d",
        {"length_mm": 60.0, "tip_load_y_n": 100.0},
    )
    assert ok_retry.ok is True
    assert h.fem_solves_used == 1

    # 3) All allowed_next_actions resolve to registered capabilities
    reg = h.registry
    for cap_entry in reg.list_capabilities():
        spec, _ = reg.get(cap_entry["name"])
        for nxt in FormaMechAgentHarness._suggest_next(spec):
            assert reg.has(nxt), f"_suggest_next recommended unknown capability {nxt!r}"

    # 4) dispatch_tool_call returns self-healing refusal on malformed JSON / missing capability
    bad_json = dispatch_tool_call(h, "formamech_invoke_capability", "{bad json")
    assert bad_json["ok"] is False and bad_json["error_code"] == "INVALID_ARGUMENTS"
    missing_cap = dispatch_tool_call(h, "formamech_invoke_capability", {})
    assert (
        missing_cap["ok"] is False and missing_cap["error_code"] == "INVALID_ARGUMENTS"
    )

    # 5) Failed resume() on tampered ledger does NOT clobber session.json,
    #    and CLI verify-ledger emits structured JSON with EXIT_REFUSED (2)
    orig_session = h.session_file.read_bytes()
    led_txt = h.ledger_file.read_text(encoding="utf-8")
    h.ledger_file.write_text(
        led_txt.replace('"ok": true', '"ok": false', 1), encoding="utf-8"
    )
    try:
        FormaMechAgentHarness.resume(ws)
        raise AssertionError("resume() must reject tampered ledger")
    except ValueError as exc:
        assert "tampered" in str(exc)
    assert h.session_file.read_bytes() == orig_session

    led_out = tmp_path / "tamper_verify.json"
    rc = cli_main(
        ["agent", "verify-ledger", "--workspace", str(ws), "--output", str(led_out)]
    )
    assert rc == 2
    led_doc = json.loads(led_out.read_text(encoding="utf-8"))
    assert led_doc["verified"] is False
    assert led_doc["errors"]


def test_harness_audit_fixes_linear_solver_receipts_and_certified_budget(
    tmp_path: Path,
) -> None:
    """Audit regressions: linear_solver pass-through, opt receipts/artifacts, and bundle.json."""
    ws = tmp_path / "audit_ws"
    h = FormaMechAgentHarness(
        ws,
        budget=HarnessBudget(max_fem_solves=8, max_dofs=50_000, max_training_epochs=10),
    )
    assert h.invoke(
        "cad.generate_sample_3d",
        {"workspace_dir": "box", "template": "box", "mesh_size_mm": 9.0},
    ).ok
    fem_pcg = h.invoke(
        "fem.solve_3d_solid",
        {"workspace_dir": "box", "load_steps": 1, "linear_solver": "pcg-ilu"},
    )
    assert fem_pcg.ok and fem_pcg.result is not None
    assert fem_pcg.result["converged"] is True

    e_simp = h.invoke(
        "optimize.topology_simp_3d",
        {
            "nx": 3,
            "ny": 2,
            "nz": 1,
            "max_iterations": 4,
            "volume_fraction": 0.6,
            "threshold_density": 0.3,
            "output_dir": "out_simp",
        },
    )
    assert e_simp.ok and e_simp.receipt_sha256 is not None
    assert (ws / "out_simp" / "simp_3d_result.npz").is_file()

    e_ffd = h.invoke(
        "optimize.ffd_shape_3d",
        {"iterations": 2, "max_displacement_limit_mm": 0.30, "output_dir": "out_ffd"},
    )
    assert e_ffd.ok and e_ffd.receipt_sha256 is not None
    assert (ws / "out_ffd" / "ffd_3d_result.npz").is_file()

    e_train = h.invoke(
        "surrogate.train_parametric_3d",
        {"output_dir": "p3d_b", "train_count": 2, "epochs": 2, "mesh_size_mm": 10.0},
    )
    assert e_train.ok
    assert any(p.endswith("bundle.json") for p in e_train.artifact_paths)
    assert (ws / "p3d_b" / "bundle.json").is_file()

    solves_before = h.fem_solves_used
    e_cert = h.invoke(
        "surrogate.predict_parametric_3d",
        {
            "bundle_dir": "p3d_b",
            "parameters": {"lx": 20.0, "ly": 12.0, "hole_radius": 2.4},
            "mode": "certified",
        },
    )
    assert e_cert.ok
    assert h.fem_solves_used == solves_before + 1
