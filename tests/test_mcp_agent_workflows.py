"""End-to-end & security verification for Phase 5 MCP Agentic Engineering Copilot (V3-5)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from piginot.contracts.state_machine import InvalidTransitionError
from piginot.export.vtu_export import export_fields_to_vtu
from piginot.geometry.mesh3d import CellType3D
from piginot.mcp_agent_workflows import AgentWorkflowState
from piginot.mcp_server import (
    FormaMechMcpServer,
    ResourceLimitError,
    SandboxViolationError,
)
from piginot.ui.workbench_3d import Workbench3DWorkspace


def _rpc_call(
    server: FormaMechMcpServer, req_id: int, tool: str, arguments: dict[str, object]
) -> dict[str, object]:
    resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
    )
    assert resp is not None
    return resp


def test_mcp_agentic_workflow_e2e_lifecycle_and_exports(tmp_path: Path) -> None:
    server = FormaMechMcpServer(tmp_path, max_dof_limit=50_000, max_fem_solves=4)

    # 1) Handshake via stdio
    req_lines = "\n".join(
        [
            json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
            ),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        ]
    )
    out = io.StringIO()
    server.serve_stdio(stdin=io.StringIO(req_lines + "\n"), stdout=out)
    assert server.workflow_state is AgentWorkflowState.IDLE

    # 2) import_cad_part (IDLE -> CAD_IMPORTED)
    r_imp = _rpc_call(
        server,
        2,
        "import_cad_part",
        {"workspace_dir": "study_box", "sample_3d_box": True, "mesh_size_mm": 8.0},
    )
    assert r_imp["result"]["isError"] is False  # type: ignore[index]
    sc_imp = r_imp["result"]["structuredContent"]  # type: ignore[index]
    assert sc_imp["workflow_state"]["state"] == "CAD_IMPORTED"
    assert sc_imp["dof_count"] > 0

    # 3) propose_physics_setup (CAD_IMPORTED -> SETUP_PROPOSED)
    r_prop = _rpc_call(
        server,
        3,
        "propose_physics_setup",
        {
            "workspace_dir": "study_box",
            "youngs_modulus_mpa": 120.0,
            "poisson_ratio": 0.30,
            "load_ux_mm": 0.30,
        },
    )
    assert r_prop["result"]["isError"] is False  # type: ignore[index]
    assert (
        r_prop["result"]["structuredContent"]["workflow_state"]["state"]
        == "SETUP_PROPOSED"
    )  # type: ignore[index]

    # 4) compile_and_qualify (SETUP_PROPOSED -> COMPILED -> QUALIFIED)
    r_qual = _rpc_call(
        server,
        4,
        "compile_and_qualify",
        {"workspace_dir": "study_box", "load_steps": 2},
    )
    assert r_qual["result"]["isError"] is False  # type: ignore[index]
    sc_qual = r_qual["result"]["structuredContent"]  # type: ignore[index]
    assert sc_qual["qualified"] is True
    assert sc_qual["workflow_state"]["state"] == "QUALIFIED"
    assert sc_qual["workflow_state"]["metadata"]["fem_solves_used"] == 1

    # 5) run_active_fem_round (budgeted: 2 solves -> 3/4 used)
    r_act = _rpc_call(
        server,
        5,
        "run_active_fem_round",
        {"workspace_dir": "study_box", "n_samples": 2, "youngs_modulus_mpa": 100.0},
    )
    assert r_act["result"]["isError"] is False  # type: ignore[index]
    sc_act = r_act["result"]["structuredContent"]  # type: ignore[index]
    assert len(sc_act["samples"]) == 2
    assert sc_act["workflow_state"]["metadata"]["fem_solves_used"] == 3
    assert sc_act["workflow_state"]["metadata"]["fem_solves_remaining"] == 1

    # 6) Budget exhaustion refusal (requesting 2 more when only 1 remains)
    r_over = _rpc_call(
        server,
        6,
        "run_active_fem_round",
        {"workspace_dir": "study_box", "n_samples": 2},
    )
    assert r_over["result"]["isError"] is True  # type: ignore[index]
    assert "ResourceLimitError" in r_over["result"]["content"][0]["text"]  # type: ignore[index]

    # 7) export_verified_artifact (QUALIFIED -> EXPORTED for both VTU and GLB)
    r_vtu = _rpc_call(
        server,
        7,
        "export_verified_artifact",
        {
            "workspace_dir": "study_box",
            "output_path": "exports/verified.vtu",
            "format": "vtu",
        },
    )
    assert r_vtu["result"]["isError"] is False  # type: ignore[index]
    assert (tmp_path / "exports" / "verified.vtu").is_file()
    assert r_vtu["result"]["structuredContent"]["workflow_state"]["state"] == "EXPORTED"  # type: ignore[index]

    r_glb = _rpc_call(
        server,
        8,
        "export_verified_artifact",
        {
            "workspace_dir": "study_box",
            "output_path": "exports/verified.glb",
            "format": "glb",
        },
    )
    assert r_glb["result"]["isError"] is False  # type: ignore[index]
    glb_bytes = (tmp_path / "exports" / "verified.glb").read_bytes()
    assert glb_bytes[:4] == b"glTF"


def test_mcp_agentic_security_fsm_and_ingest_vv10_tools(tmp_path: Path) -> None:
    server = FormaMechMcpServer(tmp_path, max_dof_limit=50_000, max_fem_solves=4)

    # 1) Illegal FSM transitions fail closed with InvalidTransitionError
    with pytest.raises(InvalidTransitionError):
        server.call_tool("propose_physics_setup", {"workspace_dir": "ws"})
    with pytest.raises(InvalidTransitionError):
        server.call_tool("compile_and_qualify", {"workspace_dir": "ws"})
    with pytest.raises(InvalidTransitionError):
        server.call_tool(
            "export_verified_artifact",
            {"workspace_dir": "ws", "output_path": "out.vtu", "format": "vtu"},
        )

    # Verify JSON-RPC surfaces structured transition audit on illegal transition
    r_bad = _rpc_call(
        server,
        99,
        "export_verified_artifact",
        {"workspace_dir": "ws", "output_path": "out.vtu", "format": "vtu"},
    )
    assert r_bad["result"]["isError"] is True  # type: ignore[index]
    assert "transition" in r_bad["result"]["structuredContent"]  # type: ignore[index]

    # 2) Path-traversal sandbox violation refused
    with pytest.raises(SandboxViolationError):
        server.call_tool(
            "import_cad_part",
            {"workspace_dir": "../escape_attempt", "sample_3d_box": True},
        )

    # 3) DOF ceiling refused on import_cad_part
    tiny_server = FormaMechMcpServer(tmp_path, max_dof_limit=12, max_fem_solves=4)
    with pytest.raises(ResourceLimitError, match="max_dof_limit"):
        tiny_server.call_tool(
            "import_cad_part",
            {"workspace_dir": "tiny_ws", "sample_3d_box": True, "mesh_size_mm": 8.0},
        )

    # 4) ingest_cae_archive + run_experimental_validation tools
    ws3 = Workbench3DWorkspace(tmp_path / "src_mesh")
    ws3.import_sample_box(mesh_size_mm=9.0, element_type=CellType3D.TET4)
    mesh = ws3.load_mesh()
    assert mesh is not None
    arch = tmp_path / "archive"
    arch.mkdir()
    for i in range(3):
        u = np.zeros((len(mesh.nodes), 3), dtype=np.float64)
        u[:, 0] = 0.01 * (i + 1) * np.asarray(mesh.nodes)[:, 0]
        export_fields_to_vtu(mesh, arch / f"c_{i}.vtu", point_fields={"u": u})

    ing = server.call_tool("ingest_cae_archive", {"archive_dir": "archive"})
    assert ing["screening_report"]["accepted_count"] == 3

    mu = 4.0
    lams = [1.05, 1.15, 1.30]
    p11 = [mu * (lam - 1.0 / (lam * lam)) for lam in lams]
    ds = {
        "model_family": "incompressible_neo_hookean",
        "coupons": [
            {
                "specimen_id": f"c{i}",
                "role": "VALIDATION_HOLDOUT" if i == 2 else "CALIBRATION",
                "test_mode": "uniaxial_tension",
                "stretches_or_strains": lams,
                "measured_stress_mpa": [s * (1.0 + 0.002 * (i - 1)) for s in p11],
                "measurement_std_mpa": 0.10,
            }
            for i in range(3)
        ],
    }
    (tmp_path / "vv10.json").write_text(json.dumps(ds), encoding="utf-8")
    vv = server.call_tool(
        "run_experimental_validation", {"dataset_path": "vv10.json", "u_num_rel": 0.05}
    )
    assert (
        vv["validation_dossier"]["experimental_validation_tier"]
        == "HOLDOUT_EXPERIMENTALLY_VALIDATED"
    )
