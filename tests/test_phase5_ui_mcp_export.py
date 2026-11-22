"""Phase 5 tests: Verification & Trust Inspector UI, MCP server, and ONNX / FMI 3.0 export."""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest
import torch
from formamech.export import (
    FMU_MANIFEST_SCHEMA,
    build_onnx_export_module,
    evaluate_fmu_step,
    export_bundle_to_fmu,
)
from formamech.mcp_server import (
    MCP_PROTOCOL_VERSION,
    FormaMechMcpServer,
    ResourceLimitError,
    SandboxViolationError,
)
from formamech.ui import (
    INSPECTOR_SCHEMA,
    build_inspector_snapshot,
    render_trust_dossier_markdown,
)

from piginot.bundle import load_bundle
from piginot.cli import main as cli_main
from piginot.cli import parse_request

ROOT = Path(__file__).resolve().parents[1]
NOTCHED_BUNDLE = ROOT / "bundles" / "notched-v1-cp-freeze"
NOTCHED_REQUEST = ROOT / "bundles" / "requests" / "notched-in-domain.json"


def _notched_raw_inputs() -> dict[str, dict[str, object]]:
    raw = json.loads(NOTCHED_REQUEST.read_text(encoding="utf-8"))
    return dict(raw["inputs"])


def test_inspector_snapshot_and_markdown_dossier_fast_and_certified() -> None:
    inputs = _notched_raw_inputs()
    snap_fast = build_inspector_snapshot(NOTCHED_BUNDLE, inputs, mode="fast")
    summary_fast = snap_fast.summary()
    assert summary_fast["schema"] == INSPECTOR_SCHEMA
    assert snap_fast.verdict == "IN_DOMAIN"
    assert snap_fast.receipt_verified is True
    assert snap_fast.receipt_errors == ()
    assert summary_fast["tier0_passed"] is True
    assert summary_fast["min_det_f_2d"] is not None
    assert summary_fast["min_det_f_2d"] > 0.0
    assert len(snap_fast.von_mises_mpa) == len(snap_fast.nodes)

    snap_cert = build_inspector_snapshot(NOTCHED_BUNDLE, inputs, mode="certified")
    assert snap_cert.verdict == "IN_DOMAIN"
    assert snap_cert.verification_tier == "DWR_BOUNDED"
    assert snap_cert.certificate is not None
    assert snap_cert.trust_report is not None
    assert snap_cert.corrected_diff_mm is not None
    assert snap_cert.receipt_verified is True

    dossier = render_trust_dossier_markdown(snap_cert)
    assert "# FormaMech Verification & Trust Dossier" in dossier
    assert snap_cert.bundle_name in dossier
    assert "DWR_BOUNDED" in dossier
    assert '"experimental_validation": false' in dossier
    assert snap_cert.limitation_paragraph in dossier


def test_mcp_server_jsonrpc_tools_sandbox_and_dof_limits(tmp_path: Path) -> None:
    server = FormaMechMcpServer(ROOT)
    # 1) Initialize + list tools + ping via JSON-RPC stdio
    req_lines = "\n".join(
        [
            json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
            ),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
            json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
        ]
    )
    out_buf = io.StringIO()
    server.serve_stdio(stdin=io.StringIO(req_lines + "\n"), stdout=out_buf)
    responses = [json.loads(line) for line in out_buf.getvalue().splitlines() if line]
    assert len(responses) == 3  # notification produces no response
    assert responses[0]["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert responses[1]["result"] == {}
    tool_names = {t["name"] for t in responses[2]["result"]["tools"]}
    assert tool_names == {
        "inspect_bundle",
        "inspect_cad_project",
        "predict_verified",
        "run_design_optimization",
        "verify_qualification",
        "verify_receipt",
    }

    # 2) inspect_bundle + predict_verified + verify_receipt round-trip
    insp = server.call_tool(
        "inspect_bundle",
        {"bundle_path": "bundles/notched-v1-cp-freeze"},
    )
    assert insp["claim_level"] == "DEVELOPMENT_VERIFIED"

    pred_resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "predict_verified",
                "arguments": {
                    "bundle_path": "bundles/notched-v1-cp-freeze",
                    "inputs": _notched_raw_inputs(),
                    "mode": "certified",
                },
            },
        }
    )
    assert pred_resp is not None
    assert pred_resp["result"]["isError"] is False
    pred_payload = pred_resp["result"]["structuredContent"]
    assert pred_payload["verdict"] == "IN_DOMAIN"
    receipt = pred_payload["trust_receipt"]
    assert receipt is not None

    ver_ok = server.call_tool("verify_receipt", {"receipt": receipt})
    assert ver_ok["verified"] is True
    assert ver_ok["errors"] == []

    # Tampered receipt must fail verification
    tampered = dict(receipt)
    tampered["verification_tier"] = "DISCRETE_FEM_VERIFIED"
    ver_bad = server.call_tool("verify_receipt", {"receipt": tampered})
    assert ver_bad["verified"] is False
    assert ver_bad["errors"]

    # 3) Path sandbox enforcement (must refuse paths escaping workspace root)
    with pytest.raises(SandboxViolationError):
        server.resolve_path("../outside-secret.json")
    sandbox_resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "inspect_bundle",
                "arguments": {"bundle_path": str(tmp_path)},
            },
        }
    )
    assert sandbox_resp is not None
    assert sandbox_resp["result"]["isError"] is True
    assert "SandboxViolationError" in sandbox_resp["result"]["content"][0]["text"]

    # 4) DOF limit enforcement
    tiny_server = FormaMechMcpServer(ROOT, max_dof_limit=4)
    with pytest.raises(ResourceLimitError, match="max_dof_limit"):
        tiny_server.call_tool(
            "predict_verified",
            {
                "bundle_path": "bundles/notched-v1-cp-freeze",
                "inputs": _notched_raw_inputs(),
                "mode": "fast",
            },
        )

    # 5) CLI `mcp list-tools`
    tools_out = tmp_path / "tools.json"
    rc = cli_main(
        ["mcp", "list-tools", "--workspace", str(ROOT), "--output", str(tools_out)]
    )
    assert rc == 0
    listed = json.loads(tools_out.read_text(encoding="utf-8"))
    assert len(listed["tools"]) == 6


def test_onnx_export_module_parity_and_fmi3_fmu_export_and_step(tmp_path: Path) -> None:
    # 1) Pure PyTorch OnnxExportModule parity against bundle.serving.displacement
    raw_inputs = _notched_raw_inputs()
    mod, sample_xy, sample_params, meta = build_onnx_export_module(
        NOTCHED_BUNDLE,
        reference_inputs=raw_inputs,
        dtype=torch.float64,
    )
    assert meta["bundle_name"] == "notched-v1-cp-freeze"

    bundle = load_bundle(NOTCHED_BUNDLE)
    parsed_inputs, _ = parse_request({"inputs": raw_inputs})
    geom = bundle.geometry(parsed_inputs)
    assert geom.mesh is not None
    from piginot.bundle.inference import _Request

    req = _Request(bundle, parsed_inputs, geom.mesh)
    with torch.no_grad():
        u_export = mod(sample_xy, sample_params)
        u_ref = bundle.serving.displacement(  # type: ignore[attr-defined]
            sample_xy,
            parameters=req.parameters,
            frame=req.frame,
            lifting=req.lifting,
            chunk=bundle.model.query_chunk,
        )
    assert torch.allclose(u_export, u_ref, atol=1.0e-12, rtol=1.0e-12)

    # 2) FMI 3.0 Co-Simulation FMU export via CLI & archive inspection
    fmu_path = tmp_path / "notched.fmu"
    rc = cli_main(
        [
            "export",
            "fmu",
            str(NOTCHED_BUNDLE),
            "--output",
            str(fmu_path),
            "--model-name",
            "NotchedPlateV1",
        ]
    )
    assert rc == 0
    assert fmu_path.is_file()

    with zipfile.ZipFile(fmu_path, "r") as zf:
        names = set(zf.namelist())
        assert {
            "modelDescription.xml",
            "resources/fmu_manifest.json",
            "resources/runner.py",
        }.issubset(names)
        xml_root = ET.fromstring(zf.read("modelDescription.xml"))
        assert xml_root.tag == "fmiModelDescription"
        assert xml_root.attrib["fmiVersion"] == "3.0"
        cs = xml_root.find("CoSimulation")
        assert cs is not None
        assert cs.attrib["canHandleVariableCommunicationStepSize"] == "true"
        mv = xml_root.find("ModelVariables")
        assert mv is not None
        var_names = {el.attrib["name"] for el in mv}
        assert {
            "time",
            "in_envelope",
            "admissible",
            "min_det_f_2d",
            "max_u_mm",
            "bundle_manifest_sha256",
        }.issubset(var_names)
        fmu_manifest = json.loads(
            zf.read("resources/fmu_manifest.json").decode("utf-8")
        )
        assert fmu_manifest["schema"] == FMU_MANIFEST_SCHEMA
        assert fmu_manifest["epistemic_scope"]["experimental_validation"] is False

    # 3) Co-Simulation step: in-envelope -> admissible=True; out-of-envelope -> refused
    in_inputs = {k: float(v["value"]) for k, v in raw_inputs.items()}  # type: ignore[arg-type]
    step_in = evaluate_fmu_step(
        fmu_path,
        bundle_path=NOTCHED_BUNDLE,
        inputs=in_inputs,
        current_time=0.0,
        step_size=0.01,
    )
    assert step_in.in_envelope is True
    assert step_in.admissible is True
    assert step_in.min_det_f_2d > 0.0
    assert step_in.max_u_mm > 0.0
    assert step_in.bundle_manifest_sha256 == bundle.manifest.manifest_sha256

    # Push one input far outside its envelope -> must set in_envelope=False, admissible=False
    first_key = next(iter(in_inputs))
    out_inputs = {**in_inputs, first_key: 1.0e9}
    step_out = evaluate_fmu_step(
        fmu_path,
        bundle_path=NOTCHED_BUNDLE,
        inputs=out_inputs,
        current_time=0.01,
        step_size=0.01,
    )
    assert step_out.in_envelope is False
    assert step_out.admissible is False

    # Direct API export_bundle_to_fmu returns deterministic instantiation token
    fmu_res = export_bundle_to_fmu(NOTCHED_BUNDLE, tmp_path / "notched-api.fmu")
    assert fmu_res.includes_onnx is False
    assert "in_envelope" in fmu_res.output_variables
