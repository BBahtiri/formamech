"""Comprehensive tests for FormaMech's explicit finite state machines (UX/DX/AX)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest
from formamech.contracts import (
    ALLOWED_CLAIM_LEVELS,
    STATE_MACHINE_SCHEMA,
    STATE_SNAPSHOT_SCHEMA,
    CheckStatus,
    ClaimLevel,
    InvalidTransitionError,
    StateMachineSpec,
    TransitionSpec,
    canonical_json_bytes,
    get_state_machine,
    registered_state_machines,
)
from formamech.mcp_server import (
    MCP_SESSION_MACHINE,
    FormaMechMcpServer,
    McpSessionState,
)
from formamech.ui import (
    JOB_LIFECYCLE_MACHINE,
    WORKBENCH_STAGE_MACHINE,
    BoundaryAssignment,
    JobLifecycleState,
    JobRecord,
    WorkbenchJobRunner,
    WorkbenchStage,
    WorkbenchWorkspace,
    generate_sample_bracket_step,
    prepare_and_plan_workspace,
)
from formamech.ui.workbench import default_draft_from_preview
from formamech.ui.workbench.cad_viewer import _classify_assignment

from piginot.certify import (
    ArtifactDigests,
    Tier0AdmissibilityResult,
    TrustReceipt,
    VerificationTier,
    verify_receipt_document,
)
from piginot.contracts import Unit
from piginot.export import (
    FMU_LIFECYCLE_MACHINE,
    ONNX_PARITY_MACHINE,
    OnnxExportResult,
    OnnxParityStatus,
    export_bundle_to_fmu,
)
from piginot.qualify import (
    DEPLOYMENT_GATE_MACHINE,
    DeploymentGateResult,
    DeploymentGateState,
    GateReasonCode,
)
from piginot.serving import (
    SERVING_FUNNEL_MACHINE,
    SPECTRAL_REPAIR_MACHINE,
    TTA_LORA_MACHINE,
    ServingMode,
    ServingPath,
    SpectralRepairReport,
    SpectralRepairStatus,
    TestTimeLoraReport,
    TestTimeLoraStatus,
    VerifiedPrediction,
)
from piginot.serving.hybrid import _build_trust_receipt
from piginot.stateful import STATEFUL_STEP_MACHINE, StepResult, StepStatus
from piginot.store import (
    STORE_CONTENT_MACHINE,
    STORE_RECIPE_MACHINE,
    ContentInfo,
    ContentLifecycleState,
    ObjectStore,
    RecipeState,
    RetentionPolicyError,
    RetentionTier,
    StoreFormatError,
)
from piginot.verification.status import (
    CheckStatus as VerificationCheckStatus,
)
from piginot.verification.status import (
    ClaimLevel as VerificationClaimLevel,
)

ROOT = Path(__file__).resolve().parents[2]
NOTCHED_BUNDLE = ROOT / "bundles" / "notched-v1-cp-freeze"
NOTCHED_REQUEST = ROOT / "bundles" / "requests" / "notched-in-domain.json"


# ---------------------------------------------------------------------------
# 1. State-machine kernel & unified ClaimLevel / CheckStatus contracts
# ---------------------------------------------------------------------------


def test_state_machine_kernel_isolation_identity_and_registry() -> None:
    # Exact vocabulary and StrEnum equality shared with byte-frozen S14 verification/status.py
    assert [c.value for c in ClaimLevel] == [c.value for c in VerificationClaimLevel]
    assert [s.value for s in CheckStatus] == [s.value for s in VerificationCheckStatus]
    assert ALLOWED_CLAIM_LEVELS == (
        "NOT_VERIFIED",
        "DEVELOPMENT_VERIFIED",
        "HOLDOUT_VERIFIED",
    )
    assert ClaimLevel.NOT_VERIFIED.rank == 0
    assert ClaimLevel.DEVELOPMENT_VERIFIED.rank == 1
    assert ClaimLevel.HOLDOUT_VERIFIED.rank == 2
    assert ClaimLevel.HOLDOUT_VERIFIED.satisfies(ClaimLevel.DEVELOPMENT_VERIFIED)
    assert ClaimLevel.HOLDOUT_VERIFIED.satisfies(
        VerificationClaimLevel.DEVELOPMENT_VERIFIED
    )
    assert not ClaimLevel.NOT_VERIFIED.satisfies(ClaimLevel.DEVELOPMENT_VERIFIED)

    # Zero heavy dependency leakage in pure stdlib contract module
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import piginot.contracts.state_machine; "
                "assert not any(m in sys.modules for m in ('torch', 'numpy', 'scipy', 'skfem'))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    # Every explicit FormaMech FSM registers itself and renders strict JSON + Mermaid
    expected_specs = (
        WORKBENCH_STAGE_MACHINE,
        JOB_LIFECYCLE_MACHINE,
        MCP_SESSION_MACHINE,
        FMU_LIFECYCLE_MACHINE,
        ONNX_PARITY_MACHINE,
        SERVING_FUNNEL_MACHINE,
        SPECTRAL_REPAIR_MACHINE,
        TTA_LORA_MACHINE,
        STORE_CONTENT_MACHINE,
        STORE_RECIPE_MACHINE,
        DEPLOYMENT_GATE_MACHINE,
        STATEFUL_STEP_MACHINE,
    )
    reg = registered_state_machines()
    for spec in expected_specs:
        assert spec.machine_id in reg
        assert get_state_machine(spec.machine_id) is spec
        desc = spec.describe()
        assert desc["schema"] == STATE_MACHINE_SCHEMA
        assert canonical_json_bytes(desc)
        mermaid = spec.to_mermaid()
        assert mermaid.startswith("stateDiagram-v2\n")


def test_state_machine_spec_validation_transitions_and_snapshots() -> None:
    class _Toy(StrEnum):
        IDLE = "IDLE"
        ACTIVE = "ACTIVE"
        DONE = "DONE"

    spec = StateMachineSpec(
        machine_id="toy.test",
        states=(_Toy.IDLE, _Toy.ACTIVE, _Toy.DONE),
        initial_states=(_Toy.IDLE,),
        terminal_states=(_Toy.DONE,),
        transitions=(
            TransitionSpec(_Toy.IDLE, _Toy.ACTIVE, "start"),
            TransitionSpec(_Toy.ACTIVE, _Toy.DONE, "finish"),
        ),
    )
    assert spec.can_transition(_Toy.IDLE, _Toy.ACTIVE, trigger="start")
    assert not spec.can_transition(_Toy.IDLE, _Toy.DONE)
    rec = spec.require_transition(_Toy.IDLE, _Toy.ACTIVE, trigger="start")
    assert rec.document()["from_state"] == "IDLE"
    assert rec.document()["to_state"] == "ACTIVE"

    with pytest.raises(InvalidTransitionError) as exc_info:
        spec.require_transition(_Toy.DONE, _Toy.ACTIVE, trigger="restart")
    err_doc = exc_info.value.document()
    assert err_doc["machine_id"] == "toy.test"
    assert err_doc["allowed_targets"] == []

    snap = spec.snapshot(
        _Toy.IDLE,
        blockers=("waiting for input",),
        stale_items=("old.json",),
        recommended_hint="Supply input before starting.",
    )
    snap_doc = snap.document()
    assert snap_doc["schema"] == STATE_SNAPSHOT_SCHEMA
    assert snap_doc["state"] == "IDLE"
    assert snap_doc["terminal"] is False
    assert snap_doc["next_triggers"] == ["start"]
    assert snap_doc["blockers"] == ["waiting for input"]
    assert snap_doc["stale_items"] == ["old.json"]

    # Rejects outgoing transition from terminal state
    with pytest.raises(ValueError, match="terminal state"):
        StateMachineSpec(
            machine_id="toy.bad-terminal",
            states=(_Toy.IDLE, _Toy.DONE),
            initial_states=(_Toy.IDLE,),
            terminal_states=(_Toy.DONE,),
            transitions=(TransitionSpec(_Toy.DONE, _Toy.IDLE, "reopen"),),
        )

    # Validation guards on ClaimLevel, TransitionSpec, StateMachineSpec, and registry
    class _Other(StrEnum):
        A = "A"

    with pytest.raises(TypeError, match="ClaimLevel"):
        ClaimLevel.NOT_VERIFIED.satisfies("NOT_VERIFIED")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="StrEnum"):
        TransitionSpec("IDLE", _Toy.ACTIVE, "x")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="StrEnum"):
        TransitionSpec(_Toy.IDLE, "ACTIVE", "x")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="same StrEnum"):
        TransitionSpec(_Toy.IDLE, _Other.A, "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="trigger"):
        TransitionSpec(_Toy.IDLE, _Toy.ACTIVE, " ")
    with pytest.raises(ValueError, match="machine_id"):
        StateMachineSpec("", (_Toy.IDLE,), (_Toy.IDLE,), (), ())
    with pytest.raises(ValueError, match="states must be non-empty"):
        StateMachineSpec("t.empty", (), (), (), ())
    with pytest.raises(ValueError, match="unique"):
        StateMachineSpec("t.dup", (_Toy.IDLE, _Toy.IDLE), (_Toy.IDLE,), (), ())
    with pytest.raises(ValueError, match="initial_states"):
        StateMachineSpec("t.noinit", (_Toy.IDLE,), (), (), ())
    with pytest.raises(ValueError, match="not in declared states"):
        StateMachineSpec("t.badinit", (_Toy.IDLE,), (_Toy.DONE,), (), ())
    with pytest.raises(ValueError, match="not in declared states"):
        StateMachineSpec("t.badterm", (_Toy.IDLE,), (_Toy.IDLE,), (_Toy.DONE,), ())
    with pytest.raises(ValueError, match="undeclared state"):
        StateMachineSpec(
            "t.badtr",
            (_Toy.IDLE,),
            (_Toy.IDLE,),
            (),
            (TransitionSpec(_Toy.IDLE, _Toy.DONE, "go"),),
        )
    with pytest.raises(ValueError, match="duplicate transition"):
        StateMachineSpec(
            "t.duptr",
            (_Toy.IDLE, _Toy.ACTIVE),
            (_Toy.IDLE,),
            (),
            (
                TransitionSpec(_Toy.IDLE, _Toy.ACTIVE, "go"),
                TransitionSpec(_Toy.IDLE, _Toy.ACTIVE, "go"),
            ),
        )
    with pytest.raises(ValueError, match="unknown state"):
        spec.allowed_transitions(_Other.A)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="unknown state machine"):
        get_state_machine("no.such.machine")
    from piginot.contracts.state_machine import register_state_machine

    with pytest.raises(ValueError, match="conflicting state machine"):
        register_state_machine(
            StateMachineSpec(
                machine_id=WORKBENCH_STAGE_MACHINE.machine_id,
                states=(_Toy.IDLE,),
                initial_states=(_Toy.IDLE,),
                terminal_states=(),
                transitions=(),
            )
        )


# ---------------------------------------------------------------------------
# 2. Workbench Workspace & Job Runner FSMs + CAD Boundary Classifier
# ---------------------------------------------------------------------------


def test_workbench_workspace_staleness_qualification_verdict_and_reimport(
    tmp_path: Path,
) -> None:
    ws = WorkbenchWorkspace(tmp_path / "ws")
    snap0 = ws.workflow_snapshot()
    assert snap0.state is WorkbenchStage.EMPTY
    assert snap0.recommended_trigger == "import_cad"

    step_path = generate_sample_bracket_step(tmp_path / "bracket.step")
    preview = ws.import_cad_path(step_path, unit=Unit.MM, mesh_size_mm=5.0)
    snap1 = ws.workflow_snapshot()
    assert snap1.state is WorkbenchStage.CAD_IMPORTED

    draft = default_draft_from_preview(preview, project_name="fsm-bracket")
    prepare_and_plan_workspace(ws, draft, execution_profile=False)
    snap2 = ws.workflow_snapshot()
    assert snap2.state is WorkbenchStage.PLAN_PREVIEWED
    assert snap2.stale_items == ()

    # Editing physics.json manually flags inputs/ as stale in workflow_snapshot
    ws.physics_path.write_text(
        ws.physics_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    snap_stale = ws.workflow_snapshot()
    assert any("stale" in item for item in snap_stale.stale_items)

    # Manifest-shaped legacy files without integrity evidence cannot qualify a bundle.
    (ws.run_dir / "candidate-bundle").mkdir(parents=True, exist_ok=True)
    (ws.run_dir / "candidate-bundle" / "bundle.json").write_text("{}", encoding="utf-8")
    ws.qualification_dir.mkdir(parents=True, exist_ok=True)
    (ws.qualification_dir / "qualification-manifest.json").write_text(
        json.dumps({"verdict": "NOT_QUALIFIED"}), encoding="utf-8"
    )
    assert ws.stage_progress().bundle_trained is True
    assert ws.stage_progress().bundle_qualified is False
    snap_nq = ws.workflow_snapshot()
    assert snap_nq.state is WorkbenchStage.BUNDLE_TRAINED
    assert snap_nq.blockers

    # Even a passing verdict cannot replace integrity and bundle binding.
    (ws.qualification_dir / "qualification-manifest.json").write_text(
        json.dumps({"verdict": "QUALIFIED"}), encoding="utf-8"
    )
    assert ws.stage_progress().bundle_qualified is False
    assert ws.workflow_snapshot().state is WorkbenchStage.BUNDLE_TRAINED

    # Failed CAD re-import must NOT destroy existing valid geometry/
    with pytest.raises((ValueError, RuntimeError)):
        ws.import_cad_bytes(b"corrupt-step-bytes", "broken.step")
    assert ws.load_preview() is not None

    # Valid CAD re-import cleanly invalidates stale downstream physics/inputs/run/qual
    ws.import_cad_path(step_path, unit=Unit.MM, mesh_size_mm=6.0)
    assert ws.stage_progress().cad_imported is True
    assert ws.stage_progress().physics_saved is False
    assert ws.stage_progress().inputs_prepared is False
    assert ws.stage_progress().bundle_trained is False
    assert ws.stage_progress().bundle_qualified is False
    assert ws.workflow_snapshot().state is WorkbenchStage.CAD_IMPORTED
    assert "boundaries" not in ws.load_state()


def test_cad_viewer_boundary_classifier_edge_cases() -> None:
    # Numeric zero variants classify as dirichlet_fixed
    role_z, _ = _classify_assignment(
        BoundaryAssignment(
            "e1", "clamp", "dirichlet", ("x", "y"), (("0.00", "mm"), ("0e0", "mm"))
        )
    )
    assert role_z == "dirichlet_fixed"
    # Incomplete empty dirichlet never masquerades as clamped
    role_empty, _ = _classify_assignment(
        BoundaryAssignment("e1", "broken", "dirichlet", (), ())
    )
    assert role_empty == "unassigned"


def test_job_runner_terminal_immutability_and_not_qualified_inference(
    tmp_path: Path,
) -> None:
    runner = WorkbenchJobRunner(tmp_path / "jobs", cwd=ROOT)
    out_dir = tmp_path / "qual-out"
    out_dir.mkdir()
    # Cross-session qualification job that finished with NOT_QUALIFIED must infer
    # REFUSED_SCIENTIFIC (exit 3), NOT SUCCEEDED (exit 0).
    (out_dir / "qualification-manifest.json").write_text(
        json.dumps({"verdict": "NOT_QUALIFIED"}), encoding="utf-8"
    )
    ext_rec = JobRecord(
        job_id="ext-nq",
        kind="qualify",
        command=("python", "-m", "formamech"),
        pid=99999999,
        state="RUNNING",
        exit_code=None,
        started_utc="2026-11-22T00:00:00Z",
        finished_utc=None,
        log_path=str(tmp_path / "missing.log"),
        output_dir=str(out_dir),
    )
    (tmp_path / "jobs" / "ext-nq.json").write_text(
        json.dumps(ext_rec.document()), encoding="utf-8"
    )
    inferred = runner.poll("ext-nq")
    assert inferred.lifecycle_state is JobLifecycleState.REFUSED_SCIENTIFIC
    assert inferred.exit_code == 3
    assert inferred.is_terminal is True
    assert inferred.snapshot().terminal is True

    # Terminal job refuses cancel() (no silent overwrite to CANCELLED)
    with pytest.raises(InvalidTransitionError):
        runner.cancel("ext-nq")

    # Terminal job never mutates latest_checkpoint_step when later runs write checkpoints
    (out_dir / "checkpoints" / "step-000000999").mkdir(parents=True)
    polled_again = runner.poll("ext-nq")
    assert polled_again.latest_checkpoint_step is None


# ---------------------------------------------------------------------------
# 3. MCP Session FSM, JSON-RPC Notifications, Schema Validation & Gate Enforcement
# ---------------------------------------------------------------------------


def test_mcp_session_fsm_schema_validation_and_require_qualified(
    tmp_path: Path,
) -> None:
    raw_inputs = dict(json.loads(NOTCHED_REQUEST.read_text(encoding="utf-8"))["inputs"])

    # Strict lifecycle mode enforces UNINITIALIZED -> INITIALIZING -> READY -> CLOSED
    strict_srv = FormaMechMcpServer(ROOT, strict_lifecycle=True)
    assert strict_srv.session_state is McpSessionState.UNINITIALIZED

    pre_init = strict_srv.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    )
    assert pre_init is not None and pre_init["error"]["code"] == -32002

    init_resp = strict_srv.handle_message(
        {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {}}
    )
    assert init_resp is not None and "result" in init_resp
    assert strict_srv.session_state is McpSessionState.INITIALIZING

    assert (
        strict_srv.handle_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        is None
    )
    assert strict_srv.session_state is McpSessionState.READY

    # JSON-RPC 2.0: unknown notifications (no 'id') must produce None (silence)
    assert (
        strict_srv.handle_message(
            {"jsonrpc": "2.0", "method": "notifications/cancelled"}
        )
        is None
    )

    # Runtime JSON Schema validation catches typos & out-of-range arguments with hints
    bad_arg = strict_srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "predict_verified",
                "arguments": {
                    "bundle_path": "bundles/notched-v1-cp-freeze",
                    "inputs": raw_inputs,
                    "escalte": True,  # typo
                },
            },
        }
    )
    assert bad_arg is not None
    assert bad_arg["result"]["isError"] is True
    assert bad_arg["result"]["structuredContent"]["category"] == "invalid_arguments"
    assert "unexpected argument" in bad_arg["result"]["structuredContent"]["message"]

    # Mode-option guard: fast + escalate is rejected cleanly
    with pytest.raises(ValueError, match="escalate=True"):
        strict_srv.call_tool(
            "predict_verified",
            {
                "bundle_path": "bundles/notched-v1-cp-freeze",
                "inputs": raw_inputs,
                "mode": "fast",
                "escalate": True,
            },
        )

    # verify_receipt rejects passing both receipt and receipt_path
    with pytest.raises(ValueError, match="not both"):
        strict_srv.call_tool(
            "verify_receipt",
            {"receipt": {}, "receipt_path": "README.md"},
        )

    # Closing session moves to terminal CLOSED and rejects subsequent requests
    strict_srv.close()
    assert strict_srv.session_state is McpSessionState.CLOSED
    after_close = strict_srv.handle_message(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/list"}
    )
    assert after_close is not None and "error" in after_close
    with pytest.raises(InvalidTransitionError):
        strict_srv.close()

    # Security fix: run_design_optimization enforces require_qualified=True
    gated_srv = FormaMechMcpServer(ROOT, require_qualified=True)
    opt_blocked = gated_srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "run_design_optimization",
                "arguments": {
                    "bundle_path": "bundles/notched-v1-cp-freeze",
                    "problem_path": "examples/advanced_workflows/notched_design_problem.json",
                },
            },
        }
    )
    assert opt_blocked is not None
    assert opt_blocked["result"]["isError"] is True
    assert (
        opt_blocked["result"]["structuredContent"]["category"]
        == "deployment_gate_refused"
    )


# ---------------------------------------------------------------------------
# 4. FMI 3.0 Runner Execution & ONNX Parity Invariants
# ---------------------------------------------------------------------------


def test_exported_fmu_runner_parses_and_enforces_fmi3_lifecycle(tmp_path: Path) -> None:
    fmu_path = tmp_path / "notched.fmu"
    export_bundle_to_fmu(NOTCHED_BUNDLE, fmu_path, model_name="NotchedFSM")

    with zipfile.ZipFile(fmu_path, "r") as zf:
        runner_src = zf.read("resources/runner.py").decode("utf-8")
        manifest_bytes = zf.read("resources/fmu_manifest.json")

    manifest_file = tmp_path / "fmu_manifest.json"
    manifest_file.write_bytes(manifest_bytes)

    # Compiles & executes cleanly (previously failed with IndentationError on L304!)
    namespace: dict[str, Any] = {}
    exec(compile(runner_src, "resources/runner.py", "exec"), namespace)
    fmu_cls = namespace["FormaMechFmuInstance"]
    inst = fmu_cls(NOTCHED_BUNDLE, manifest_file)

    assert inst.state.value == "INSTANTIATED"
    with pytest.raises(ValueError, match="STEP_MODE"):
        inst.do_step(0.0, 0.01)
    with pytest.raises(ValueError, match="before initialization"):
        inst.get_outputs()

    inst.enter_initialization_mode()
    assert inst.state.value == "INITIALIZATION_MODE"
    with pytest.raises(KeyError, match="unknown FMU input"):
        inst.set_float64("no_such_var", 1.0)
    with pytest.raises(ValueError, match="finite"):
        first_var = next(iter(inst.manifest["envelope_bounds"]))
        inst.set_float64(first_var, float("nan"))

    inst.exit_initialization_mode()
    assert inst.state.value == "STEP_MODE"
    with pytest.raises(ValueError, match="step_size"):
        inst.do_step(0.0, 0.0)

    out1 = inst.do_step(0.0, 0.05)
    assert out1["in_envelope"] is True
    assert out1["admissible"] is True
    assert out1["min_det_f_2d"] > 0.0
    assert inst.get_outputs() == out1

    # Non-monotonic time refused
    with pytest.raises(ValueError, match="non-monotonic"):
        inst.do_step(0.01, 0.05)

    inst.terminate()
    assert inst.state.value == "TERMINATED"
    with pytest.raises(ValueError, match="TERMINATED"):
        inst.set_float64(first_var, 0.1)
    with pytest.raises(ValueError, match="STEP_MODE"):
        inst.do_step(0.1, 0.05)


def test_onnx_export_result_invariants() -> None:
    ok = OnnxExportResult(
        onnx_path=Path("m.onnx"),
        metadata_path=Path("m.onnx.json"),
        bundle_manifest_sha256="a" * 64,
        onnx_sha256="b" * 64,
        opset_version=17,
        input_names=("strain",),
        parity_checked=True,
        parity_max_abs_error=1.0e-7,
    )
    assert ok.parity_status is OnnxParityStatus.VERIFIED_PASS
    assert ok.document()["parity_status"] == "VERIFIED_PASS"

    with pytest.raises(ValueError, match="parity_max_abs_error"):
        OnnxExportResult(
            onnx_path=Path("m.onnx"),
            metadata_path=Path("m.onnx.json"),
            bundle_manifest_sha256="a" * 64,
            onnx_sha256="b" * 64,
            opset_version=17,
            input_names=("strain",),
            parity_checked=True,
            parity_max_abs_error=None,
        )
    with pytest.raises(ValueError, match="inconsistent"):
        OnnxExportResult(
            onnx_path=Path("m.onnx"),
            metadata_path=Path("m.onnx.json"),
            bundle_manifest_sha256="a" * 64,
            onnx_sha256="b" * 64,
            opset_version=17,
            input_names=("strain",),
            parity_checked=False,
            parity_max_abs_error=None,
            parity_status=OnnxParityStatus.VERIFIED_PASS,
        )


# ---------------------------------------------------------------------------
# 5. Serving Funnel, Spectral Repair, TTA, Tier 0 & Trust Receipt Invariants
# ---------------------------------------------------------------------------


def test_serving_and_receipt_invariants_and_epistemic_scope_tamper_guard() -> None:
    # VerifiedPrediction rejects illegal mode/path and exact/path combinations
    with pytest.raises(ValueError, match="FAST mode"):
        VerifiedPrediction(ServingMode.FAST, ServingPath.COLD_FEM, prediction={})
    with pytest.raises(ValueError, match="converged exact FemResult"):
        VerifiedPrediction(ServingMode.EXACT, ServingPath.WARM_NEWTON, prediction={})
    with pytest.raises(ValueError, match="SOLVE_FAILED"):
        VerifiedPrediction(
            ServingMode.EXACT,
            ServingPath.SOLVE_FAILED,
            prediction={},
            exact=object(),  # type: ignore[arg-type]
        )

    # SOLVE_FAILED never masks failure under DWR_BOUNDED or SURROGATE_ADMISSIBLE
    class _FakeBundle:
        claim_level = ClaimLevel.NOT_VERIFIED
        manifest = type("M", (), {"manifest_sha256": "a" * 64})()
        identities = {"weights_sha256": "b" * 64}
        reference_mesh = type("Mesh", (), {"mesh_id": "m1"})()
        model = type("Model", (), {"problem": None})()

    rc_doc, _ = _build_trust_receipt(
        bundle=_FakeBundle(),
        mode=ServingMode.EXACT,
        path=ServingPath.SOLVE_FAILED,
        request=None,
        solver=None,
        tier0={"passed": True},
        conformal_doc=None,
        certificate_doc={"verdict": "OK", "eta_energy": 0.01},
        trust_report_doc=None,
        attempts=(),
        exact_converged=False,
        spectral_repair_doc=None,
        tta_doc=None,
        reason="every rung failed",
    )
    assert rc_doc["verification_tier"] == VerificationTier.SURROGATE_UNCHECKED.value
    assert rc_doc["epistemic_scope"]["discrete_fem_agreement"] is False

    # verify_receipt_document catches forged-then-rehashed epistemic_scope
    base_rc = TrustReceipt(
        mode="fast",
        serving_path="operator",
        claim_level="NOT_VERIFIED",
        verification_tier=VerificationTier.SURROGATE_UNCHECKED,
        artifact_digests=ArtifactDigests(),
    ).document()
    forged = dict(base_rc)
    forged["epistemic_scope"] = {
        **base_rc["epistemic_scope"],
        "discrete_fem_agreement": True,
    }
    body = {k: v for k, v in forged.items() if k != "receipt_sha256"}
    forged["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    ok_forged, errs_forged = verify_receipt_document(forged)
    assert ok_forged is False
    assert any("epistemic_scope does not match" in e for e in errs_forged)

    # Tier0AdmissibilityResult, SpectralRepairReport, TestTimeLoraReport invariants
    with pytest.raises(ValueError, match="invariant violated"):
        Tier0AdmissibilityResult(
            passed=True,
            all_finite=True,
            min_det_f_2d=-0.5,
            min_det_f_3d=None,
            dirichlet_deviation=None,
            equilibrium_residual_rms=None,
            free_traction_rms=None,
            section_force_imbalance=None,
            failed_checks=("det_f_2d_below_floor",),
        )
    rep_safe = SpectralRepairReport(
        needed_repair=False,
        succeeded=True,
        steps_taken=0,
        initial_admissible=True,
        initial_positive_definite=True,
        final_positive_definite=True,
        initial_min_det_f=1.0,
        final_min_det_f=1.0,
        initial_residual_norm=0.0,
        final_residual_norm=0.0,
        wall_seconds=0.0,
        reason="ok",
    )
    assert rep_safe.status is SpectralRepairStatus.ALREADY_SAFE
    with pytest.raises(ValueError, match="needed_repair=False"):
        SpectralRepairReport(
            needed_repair=False,
            succeeded=False,
            steps_taken=0,
            initial_admissible=True,
            initial_positive_definite=True,
            final_positive_definite=True,
            initial_min_det_f=1.0,
            final_min_det_f=1.0,
            initial_residual_norm=0.0,
            final_residual_norm=0.0,
            wall_seconds=0.0,
            reason="contradiction",
        )
    tta_ok = TestTimeLoraReport(
        steps_taken=2,
        adapted_layers=("l1",),
        adapter_parameter_count=8,
        initial_energy=2.0,
        final_energy=1.5,
        energy_drop=0.5,
        final_min_det_f=0.9,
        improved=True,
        rolled_back=False,
        adapter_digest="blake2b256:" + "0" * 64,
        wall_seconds=0.01,
    )
    assert tta_ok.status is TestTimeLoraStatus.APPLIED
    with pytest.raises(ValueError, match="strictly opposite"):
        TestTimeLoraReport(
            steps_taken=2,
            adapted_layers=("l1",),
            adapter_parameter_count=8,
            initial_energy=2.0,
            final_energy=1.5,
            energy_drop=0.5,
            final_min_det_f=0.9,
            improved=True,
            rolled_back=True,
            adapter_digest="blake2b256:" + "0" * 64,
            wall_seconds=0.01,
        )


# ---------------------------------------------------------------------------
# 6. ObjectStore, DeploymentGate, Stateful StepResult, Active & 3D Mode Guards
# ---------------------------------------------------------------------------


def test_store_content_and_recipe_state_machines_block_paradoxical_states(
    tmp_path: Path,
) -> None:
    store = ObjectStore.create(tmp_path / "store")
    missing_hash = "0" * 64
    assert store.content_state(missing_hash) is ContentLifecycleState.UNCOMMITTED
    assert store.recipe_state(missing_hash) is RecipeState.UNRECORDED

    digest = store.put(b"ephemeral", tier=RetentionTier.PRUNABLE)
    assert store.content_state(digest) is ContentLifecycleState.PRUNABLE_MATERIALIZED
    store.prune()
    assert store.content_state(digest) is ContentLifecycleState.PRUNABLE_PRUNED

    # Paradoxical promotion of pruned content to PERMANENT or RETAINED must fail closed
    with pytest.raises(RetentionPolicyError, match="rehydration"):
        store.set_retention(digest, RetentionTier.PERMANENT)
    with pytest.raises(RetentionPolicyError, match="rehydration"):
        store.set_retention(digest, RetentionTier.RETAINED)

    # Direct construction & on-disk paradoxical metadata are both rejected
    with pytest.raises(ValueError, match="unmaterialized"):
        ContentInfo(digest, 9, RetentionTier.PERMANENT, materialized=False)

    meta_path = store._content_metadata_path(digest)
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw["retention"] = "PERMANENT"
    meta_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(StoreFormatError, match="inconsistent"):
        store.info(digest)

    # Empty recipe metadata list is rejected instead of mimicking UNRECORDED
    r_hash = "1" * 64
    r_path = store._recipe_metadata_path(r_hash)
    r_path.parent.mkdir(parents=True, exist_ok=True)
    r_path.write_text(
        json.dumps({"schema_version": 1, "recipe_hash": r_hash, "content_hashes": []}),
        encoding="utf-8",
    )
    with pytest.raises(StoreFormatError, match="non-empty"):
        store.recipe_state(r_hash)


def test_deployment_gate_stateful_step_and_3d_mode_invariants() -> None:
    ok_gate = DeploymentGateResult(
        allowed=True,
        bundle_name="demo",
        bundle_manifest_sha256="a" * 64,
        qualification_manifest_sha256="b" * 64,
        granted_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
        min_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
        reason_codes=(),
        detail="ok",
    )
    assert ok_gate.state is DeploymentGateState.ALLOWED
    with pytest.raises(ValueError, match="forbids non-empty"):
        DeploymentGateResult(
            allowed=True,
            bundle_name="demo",
            bundle_manifest_sha256=None,
            qualification_manifest_sha256=None,
            granted_claim_level=ClaimLevel.NOT_VERIFIED,
            min_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
            reason_codes=(GateReasonCode.NOT_QUALIFIED,),
            detail="bad",
        )
    with pytest.raises(ValueError, match="at least one"):
        DeploymentGateResult(
            allowed=False,
            bundle_name="demo",
            bundle_manifest_sha256=None,
            qualification_manifest_sha256=None,
            granted_claim_level=ClaimLevel.NOT_VERIFIED,
            min_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
            reason_codes=(),
            detail="bad",
        )

    with pytest.raises(ValueError, match="accepted StepResult"):
        StepResult(
            status=StepStatus.ACCEPTED_OPERATOR,
            time_index=1,
            time_value=0.1,
            dt_used=0.1,
            substeps_taken=1,
            displacement=None,
            state_new=None,
            bound_state=None,
            min_det_f=1.0,
            total_dissipation=0.0,
            residual_norm=0.0,
            reason="missing payload",
        )


def test_active_campaign_preserves_joint_geometry_token_anchors_across_rounds() -> None:
    from piginot.active.acquisition import AcquisitionCandidate, AcquisitionPlan
    from piginot.active.campaign import ActiveFemSolveOutcome, run_active_fem_campaign

    pool = [
        AcquisitionCandidate(
            case_id=f"c{i}",
            partition="TRAIN",
            feature_vector=(float(i), 0.1 * i),
            eta_energy=1.0,
            dwr_sensitivity=1.0,
            geometry_token_vector=(float(i * 10), 0.0),
        )
        for i in range(4)
    ]
    plan = AcquisitionPlan(
        plan_id="joint-anchor-test",
        arm_kind="physics_dwr_diversity",
        batch_size=1,
        budget_solves=3,
        seed=0,
    )
    res = run_active_fem_campaign(
        plan=plan,
        score_pool=lambda _r, _ids: pool,
        solve_train_case=lambda c: ActiveFemSolveOutcome(
            succeeded=True, wall_seconds=0.01, dof_count=8, result_hash="a" * 64
        ),
    )
    # Round 0 selects c0; round 1 must pick farthest anchor (c3) using joint geometry
    # tokens rather than dropping anchors and falling back to tie-broken c1.
    assert res.acquired_case_ids[0] == "c0"
    assert res.acquired_case_ids[1] == "c3"


def test_wave2_unified_gate_status_and_registered_machines() -> None:
    from piginot.bounds import CAD_CALIBRATION_MACHINE
    from piginot.cad import PARAMETRIC_PREDICT_MACHINE
    from piginot.contracts import GateStatus, claim_level_rank
    from piginot.datasets.gate import GateStatus as DatasetsGateStatus
    from piginot.doe.gate import GateStatus as DoeGateStatus
    from piginot.fem.s8_gate import GateStatus as FemS8GateStatus
    from piginot.objectives.gate import GateStatus as ObjectivesGateStatus
    from piginot.operators import LORA_ADAPTATION_MACHINE
    from piginot.operators.gate import GateStatus as OperatorsGateStatus
    from piginot.optimization import DESIGN_OPTIMIZATION_MACHINE

    assert GateStatus is DoeGateStatus
    assert GateStatus is DatasetsGateStatus
    assert GateStatus is OperatorsGateStatus
    assert GateStatus is ObjectivesGateStatus
    assert GateStatus is FemS8GateStatus
    assert [g.value for g in GateStatus] == ["PASS", "FAIL", "ERROR", "NOT_RUN"]

    assert claim_level_rank(ClaimLevel.NOT_VERIFIED) == 0
    assert claim_level_rank(VerificationClaimLevel.HOLDOUT_VERIFIED) == 2
    with pytest.raises(TypeError, match="ClaimLevel"):
        claim_level_rank("NOT_VERIFIED")  # type: ignore[arg-type]

    reg = registered_state_machines()
    for spec in (
        PARAMETRIC_PREDICT_MACHINE,
        CAD_CALIBRATION_MACHINE,
        LORA_ADAPTATION_MACHINE,
        DESIGN_OPTIMIZATION_MACHINE,
    ):
        assert reg[spec.machine_id] == spec
        assert spec.describe()["machine_id"] == spec.machine_id
        assert "stateDiagram-v2" in spec.to_mermaid()
