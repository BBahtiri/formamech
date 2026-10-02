from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import rfc8785
import torch

from campaigns.dogbone_v1 import s13_joint_hybrid_run as runner
from campaigns.dogbone_v1 import s13_joint_hybrid_runtime as runtime
from campaigns.dogbone_v1.s11_contingency_models import model_state_hash
from campaigns.dogbone_v1.s13_joint_hybrid_metrics import JointHybridSnapshot
from campaigns.dogbone_v1.s13_joint_hybrid_protocol import JointHybridPlan
from campaigns.dogbone_v1.s13_joint_hybrid_training import (
    JointArmTrainingResult,
    JointStepTrace,
    JointTrainingFailure,
)
from campaigns.dogbone_v1.s13_overfit_diagnostic import classify_s13_overfit
from campaigns.dogbone_v1.s13_quadrature_metrics import summarize_quadrature_case
from piginot.contracts import canonical_json_bytes, hash_spec, strict_json_loads
from piginot.doe import EvidenceRole
from piginot.operators import module_digest
from tests.campaigns.test_dogbone_s13_objective import _model
from tests.campaigns.test_dogbone_s13_overfit_diagnostic import _metrics
from tests.campaigns.test_dogbone_s13_quadrature_metrics import _fields
from tests.campaigns.test_dogbone_s13_screen import _case
from tests.fem.test_verification_norms import _synthetic_exact_result

ROOT = Path(__file__).resolve().parents[2]


def test_ratification_preserves_exact_approved_proposal_without_rewriting_draft() -> (
    None
):
    approval = runner.JointHybridApproval()
    assert approval.proposal_sha256 == (
        "b613b4dc884e0b34d5a253fd81154abd11406e3b5d67ef1d89ab1e88f6cf7e0a"
    )
    assert approval.plan_hash == (
        "c7e698556c23b4163d5ff6a22a1a30cd600904312e7f7e2a445bafdf58b718e2"
    )
    assert approval.human_message == "i approve and continue"
    assert approval.status == "APPROVED"
    assert approval.ratification_date == "2026-10-01"
    assert approval.approval_is_cryptographic_attestation is False
    with pytest.raises(FrozenInstanceError):
        approval.status = "DRAFT"
    for change in (
        {"plan_hash": "a" * 64},
        {"status": "DRAFT"},
        {"approval_is_cryptographic_attestation": True},
    ):
        with pytest.raises(ValueError, match="ratification"):
            replace(approval, **change)
    proposal_path = (
        ROOT / "artifacts/campaigns/dogbone-v1/S13-JOINT-HYBRID-PROPOSAL.json"
    )
    assert (
        hashlib.sha256(proposal_path.read_bytes()).hexdigest()
        == approval.proposal_sha256
    )
    assert (
        strict_json_loads(proposal_path.read_bytes())["execution_authorized"] is False
    )


def test_approval_parser_refuses_missing_changed_noncanonical_or_relative_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "approval.json"
    with pytest.raises(FileNotFoundError):
        runner.load_approval(path)
    encoded = canonical_json_bytes(runner.JointHybridApproval()) + b"\n"
    path.write_bytes(encoded)
    assert runner.load_approval(path) == runner.JointHybridApproval()
    path.write_bytes(encoded.replace(b"APPROVED", b"DRAFT"))
    with pytest.raises(ValueError, match="ratification"):
        runner.load_approval(path)
    path.write_bytes(encoded.rstrip())
    with pytest.raises(ValueError, match="canonical"):
        runner.load_approval(path)
    with pytest.raises(ValueError, match="absolute"):
        runner.load_approval(Path("relative.json"))


def test_source_pin_covers_real_production_sources_not_experiments() -> None:
    pins = runner.implementation_hashes()
    assert len(pins) > 20
    assert "campaigns/dogbone_v1/s13_joint_hybrid_run.py" in dict(pins)
    assert "campaigns/dogbone_v1/s13_joint_hybrid_training.py" in dict(pins)
    assert "src/piginot/operators/architecture.py" in dict(pins)
    assert all(not path.startswith("experiments/") for path, _ in pins)
    assert all(len(digest) == 64 for _, digest in pins)
    assert pins == runner.implementation_hashes()


def test_authority_failure_precedes_any_real_source_or_optimizer_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse() -> object:
        raise PermissionError("ratification unavailable")

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("source/checkpoint/replay reached before authority verification")

    monkeypatch.setattr(runner, "validate_execution_pin", refuse)
    monkeypatch.setattr(runner, "open_selective_s10_source", forbidden)
    monkeypatch.setattr(runner, "load_selected_s11_initialization", forbidden)
    monkeypatch.setattr(runner, "_train_one_case", forbidden)
    with pytest.raises(PermissionError, match="ratification"):
        runner.run_joint_hybrid(
            s10_run_root=tmp_path / "s10",
            s11_checkpoint_root=tmp_path / "s11",
            development_access_record=tmp_path / "access.json",
            output_path=tmp_path / "result.json",
        )
    assert not (tmp_path / "result.json").exists()


def _canonical(value: object) -> Any:
    encoded = (
        rfc8785.dumps(value)
        if isinstance(value, dict) and "$type" in value
        else canonical_json_bytes(value)
    )
    return strict_json_loads(encoded)


def _write_record(path: Path, value: object) -> None:
    path.write_bytes(rfc8785.dumps(_canonical(value)) + b"\n")


def _patch(monkeypatch: pytest.MonkeyPatch, target: object, **values: object) -> None:
    for name, value in values.items():
        monkeypatch.setattr(target, name, value)


def _runner_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Mock all source reads, replay, optimization and measurement on a typed square."""
    s10, s11, out = (tmp_path / name for name in ("s10", "s11", "out"))
    for path in (s10, s11, out):
        path.mkdir()
    model, case = _model(), _case("candidate-002485")
    # Only this test's factory returns a forged dataclass; production construction
    # and the unchanged 20-QoI population guard remain independently tested.
    plan = object.__new__(JointHybridPlan)
    real = JointHybridPlan()
    for item in fields(real):
        object.__setattr__(plan, item.name, getattr(real, item.name))
    for name, value in (
        ("expected_vertices", 4),
        ("quadrature_layout", (2, 7)),
        ("v5_state_hash", model_state_hash(model)),
    ):
        object.__setattr__(plan, name, value)
    source = SimpleNamespace(
        development_opened=False, bindings=SimpleNamespace(content_hash="a" * 64)
    )
    initialization = SimpleNamespace(
        model=model,
        evidence_file_blake2b256="b" * 64,
        checkpoint_file_blake2b256="c" * 64,
    )
    verified = SimpleNamespace(
        case=SimpleNamespace(case_id=plan.case_id, role=EvidenceRole.TRAIN),
        request=case.request,
        result=_synthetic_exact_result(case.request),
    )
    loaded: list[str] = []
    events: list[object] = []

    def load(case_id: str) -> tuple[SimpleNamespace, None]:
        loaded.append(case_id)
        return verified, None

    source.load_fit_with_collocation = load
    # Measurement and reconstruction are mocked: this opaque reference is never
    # queried. The canonical snapshot uses only pure, synthetic metric fields.
    reference = object()
    inputs = {
        **_fields(),
        "u_model": torch.ones((2, 2, 2), dtype=torch.float64) * 1.1,
        "u_fem": torch.ones((2, 2, 2), dtype=torch.float64),
    }
    quadrature = summarize_quadrature_case(**inputs)
    original, baseline = _metrics(data_mse=1), _metrics(data_mse=0.1)
    snapshot = JointHybridSnapshot(
        "v5",
        0,
        plan.v5_state_hash,
        "d" * 64,
        baseline,
        classify_s13_overfit(
            runner.build_s13_overfit_plan(version="v5"), original, baseline
        ),
        0.0,
        None,
        "synthetic_fixture",
        quadrature,
    )
    sampler = SimpleNamespace(plan_hash=plan.vertex_sampler_plan_hash)
    implementation = (("historical.py", "e" * 64),)
    v5 = {
        "source_bindings_hash": source.bindings.content_hash,
        "s11_evidence_file_hash": initialization.evidence_file_blake2b256,
        "initial_checkpoint_file_hash": initialization.checkpoint_file_blake2b256,
        "initial_model_state_hash": plan.v5_state_hash,
        "vertex_sampler_plan_hash": sampler.plan_hash,
        "plan_hash": plan.v5_plan_hash,
        "before": _canonical(original),
        "after": _canonical(baseline),
        "optimizer_loss_trace": [0.2] * 4000,
    }
    quad = {
        "implementation_sha256": _canonical(implementation),
        "source_bindings_hash": source.bindings.content_hash,
        "request_content_hash": hash_spec(case.request, "content"),
        "result_content_hash": hash_spec(verified.result, "content"),
        "arms": [
            {
                "arm_id": "v5",
                "state_hash": plan.v5_state_hash,
                "module_digest": module_digest(model),
                "metrics": _canonical(quadrature),
            }
        ],
    }
    pin = runner.JointHybridExecutionPin(
        EvidenceRole.TRAIN, True, real.content_hash, "a" * 64, "b" * 64, (), ()
    )
    states = tuple(
        SimpleNamespace(arm_id=arm, model=_model()) for arm in ("control", "candidate")
    )
    objective = object()
    step_trace = (
        JointStepTrace(
            4000,
            (0, 1),
            (("data_u", 0.1),),
            (("data_u", 0.1),),
            0.1,
            0.5,
            1.0,
            4000,
            (("loss", True),),
        ),
    )

    def evaluate(model: object, *args: object, **kwargs: Any) -> JointHybridSnapshot:
        events.append(("evaluate", kwargs["arm_id"], kwargs["step"]))
        assert kwargs["original_s11_metrics"] is original
        assert kwargs["pinned_v5_metrics"] is baseline
        return replace(snapshot, arm_id=kwargs["arm_id"], step=kwargs["step"])

    def train(
        state: Any,
        received_objective: object,
        received_sampler: object,
        received_plan: object,
        *,
        evaluate: Any,
    ) -> JointArmTrainingResult:
        assert received_objective is objective and received_sampler is sampler
        assert received_plan is plan
        events.append(("train", state.arm_id))
        evaluations = tuple(
            (step, evaluate(state.model, step)) for step in plan.evaluation_steps
        )
        return JointArmTrainingResult(
            state.arm_id,
            step_trace,
            evaluations,
            sampler.plan_hash,
            4000,
            (),
            hash_spec(model.config, "content"),
            module_digest(model),
        )

    def replay(*args: object) -> tuple[float, ...]:
        events.append("mock_reconstruction")
        assert args[0] is model and args[1] is case and args[3] is sampler
        return tuple(v5["optimizer_loss_trace"])

    measured = iter((original, baseline))
    _patch(
        monkeypatch,
        runner,
        JointHybridPlan=lambda: plan,
        validate_execution_pin=lambda: pin,
        _clean_revision=lambda: "f" * 40,
        _validate_registered_population=lambda *args: None,
        load_selected_s11_initialization=lambda *args: initialization,
        open_selective_s10_source=lambda *args, **kwargs: source,
        materialize_dogbone_s13_training_case=lambda *args: case,
        load_pinned_report=lambda path, *args: v5 if "V5" in path.name else quad,
        audit_implementation_hashes=lambda: implementation,
        qoi_vertex_indices=lambda *args, **kwargs: tuple(
            case.tensors.qoi_indices.tolist()
        ),
        rebuild_fem_quadrature=lambda *args: reference,
        build_joint_vertex_sampler=lambda *args: sampler,
        _measure_case=lambda *args: next(measured),
        _train_one_case=replay,
        evaluate_joint_snapshot=evaluate,
        clone_joint_arms=lambda *args: states,
        SampledJointObjective=lambda *args: objective,
        train_joint_arm=train,
    )
    return SimpleNamespace(
        model=model,
        case=case,
        plan=plan,
        source=source,
        verified=verified,
        snapshot=snapshot,
        v5=v5,
        quad=quad,
        pin=pin,
        states=states,
        events=events,
        loaded=loaded,
        step_trace=step_trace,
        train=train,
        arguments={
            "s10_run_root": s10,
            "s11_checkpoint_root": s11,
            "development_access_record": out / "access.json",
            "output_path": out / "result.json",
        },
    )


def test_mocked_coordinator_completes_fixed_arms_and_writes_canonical_exclusively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    report = runner.run_joint_hybrid(**fixture.arguments)
    assert report.status == "COMPLETED" and report.postflight_verified
    assert report.warmstart_optimizer_updates == 4000
    assert report.plan_hash == fixture.plan.content_hash
    assert report.execution_pin_hash == hash_spec(fixture.pin, "content")
    assert report.source_bindings_hash == fixture.source.bindings.content_hash
    assert report.original_s11_state_hash == model_state_hash(fixture.model)
    assert report.warmstart == fixture.snapshot
    assert tuple(arm.arm_id for arm in report.arms) == ("control", "candidate")
    assert fixture.events.count("mock_reconstruction") == 1
    assert [
        event
        for event in fixture.events
        if isinstance(event, tuple) and event[0] == "train"
    ] == [("train", "control"), ("train", "candidate")]
    assert fixture.loaded == [fixture.plan.case_id, fixture.plan.case_id]
    assert report.comparison.comparison_step == 4000
    assert report.failure_stage is None and report.failure_reason is None
    assert report.development_access_absent
    assert (report.g13_status, report.cp_freeze_status, report.s14_status) == (
        "NOT_PASSED",
        "WITHHELD",
        "BLOCKED",
    )
    path = fixture.arguments["output_path"]
    assert path.read_bytes() == canonical_json_bytes(report) + b"\n"
    with pytest.raises(FileExistsError):
        runner.run_joint_hybrid(**fixture.arguments)


@pytest.mark.parametrize(
    "defect",
    (
        "relative",
        "existing_output",
        "source_output",
        "access_exists",
        "same_output_access",
    ),
)
def test_path_failures_precede_private_reads_and_any_mocked_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    arguments = fixture.arguments
    if defect == "relative":
        arguments["s10_run_root"] = Path("relative")
    elif defect == "existing_output":
        arguments["output_path"].write_bytes(b"existing")
    elif defect == "source_output":
        arguments["output_path"] = arguments["s10_run_root"] / "output.json"
    elif defect == "access_exists":
        arguments["development_access_record"].write_bytes(b"existing")
    else:
        arguments["output_path"] = arguments["development_access_record"]
    monkeypatch.setattr(
        runner,
        "load_selected_s11_initialization",
        lambda *args: pytest.fail("private initialization reached"),
    )
    monkeypatch.setattr(
        runner,
        "open_selective_s10_source",
        lambda *args, **kwargs: pytest.fail("private source reached"),
    )
    with pytest.raises((ValueError, FileExistsError)):
        runner.run_joint_hybrid(**arguments)
    assert not fixture.events


@pytest.mark.parametrize(
    "defect",
    (
        "initialization development source_binding verified_role case_id request "
        "replay reconstruction_failure baseline_metrics quad_record quad_metrics clone_state"
    ).split(),
)
def test_coordinator_failures_close_once_without_retry_or_completed_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    expected_stage = "TRAIN-case-verification"
    if defect == "initialization":

        def fail(*args: object) -> None:
            raise ValueError("synthetic initialization failure")

        monkeypatch.setattr(runner, "load_selected_s11_initialization", fail)
        expected_stage = "initialization"
    elif defect == "development":
        fixture.source.development_opened = True
        expected_stage = "initialization"
    elif defect == "source_binding":
        fixture.v5["source_bindings_hash"] = "0" * 64
        expected_stage = "initialization"
    elif defect == "verified_role":
        fixture.verified.case.role = EvidenceRole.DEVELOPMENT
    elif defect == "case_id":
        fixture.verified.case.case_id = "other"
    elif defect == "request":
        fixture.verified.request = replace(fixture.case.request, request_id="other")
    elif defect == "replay":
        fixture.v5["optimizer_loss_trace"] = [0.3]
        monkeypatch.setattr(runner, "_train_one_case", lambda *args: (0.2,))
        expected_stage = "V5-exact-reconstruction"
    elif defect == "reconstruction_failure":

        def partial_reconstruction(*args: object) -> None:
            raise FloatingPointError("synthetic partial reconstruction failure")

        monkeypatch.setattr(runner, "_train_one_case", partial_reconstruction)
        expected_stage = "V5-exact-reconstruction"
    elif defect == "baseline_metrics":
        fixture.v5["after"] = _canonical(_metrics(data_mse=0.2))
        expected_stage = "V5-exact-reconstruction"
    elif defect == "quad_record":
        fixture.quad["arms"] = []
        expected_stage = "V5-exact-reconstruction"
    elif defect == "quad_metrics":
        fixture.quad["arms"][0]["metrics"] = {}
        expected_stage = "V5-exact-reconstruction"
    else:
        monkeypatch.setattr(
            runner,
            "clone_joint_arms",
            lambda *args: (SimpleNamespace(arm_id="control", model=_model().float()),),
        )
        expected_stage = "control-joint-training"
    report = runner.run_joint_hybrid(**fixture.arguments)
    assert report.status == "FAILED" and not report.postflight_verified
    assert report.failure_stage == expected_stage
    assert report.failure_reason
    assert report.comparison is None
    assert report.arms == ()
    assert fixture.events.count("mock_reconstruction") <= 1
    if defect == "reconstruction_failure":
        assert report.warmstart_optimizer_updates is None
    elif expected_stage != "V5-exact-reconstruction" and defect != "clone_state":
        assert report.warmstart_optimizer_updates == 0
    assert (
        fixture.arguments["output_path"].read_bytes()
        == canonical_json_bytes(report) + b"\n"
    )


def test_failed_candidate_preserves_completed_control_and_partial_attempt_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    partial = replace(fixture.step_trace[0], step=1, optimizer_updates=1)

    def train(state: Any, *args: Any, **kwargs: Any) -> Any:
        if state.arm_id == "control":
            return fixture.train(state, *args, **kwargs)
        evaluated = kwargs["evaluate"](state.model, 0)
        raise JointTrainingFailure(
            FloatingPointError("synthetic nonfinite gradient"),
            arm_id="candidate",
            attempted_step=2,
            step_trace=(partial,),
            evaluations=((0, evaluated),),
            optimizer_updates=1,
        )

    monkeypatch.setattr(runner, "train_joint_arm", train)
    report = runner.run_joint_hybrid(**fixture.arguments)
    assert report.status == "FAILED"
    assert report.failure_stage == "candidate-joint-training"
    assert tuple(arm.arm_id for arm in report.arms) == ("control",)
    details = dict(report.failure_details)
    assert details["arm_id"] == "candidate" and details["attempted_step"] == 2
    assert details["optimizer_updates"] == 1
    assert details["completed_step_trace"] == (partial,)
    assert details["completed_evaluations"][0][0] == 0
    assert details["cause_type"] == "FloatingPointError"
    assert fixture.events.count("mock_reconstruction") == 1


@pytest.mark.parametrize(
    "defect", "clone_config clone_module result_config result_module".split()
)
def test_config_and_module_bindings_are_checked_before_and_after_each_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    if defect == "clone_config":
        fixture.states[0].model.config = replace(fixture.model.config, hidden_dim=16)
    elif defect == "clone_module":
        original_digest = runner.module_digest
        monkeypatch.setattr(
            runner,
            "module_digest",
            lambda model: (
                "0" * 64 if model is fixture.states[0].model else original_digest(model)
            ),
        )
    else:

        def changed_result(*args: Any, **kwargs: Any) -> JointArmTrainingResult:
            result = fixture.train(*args, **kwargs)
            return replace(
                result,
                **{
                    (
                        "config_hash" if defect == "result_config" else "module_digest"
                    ): "0" * 64
                },
            )

        monkeypatch.setattr(runner, "train_joint_arm", changed_result)
    report = runner.run_joint_hybrid(**fixture.arguments)
    assert (
        report.status == "FAILED" and report.failure_stage == "control-joint-training"
    )
    assert "config/module" in report.failure_reason
    assert len(report.arms) == (0 if defect.startswith("clone") else 1)


@pytest.mark.parametrize(
    "defect",
    "source_result baseline baseline_config pin revision opened".split(),
)
def test_postflight_refuses_changed_source_baseline_authority_or_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    if defect == "source_result":
        calls = 0

        def changed(case_id: str) -> tuple[SimpleNamespace, None]:
            nonlocal calls
            calls += 1
            latest = (
                fixture.verified
                if calls == 1
                else SimpleNamespace(
                    request=fixture.verified.request,
                    result=replace(fixture.verified.result, total_strain_energy=0.5),
                )
            )
            return latest, None

        fixture.source.load_fit_with_collocation = changed
    elif defect in ("baseline", "baseline_config", "opened"):
        original = fixture.train

        def changed_train(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            if result.arm_id == "candidate":
                if defect == "opened":
                    fixture.source.development_opened = True
                elif defect == "baseline":
                    with torch.no_grad():
                        next(fixture.model.parameters()).add_(1)
                else:
                    fixture.model.config = replace(fixture.model.config, hidden_dim=16)
            return result

        monkeypatch.setattr(runner, "train_joint_arm", changed_train)
    elif defect in ("pin", "revision"):
        values = iter(
            (fixture.pin, {"changed": True})
            if defect == "pin"
            else ("f" * 40, "0" * 40)
        )
        monkeypatch.setattr(
            runner,
            "validate_execution_pin" if defect == "pin" else "_clean_revision",
            lambda: next(values),
        )
    report = runner.run_joint_hybrid(**fixture.arguments)
    assert report.status == "FAILED" and not report.postflight_verified
    assert report.failure_stage == "postflight"
    assert len(report.arms) == 2
    assert report.failure_reason


def test_production_population_guard_still_requires_969_vertices_1792_cells_and_20_qoi() -> (
    None
):
    plan = JointHybridPlan()
    case = _case(plan.case_id)
    with pytest.raises(ValueError, match="population"):
        runner._validate_registered_population(
            case, tuple(case.tensors.qoi_indices.tolist()), plan
        )
    request = SimpleNamespace(mesh=SimpleNamespace(nodes=range(969), cells=range(1792)))
    tensors = SimpleNamespace(qoi_indices=torch.arange(20))
    synthetic = SimpleNamespace(request=request, tensors=tensors)
    runner._validate_registered_population(synthetic, tuple(range(20)), plan)
    with pytest.raises(ValueError, match="population"):
        runner._validate_registered_population(synthetic, tuple(range(1, 21)), plan)
    with pytest.raises(ValueError, match="population"):
        runner._validate_registered_population(synthetic, tuple(range(19)), plan)


def _pin_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    proof = {
        "$type": "piginot.campaigns.JointRuntimeProof",
        "audit_status": "PASS",
        "dependency_check_status": "PASS",
        "shared_runtime_unchanged": True,
        "scientific_packages_unchanged": True,
        "scientific_package_hashes": (("torch", "a" * 64),),
    }
    _patch(
        monkeypatch,
        runner,
        _PUBLIC=tmp_path,
        implementation_hashes=lambda: (("synthetic.py", "b" * 64),),
        dependency_versions=lambda: (("torch", "2.13.0+cpu"),),
    )
    _write_record(tmp_path / "S13-JOINT-RUNTIME-PROOF.json", proof)
    _write_record(
        tmp_path / "S13-JOINT-HYBRID-APPROVAL.json", runner.JointHybridApproval()
    )
    pin = runner.build_execution_pin()
    _write_record(tmp_path / "S13-JOINT-HYBRID-EXECUTION-PIN.json", pin)
    _patch(
        monkeypatch,
        runner,
        build_joint_hybrid_proposal=lambda *args: SimpleNamespace(
            plan=JointHybridPlan()
        ),
        validate_replay_runtime=lambda *args: None,
    )
    _patch(
        monkeypatch,
        runtime,
        assert_project_origins=lambda *args: None,
        scientific_runtime=lambda: None,
        scientific_package_hashes=lambda: proof["scientific_package_hashes"],
    )
    return SimpleNamespace(
        proof=proof, pin=_canonical(pin), expected_pin=pin, root=tmp_path
    )


def test_execution_pin_binds_approval_sources_runtime_and_scientific_binaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _pin_fixture(tmp_path, monkeypatch)
    assert runner.validate_execution_pin() == fixture.expected_pin
    assert fixture.pin["plan_hash"] == JointHybridPlan().content_hash
    assert (
        fixture.pin["runtime_proof_sha256"]
        == hashlib.sha256(
            (tmp_path / "S13-JOINT-RUNTIME-PROOF.json").read_bytes()
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "defect",
    "type audit dependency shared scientific code approval_plan binary".split(),
)
def test_invalid_runtime_proof_or_execution_pin_cannot_authorize_private_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    fixture = _pin_fixture(tmp_path, monkeypatch)
    if defect in ("type", "audit", "dependency", "shared", "scientific"):
        key = {
            "type": "$type",
            "audit": "audit_status",
            "dependency": "dependency_check_status",
            "shared": "shared_runtime_unchanged",
            "scientific": "scientific_packages_unchanged",
        }[defect]
        fixture.proof[key] = False if defect in ("shared", "scientific") else "FAIL"
        _write_record(tmp_path / "S13-JOINT-RUNTIME-PROOF.json", fixture.proof)
    elif defect == "code":
        fixture.pin["implementation_sha256"] = ()
        _write_record(tmp_path / "S13-JOINT-HYBRID-EXECUTION-PIN.json", fixture.pin)
    elif defect == "approval_plan":
        monkeypatch.setattr(
            runner,
            "build_joint_hybrid_proposal",
            lambda *args: SimpleNamespace(plan=SimpleNamespace(content_hash="0" * 64)),
        )
    else:
        monkeypatch.setattr(
            runtime, "scientific_package_hashes", lambda: (("torch", "0" * 64),)
        )
    monkeypatch.setattr(
        runner,
        "open_selective_s10_source",
        lambda *args, **kwargs: pytest.fail("private source reached"),
    )
    with pytest.raises(ValueError):
        runner.run_joint_hybrid(
            s10_run_root=tmp_path / "s10",
            s11_checkpoint_root=tmp_path / "s11",
            development_access_record=tmp_path / "access.json",
            output_path=tmp_path / "result.json",
        )
    assert not (tmp_path / "result.json").exists()


def test_public_record_and_source_pin_refuse_symlink_paths_without_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "linked.json"
    original = Path.is_symlink
    monkeypatch.setattr(
        Path, "is_symlink", lambda current: current == path or original(current)
    )
    with pytest.raises(ValueError, match="symlinks"):
        runner._public_record(path)
    source = tmp_path / "src/piginot/file.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"# synthetic source")
    monkeypatch.setattr(runner, "_ROOT", tmp_path)
    monkeypatch.setattr(Path, "is_symlink", lambda current: current == source)
    with pytest.raises(ValueError, match="symlink"):
        runner.implementation_hashes()


def test_dependency_versions_normalize_names_and_clean_revision_checks_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.importlib.metadata,
        "distributions",
        lambda: (
            SimpleNamespace(metadata={"Name": "Z_Package"}, version="1"),
            SimpleNamespace(metadata={"Name": "a-package"}, version="2"),
        ),
    )
    assert runner.dependency_versions() == (("a-package", "2"), ("z-package", "1"))
    outputs = iter(
        (SimpleNamespace(stdout=""), SimpleNamespace(stdout="f" * 40 + "\n"))
    )
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: next(outputs))
    assert runner._clean_revision() == "f" * 40
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M changed.py\n"),
    )
    with pytest.raises(ValueError, match="clean worktree"):
        runner._clean_revision()


@pytest.mark.parametrize(("status", "exit_code"), (("COMPLETED", 0), ("FAILED", 1)))
def test_cli_passes_absolute_paths_and_fixed_threads_without_real_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    exit_code: int,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        runner.torch, "set_num_threads", lambda value: calls.append(("threads", value))
    )
    monkeypatch.setattr(
        runner.torch,
        "set_num_interop_threads",
        lambda value: calls.append(("interop", value)),
    )

    def run(**kwargs: Path) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(status=status)

    monkeypatch.setattr(runner, "run_joint_hybrid", run)
    arguments = tuple(
        item
        for name in (
            "s10-run-root",
            "s11-checkpoint-root",
            "development-access-record",
            "output",
        )
        for item in (f"--{name}", str(tmp_path / name))
    )
    assert runner.main(arguments) == exit_code
    assert calls[:2] == [("threads", 8), ("interop", 8)]
    assert calls[-1]["output_path"] == tmp_path / "output"
    with pytest.raises(SystemExit):
        runner.main(())
