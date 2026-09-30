from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from campaigns.dogbone_v1 import s13_stress_capacity as capacity
from campaigns.dogbone_v1.s11_contingency_models import model_state_hash
from campaigns.dogbone_v1.s13_overfit_diagnostic import (
    _build_vertex_sampler_plan,
    _predict_case,
    _train_one_case,
    build_s13_overfit_plan,
)
from campaigns.dogbone_v1.s13_spatial_audit import _vertex_prediction_and_physics
from piginot.contracts import canonical_json_bytes
from piginot.doe import EvidenceRole
from piginot.operators import module_digest
from tests.campaigns.test_dogbone_s13_objective import _model
from tests.campaigns.test_dogbone_s13_overfit_diagnostic import _metrics
from tests.campaigns.test_dogbone_s13_screen import _case


def test_capacity_plan_binds_budget_migration_and_frozen_thresholds() -> None:
    plan = capacity.StressCapacityPlan()
    raw = canonical_json_bytes(plan)
    assert b'"kernel_scales":[0.08,0.25]' in raw
    assert b'"stress_layers":2' in raw
    assert b'"initialization_seed":13002' in raw
    assert plan.fine_tune_plan.optimizer_steps == 4000
    assert plan.fine_tune_plan.field_p_loss_weight == 1.0
    assert plan.fine_tune_plan.peak_loss_weight == 0.0
    for changes in (
        {"kernel_scales": (0.1, 0.25)},
        {"stress_layers": 1},
        {"minimum_root_improvement": 0.1},
        {"maximum_nonroot_regression": 0.1},
        {"initialization_seed": 1},
        {"non_gate": False},
    ):
        with pytest.raises(ValueError, match="frozen"):
            replace(plan, **changes)


def test_neutral_migration_preserves_outputs_gradients_and_source() -> None:
    source = _model()
    state = model_state_hash(source)
    case = _case("candidate-002485")
    plan = capacity.StressCapacityPlan()

    control = capacity.build_stress_arm(source, plan, enriched=False)
    enriched = capacity.build_stress_arm(source, plan, enriched=True)

    assert model_state_hash(source) == state
    assert model_state_hash(control) == state
    assert module_digest(enriched) != module_digest(source)
    expected = _predict_case(source, case)
    for arm in (control, enriched):
        actual = _predict_case(arm, case)
        torch.testing.assert_close(actual.flat(), expected.flat(), rtol=0, atol=0)
        _, _, det = _vertex_prediction_and_physics(arm, case)
        _, _, expected_det = _vertex_prediction_and_physics(source, case)
        torch.testing.assert_close(det, expected_det, rtol=0, atol=0)
    assert capacity.frozen_state_hash(control) == capacity.frozen_state_hash(enriched)
    assert capacity.trainable_names(control) == ("p_head.weight", "p_head.bias")
    assert all(
        name.startswith(capacity.STRESS_PREFIXES)
        for name in capacity.trainable_names(enriched)
    )


def test_migration_refuses_existing_enrichment_or_unexpected_state() -> None:
    source = _model()
    plan = capacity.StressCapacityPlan()
    enriched = capacity.build_stress_arm(source, plan, enriched=True)
    with pytest.raises(ValueError, match="baseline"):
        capacity.build_stress_arm(enriched, plan, enriched=True)
    source.register_buffer("surprise", torch.zeros(1, dtype=torch.float64))
    with pytest.raises(ValueError, match="state schema"):
        capacity.build_stress_arm(source, plan, enriched=False)


def test_stress_only_optimizer_changes_stress_and_preserves_all_u_derivatives() -> None:
    case = _case("candidate-002485")
    model = capacity.build_stress_arm(
        _model(), capacity.StressCapacityPlan(), enriched=True
    )
    fixed_hash = capacity.frozen_state_hash(model)
    before = capacity.displacement_signature(model, case)
    final_layer = model.stress_enrichment_blocks[0][3].weight.detach().clone()
    first_layer = model.stress_enrichment_blocks[0][1].weight.detach().clone()
    plan = build_s13_overfit_plan(version="v5", optimizer_steps=2)
    _train_one_case(model, case, plan, _build_vertex_sampler_plan(plan, 4))

    assert capacity.frozen_state_hash(model) == fixed_hash
    assert not torch.equal(model.stress_enrichment_blocks[0][3].weight, final_layer)
    assert not torch.equal(model.stress_enrichment_blocks[0][1].weight, first_layer)
    after = capacity.displacement_signature(model, case)
    capacity.assert_fixed_displacement(before, after)
    bad = (*after[:-1], after[-1] + 1e-12)
    with pytest.raises(ValueError, match="displacement"):
        capacity.assert_fixed_displacement(before, bad)
    model.u_head.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable"):
        capacity.assert_trainable_scope(model)


@pytest.mark.parametrize(
    ("root", "nonroot", "expected"),
    ((0.075, 0.0525, True), (0.075001, 0.0525, False), (0.075, 0.052501, False)),
)
def test_capacity_contrast_keeps_exact_boundary_thresholds(
    root: float, nonroot: float, expected: bool
) -> None:
    plan = capacity.StressCapacityPlan()
    control = SimpleNamespace(root_p_relative_l2=0.1, nonroot_p_relative_l2=0.05)
    candidate = SimpleNamespace(root_p_relative_l2=root, nonroot_p_relative_l2=nonroot)
    decision = capacity.classify_capacity_contrast(
        plan, control, candidate, candidate_primary_success=True
    )
    assert decision.success is expected
    assert not capacity.classify_capacity_contrast(
        plan, control, candidate, candidate_primary_success=False
    ).success


@pytest.mark.parametrize(
    "failure",
    ("replay", "bindings", "development_before", "development_during", "frozen", None),
)
def test_runner_checks_replay_scope_and_frozen_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    s10, s11, out = (tmp_path / name for name in ("s10", "s11", "out"))
    for path in (s10, s11, out):
        path.mkdir()
    case = _case("candidate-002485")
    case = replace(
        case,
        tensors=replace(
            case.tensors,
            query_xy=torch.ones((969, 2), dtype=torch.float64),
            target_u=torch.ones((969, 2), dtype=torch.float64),
            target_P=torch.ones((969, 2, 2), dtype=torch.float64),
            qoi_indices=torch.arange(20),
        ),
    )
    model = _model()
    source = SimpleNamespace(
        development_opened=failure == "development_before",
        bindings=SimpleNamespace(content_hash="a" * 64),
    )
    initialization = SimpleNamespace(
        model=model,
        evidence_file_blake2b256="b" * 64,
        checkpoint_file_blake2b256="c" * 64,
    )
    monkeypatch.setattr(
        capacity, "load_selected_s11_initialization", lambda *a: initialization
    )
    monkeypatch.setattr(capacity, "open_selective_s10_source", lambda *a, **k: source)
    monkeypatch.setattr(
        capacity, "materialize_dogbone_s13_training_case", lambda *a: case
    )
    plan = build_s13_overfit_plan(version="v5")
    report = {
        "plan_hash": plan.content_hash,
        "vertex_sampler_plan_hash": _build_vertex_sampler_plan(plan, 969).plan_hash,
        "source_bindings_hash": "a" * 64,
        "s11_evidence_file_hash": "b" * 64,
        "initial_checkpoint_file_hash": "c" * 64,
        "initial_model_state_hash": model_state_hash(model),
        "final_model_state_hash": model_state_hash(model),
        "optimizer_loss_trace": [0.1] * 4000,
    }
    if failure == "bindings":
        report["source_bindings_hash"] = "e" * 64
    monkeypatch.setattr(capacity, "_load_pinned_report", lambda *a: report)
    monkeypatch.setattr(capacity, "_measure_case", lambda *a: _metrics(data_mse=1.0))
    monkeypatch.setattr(
        capacity, "displacement_signature", lambda *a: (torch.ones(1),) * 6
    )
    monkeypatch.setattr(
        capacity,
        "_vertex_prediction_and_physics",
        lambda *a: (
            case.tensors.target_P * 1.01,
            torch.zeros_like(case.tensors.target_P),
            torch.ones(969, dtype=torch.float64),
        ),
    )
    trained: list[object] = []

    def train(*arguments: object) -> tuple[float, ...]:
        trained.append(arguments[0])
        if failure == "development_during":
            source.development_opened = True
        if failure == "frozen" and len(trained) == 2:
            with torch.no_grad():
                arguments[0].u_head.weight.add_(1.0)  # type: ignore[attr-defined]
        return (0.2 if failure == "replay" else 0.1,) * 4000

    monkeypatch.setattr(capacity, "_train_one_case", train)
    arguments = dict(
        s10_run_root=s10,
        s11_checkpoint_root=s11,
        development_access_record=out / "access.json",
        v5_report_path=out / "v5.json",
        output_path=out / "result.json",
    )
    if failure is None:
        result = capacity.run_stress_capacity_contrast(**arguments)
        assert len(trained) == 3
        assert tuple(arm.arm_id for arm in result.arms) == ("control", "enriched")
        assert result.role is EvidenceRole.TRAIN and result.non_gate is True
        assert result.decision.success is False
        assert result.arms[0].frozen_state_hash == result.arms[1].frozen_state_hash
        assert (
            result.arms[0].optimizer_loss_trace == result.arms[1].optimizer_loss_trace
        )
        assert result.plan_hash == result.plan.content_hash
        assert len(result.implementation_sha256) == 6
        assert result.fixed_displacement_verified is True
        canonical_json_bytes(result)
    else:
        expected_message = {
            "replay": "replay",
            "bindings": "binding",
            "frozen": "frozen",
            "development_before": "DEVELOPMENT",
            "development_during": "DEVELOPMENT",
        }[failure]
        with pytest.raises((ValueError, RuntimeError), match=expected_message):
            capacity.run_stress_capacity_contrast(**arguments)
        assert (
            len(trained)
            == {
                "replay": 1,
                "bindings": 0,
                "development_before": 0,
                "development_during": 1,
                "frozen": 2,
            }[failure]
        )
    assert not (out / "result.json").exists()


def test_cli_writes_report_exclusively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {"role": EvidenceRole.TRAIN, "non_gate": True}
    monkeypatch.setattr(capacity, "run_stress_capacity_contrast", lambda **k: report)
    output = tmp_path / "report.json"
    args = (
        "--s10-run-root",
        str(tmp_path / "s10"),
        "--s11-checkpoint-root",
        str(tmp_path / "s11"),
        "--development-access-record",
        str(tmp_path / "access.json"),
        "--v5-report",
        str(tmp_path / "v5.json"),
        "--output",
        str(output),
    )
    assert capacity.main(args) == 0
    assert output.read_bytes() == canonical_json_bytes(report) + b"\n"
    with pytest.raises(FileExistsError):
        capacity.main(args)
