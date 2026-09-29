from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from campaigns.dogbone_v1 import s13_overfit_diagnostic as diagnostic
from campaigns.dogbone_v1.s11_contingency_models import model_state_hash
from piginot.contracts import canonical_json_bytes
from piginot.doe import EvidenceRole
from tests.campaigns.test_dogbone_s13_objective import _model
from tests.campaigns.test_dogbone_s13_screen import _case


def _metrics(
    *,
    data_mse: float,
    u_relative_l2: float = 0.01,
    p_relative_l2: float = 0.04,
    peak_error: float = 0.03,
    invalid: int = 0,
    invalid_vertices: int = 0,
) -> diagnostic.OverfitCaseMetrics:
    return diagnostic.OverfitCaseMetrics(
        normalized_data_mse=data_mse,
        normalized_u_mse=data_mse / 2.0,
        normalized_p_mse=data_mse / 2.0,
        u_relative_l2=u_relative_l2,
        p_relative_l2=p_relative_l2,
        absolute_peak_error=abs(peak_error),
        signed_peak_error=peak_error,
        min_vertex_determinant=0.5 if invalid_vertices == 0 else -0.1,
        invalid_vertex_count=invalid_vertices,
        min_collocation_determinant=0.5 if invalid == 0 else -0.1,
        invalid_collocation_count=invalid,
    )


def test_default_plan_preregisters_bounded_one_case_success_contract() -> None:
    plan = diagnostic.build_s13_overfit_plan()

    assert plan.role is EvidenceRole.TRAIN
    assert plan.non_gate is True
    assert plan.case_id == "candidate-002485"
    assert plan.seed == 13_001
    assert plan.learning_rate == pytest.approx(1.0e-4)
    assert plan.optimizer_steps == 800
    assert plan.maximum_optimizer_steps == 800
    assert plan.minimum_data_mse_drop_fraction == pytest.approx(0.80)
    assert plan.maximum_final_u_relative_l2 == pytest.approx(0.02)
    assert plan.maximum_final_p_relative_l2 == pytest.approx(0.05)
    assert plan.maximum_final_absolute_peak_error == pytest.approx(0.05)
    assert plan.maximum_final_invalid_collocation_count == 0
    assert len(plan.content_hash) == 64

    assert replace(plan, optimizer_steps=3).optimizer_steps == 3
    with pytest.raises(ValueError, match="maximum"):
        replace(plan, optimizer_steps=801)
    with pytest.raises(PermissionError, match="TRAIN"):
        replace(plan, role=EvidenceRole.DEVELOPMENT)


@pytest.mark.parametrize(
    ("change", "error"),
    (
        ({"diagnostic_id": "changed"}, ValueError),
        ({"non_gate": False}, ValueError),
        ({"case_id": "other"}, ValueError),
        ({"seed": -1}, ValueError),
        ({"learning_rate": float("nan")}, ValueError),
        ({"maximum_optimizer_steps": 801}, ValueError),
        ({"optimizer_steps": 0}, ValueError),
        ({"minimum_data_mse_drop_fraction": 1.1}, ValueError),
        ({"maximum_final_u_relative_l2": -0.1}, ValueError),
        ({"maximum_final_invalid_collocation_count": 1}, ValueError),
    ),
)
def test_plan_rejects_changes_to_the_frozen_contract(
    change: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        replace(diagnostic.build_s13_overfit_plan(), **change)  # type: ignore[arg-type]


def test_metric_and_classification_contracts_fail_closed() -> None:
    valid = _metrics(data_mse=1.0)
    with pytest.raises(ValueError, match="equal"):
        replace(valid, normalized_data_mse=2.0)
    with pytest.raises(ValueError, match="signed"):
        replace(valid, absolute_peak_error=0.5)
    with pytest.raises(ValueError, match="non-negative"):
        replace(valid, invalid_collocation_count=-1)
    with pytest.raises(ValueError, match="positive"):
        diagnostic.classify_s13_overfit(
            diagnostic.build_s13_overfit_plan(),
            _metrics(data_mse=0.0),
            valid,
        )


def test_success_classification_requires_every_preregistered_condition() -> None:
    plan = diagnostic.build_s13_overfit_plan(optimizer_steps=2)
    before = _metrics(data_mse=1.0)
    after = _metrics(data_mse=0.19)

    successful = diagnostic.classify_s13_overfit(plan, before, after)

    assert successful.data_mse_drop_fraction == pytest.approx(0.81)
    assert successful.success is True
    assert successful.failed_criteria == ()

    failed = diagnostic.classify_s13_overfit(
        plan,
        before,
        _metrics(
            data_mse=0.21,
            u_relative_l2=0.021,
            p_relative_l2=0.051,
            peak_error=-0.051,
            invalid=1,
        ),
    )

    assert failed.success is False
    assert failed.failed_criteria == (
        "combined_data_mse_drop",
        "final_u_relative_l2",
        "final_p_relative_l2",
        "final_absolute_peak_error",
        "final_invalid_collocation_count",
    )


def test_data_only_training_uses_full_vertices_and_is_deterministic() -> None:
    case = _case("candidate-002485")
    left = _model()
    right = _model()
    initial_hash = model_state_hash(left)

    left_trace = diagnostic._train_one_case(  # noqa: SLF001
        left,
        case,
        diagnostic.build_s13_overfit_plan(optimizer_steps=2),
    )
    right_trace = diagnostic._train_one_case(  # noqa: SLF001
        right,
        case,
        diagnostic.build_s13_overfit_plan(optimizer_steps=2),
    )

    assert left_trace == right_trace
    assert len(left_trace) == 2
    assert all(value >= 0.0 for value in left_trace)
    assert model_state_hash(left) == model_state_hash(right)
    assert model_state_hash(left) != initial_hash


def test_case_metrics_use_all_vertices_and_physical_vertex_determinants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case("candidate-002485")
    prediction = SimpleNamespace(
        u=(case.tensors.target_u * 0.9).unsqueeze(0),
        P=(case.tensors.target_P * 1.1).unsqueeze(0),
    )
    monkeypatch.setattr(diagnostic, "_predict_case", lambda *args: prediction)
    monkeypatch.setattr(
        diagnostic,
        "_vertex_determinants",
        lambda *args: torch.tensor([0.5, 0.2, -0.1, 0.4], dtype=torch.float64),
    )
    monkeypatch.setattr(
        diagnostic,
        "_collocation_determinants",
        lambda *args: torch.tensor(
            [0.6, 0.3, -0.2, 0.7, 0.8, 0.9, 1.0, 1.1],
            dtype=torch.float64,
        ),
    )

    metrics = diagnostic._measure_case(_model(), case)  # noqa: SLF001

    expected_u_mse = float(
        torch.mean((prediction.u[0] - case.tensors.target_u).square())
    )
    expected_p_mse = float(
        torch.mean((prediction.P[0] - case.tensors.target_P).square())
    )
    assert metrics.normalized_u_mse == pytest.approx(expected_u_mse)
    assert metrics.normalized_p_mse == pytest.approx(expected_p_mse)
    assert metrics.normalized_data_mse == pytest.approx(expected_u_mse + expected_p_mse)
    assert metrics.u_relative_l2 == pytest.approx(0.1)
    assert metrics.p_relative_l2 == pytest.approx(0.1)
    assert metrics.signed_peak_error == pytest.approx(0.1)
    assert metrics.min_vertex_determinant == pytest.approx(-0.1)
    assert metrics.invalid_vertex_count == 1
    assert metrics.min_collocation_determinant == pytest.approx(-0.2)
    assert metrics.invalid_collocation_count == 1


def test_success_uses_collocation_invalid_count_not_vertex_count() -> None:
    plan = diagnostic.build_s13_overfit_plan(optimizer_steps=2)
    result = diagnostic.classify_s13_overfit(
        plan,
        _metrics(data_mse=1.0),
        _metrics(data_mse=0.1, invalid=0, invalid_vertices=1),
    )

    assert result.success is True
    assert result.failed_criteria == ()


def test_actual_case_measurement_runs_full_vertex_prediction_and_gradient() -> None:
    case = _case("candidate-002485")
    model = _model()
    model.train()

    metrics = diagnostic._measure_case(model, case)  # noqa: SLF001

    assert metrics.normalized_data_mse >= 0.0
    assert metrics.u_relative_l2 >= 0.0
    assert metrics.p_relative_l2 >= 0.0
    assert metrics.absolute_peak_error == pytest.approx(abs(metrics.signed_peak_error))
    assert math.isfinite(metrics.min_vertex_determinant)
    assert model.training


def test_orchestrator_freezes_s11_lineage_before_opening_train_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s10 = tmp_path / "s10"
    s11 = tmp_path / "s11"
    access = tmp_path / "audit" / "development-access.json"
    for path in (s10, s11, access.parent):
        path.mkdir()
    model = _model()
    initial_hash = model_state_hash(model)
    calls: list[str] = []
    initialization = SimpleNamespace(
        candidate_id="baseline-raw-parameters",
        model=model,
        source_bindings_hash="a" * 64,
        evidence_file_blake2b256="b" * 64,
        checkpoint_file_blake2b256="c" * 64,
    )

    def load_initialization(path: Path) -> object:
        calls.append("initialization")
        assert path == s11
        return initialization

    monkeypatch.setattr(
        diagnostic, "load_selected_s11_initialization", load_initialization
    )
    source = SimpleNamespace(
        development_opened=False,
        bindings=SimpleNamespace(content_hash="a" * 64),
    )

    def open_source(path: Path, *, development_access_record: Path) -> object:
        assert calls == ["initialization"]
        assert path == s10
        assert development_access_record == access
        calls.append("source")
        return source

    monkeypatch.setattr(diagnostic, "open_selective_s10_source", open_source)
    case = _case("candidate-002485")
    data = SimpleNamespace(cases=(case,))
    monkeypatch.setattr(
        diagnostic,
        "materialize_dogbone_s13_training_data",
        lambda opened, loaded: data,
    )
    monkeypatch.setattr(
        diagnostic, "_measure_case", lambda *args: _metrics(data_mse=1.0)
    )
    monkeypatch.setattr(diagnostic, "_train_one_case", lambda *args: (1.0, 0.1))
    final_metrics = _metrics(data_mse=0.1)
    measurements = iter((_metrics(data_mse=1.0), final_metrics))
    monkeypatch.setattr(diagnostic, "_measure_case", lambda *args: next(measurements))
    monkeypatch.setattr(
        diagnostic,
        "model_state_hash",
        lambda current: initial_hash if len(calls) < 2 else "d" * 64,
    )

    report = diagnostic.run_s13_overfit_diagnostic(
        s10_run_root=s10,
        s11_checkpoint_root=s11,
        development_access_record=access,
        plan=diagnostic.build_s13_overfit_plan(optimizer_steps=2),
    )

    assert calls == ["initialization", "source"]
    assert report.role is EvidenceRole.TRAIN
    assert report.non_gate is True
    assert report.case_id == "candidate-002485"
    assert report.initial_model_state_hash == initial_hash
    assert report.final_model_state_hash == "d" * 64
    assert report.initial_checkpoint_file_hash == "c" * 64
    assert report.optimizer_loss_trace == (1.0, 0.1)
    assert report.classification.success is True
    assert source.development_opened is False


def test_preflight_rejects_unsafe_or_existing_access_and_output_paths(
    tmp_path: Path,
) -> None:
    s10 = tmp_path / "s10"
    s11 = tmp_path / "s11"
    outside = tmp_path / "outside"
    for path in (s10, s11, outside):
        path.mkdir()

    diagnostic._preflight_paths(  # noqa: SLF001
        s10,
        s11,
        outside / "access.json",
        outside / "report.json",
    )
    with pytest.raises(ValueError, match="absolute"):
        diagnostic._preflight_paths(  # noqa: SLF001
            Path("relative"), s11, outside / "access.json", None
        )
    with pytest.raises(ValueError, match="outside source roots"):
        diagnostic._preflight_paths(  # noqa: SLF001
            s10, s11, s10 / "access.json", None
        )
    existing = outside / "existing.json"
    existing.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not already exist"):
        diagnostic._preflight_paths(  # noqa: SLF001
            s10, s11, outside / "access.json", existing
        )


def test_cli_emits_canonical_json_and_exclusively_writes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = diagnostic.build_s13_overfit_plan(optimizer_steps=2)
    classification = diagnostic.classify_s13_overfit(
        plan, _metrics(data_mse=1.0), _metrics(data_mse=0.1)
    )
    report = diagnostic.S13OverfitDiagnosticReport(
        role=EvidenceRole.TRAIN,
        non_gate=True,
        plan=plan,
        plan_hash=plan.content_hash,
        case_id=plan.case_id,
        source_bindings_hash="a" * 64,
        s11_evidence_file_hash="b" * 64,
        initial_checkpoint_file_hash="c" * 64,
        initial_model_state_hash="d" * 64,
        final_model_state_hash="e" * 64,
        before=_metrics(data_mse=1.0),
        after=_metrics(data_mse=0.1),
        classification=classification,
        optimizer_loss_trace=(1.0, 0.1),
    )
    monkeypatch.setattr(
        diagnostic, "run_s13_overfit_diagnostic", lambda **kwargs: report
    )
    output = tmp_path / "report.json"
    access = tmp_path / "access.json"

    result = diagnostic.main(
        (
            "--s10-run-root",
            str(tmp_path / "s10"),
            "--s11-checkpoint-root",
            str(tmp_path / "s11"),
            "--development-access-record",
            str(access),
            "--optimizer-steps",
            "2",
            "--output",
            str(output),
        )
    )

    assert result == 0
    expected = canonical_json_bytes(report) + b"\n"
    assert output.read_bytes() == expected
    assert capsys.readouterr().out.encode("utf-8") == expected
    with pytest.raises(FileExistsError):
        diagnostic._exclusive_write(output, expected)  # noqa: SLF001
