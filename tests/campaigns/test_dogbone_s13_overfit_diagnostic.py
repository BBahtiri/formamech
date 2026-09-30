from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from campaigns.dogbone_v1 import s13_overfit_diagnostic as diagnostic
from campaigns.dogbone_v1.family import compiled_family
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

    assert plan.diagnostic_id == "dogbone-s13-one-case-overfit-v3"
    assert plan.role is EvidenceRole.TRAIN
    assert plan.non_gate is True
    assert plan.case_id == "candidate-002485"
    assert plan.seed == 13_001
    assert plan.learning_rate == pytest.approx(5.0e-6)
    assert plan.optimizer_steps == 800
    assert plan.maximum_optimizer_steps == 800
    assert plan.expected_vertex_count == 969
    assert plan.vertex_batch_size == 57
    assert (
        plan.vertex_sampler_policy_id
        == "dogbone-s13-overfit-blake2b-epoch-permutation-v1"
    )
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


def test_vertex_contract_matches_the_operator_reference_mesh() -> None:
    plan = diagnostic.build_s13_overfit_plan()
    vertex_count = len(compiled_family().reference_mesh.nodes)

    assert vertex_count == 969
    assert plan.expected_vertex_count == vertex_count


def test_v4_changes_only_learning_rate_at_the_same_budget() -> None:
    control = diagnostic.build_s13_overfit_plan()
    v4 = diagnostic.build_s13_overfit_plan(version="v4")

    assert v4.diagnostic_id == "dogbone-s13-one-case-overfit-v4"
    assert v4.learning_rate == pytest.approx(2.0e-5)
    assert (
        replace(v4, diagnostic_id=control.diagnostic_id, learning_rate=5.0e-6)
        == control
    )
    assert (
        diagnostic._build_vertex_sampler_plan(  # noqa: SLF001
            v4, v4.expected_vertex_count
        ).plan_hash
        == diagnostic._build_vertex_sampler_plan(  # noqa: SLF001
            control, control.expected_vertex_count
        ).plan_hash
    )
    with pytest.raises(ValueError, match="learning_rate"):
        replace(v4, learning_rate=control.learning_rate)
    with pytest.raises(ValueError, match="learning_rate"):
        replace(control, learning_rate=v4.learning_rate)


def test_v5_extends_the_v4_schedule_without_changing_its_first_800_steps() -> None:
    v4 = diagnostic.build_s13_overfit_plan(version="v4")
    v5 = diagnostic.build_s13_overfit_plan(version="v5")

    assert v5.diagnostic_id == "dogbone-s13-one-case-overfit-v5"
    assert v5.learning_rate == v4.learning_rate
    assert v5.optimizer_steps == 4_000
    assert v5.maximum_optimizer_steps == 4_000
    assert (
        replace(
            v5,
            diagnostic_id=v4.diagnostic_id,
            optimizer_steps=800,
            maximum_optimizer_steps=800,
        )
        == v4
    )
    assert (
        diagnostic._build_vertex_sampler_plan(  # noqa: SLF001
            v5, v5.expected_vertex_count
        ).batches[:800]
        == diagnostic._build_vertex_sampler_plan(  # noqa: SLF001
            v4, v4.expected_vertex_count
        ).batches
    )
    with pytest.raises(ValueError, match="maximum_optimizer_steps"):
        replace(v4, maximum_optimizer_steps=4_000)
    with pytest.raises(ValueError, match="maximum_optimizer_steps"):
        replace(v5, maximum_optimizer_steps=800)


def test_v6_adds_only_a_fixed_root_peak_term() -> None:
    v5 = diagnostic.build_s13_overfit_plan(version="v5")
    v6 = diagnostic.build_s13_overfit_plan(version="v6")

    assert v6.diagnostic_id == "dogbone-s13-one-case-overfit-v6"
    assert v5.peak_loss_weight == 0.0
    assert v6.peak_loss_weight == pytest.approx(0.1)
    assert replace(v6, diagnostic_id=v5.diagnostic_id) == v5
    assert (
        diagnostic._build_vertex_sampler_plan(  # noqa: SLF001
            v6, v6.expected_vertex_count
        ).batches
        == diagnostic._build_vertex_sampler_plan(  # noqa: SLF001
            v5, v5.expected_vertex_count
        ).batches
    )
    with pytest.raises(ValueError, match="version"):
        diagnostic.build_s13_overfit_plan(version="v7")


@pytest.mark.parametrize(
    ("change", "error"),
    (
        ({"diagnostic_id": "changed"}, ValueError),
        ({"non_gate": False}, ValueError),
        ({"case_id": "other"}, ValueError),
        ({"seed": -1}, ValueError),
        ({"seed": 13_002}, ValueError),
        ({"learning_rate": float("nan")}, ValueError),
        ({"learning_rate": 1.0e-5}, ValueError),
        ({"maximum_optimizer_steps": 801}, ValueError),
        ({"optimizer_steps": 0}, ValueError),
        ({"expected_vertex_count": 968}, ValueError),
        ({"vertex_batch_size": 56}, ValueError),
        ({"vertex_sampler_policy_id": "changed"}, ValueError),
        ({"minimum_data_mse_drop_fraction": 1.1}, ValueError),
        ({"minimum_data_mse_drop_fraction": 0.79}, ValueError),
        ({"maximum_final_u_relative_l2": -0.1}, ValueError),
        ({"maximum_final_u_relative_l2": 0.03}, ValueError),
        ({"maximum_final_p_relative_l2": 0.06}, ValueError),
        ({"maximum_final_absolute_peak_error": 0.06}, ValueError),
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


def test_minibatch_schedule_is_deterministic_and_covers_complete_epochs() -> None:
    plan = diagnostic.build_s13_overfit_plan(optimizer_steps=34)

    left = diagnostic._build_vertex_sampler_plan(plan, 969)  # noqa: SLF001
    right = diagnostic._build_vertex_sampler_plan(plan, 969)  # noqa: SLF001

    assert left == right
    assert left.population_size == 969
    assert len(left.batches) == 34
    assert all(len(batch) == 57 for batch in left.batches)
    for epoch in range(2):
        start = epoch * 17
        observed = tuple(
            sorted(
                index for batch in left.batches[start : start + 17] for index in batch
            )
        )
        assert observed == tuple(range(969))
    assert len(left.plan_hash) == 64


def test_chunked_prediction_matches_full_vertex_prediction() -> None:
    case = _case("candidate-002485")
    model = _model()

    full = diagnostic._predict_case(model, case)  # noqa: SLF001
    first = diagnostic._predict_case_vertices(  # noqa: SLF001
        model,
        case,
        torch.tensor((0, 2), dtype=torch.long),
    )
    second = diagnostic._predict_case_vertices(  # noqa: SLF001
        model,
        case,
        torch.tensor((1, 3), dtype=torch.long),
    )

    assert first.u.shape == (1, 2, 2)
    assert second.P.shape == (1, 2, 2, 2)
    torch.testing.assert_close(first.u[0, 0], full.u[0, 0])
    torch.testing.assert_close(first.u[0, 1], full.u[0, 2])
    torch.testing.assert_close(first.P[0, 0], full.P[0, 0])
    torch.testing.assert_close(first.P[0, 1], full.P[0, 2])
    torch.testing.assert_close(second.u[0, 0], full.u[0, 1])
    torch.testing.assert_close(second.u[0, 1], full.u[0, 3])
    torch.testing.assert_close(second.P[0, 0], full.P[0, 1])
    torch.testing.assert_close(second.P[0, 1], full.P[0, 3])


def test_data_only_training_uses_vertex_minibatches_and_is_deterministic() -> None:
    case = _case("candidate-002485")
    left = _model()
    right = _model()
    initial_hash = model_state_hash(left)
    plan = diagnostic.build_s13_overfit_plan(optimizer_steps=2)

    left_trace = diagnostic._train_one_case(  # noqa: SLF001
        left,
        case,
        plan,
    )
    right_trace = diagnostic._train_one_case(  # noqa: SLF001
        right,
        case,
        plan,
    )

    assert left_trace == right_trace
    assert len(left_trace) == 2
    assert all(value >= 0.0 for value in left_trace)
    assert model_state_hash(left) == model_state_hash(right)
    assert model_state_hash(left) != initial_hash


def test_v6_training_queries_exact_root_qoi_and_uses_peak_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case("candidate-002485")
    v5_model = _model()
    v6_model = _model()
    v5 = diagnostic.build_s13_overfit_plan(optimizer_steps=1, version="v5")
    v6 = diagnostic.build_s13_overfit_plan(optimizer_steps=1, version="v6")
    original = diagnostic._predict_case_vertices  # noqa: SLF001
    queries: list[tuple[int, ...]] = []

    def record_queries(
        model: object,
        selected: object,
        indices: torch.Tensor,
    ) -> object:
        queries.append(tuple(int(index) for index in indices.tolist()))
        return original(model, selected, indices)  # type: ignore[arg-type]

    monkeypatch.setattr(diagnostic, "_predict_case_vertices", record_queries)

    v5_trace = diagnostic._train_one_case(v5_model, case, v5)  # noqa: SLF001
    v6_trace = diagnostic._train_one_case(v6_model, case, v6)  # noqa: SLF001

    assert len(queries[0]) == 4
    assert queries[1][:4] == queries[0]
    assert queries[1][4:] == tuple(case.tensors.qoi_indices.tolist())
    assert v5_trace != v6_trace
    assert model_state_hash(v5_model) != model_state_hash(v6_model)


def test_v6_first_loss_matches_field_plus_weighted_exact_qoi_peak() -> None:
    case = _case("candidate-002485")
    model = _model()
    plan = diagnostic.build_s13_overfit_plan(optimizer_steps=1, version="v6")
    sampler = diagnostic._build_vertex_sampler_plan(  # noqa: SLF001
        plan, len(case.tensors.query_xy)
    )
    indices = torch.tensor(sampler.batches[0], dtype=torch.long)
    query_indices = torch.cat((indices, case.tensors.qoi_indices))
    output = diagnostic._predict_case_vertices(  # noqa: SLF001
        model, case, query_indices
    )
    target_u = case.tensors.target_u.index_select(0, indices)
    target_p = case.tensors.target_P.index_select(0, indices)
    predicted_peak = torch.linalg.vector_norm(
        output.P[0, len(indices) :], dim=(-2, -1)
    ).amax()
    true_peak = torch.linalg.vector_norm(
        case.tensors.target_P[case.tensors.qoi_indices], dim=(-2, -1)
    ).amax()
    expected = (
        torch.mean((output.u[0, : len(indices)] - target_u).square())
        + torch.mean((output.P[0, : len(indices)] - target_p).square())
        + 0.1 * (predicted_peak - true_peak).square()
    )

    trace = diagnostic._train_one_case(model, case, plan)  # noqa: SLF001

    assert trace[0] == pytest.approx(float(expected.detach()), rel=1.0e-12)


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
    monkeypatch.setattr(diagnostic, "_EXPECTED_VERTEX_COUNT", 4)
    monkeypatch.setattr(diagnostic, "_VERTEX_BATCH_SIZE", 4)
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
    real_sampler_builder = diagnostic._build_vertex_sampler_plan  # noqa: SLF001

    def build_sampler(plan: diagnostic.S13OverfitPlan, vertex_count: int) -> object:
        calls.append("sampler")
        return real_sampler_builder(plan, vertex_count)

    monkeypatch.setattr(diagnostic, "_build_vertex_sampler_plan", build_sampler)

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
        assert calls == ["sampler", "initialization"]
        assert path == s10
        assert development_access_record == access
        calls.append("source")
        return source

    monkeypatch.setattr(diagnostic, "open_selective_s10_source", open_source)
    case = _case("candidate-002485")

    def materialize(opened: object, loaded: object, case_id: str) -> object:
        assert opened is source
        assert loaded is initialization
        assert case_id == "candidate-002485"
        calls.append("case")
        return case

    monkeypatch.setattr(
        diagnostic,
        "materialize_dogbone_s13_training_case",
        materialize,
    )
    monkeypatch.setattr(
        diagnostic, "_measure_case", lambda *args: _metrics(data_mse=1.0)
    )
    monkeypatch.setattr(diagnostic, "_train_one_case", lambda *args: (1.0, 0.1))
    final_metrics = _metrics(data_mse=0.1)
    measurements = iter((_metrics(data_mse=1.0), final_metrics))
    monkeypatch.setattr(diagnostic, "_measure_case", lambda *args: next(measurements))
    state_hashes = iter((initial_hash, "d" * 64))
    monkeypatch.setattr(
        diagnostic, "model_state_hash", lambda current: next(state_hashes)
    )

    report = diagnostic.run_s13_overfit_diagnostic(
        s10_run_root=s10,
        s11_checkpoint_root=s11,
        development_access_record=access,
        plan=diagnostic.build_s13_overfit_plan(optimizer_steps=2),
    )

    assert calls == ["sampler", "initialization", "source", "case", "sampler"]
    assert report.role is EvidenceRole.TRAIN
    assert report.non_gate is True
    assert report.case_id == "candidate-002485"
    assert report.initial_model_state_hash == initial_hash
    assert report.final_model_state_hash == "d" * 64
    assert report.initial_checkpoint_file_hash == "c" * 64
    assert report.vertex_count == 4
    assert len(report.vertex_sampler_plan_hash) == 64
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
    plan = diagnostic.build_s13_overfit_plan(optimizer_steps=2, version="v6")
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
        vertex_count=plan.expected_vertex_count,
        vertex_sampler_plan_hash=diagnostic._build_vertex_sampler_plan(  # noqa: SLF001
            plan, plan.expected_vertex_count
        ).plan_hash,
        before=_metrics(data_mse=1.0),
        after=_metrics(data_mse=0.1),
        classification=classification,
        optimizer_loss_trace=(1.0, 0.1),
    )
    with pytest.raises(ValueError, match="sampler"):
        replace(report, vertex_sampler_plan_hash="f" * 64)
    with pytest.raises(ValueError, match="classification"):
        replace(
            report,
            classification=diagnostic.OverfitClassification(0.9, False, ("forged",)),
        )

    def run_v6(**kwargs: object) -> diagnostic.S13OverfitDiagnosticReport:
        assert kwargs["plan"] == plan
        return report

    monkeypatch.setattr(diagnostic, "run_s13_overfit_diagnostic", run_v6)
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
            "--version",
            "v6",
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
