from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from campaigns.dogbone_v1.s11_contingency_source import SelectiveS10Source
from campaigns.dogbone_v1.s13_data import DogboneS13TrainingData
from campaigns.dogbone_v1.s13_initialization import (
    load_selected_s11_checkpoint_evidence,
)
from campaigns.dogbone_v1.s13_objective import DogboneS13TrainingCase
from campaigns.dogbone_v1.s13_protocol import build_dogbone_s13_protocol
from campaigns.dogbone_v1.s13_screen import (
    _count_invalid_collocation_points,
    _determinants_for_case,
    _screen_metrics_from_cases,
    evaluate_dogbone_s13_train_screen,
)
from piginot.doe import EvidenceRole
from piginot.objectives import EquilibriumForm, ObjectiveMode, ObjectiveRegistry
from piginot.operators import OperatorOutput
from piginot.training import SelectionMetrics, TrainingPopulation, TrialConfig
from tests.campaigns.test_dogbone_s13_objective import (
    _collocation,
    _model,
    _request,
    _tensor_case,
)


def _case(case_id: str) -> DogboneS13TrainingCase:
    request = _request(case_id)
    tensors = _tensor_case(request)
    tensors = replace(
        tensors,
        leakage_group_id=f"group-{case_id}",
        target_u=torch.ones_like(tensors.target_u),
        target_P=torch.ones_like(tensors.target_P),
    )
    return DogboneS13TrainingCase(
        tensors=tensors,
        request=request,
        collocation=_collocation(request),
    )


def _trial() -> TrialConfig:
    return TrialConfig(
        learning_rate=1.0e-5,
        physics_weight_scale=1.0e-3,
        optimizer_steps=1_500,
        case_batch_size=4,
    )


def _registry() -> ObjectiveRegistry:
    return ObjectiveRegistry(
        mode=ObjectiveMode.HYBRID,
        equilibrium_form=EquilibriumForm.STRONG,
    )


def test_orientation_count_includes_fit_and_screen_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _case("fit-001")
    screen = _case("screen-001")
    seen: list[str] = []

    def determinants(model: object, case: DogboneS13TrainingCase) -> torch.Tensor:
        seen.append(case.tensors.case_id)
        values = [1.0] * 8
        if case is fit:
            values[0] = 0.0
        else:
            values[1] = -0.1
        return torch.tensor(values, dtype=torch.float64)

    monkeypatch.setattr(
        "campaigns.dogbone_v1.s13_screen._determinants_for_case", determinants
    )

    invalid = _count_invalid_collocation_points(
        _model(), (fit, screen), expected_interior_count=8
    )

    assert invalid == 2
    assert seen == ["fit-001", "screen-001"]


@pytest.mark.parametrize(
    ("axial_gradient", "expected_determinant"),
    ((0.0, 1.0), (-1.0, 0.0), (-1.2, -0.2)),
)
def test_actual_screen_determinant_uses_physical_displacement_gradient(
    monkeypatch: pytest.MonkeyPatch,
    axial_gradient: float,
    expected_determinant: float,
) -> None:
    case = _case("fit-001")
    model = _model()

    def forward(**kwargs: torch.Tensor) -> OperatorOutput:
        coordinates = kwargs["query_xy"]
        displacement = torch.stack(
            (
                10.0 * axial_gradient * coordinates[..., 0],
                0.0 * coordinates[..., 1],
            ),
            dim=-1,
        )
        stress = torch.zeros((*coordinates.shape[:2], 2, 2), dtype=torch.float64)
        return OperatorOutput(u=displacement, P=stress)

    monkeypatch.setattr(model, "forward", forward)

    determinant = _determinants_for_case(model, case)

    assert determinant.shape == (8,)
    torch.testing.assert_close(
        determinant,
        torch.full((8,), expected_determinant, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_orientation_count_fails_on_nonfinite_or_missing_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case("fit-001")
    monkeypatch.setattr(
        "campaigns.dogbone_v1.s13_screen._determinants_for_case",
        lambda model, case: torch.tensor([float("nan")] * 8),
    )
    with pytest.raises(ValueError, match="finite"):
        _count_invalid_collocation_points(_model(), (case,), expected_interior_count=8)

    monkeypatch.setattr(
        "campaigns.dogbone_v1.s13_screen._determinants_for_case",
        lambda model, case: torch.ones(7, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="interior"):
        _count_invalid_collocation_points(_model(), (case,), expected_interior_count=8)


def test_synthetic_screen_metrics_include_all_fit_folds_and_fixed_hybrid_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _case("fit-001")
    screen = _case("screen-001")
    seen: list[tuple[str, ...]] = []

    def count(
        model: object,
        cases: tuple[DogboneS13TrainingCase, ...],
        *,
        expected_interior_count: int,
    ) -> int:
        seen.append(tuple(case.tensors.case_id for case in cases))
        assert expected_interior_count == 8
        return 1

    monkeypatch.setattr(
        "campaigns.dogbone_v1.s13_screen._count_invalid_collocation_points",
        count,
    )
    model = _model()
    model.train()

    metrics = _screen_metrics_from_cases(
        model,
        fit_cases=(fit,),
        screen_cases=(screen,),
        trial=_trial(),
        base_registry=_registry(),
        expected_fit_ids=("fit-001",),
        expected_screen_ids=("screen-001",),
        expected_interior_count=8,
    )

    assert seen == [("fit-001", "screen-001")]
    assert metrics.invalid_collocation_points == 1
    assert metrics.hybrid_loss > 0.0
    assert metrics.median_absolute_peak_error >= 0.0
    assert metrics.p90_absolute_peak_error >= 0.0
    assert metrics.u_relative_l2 >= 0.0
    assert metrics.p_relative_l2 >= 0.0
    assert model.training


def test_synthetic_screen_refuses_missing_or_overlapping_population() -> None:
    fit = _case("fit-001")
    screen = _case("screen-001")
    arguments = dict(
        trial=_trial(),
        base_registry=_registry(),
        expected_fit_ids=("fit-001",),
        expected_screen_ids=("screen-001",),
        expected_interior_count=8,
    )

    with pytest.raises(ValueError, match="screen"):
        _screen_metrics_from_cases(
            _model(), fit_cases=(fit,), screen_cases=(), **arguments
        )
    with pytest.raises(ValueError, match="overlap"):
        _screen_metrics_from_cases(
            _model(),
            fit_cases=(fit,),
            screen_cases=(fit,),
            **{**arguments, "expected_screen_ids": ("fit-001",)},
        )
    with pytest.raises(ValueError, match="fit"):
        _screen_metrics_from_cases(
            _model(), fit_cases=(screen,), screen_cases=(fit,), **arguments
        )


def test_production_screen_refuses_opened_development_before_access(
    tmp_path: Path,
) -> None:
    source = object.__new__(SelectiveS10Source)
    object.__setattr__(source, "_development_opened", True)
    object.__setattr__(source, "_development_access_record", tmp_path / "access")

    with pytest.raises(RuntimeError, match="DEVELOPMENT"):
        evaluate_dogbone_s13_train_screen(
            protocol=object(),  # type: ignore[arg-type]
            source=source,
            training_data=object(),  # type: ignore[arg-type]
            model=_model(),
            trial=_trial(),
        )


def test_production_screen_replays_only_registered_train_screen_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = build_dogbone_s13_protocol()
    training_data = object.__new__(DogboneS13TrainingData)
    object.__setattr__(training_data, "cases", ())
    object.__setattr__(
        training_data,
        "population",
        TrainingPopulation(
            dataset_snapshot_hash=protocol.dataset_snapshot_hash,
            public_manifest_hash=protocol.public_manifest_hash,
            role=EvidenceRole.TRAIN,
            sample_ids=protocol.fit_case_ids,
        ),
    )
    object.__setattr__(
        training_data,
        "normalization",
        load_selected_s11_checkpoint_evidence().normalization,
    )
    object.__setattr__(
        training_data, "source_bindings_hash", protocol.source_bindings_hash
    )
    object.__setattr__(
        training_data, "selected_candidate_id", protocol.selected_candidate_id
    )
    object.__setattr__(
        training_data,
        "selected_checkpoint_file_hash",
        protocol.initial_checkpoint_file_hash,
    )
    source = object.__new__(SelectiveS10Source)
    screen_request = _request(protocol.screen_case_ids[0])
    verified = SimpleNamespace(
        case=SimpleNamespace(case_id=screen_request.request_id),
        request=screen_request,
        result=object(),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        SelectiveS10Source,
        "development_opened",
        property(lambda self: False),
    )
    monkeypatch.setattr(
        SelectiveS10Source,
        "bindings",
        property(
            lambda self: SimpleNamespace(content_hash=protocol.source_bindings_hash)
        ),
    )
    monkeypatch.setattr(
        SelectiveS10Source,
        "fit_case_ids",
        property(lambda self: protocol.fit_case_ids),
    )
    monkeypatch.setattr(
        SelectiveS10Source,
        "train_screen_case_ids",
        property(lambda self: protocol.screen_case_ids),
    )
    monkeypatch.setattr(
        SelectiveS10Source,
        "train_snapshot",
        property(
            lambda self: SimpleNamespace(content_hash=protocol.dataset_snapshot_hash)
        ),
    )

    def iter_screen(self: SelectiveS10Source):  # type: ignore[no-untyped-def]
        calls.append("iter_train_screen")
        yield verified

    def replay(self: SelectiveS10Source, item: object):  # type: ignore[no-untyped-def]
        assert item is verified
        calls.append("replay_train_collocation")
        return _collocation(screen_request)

    monkeypatch.setattr(SelectiveS10Source, "iter_train_screen", iter_screen)
    monkeypatch.setattr(SelectiveS10Source, "replay_train_collocation", replay)
    monkeypatch.setattr(
        "campaigns.dogbone_v1.s13_screen.module_digest",
        lambda model: protocol.architecture_module_digest,
    )
    monkeypatch.setattr(
        "campaigns.dogbone_v1.s13_screen.prepare_operator_case",
        lambda *args: object(),
    )
    monkeypatch.setattr(
        "campaigns.dogbone_v1.s13_screen.normalize_case_values",
        lambda *args: _tensor_case(screen_request),
    )
    sentinel = SelectionMetrics(0.07, 0.1, -0.06, 0.18, 0.03, 1.0, 0)

    def reduce_cases(model: object, **kwargs: object) -> SelectionMetrics:
        cases = kwargs["screen_cases"]
        assert isinstance(cases, tuple)
        assert len(cases) == 1
        assert cases[0].tensors.case_id == protocol.screen_case_ids[0]
        assert kwargs["expected_screen_ids"] == protocol.screen_case_ids
        calls.append("reduce")
        return sentinel

    monkeypatch.setattr(
        "campaigns.dogbone_v1.s13_screen._screen_metrics_from_cases",
        reduce_cases,
    )

    metrics = evaluate_dogbone_s13_train_screen(
        protocol=protocol,
        source=source,
        training_data=training_data,
        model=_model(),
        trial=protocol.search_space.trials[0],
    )

    assert metrics is sentinel
    assert calls == ["iter_train_screen", "replay_train_collocation", "reduce"]
    assert not (tmp_path / "development-access.json").exists()
