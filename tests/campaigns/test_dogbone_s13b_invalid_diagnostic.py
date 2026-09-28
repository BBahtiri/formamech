"""Locate the exact TRAIN collocation points behind S13b's invalid count."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from campaigns.dogbone_v1 import s13b_invalid_diagnostic as diagnostic
from campaigns.dogbone_v1.s13b_protocol import build_dogbone_s13b_pilot_plan
from piginot.datasets import SemanticBCClass
from piginot.doe import EvidenceRole
from piginot.physics._validation import DETERMINANT_MARGIN
from tests.campaigns.test_dogbone_s13_objective import _model
from tests.campaigns.test_dogbone_s13_screen import _case


def test_locator_reports_all_cases_and_every_invalid_interior_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _case("fit-001")
    screen = _case("screen-001")
    seen: list[str] = []

    def determinants(model: object, case: object) -> torch.Tensor:
        case_id = case.tensors.case_id  # type: ignore[attr-defined]
        seen.append(case_id)
        values = [1.0] * 8
        if case_id == "fit-001":
            values[0] = DETERMINANT_MARGIN
            values[3] = -0.2
        else:
            values[5] = 0.1
        return torch.tensor(values, dtype=torch.float64)

    monkeypatch.setattr(diagnostic, "_determinants_for_case", determinants)
    model = _model()
    model.train()

    cases = diagnostic._locate_invalid_points(
        model,
        fit_cases=(fit,),
        screen_cases=(screen,),
        expected_fit_ids=("fit-001",),
        expected_screen_ids=("screen-001",),
        expected_interior_count=8,
    )

    interior = tuple(
        (index, point)
        for index, point in enumerate(fit.collocation.points)
        if point.semantic_class is SemanticBCClass.INTERIOR
    )
    assert seen == ["fit-001", "screen-001"]
    assert model.training
    assert len(cases) == 2
    assert cases[0].population == "TRAIN_FIT"
    assert cases[1].population == "TRAIN_SCREEN"
    assert cases[0].min_determinant == -0.2
    assert cases[1].min_determinant == 0.1
    assert len(cases[0].invalid_points) == 2
    assert cases[1].invalid_points == ()
    assert cases[0].invalid_points[0].point_index == interior[0][0]
    assert cases[0].invalid_points[0].coordinates == interior[0][1].coordinates
    assert cases[0].invalid_points[0].determinant == DETERMINANT_MARGIN
    assert cases[0].invalid_points[1].point_index == interior[3][0]
    assert cases[0].invalid_points[1].determinant == -0.2


@pytest.mark.parametrize("values", ([1.0] * 7, [float("nan")] * 8))
def test_locator_rejects_missing_or_nonfinite_determinants(
    monkeypatch: pytest.MonkeyPatch, values: list[float]
) -> None:
    case = _case("fit-001")
    screen = _case("screen-001")
    monkeypatch.setattr(
        diagnostic,
        "_determinants_for_case",
        lambda *args: torch.tensor(values, dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="interior|finite"):
        diagnostic._locate_invalid_points(
            _model(),
            fit_cases=(case,),
            screen_cases=(screen,),
            expected_fit_ids=("fit-001",),
            expected_screen_ids=("screen-001",),
            expected_interior_count=8,
        )


def test_diagnostic_validates_checkpoint_before_train_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pilot = build_dogbone_s13b_pilot_plan()
    s10 = tmp_path / "s10"
    s11 = tmp_path / "s11"
    run = tmp_path / "peak"
    for path in (s10, s11, run):
        path.mkdir()
    checkpoint = run / "checkpoint-step-0800"
    checkpoint.mkdir()
    model = _model()
    calls: list[str] = []
    monkeypatch.setattr(
        diagnostic,
        "load_selected_s11_initialization",
        lambda path: SimpleNamespace(model=model),
    )

    def binding(*args: object, **kwargs: object) -> object:
        assert kwargs["implementation_revision"] == (
            "65b091f136d72697698c1209427e96e0668a43e6"
        )
        return SimpleNamespace(run_hash="1" * 64)

    monkeypatch.setattr(diagnostic, "_build_s13b_binding", binding)

    def checked(*args: object, **kwargs: object) -> object:
        calls.append("checkpoint")
        assert kwargs["expected_manifest_hash"] is None
        return SimpleNamespace(manifest_hash="2" * 64)

    monkeypatch.setattr(diagnostic, "_load_checked_checkpoint", checked)
    source = SimpleNamespace(
        development_opened=False,
        bindings=SimpleNamespace(content_hash=pilot.base_protocol.source_bindings_hash),
    )

    def open_source(path: Path, *, development_access_record: Path) -> object:
        assert calls == ["checkpoint"]
        assert development_access_record == run / "development-access.json"
        calls.append("source")
        return source

    monkeypatch.setattr(diagnostic, "open_selective_s10_source", open_source)
    data = SimpleNamespace(cases=(object(),), normalization=object())
    monkeypatch.setattr(
        diagnostic, "materialize_dogbone_s13_training_data", lambda *args: data
    )
    monkeypatch.setattr(diagnostic, "_verify_training_data", lambda *args: None)
    monkeypatch.setattr(diagnostic, "_screen_cases", lambda *args: (object(),))
    monkeypatch.setattr(
        diagnostic, "_locate_invalid_points", lambda *args, **kwargs: ()
    )

    report = diagnostic.locate_s13b_invalid_collocation(
        s10_run_root=s10,
        s11_checkpoint_root=s11,
        peak_checkpoint_path=checkpoint,
    )

    assert calls == ["checkpoint", "source"]
    assert report.role is EvidenceRole.TRAIN
    assert report.checkpoint_manifest_hash == "2" * 64
    assert report.invalid_point_count == 0
    assert source.development_opened is False


def test_cli_emits_train_only_non_gate_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = diagnostic.InvalidCollocationReport(
        role=EvidenceRole.TRAIN,
        checkpoint_manifest_hash="a" * 64,
        determinant_margin=DETERMINANT_MARGIN,
        interior_points_per_case=128,
        cases=(
            diagnostic.CaseDeterminants("valid", "TRAIN_FIT", 0.2, ()),
            diagnostic.CaseDeterminants(
                "invalid",
                "TRAIN_SCREEN",
                -0.01,
                (diagnostic.InvalidCollocationPoint(4, (1.0, 2.0), -0.01),),
            ),
        ),
    )
    monkeypatch.setattr(
        diagnostic, "locate_s13b_invalid_collocation", lambda **kwargs: report
    )

    result = diagnostic.main(
        (
            "--s10-run-root",
            str(tmp_path / "s10"),
            "--s11-checkpoint-root",
            str(tmp_path / "s11"),
            "--peak-checkpoint-path",
            str(tmp_path / "checkpoint-step-0800"),
        )
    )

    assert result == 0
    output = capsys.readouterr().out
    assert '"role":"TRAIN"' in output
    assert '"non_gate":true' in output
    assert '"checked_case_count":2' in output
    assert '"invalid_point_count":1' in output
    assert '"min_determinant":-0.01' in output
    assert '"case_id":"valid"' not in output
