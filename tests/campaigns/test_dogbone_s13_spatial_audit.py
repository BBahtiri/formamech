from __future__ import annotations

import math
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from campaigns.dogbone_v1 import s13_spatial_audit as audit
from campaigns.dogbone_v1.s13_overfit_diagnostic import (
    _build_vertex_sampler_plan,
    _predict_case,
    build_s13_overfit_plan,
)
from piginot.contracts import canonical_json_bytes
from piginot.doe import EvidenceRole
from tests.campaigns.test_dogbone_s13_objective import _model
from tests.campaigns.test_dogbone_s13_screen import _case

summarize_spatial_case = audit.summarize_spatial_case


def test_spatial_summary_separates_root_and_nonroot_error() -> None:
    target = torch.zeros((4, 2, 2), dtype=torch.float64)
    target[:, 0, 0] = torch.tensor((1.0, 2.0, 1.0, 1.0))
    predicted = target.clone()
    predicted[0, 0, 0] = 1.5
    predicted[3, 0, 0] = 3.0
    coordinates = torch.tensor(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)),
        dtype=torch.float64,
    )
    constitutive = torch.ones_like(target)
    constitutive[[1, 3]] = 2.0

    result = summarize_spatial_case(
        predicted,
        target,
        torch.tensor((1, 3), dtype=torch.long),
        coordinates,
        constitutive,
        torch.tensor((0.8, 0.7, 0.9, 0.6), dtype=torch.float64),
    )

    assert result.root_vertex_count == 2
    assert result.nonroot_vertex_count == 2
    assert result.root_p_relative_l2 == pytest.approx(2 / math.sqrt(5))
    assert result.nonroot_p_relative_l2 == pytest.approx(0.5 / math.sqrt(2))
    assert result.root_squared_p_error_share == pytest.approx(16 / 17)
    assert result.root_constitutive_rms == pytest.approx(2.0)
    assert result.nonroot_constitutive_rms == pytest.approx(1.0)
    assert result.fem_peak_vertex_index == 1
    assert result.predicted_peak_vertex_index == 3
    assert result.fem_peak_xy_mm == (1.0, 0.0)
    assert result.predicted_peak_xy_mm == (3.0, 0.0)
    assert result.fem_peak_p_frobenius == pytest.approx(2.0)
    assert result.predicted_peak_p_frobenius == pytest.approx(3.0)
    assert result.fem_peak_p_components == ((2.0, 0.0), (0.0, 0.0))
    assert result.predicted_peak_p_components == ((3.0, 0.0), (0.0, 0.0))
    assert result.root_min_determinant == pytest.approx(0.6)
    assert result.nonroot_min_determinant == pytest.approx(0.8)


def test_spatial_summary_rejects_invalid_or_misaligned_inputs() -> None:
    field = torch.ones((4, 2, 2), dtype=torch.float64)
    coordinates = torch.zeros((4, 2), dtype=torch.float64)
    determinants = torch.ones(4, dtype=torch.float64)
    qoi = torch.tensor((1, 3), dtype=torch.long)

    with pytest.raises(ValueError, match="sorted and unique"):
        summarize_spatial_case(
            field, field, torch.tensor((3, 1)), coordinates, field, determinants
        )
    with pytest.raises(ValueError, match="positive"):
        summarize_spatial_case(
            field,
            field,
            qoi,
            coordinates,
            field,
            torch.tensor((1.0, 1.0, 0.0, 1.0), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="aligned"):
        summarize_spatial_case(field[:3], field, qoi, coordinates, field, determinants)


def test_pinned_report_rejects_tampering_and_nontrain_role(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report = {"role": "TRAIN", "non_gate": True, "case_id": "candidate-002485"}
    raw = canonical_json_bytes(report) + b"\n"
    path.write_bytes(raw)

    assert audit._load_pinned_report(path, sha256(raw).hexdigest()) == report  # noqa: SLF001
    with pytest.raises(ValueError, match="SHA-256"):
        audit._load_pinned_report(path, "0" * 64)  # noqa: SLF001
    bad = canonical_json_bytes({**report, "role": "DEVELOPMENT"}) + b"\n"
    path.write_bytes(bad)
    with pytest.raises(ValueError, match="TRAIN"):
        audit._load_pinned_report(path, sha256(bad).hexdigest())  # noqa: SLF001


def test_vertex_physics_uses_model_displacement_gradient() -> None:
    case = _case("candidate-002485")
    model = _model()
    model.train()

    predicted, constitutive, determinants = (  # noqa: SLF001
        audit._vertex_prediction_and_physics(model, case)
    )

    torch.testing.assert_close(predicted, _predict_case(model, case).P[0])
    assert constitutive.shape == predicted.shape == case.tensors.target_P.shape
    assert torch.isfinite(constitutive).all()
    assert determinants.shape == (len(case.tensors.query_xy),)
    assert float(determinants.min()) > 0.0
    assert model.training is True


@pytest.mark.parametrize("tampered_trace", (False, True))
def test_spatial_runner_verifies_every_replay_before_reporting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tampered_trace: bool
) -> None:
    s10, s11, artifacts, audit_dir = (
        tmp_path / name for name in ("s10", "s11", "artifacts", "audit")
    )
    for path in (s10, s11, artifacts, audit_dir):
        path.mkdir()
    base = _case("candidate-002485")
    tensors = replace(
        base.tensors,
        query_xy=torch.zeros((969, 2), dtype=torch.float64),
        target_u=torch.ones((969, 2), dtype=torch.float64),
        target_P=torch.ones((969, 2, 2), dtype=torch.float64),
        qoi_indices=torch.arange(20, dtype=torch.long),
    )
    case = replace(base, tensors=tensors)
    source = SimpleNamespace(
        development_opened=False,
        bindings=SimpleNamespace(content_hash="a" * 64),
    )
    monkeypatch.setattr(audit, "open_selective_s10_source", lambda *args, **kw: source)
    monkeypatch.setattr(
        audit, "materialize_dogbone_s13_training_case", lambda *args: case
    )
    monkeypatch.setattr(
        audit,
        "load_selected_s11_initialization",
        lambda *args: SimpleNamespace(
            model=_model(),
            evidence_file_blake2b256="b" * 64,
            checkpoint_file_blake2b256="c" * 64,
        ),
    )
    monkeypatch.setattr(audit, "model_state_hash", lambda *args: "d" * 64)
    monkeypatch.setattr(
        audit,
        "build_s13_overfit_plan",
        lambda *, version: build_s13_overfit_plan(version=version, optimizer_steps=1),
    )
    trained: list[str] = []

    def fake_train(
        model: object, selected: object, plan: object, sampler: object
    ) -> tuple[float, ...]:
        trained.append(plan.diagnostic_id[-2:])  # type: ignore[attr-defined]
        return (0.1,)

    monkeypatch.setattr(audit, "_train_one_case", fake_train)
    monkeypatch.setattr(
        audit,
        "_vertex_prediction_and_physics",
        lambda *args: (
            tensors.target_P * 1.01,
            torch.ones_like(tensors.target_P),
            torch.ones(969, dtype=torch.float64),
        ),
    )
    pinned: list[tuple[str, str, str]] = []
    for version in ("v5", "v6", "v7"):
        plan = build_s13_overfit_plan(version=version, optimizer_steps=1)
        sampler = _build_vertex_sampler_plan(plan, 969)
        prior = {
            "role": "TRAIN",
            "non_gate": True,
            "case_id": "candidate-002485",
            "plan_hash": plan.content_hash,
            "vertex_sampler_plan_hash": sampler.plan_hash,
            "source_bindings_hash": "a" * 64,
            "s11_evidence_file_hash": "b" * 64,
            "initial_checkpoint_file_hash": "c" * 64,
            "initial_model_state_hash": "d" * 64,
            "final_model_state_hash": "d" * 64,
            "optimizer_loss_trace": [
                0.2 if tampered_trace and version == "v6" else 0.1
            ],
        }
        name = f"{version}.json"
        raw = canonical_json_bytes(prior) + b"\n"
        (artifacts / name).write_bytes(raw)
        pinned.append((version, name, sha256(raw).hexdigest()))
    monkeypatch.setattr(audit, "_FROZEN_REPORTS", tuple(pinned))

    arguments = dict(
        s10_run_root=s10,
        s11_checkpoint_root=s11,
        development_access_record=audit_dir / "development-access.json",
        artifact_root=artifacts,
        output_path=audit_dir / "spatial.json",
    )
    if tampered_trace:
        with pytest.raises(ValueError, match="v6 deterministic training replay"):
            audit.run_s13_spatial_audit(**arguments)
        assert trained == ["v5", "v6"]
        assert not (audit_dir / "spatial.json").exists()
        return
    report = audit.run_s13_spatial_audit(**arguments)

    assert report.role is EvidenceRole.TRAIN
    assert report.non_gate is True
    assert tuple(arm.version for arm in report.arms) == ("v5", "v6", "v7")
    assert trained == ["v5", "v6", "v7"]
    assert all(arm.metrics.root_vertex_count == 20 for arm in report.arms)
    assert not (audit_dir / "development-access.json").exists()


def test_cli_writes_canonical_audit_exclusively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = audit.S13SpatialAuditReport(
        role=EvidenceRole.TRAIN,
        non_gate=True,
        case_id="candidate-002485",
        source_bindings_hash="a" * 64,
        arms=(),
    )
    monkeypatch.setattr(audit, "run_s13_spatial_audit", lambda **kwargs: report)
    output = tmp_path / "audit.json"
    args = (
        "--s10-run-root",
        str(tmp_path / "s10"),
        "--s11-checkpoint-root",
        str(tmp_path / "s11"),
        "--development-access-record",
        str(tmp_path / "access.json"),
        "--artifact-root",
        str(tmp_path),
        "--output",
        str(output),
    )

    assert audit.main(args) == 0
    assert output.read_bytes() == canonical_json_bytes(report) + b"\n"
    with pytest.raises(FileExistsError):
        audit.main(args)
