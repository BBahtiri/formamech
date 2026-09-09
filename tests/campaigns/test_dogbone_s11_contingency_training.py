from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch

import campaigns.dogbone_v1.s11_contingency_training as training_module
from campaigns.dogbone_v1.s11_contingency import (
    DERIVED_PARAMETER_NAMES,
    RAW_PARAMETER_NAMES,
    ArchitectureScreenProtocol,
    build_architecture_screen_protocol,
)
from campaigns.dogbone_v1.s11_contingency_models import build_candidate_models
from campaigns.dogbone_v1.s11_contingency_training import (
    ArchitectureCaseTensors,
    CandidateTrainingSummary,
    TrainingSchedule,
    _train_candidate_prefix_for_test,
    architecture_screen_loss,
    build_training_schedule,
    calculate_architecture_screen_metrics,
    evaluate_development_candidate,
    evaluate_train_screen_candidate,
    train_candidate_model,
)
from piginot.doe import EvidenceRole, PublishedCaseManifest, PublishedCaseStatus
from piginot.operators import OperatorOutput

_DTYPE = torch.float64
_TAGS = ("bottom", "gauge_top", "left_symmetry", "right_arc", "right_grip")


@pytest.fixture(scope="module")
def protocol() -> ArchitectureScreenProtocol:
    root = Path(__file__).resolve().parents[2]
    manifest = PublishedCaseManifest.from_bytes(
        (
            root
            / "artifacts"
            / "campaigns"
            / "dogbone-v1"
            / "public-case-manifest.json"
        ).read_bytes()
    )
    return build_architecture_screen_protocol(manifest)


def _case(
    case_id: str,
    *,
    scale: float = 1.0,
    role: EvidenceRole = EvidenceRole.TRAIN,
) -> ArchitectureCaseTensors:
    boundary_xy = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=_DTYPE,
    )
    boundary_tags = torch.zeros((4, len(_TAGS)), dtype=_DTYPE)
    boundary_tags[torch.arange(4), torch.arange(4)] = 1.0
    query_xy = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
        dtype=_DTYPE,
    )
    target_u = scale * (0.1 + query_xy)
    target_p = torch.zeros((6, 2, 2), dtype=_DTYPE)
    target_p[:, 0, 0] = scale * torch.linspace(0.5, 1.0, 6, dtype=_DTYPE)
    target_p[:, 1, 1] = scale * 0.25
    return ArchitectureCaseTensors(
        case_id=case_id,
        leakage_group_id=f"group-{case_id}",
        role=role,
        status=PublishedCaseStatus.ACTIVE,
        parameter_names=(*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES),
        normalized_parameters=torch.linspace(-1.0, 1.0, 11, dtype=_DTYPE),
        normalization_hash="1" * 64,
        boundary_tag_names=_TAGS,
        boundary_xy=boundary_xy,
        boundary_tags=boundary_tags,
        boundary_normals=torch.tensor(
            [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=_DTYPE,
        ),
        boundary_weights=torch.full((4,), 0.25, dtype=_DTYPE),
        boundary_mask=torch.ones(4, dtype=torch.bool),
        query_xy=query_xy,
        target_u=target_u,
        target_P=target_p,
        qoi_indices=torch.tensor([4, 5], dtype=torch.int64),
    )


def _population(
    case_ids: tuple[str, ...],
    *,
    role: EvidenceRole = EvidenceRole.TRAIN,
) -> tuple[ArchitectureCaseTensors, ...]:
    return tuple(
        _case(case_id, scale=1.0 + index / 1000.0, role=role)
        for index, case_id in enumerate(case_ids)
    )


def test_case_contract_is_frozen_owned_float64_and_public_evaluation_only() -> None:
    source = torch.linspace(-1.0, 1.0, 11, dtype=_DTYPE)
    case = _case("train-case")
    owned = ArchitectureCaseTensors(
        case_id="owned-case",
        leakage_group_id="owned-group",
        role=EvidenceRole.TRAIN,
        status=PublishedCaseStatus.ACTIVE,
        parameter_names=(*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES),
        normalized_parameters=source,
        normalization_hash="2" * 64,
        boundary_tag_names=_TAGS,
        boundary_xy=case.boundary_xy,
        boundary_tags=case.boundary_tags,
        boundary_normals=case.boundary_normals,
        boundary_weights=case.boundary_weights,
        boundary_mask=case.boundary_mask,
        query_xy=case.query_xy,
        target_u=case.target_u,
        target_P=case.target_P,
        qoi_indices=case.qoi_indices,
    )
    source[0] = 99.0

    assert owned.normalized_parameters[0].item() == -1.0
    assert owned.target_P.shape == (6, 2, 2)
    assert owned.qoi_indices.tolist() == [4, 5]
    assert owned.parameters_for(RAW_PARAMETER_NAMES).shape == (7,)
    assert owned.parameters_for((*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES)).shape == (
        11,
    )
    with pytest.raises(ValueError, match="not registered"):
        owned.parameters_for(tuple(reversed(RAW_PARAMETER_NAMES)))
    with pytest.raises(FrozenInstanceError):
        owned.case_id = "changed"  # type: ignore[misc]
    assert replace(owned, role=EvidenceRole.DEVELOPMENT).role is EvidenceRole.DEVELOPMENT
    with pytest.raises(ValueError, match="TRAIN or DEVELOPMENT"):
        replace(owned, role=EvidenceRole.PILOT)
    with pytest.raises(ValueError, match="ACTIVE"):
        replace(owned, status=PublishedCaseStatus.RESERVE)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"normalized_parameters": torch.ones(11)}, "float64"),
        ({"target_u": torch.full((6, 2), float("nan"), dtype=_DTYPE)}, "finite"),
        ({"qoi_indices": torch.tensor([4, 4])}, "unique"),
        ({"qoi_indices": torch.tensor([6])}, "bounds"),
        ({"boundary_mask": torch.zeros(4, dtype=torch.bool)}, "visible"),
        ({"parameter_names": (*RAW_PARAMETER_NAMES, RAW_PARAMETER_NAMES[0])}, "unique"),
        ({"parameter_names": RAW_PARAMETER_NAMES}, "raw-plus-derived"),
        ({"normalization_hash": "not-a-hash"}, "256-bit"),
        (
            {"normalized_parameters": torch.ones((11, 1), dtype=_DTYPE)},
            "align",
        ),
        ({"boundary_xy": torch.ones((4, 3), dtype=_DTYPE)}, "boundary_xy"),
        ({"boundary_xy": torch.empty((0, 2), dtype=_DTYPE)}, "non-empty"),
        ({"boundary_tags": torch.ones((4, 4), dtype=_DTYPE)}, "boundary_tags"),
        ({"boundary_normals": torch.ones((4, 3), dtype=_DTYPE)}, "normals"),
        ({"boundary_weights": torch.ones(3, dtype=_DTYPE)}, "weights"),
        ({"boundary_mask": torch.ones(3, dtype=torch.bool)}, "mask"),
        (
            {"boundary_weights": torch.tensor([0.25, -0.1, 0.25, 0.25], dtype=_DTYPE)},
            "non-negative",
        ),
        (
            {"boundary_weights": torch.tensor([0.25, 0.0, 0.25, 0.25], dtype=_DTYPE)},
            "positive weight",
        ),
        (
            {
                "boundary_tags": torch.tensor(
                    [
                        [0.5, 0.5, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                    ],
                    dtype=_DTYPE,
                )
            },
            "one-hot",
        ),
        (
            {
                "boundary_tags": torch.tensor(
                    [
                        [1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                    ],
                    dtype=_DTYPE,
                )
            },
            "exactly one",
        ),
        ({"boundary_normals": torch.ones((4, 2), dtype=_DTYPE)}, "unit"),
        (
            {
                "boundary_xy": torch.tensor(
                    [[0.0, 0.0], [0.0, 0.2], [0.0, 0.8], [0.0, 1.0]],
                    dtype=_DTYPE,
                )
            },
            "span",
        ),
        ({"query_xy": torch.ones((6, 3), dtype=_DTYPE)}, "query_xy"),
        ({"target_u": torch.ones((6, 3), dtype=_DTYPE)}, "target_u"),
        ({"target_P": torch.ones((6, 4), dtype=_DTYPE)}, "target_P"),
        ({"qoi_indices": torch.empty(0, dtype=torch.int64)}, "non-empty"),
        ({"qoi_indices": torch.tensor([4.0, 5.0])}, "int64"),
    ],
)
def test_case_contract_fails_closed(change: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        replace(_case("invalid"), **change)  # type: ignore[arg-type]


def test_schedule_is_reproducible_candidate_independent_and_qoi_stratified(
    protocol: ArchitectureScreenProtocol,
) -> None:
    cases = _population(protocol.fit_case_ids)

    left = build_training_schedule(protocol, cases)
    right = build_training_schedule(protocol, tuple(reversed(cases)))

    assert left.optimizer_steps == protocol.optimizer_steps == 1_500
    assert left.case_batch_size == protocol.case_batch_size == 4
    assert left.query_points_per_case == protocol.query_points_per_case == 256
    assert left.qoi_query_count == 64
    assert left.content_hash == right.content_hash
    assert torch.equal(left.case_indices, right.case_indices)
    assert torch.equal(left.query_indices, right.query_indices)
    for row, case_index in zip(
        left.query_indices[0], left.case_indices[0], strict=True
    ):
        case = cases[int(case_index)]
        qoi = set(case.qoi_indices.tolist())
        assert all(index not in qoi for index in row[:-64].tolist())
        assert all(index in qoi for index in row[-64:].tolist())


def test_schedule_refuses_nonregistered_cases_and_no_generic_pool(
    protocol: ArchitectureScreenProtocol,
) -> None:
    cases = _population(protocol.fit_case_ids)
    with pytest.raises(ValueError, match="fit_case_ids"):
        build_training_schedule(protocol, cases[:-1])
    all_qoi = replace(
        cases[0],
        qoi_indices=torch.arange(cases[0].query_xy.shape[0], dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="non-QoI"):
        build_training_schedule(protocol, (all_qoi, *cases[1:]))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"protocol_id": ""}, "protocol_id"),
        ({"sampler_seed": -1}, "sampler_seed"),
        ({"case_ids": ("b", "a")}, "sorted"),
        ({"case_ids": ("a", "a")}, "unique"),
        ({"case_indices": torch.tensor([0], dtype=torch.int64)}, "case_indices"),
        ({"case_indices": torch.tensor([[2, 1]], dtype=torch.int64)}, "exceed"),
        ({"query_indices": torch.tensor([[0]], dtype=torch.int64)}, "query_indices"),
        (
            {"query_indices": torch.tensor([[[-1], [0]]], dtype=torch.int64)},
            "non-negative",
        ),
        ({"qoi_query_count": 0}, "qoi_query_count"),
    ],
)
def test_schedule_contract_fails_closed(
    change: dict[str, object], message: str
) -> None:
    schedule = TrainingSchedule(
        protocol_id="protocol",
        sampler_seed=1,
        case_ids=("a", "b"),
        case_indices=torch.tensor([[0, 1]], dtype=torch.int64),
        query_indices=torch.tensor([[[0, 1], [0, 1]]], dtype=torch.int64),
        qoi_query_count=1,
    )
    with pytest.raises((TypeError, ValueError), match=message):
        replace(schedule, **change)  # type: ignore[arg-type]


def test_schedule_rejects_negative_nonfirst_case_index() -> None:
    with pytest.raises(ValueError, match="registered fit population"):
        TrainingSchedule(
            protocol_id="protocol",
            sampler_seed=1,
            case_ids=("a", "b"),
            case_indices=torch.tensor([[0, 1], [0, -1]], dtype=torch.int64),
            query_indices=torch.tensor(
                [[[0, 1], [0, 1]], [[0, 1], [0, 1]]], dtype=torch.int64
            ),
            qoi_query_count=1,
        )


def test_training_summary_fails_closed() -> None:
    valid = CandidateTrainingSummary(
        candidate_id="A-s11-baseline",
        optimizer_steps=1,
        schedule_hash="a" * 64,
        final_total_loss=1.0,
        final_u_mse=0.2,
        final_P_mse=0.3,
        final_peak_P_mse=0.5,
    )
    assert valid.final_total_loss == 1.0
    with pytest.raises(ValueError, match="optimizer_steps"):
        replace(valid, optimizer_steps=0)
    with pytest.raises(ValueError, match="schedule_hash"):
        replace(valid, schedule_hash="bad")
    with pytest.raises(ValueError, match="finite"):
        replace(valid, final_P_mse=float("inf"))
    with pytest.raises(ValueError, match="non-negative"):
        replace(valid, final_u_mse=-1.0)


def test_loss_is_exact_sum_of_u_p_and_root_peak_terms(
    protocol: ArchitectureScreenProtocol,
) -> None:
    output = OperatorOutput(
        u=torch.zeros((1, 3, 2), dtype=_DTYPE),
        P=torch.zeros((1, 3, 2, 2), dtype=_DTYPE),
    )
    target_u = torch.ones((1, 3, 2), dtype=_DTYPE)
    target_p = torch.ones((1, 3, 2, 2), dtype=_DTYPE)

    loss = architecture_screen_loss(
        protocol,
        output,
        target_u=target_u,
        target_P=target_p,
        qoi_query_count=1,
    )

    assert loss.u_mse.item() == pytest.approx(1.0)
    assert loss.P_mse.item() == pytest.approx(1.0)
    assert loss.peak_P_mse.item() == pytest.approx(4.0)
    assert loss.total.item() == pytest.approx(6.0)
    with pytest.raises(ValueError, match="qoi_query_count"):
        architecture_screen_loss(
            protocol,
            output,
            target_u=target_u,
            target_P=target_p,
            qoi_query_count=0,
        )


def test_loss_rejects_wrong_types_shapes_dtypes_and_nonfinite_values(
    protocol: ArchitectureScreenProtocol,
) -> None:
    output = OperatorOutput(
        u=torch.zeros((1, 3, 2), dtype=_DTYPE),
        P=torch.zeros((1, 3, 2, 2), dtype=_DTYPE),
    )
    target_u = torch.ones((1, 3, 2), dtype=_DTYPE)
    target_P = torch.ones((1, 3, 2, 2), dtype=_DTYPE)
    with pytest.raises(TypeError, match="OperatorOutput"):
        architecture_screen_loss(
            protocol,
            object(),  # type: ignore[arg-type]
            target_u=target_u,
            target_P=target_P,
            qoi_query_count=1,
        )
    with pytest.raises(ValueError, match="float64"):
        architecture_screen_loss(
            protocol,
            output,
            target_u=target_u.float(),
            target_P=target_P,
            qoi_query_count=1,
        )
    with pytest.raises(ValueError, match="batch shape"):
        architecture_screen_loss(
            protocol,
            output,
            target_u=target_u[:, :, :1],
            target_P=target_P,
            qoi_query_count=1,
        )
    with pytest.raises(ValueError, match="align"):
        architecture_screen_loss(
            protocol,
            output,
            target_u=torch.ones((1, 2, 2), dtype=_DTYPE),
            target_P=torch.ones((1, 2, 2, 2), dtype=_DTYPE),
            qoi_query_count=1,
        )
    nonfinite_output = OperatorOutput(output.u, output.P.clone())
    nonfinite_output.P[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        architecture_screen_loss(
            protocol,
            nonfinite_output,
            target_u=target_u,
            target_P=target_P,
            qoi_query_count=1,
        )


def test_metric_calculation_uses_relative_root_peaks_and_linear_quantiles() -> None:
    signed_errors = (-0.5, -0.25, 0.25, 1.0)
    cases: list[ArchitectureCaseTensors] = []
    predictions: list[OperatorOutput] = []
    for index, signed_error in enumerate(signed_errors):
        case = _case(f"metric-{index}", scale=2.0)
        target_p = torch.full_like(case.target_P, 2.0)
        target_u = torch.ones_like(case.target_u)
        case = replace(case, target_P=target_p, target_u=target_u)
        predictions.append(
            OperatorOutput(
                u=(1.0 + 0.1 * (index + 1)) * target_u.unsqueeze(0),
                P=(1.0 + signed_error) * target_p.unsqueeze(0),
            )
        )
        cases.append(case)

    metrics = calculate_architecture_screen_metrics(
        "A-s11-baseline",
        tuple(cases),
        tuple(predictions),
        trainable_parameter_count=123,
        elapsed_seconds=2.0,
    )

    assert metrics.median_signed_peak_error == pytest.approx(0.0)
    assert metrics.median_absolute_peak_error == pytest.approx(0.375)
    assert metrics.p90_absolute_peak_error == pytest.approx(0.85)
    assert metrics.median_u_relative_l2 == pytest.approx(0.25)
    assert metrics.median_p_relative_l2 == pytest.approx(0.375)
    assert metrics.trainable_parameter_count == 123
    assert metrics.elapsed_seconds == 2.0


def test_metric_calculation_rejects_incomplete_or_undefined_evidence() -> None:
    case = _case("metric-invalid")
    prediction = OperatorOutput(
        case.target_u.unsqueeze(0).clone(), case.target_P.unsqueeze(0).clone()
    )
    kwargs = {"trainable_parameter_count": 1, "elapsed_seconds": 0.0}
    with pytest.raises(ValueError, match="non-empty"):
        calculate_architecture_screen_metrics("candidate", (), (), **kwargs)
    with pytest.raises(ValueError, match="one-to-one"):
        calculate_architecture_screen_metrics("candidate", (case,), (), **kwargs)
    with pytest.raises(ValueError, match="align"):
        calculate_architecture_screen_metrics(
            "candidate",
            (case,),
            (
                OperatorOutput(
                    case.target_u[:-1].unsqueeze(0), case.target_P[:-1].unsqueeze(0)
                ),
            ),
            **kwargs,
        )
    nonfinite = OperatorOutput(prediction.u.clone(), prediction.P.clone())
    nonfinite.P[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        calculate_architecture_screen_metrics(
            "candidate", (case,), (nonfinite,), **kwargs
        )
    with pytest.raises(ValueError, match="positive norms"):
        calculate_architecture_screen_metrics(
            "candidate",
            (replace(case, target_u=torch.zeros_like(case.target_u)),),
            (prediction,),
            **kwargs,
        )
    zero_qoi_target = case.target_P.clone()
    zero_qoi_target[case.qoi_indices] = 0.0
    zero_qoi_case = replace(case, target_P=zero_qoi_target)
    with pytest.raises(ValueError, match="positive stress peak"):
        calculate_architecture_screen_metrics(
            "candidate",
            (zero_qoi_case,),
            (
                OperatorOutput(
                    zero_qoi_case.target_u.unsqueeze(0),
                    zero_qoi_case.target_P.unsqueeze(0),
                ),
            ),
            **kwargs,
        )


def test_training_prefix_and_train_screen_evaluation_are_bit_reproducible(
    protocol: ArchitectureScreenProtocol,
) -> None:
    fit_cases = _population(protocol.fit_case_ids)
    screen_cases = _population(protocol.screen_case_ids)
    schedule = build_training_schedule(protocol, fit_cases)
    left_model = build_candidate_models(protocol, boundary_tag_names=_TAGS)[0]
    right_model = build_candidate_models(protocol, boundary_tag_names=_TAGS)[0]
    candidate = protocol.candidates[0]

    progress: list[tuple[int, float]] = []
    left_summary = _train_candidate_prefix_for_test(
        protocol,
        candidate,
        left_model,
        fit_cases,
        schedule,
        optimizer_steps=2,
        progress=lambda step, loss: progress.append((step, loss)),
    )
    right_summary = _train_candidate_prefix_for_test(
        protocol,
        candidate,
        right_model,
        fit_cases,
        schedule,
        optimizer_steps=2,
    )

    assert left_summary == right_summary
    assert tuple(step for step, _ in progress) == (1, 2)
    assert all(math.isfinite(loss) for _, loss in progress)
    assert left_summary.optimizer_steps == 2
    assert left_summary.schedule_hash == schedule.content_hash
    for left, right in zip(
        left_model.state_dict().values(),
        right_model.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(left, right)
    left_metrics = evaluate_train_screen_candidate(
        protocol,
        candidate,
        left_model,
        screen_cases,
        elapsed_seconds=0.0,
    )
    right_metrics = evaluate_train_screen_candidate(
        protocol,
        candidate,
        right_model,
        screen_cases,
        elapsed_seconds=0.0,
    )
    assert left_metrics == right_metrics


def test_development_evaluation_preserves_role_and_exact_population(
    protocol: ArchitectureScreenProtocol,
) -> None:
    cases = _population(
        protocol.development_case_ids,
        role=EvidenceRole.DEVELOPMENT,
    )
    candidate = protocol.candidates[0]
    model = build_candidate_models(protocol, boundary_tag_names=_TAGS)[0]

    metrics = evaluate_development_candidate(
        protocol,
        candidate,
        model,
        cases,
        elapsed_seconds=0.0,
    )

    assert metrics.candidate_id == candidate.candidate_id
    with pytest.raises(ValueError, match="DEVELOPMENT"):
        evaluate_development_candidate(
            protocol,
            candidate,
            model,
            tuple(replace(case, role=EvidenceRole.TRAIN) for case in cases),
            elapsed_seconds=0.0,
        )
    with pytest.raises(ValueError, match="TRAIN"):
        build_training_schedule(
            protocol,
            _population(
                protocol.fit_case_ids,
                role=EvidenceRole.DEVELOPMENT,
            ),
        )


def test_production_training_has_no_step_override_and_rejects_short_schedule(
    protocol: ArchitectureScreenProtocol,
) -> None:
    fit_cases = _population(protocol.fit_case_ids)
    schedule = build_training_schedule(protocol, fit_cases)
    shortened = replace(
        schedule,
        case_indices=schedule.case_indices[:2],
        query_indices=schedule.query_indices[:2],
    )
    model = build_candidate_models(protocol, boundary_tag_names=_TAGS)[0]

    with pytest.raises(ValueError, match="1,500"):
        train_candidate_model(
            protocol,
            protocol.candidates[0],
            model,
            fit_cases,
            shortened,
        )


def test_production_training_requires_registered_fresh_full_state(
    protocol: ArchitectureScreenProtocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_cases = _population(protocol.fit_case_ids)
    schedule = build_training_schedule(protocol, fit_cases)
    candidate = protocol.candidates[0]
    fresh_model = build_candidate_models(protocol, boundary_tag_names=_TAGS)[0]
    expected = CandidateTrainingSummary(
        candidate_id=candidate.candidate_id,
        optimizer_steps=protocol.optimizer_steps,
        schedule_hash=schedule.content_hash,
        final_total_loss=1.0,
        final_u_mse=0.2,
        final_P_mse=0.3,
        final_peak_P_mse=0.5,
    )

    def fake_train_candidate(*args: object, **kwargs: object) -> CandidateTrainingSummary:
        return expected

    monkeypatch.setattr(training_module, "_train_candidate", fake_train_candidate)

    assert (
        train_candidate_model(
            protocol,
            candidate,
            fresh_model,
            fit_cases,
            schedule,
        )
        == expected
    )

    perturbed_model = build_candidate_models(protocol, boundary_tag_names=_TAGS)[0]
    with torch.no_grad():
        next(perturbed_model.parameters()).add_(1.0e-12)
    with pytest.raises(ValueError, match="registered fresh initialization"):
        train_candidate_model(
            protocol,
            candidate,
            perturbed_model,
            fit_cases,
            schedule,
        )
