from __future__ import annotations

import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import campaigns.dogbone_v1.s11_contingency_execute as execute_module
from campaigns.dogbone_v1.s11_contingency import (
    BASELINE_CANDIDATE_ID,
    DERIVED_PARAMETER_NAMES,
    RAW_PARAMETER_NAMES,
    ArchitectureScreenMetrics,
    ArchitectureScreenProtocol,
    build_architecture_screen_protocol,
)
from campaigns.dogbone_v1.s11_contingency_data import (
    CaseNormalizationStatistics,
)
from campaigns.dogbone_v1.s11_contingency_run import TrainScreenDecision
from campaigns.dogbone_v1.s11_contingency_source import (
    S10SourceBindings,
    SelectiveS10Source,
    _FixtureSelectiveS10Source,
)
from campaigns.dogbone_v1.s11_contingency_training import (
    ArchitectureCaseTensors,
    CandidateTrainingSummary,
)
from piginot.doe import EvidenceRole, PublishedCaseManifest, PublishedCaseStatus

_DTYPE = torch.float64
_REVISION = "1" * 40
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
    normalization_hash: str,
    *,
    role: EvidenceRole,
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
    target_p = torch.zeros((6, 2, 2), dtype=_DTYPE)
    target_p[:, 0, 0] = torch.linspace(0.5, 1.0, 6, dtype=_DTYPE)
    target_p[:, 1, 1] = 0.25
    return ArchitectureCaseTensors(
        case_id=case_id,
        leakage_group_id=f"group-{case_id}",
        role=role,
        status=PublishedCaseStatus.ACTIVE,
        parameter_names=(*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES),
        normalized_parameters=torch.linspace(-1.0, 1.0, 11, dtype=_DTYPE),
        normalization_hash=normalization_hash,
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
        target_u=0.1 + query_xy,
        target_P=target_p,
        qoi_indices=torch.tensor([4, 5], dtype=torch.int64),
    )


def _screen_metrics(
    protocol: ArchitectureScreenProtocol,
    *,
    eligible: bool,
) -> dict[str, ArchitectureScreenMetrics]:
    result: dict[str, ArchitectureScreenMetrics] = {}
    for index, candidate in enumerate(protocol.candidates):
        if index == 0:
            signed, absolute, p90 = -0.20, 0.20, 0.40
        elif eligible and index == 2:
            signed, absolute, p90 = -0.10, 0.15, 0.35
        else:
            signed, absolute, p90 = -0.19, 0.19, 0.41
        result[candidate.candidate_id] = ArchitectureScreenMetrics(
            candidate.candidate_id,
            signed,
            absolute,
            p90,
            0.10,
            0.20,
            400_000 + index,
            1.0,
        )
    return result


def _development_metrics(
    protocol: ArchitectureScreenProtocol,
    *,
    selected_passes: bool,
) -> dict[str, ArchitectureScreenMetrics]:
    selected_id = protocol.candidates[2].candidate_id
    return {
        BASELINE_CANDIDATE_ID: ArchitectureScreenMetrics(
            BASELINE_CANDIDATE_ID,
            -0.20,
            0.20,
            0.40,
            0.10,
            0.20,
            400_000,
            1.0,
        ),
        selected_id: ArchitectureScreenMetrics(
            selected_id,
            -0.10 if selected_passes else -0.19,
            0.15 if selected_passes else 0.19,
            0.35 if selected_passes else 0.41,
            0.101,
            0.201,
            400_002,
            1.0,
        ),
    }


@dataclass
class _Counters:
    fit_reads: int = 0
    screen_reads: int = 0
    development_reads: int = 0
    fit_stream_calls: int = 0
    screen_stream_calls: int = 0
    development_stream_calls: int = 0
    training_calls: int = 0
    screen_evaluations: int = 0
    development_evaluations: int = 0
    decision_seen: TrainScreenDecision | None = None


def _install_harness(
    monkeypatch: pytest.MonkeyPatch,
    protocol: ArchitectureScreenProtocol,
    *,
    train_winner: bool,
    development_selected_passes: bool,
    mutate_during_screen: bool = False,
) -> tuple[SelectiveS10Source, _Counters]:
    bindings = S10SourceBindings(
        report_content_hash="1" * 64,
        public_case_manifest_hash=protocol.public_case_manifest_hash,
        request_plan_hash="2" * 64,
        execution_profile_hash="3" * 64,
        ledger_content_hash="4" * 64,
        active_case_count=344,
        successful_case_count=344,
    )
    source = object.__new__(SelectiveS10Source)
    object.__setattr__(source, "_bindings", bindings)
    object.__setattr__(source, "_development_opened", False)
    object.__setattr__(
        source,
        "_development_access_record",
        Path(tempfile.mkdtemp()) / "development-access.json",
    )
    object.__setattr__(source, "_protocol", protocol)

    statistics = CaseNormalizationStatistics(
        parameter_names=(*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES),
        mean=(0.0,) * 11,
        standard_deviation=(1.0,) * 11,
        fit_case_ids=protocol.fit_case_ids,
        training_snapshot_hash=protocol.training_snapshot_hash,
    )
    populations = {
        "fit": tuple(
            _case(case_id, statistics.content_hash, role=EvidenceRole.TRAIN)
            for case_id in protocol.fit_case_ids
        ),
        "screen": tuple(
            _case(case_id, statistics.content_hash, role=EvidenceRole.TRAIN)
            for case_id in protocol.screen_case_ids
        ),
        "development": tuple(
            _case(case_id, statistics.content_hash, role=EvidenceRole.DEVELOPMENT)
            for case_id in protocol.development_case_ids
        ),
    }
    counters = _Counters()

    def _verified_stream(
        cases: tuple[ArchitectureCaseTensors, ...],
        counter_name: str,
    ) -> Iterator[SimpleNamespace]:
        for case in cases:
            setattr(counters, counter_name, getattr(counters, counter_name) + 1)
            yield SimpleNamespace(case=case, request=None, result=None)

    def iter_fit(self):  # type: ignore[no-untyped-def]
        assert self is source
        counters.fit_stream_calls += 1
        assert counters.screen_reads == 0
        assert counters.development_reads == 0
        return _verified_stream(populations["fit"], "fit_reads")

    def iter_train_screen(self):  # type: ignore[no-untyped-def]
        assert self is source
        counters.screen_stream_calls += 1
        assert counters.training_calls == len(protocol.candidates)
        assert counters.development_reads == 0
        return _verified_stream(populations["screen"], "screen_reads")

    def open_development(self, decision):  # type: ignore[no-untyped-def]
        assert self is source
        counters.development_stream_calls += 1
        assert type(decision) is TrainScreenDecision
        assert counters.screen_reads == len(protocol.screen_case_ids)
        assert counters.screen_evaluations == len(protocol.candidates)
        counters.decision_seen = decision
        return _verified_stream(populations["development"], "development_reads")

    monkeypatch.setattr(SelectiveS10Source, "iter_fit", iter_fit)
    monkeypatch.setattr(SelectiveS10Source, "iter_train_screen", iter_train_screen)
    monkeypatch.setattr(SelectiveS10Source, "_open_development", open_development)
    monkeypatch.setattr(
        execute_module,
        "prepare_operator_case",
        lambda case, request, result: case,
    )

    def fit_normalization(cases, *, fit_case_ids, training_snapshot_hash):  # type: ignore[no-untyped-def]
        assert tuple(case.case_id for case in cases) == protocol.fit_case_ids
        assert fit_case_ids == protocol.fit_case_ids
        assert training_snapshot_hash == protocol.training_snapshot_hash
        assert counters.fit_reads == len(protocol.fit_case_ids)
        assert counters.screen_reads == 0
        return statistics

    monkeypatch.setattr(execute_module, "fit_case_normalization", fit_normalization)
    monkeypatch.setattr(
        execute_module,
        "normalize_case_values",
        lambda case, observed: case if observed is statistics else None,
    )

    def train_candidate(
        protocol_arg, candidate, model, fit_cases, schedule, *, progress=None
    ):  # type: ignore[no-untyped-def]
        assert protocol_arg is protocol
        assert tuple(case.case_id for case in fit_cases) == protocol.fit_case_ids
        assert schedule.optimizer_steps == protocol.optimizer_steps
        assert counters.screen_reads == 0
        assert counters.development_reads == 0
        counters.training_calls += 1
        if progress is not None:
            progress(1, 1.0)
            progress(protocol.optimizer_steps, 0.3)
        with torch.no_grad():
            next(model.parameters()).add_(counters.training_calls * 1.0e-9)
        return CandidateTrainingSummary(
            candidate_id=candidate.candidate_id,
            optimizer_steps=protocol.optimizer_steps,
            schedule_hash=schedule.content_hash,
            final_total_loss=1.0,
            final_u_mse=0.3,
            final_P_mse=0.4,
            final_peak_P_mse=0.3,
        )

    train_metrics = _screen_metrics(protocol, eligible=train_winner)
    development_metrics = _development_metrics(
        protocol,
        selected_passes=development_selected_passes,
    )

    def evaluate_screen(protocol_arg, candidate, model, cases, *, elapsed_seconds):  # type: ignore[no-untyped-def]
        assert protocol_arg is protocol
        assert tuple(case.case_id for case in cases) == protocol.screen_case_ids
        assert counters.screen_reads == len(protocol.screen_case_ids)
        assert counters.development_reads == 0
        counters.screen_evaluations += 1
        if mutate_during_screen and counters.screen_evaluations == 1:
            with torch.no_grad():
                next(model.parameters()).add_(1.0)
        return train_metrics[candidate.candidate_id]

    def evaluate_development(protocol_arg, candidate, model, cases, *, elapsed_seconds):  # type: ignore[no-untyped-def]
        assert protocol_arg is protocol
        assert tuple(case.case_id for case in cases) == protocol.development_case_ids
        assert counters.development_reads == len(protocol.development_case_ids)
        counters.development_evaluations += 1
        return development_metrics[candidate.candidate_id]

    monkeypatch.setattr(execute_module, "train_candidate_model", train_candidate)
    monkeypatch.setattr(
        execute_module,
        "evaluate_train_screen_candidate",
        evaluate_screen,
    )
    monkeypatch.setattr(
        execute_module,
        "evaluate_development_candidate",
        evaluate_development,
    )
    return source, counters


def test_orchestrator_opens_development_once_only_after_train_decision(
    monkeypatch: pytest.MonkeyPatch,
    protocol: ArchitectureScreenProtocol,
) -> None:
    source, counters = _install_harness(
        monkeypatch,
        protocol,
        train_winner=True,
        development_selected_passes=True,
    )

    events: list[str] = []
    result = execute_module.execute_architecture_contingency(
        protocol,
        source,
        implementation_revision=_REVISION,
        progress=events.append,
    )

    assert counters.fit_stream_calls == 1
    assert counters.fit_reads == 192
    assert counters.training_calls == 4
    assert counters.screen_stream_calls == 1
    assert counters.screen_reads == 64
    assert counters.screen_evaluations == 4
    assert counters.development_stream_calls == 1
    assert counters.development_reads == 64
    assert counters.development_evaluations == 2
    assert counters.decision_seen is result.train_screen_decision
    assert result.train_screen_decision.selection.selected_candidate_id == (
        protocol.candidates[2].candidate_id
    )
    assert result.development_confirmation is not None
    assert result.development_confirmation.final_candidate_id == (
        protocol.candidates[2].candidate_id
    )
    assert not result.development_confirmation.fallback_to_baseline
    assert result.training_schedule.optimizer_steps == 1_500
    assert result.training_schedule.content_hash == (
        result.train_screen_decision.training_schedule_hash
    )
    assert result.normalization.content_hash == (
        result.train_screen_decision.normalization_hash
    )
    assert all(
        evidence.initial_state_hash != evidence.final_state_hash
        for evidence in result.train_screen_decision.model_states
    )
    assert events[0] == "phase=TRAIN_FIT event=load_start"
    assert events[-1].startswith("phase=DEVELOPMENT event=complete")
    assert sum("event=train_start" in event for event in events) == 4
    assert sum("step=1500" in event for event in events) == 4
    assert events.index("phase=TRAIN_SCREEN event=load_start") > max(
        index for index, event in enumerate(events) if "event=train_complete" in event
    )
    _assert_result_contract_rejects_drift(result)


def _assert_result_contract_rejects_drift(
    result: execute_module.ContingencyExecutionResult,
) -> None:
    invalid_types = (
        ("normalization", object(), "normalization"),
        ("training_schedule", object(), "training_schedule"),
        ("train_screen_decision", object(), "train_screen_decision"),
        ("training_summaries", [], "training_summaries"),
        ("trained_models", [], "trained_models"),
        ("development_confirmation", object(), "development_confirmation"),
    )
    for field_name, value, message in invalid_types:
        with pytest.raises(TypeError, match=message):
            replace(result, **{field_name: value})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="candidate order"):
        replace(result, training_summaries=tuple(reversed(result.training_summaries)))
    with pytest.raises(ValueError, match="model population"):
        replace(result, trained_models=result.trained_models[:-1])
    wrong_summary = replace(result.training_summaries[0], schedule_hash="f" * 64)
    with pytest.raises(ValueError, match="summary schedule"):
        replace(
            result,
            training_summaries=(wrong_summary, *result.training_summaries[1:]),
        )
    wrong_decision = replace(
        result.train_screen_decision,
        training_schedule_hash="f" * 64,
    )
    with pytest.raises(ValueError, match="result identities"):
        replace(result, train_screen_decision=wrong_decision)
    with pytest.raises(ValueError, match="confirmation does not match"):
        replace(result, development_confirmation=None)
    assert result.development_confirmation is not None
    wrong_confirmation = replace(
        result.development_confirmation,
        train_screen_decision_hash="f" * 64,
    )
    with pytest.raises(ValueError, match="another decision"):
        replace(result, development_confirmation=wrong_confirmation)
    with torch.no_grad():
        next(result.trained_models[0].parameters()).add_(1.0)
    with pytest.raises(ValueError, match="model state differs"):
        replace(result)


def test_development_confirmation_contract_rejects_identity_drift(
    protocol: ArchitectureScreenProtocol,
) -> None:
    metrics = _development_metrics(protocol, selected_passes=True)
    baseline = metrics[BASELINE_CANDIDATE_ID]
    selected = metrics[protocol.candidates[2].candidate_id]
    confirmation = execute_module.DevelopmentConfirmation(
        train_screen_decision_hash="a" * 64,
        baseline_metrics=baseline,
        selected_metrics=selected,
        final_candidate_id=selected.candidate_id,
        fallback_to_baseline=False,
        development_case_count=64,
    )

    invalid_changes = (
        ({"baseline_metrics": object()}, TypeError, "baseline_metrics"),
        ({"selected_metrics": object()}, TypeError, "selected_metrics"),
        ({"baseline_metrics": selected}, ValueError, "baseline candidate"),
        ({"selected_metrics": baseline}, ValueError, "non-baseline"),
        ({"fallback_to_baseline": 1}, TypeError, "boolean"),
        ({"final_candidate_id": BASELINE_CANDIDATE_ID}, ValueError, "disagree"),
        ({"development_case_count": 63}, ValueError, "exactly 64"),
    )
    for changes, error, message in invalid_changes:
        with pytest.raises(error, match=message):
            replace(confirmation, **changes)  # type: ignore[arg-type]


def test_entry_validation_fails_before_any_label_access(
    tmp_path: Path,
    protocol: ArchitectureScreenProtocol,
) -> None:
    bindings = S10SourceBindings(
        report_content_hash="1" * 64,
        public_case_manifest_hash=protocol.public_case_manifest_hash,
        request_plan_hash="2" * 64,
        execution_profile_hash="3" * 64,
        ledger_content_hash="4" * 64,
        active_case_count=344,
        successful_case_count=344,
    )

    def source_with(**values: object) -> SelectiveS10Source:
        source = object.__new__(SelectiveS10Source)
        object.__setattr__(source, "_bindings", values.get("bindings", bindings))
        object.__setattr__(
            source,
            "_development_opened",
            values.get("development_opened", False),
        )
        object.__setattr__(source, "_protocol", values.get("protocol", protocol))
        object.__setattr__(
            source,
            "_development_access_record",
            values.get("access_record", tmp_path / "absent-development-access.json"),
        )
        return source

    with pytest.raises(ValueError, match="Git revision"):
        execute_module.execute_architecture_contingency(
            protocol,
            source_with(),
            implementation_revision="not-a-revision",
        )
    with pytest.raises(RuntimeError, match="already opened"):
        execute_module.execute_architecture_contingency(
            protocol,
            source_with(development_opened=True),
            implementation_revision=_REVISION,
        )
    access_record = tmp_path / "development-opened.json"
    access_record.write_text("sealed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already opened"):
        execute_module.execute_architecture_contingency(
            protocol,
            source_with(access_record=access_record),
            implementation_revision=_REVISION,
        )
    with pytest.raises(TypeError, match="source bindings"):
        execute_module.execute_architecture_contingency(
            protocol,
            source_with(bindings=object()),
            implementation_revision=_REVISION,
        )
    other_bindings = replace(bindings, public_case_manifest_hash="f" * 64)
    with pytest.raises(ValueError, match="public manifests"):
        execute_module.execute_architecture_contingency(
            protocol,
            source_with(bindings=other_bindings),
            implementation_revision=_REVISION,
        )
    with pytest.raises(ValueError, match="protocols differ"):
        execute_module.execute_architecture_contingency(
            protocol,
            source_with(protocol=None),
            implementation_revision=_REVISION,
        )


def test_train_fallback_keeps_development_closed(
    monkeypatch: pytest.MonkeyPatch,
    protocol: ArchitectureScreenProtocol,
) -> None:
    source, counters = _install_harness(
        monkeypatch,
        protocol,
        train_winner=False,
        development_selected_passes=False,
    )

    result = execute_module.execute_architecture_contingency(
        protocol,
        source,
        implementation_revision=_REVISION,
    )

    assert result.train_screen_decision.selection.fallback_to_baseline
    assert result.development_confirmation is None
    assert result.final_candidate_id == BASELINE_CANDIDATE_ID
    assert counters.development_stream_calls == 0
    assert counters.development_reads == 0
    assert counters.development_evaluations == 0


def test_development_uses_same_guardrails_and_falls_back_to_baseline(
    monkeypatch: pytest.MonkeyPatch,
    protocol: ArchitectureScreenProtocol,
) -> None:
    source, counters = _install_harness(
        monkeypatch,
        protocol,
        train_winner=True,
        development_selected_passes=False,
    )

    result = execute_module.execute_architecture_contingency(
        protocol,
        source,
        implementation_revision=_REVISION,
    )

    assert result.development_confirmation is not None
    assert result.development_confirmation.fallback_to_baseline
    assert result.development_confirmation.final_candidate_id == BASELINE_CANDIDATE_ID
    assert result.final_candidate_id == BASELINE_CANDIDATE_ID
    assert counters.development_reads == 64


def test_screen_state_mutation_aborts_before_development(
    monkeypatch: pytest.MonkeyPatch,
    protocol: ArchitectureScreenProtocol,
) -> None:
    source, counters = _install_harness(
        monkeypatch,
        protocol,
        train_winner=True,
        development_selected_passes=True,
        mutate_during_screen=True,
    )

    with pytest.raises(RuntimeError, match="changed after training"):
        execute_module.execute_architecture_contingency(
            protocol,
            source,
            implementation_revision=_REVISION,
        )

    assert counters.development_stream_calls == 0
    assert counters.development_reads == 0


def test_orchestrator_refuses_nonproduction_source_before_any_access(
    protocol: ArchitectureScreenProtocol,
) -> None:
    fixture_source = object.__new__(_FixtureSelectiveS10Source)

    with pytest.raises(TypeError, match="exact production SelectiveS10Source"):
        execute_module.execute_architecture_contingency(
            protocol,
            fixture_source,  # type: ignore[arg-type]
            implementation_revision=_REVISION,
        )
