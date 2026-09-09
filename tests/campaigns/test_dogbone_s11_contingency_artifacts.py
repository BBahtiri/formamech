from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from campaigns.dogbone_v1.s11_contingency import (
    BASELINE_CANDIDATE_ID,
    ArchitectureScreenMetrics,
    ArchitectureScreenProtocol,
)
from campaigns.dogbone_v1.s11_contingency_artifacts import (
    CONTINGENCY_EVIDENCE_NAME,
    DEVELOPMENT_ACCESS_NAME,
    RUN_RESERVATION_NAME,
    ModelFileEvidence,
    OutputReservation,
    TrainingScheduleEvidence,
    persist_contingency_result,
    reserve_output_root,
)
from campaigns.dogbone_v1.s11_contingency_data import (
    CaseNormalizationStatistics,
)
from campaigns.dogbone_v1.s11_contingency_execute import (
    ContingencyExecutionResult,
    DevelopmentConfirmation,
)
from campaigns.dogbone_v1.s11_contingency_models import (
    build_candidate_models,
    model_state_hash,
)
from campaigns.dogbone_v1.s11_contingency_run import (
    CandidateModelStateEvidence,
    complete_train_screen,
    load_committed_protocol,
)
from campaigns.dogbone_v1.s11_contingency_source import S10SourceBindings
from campaigns.dogbone_v1.s11_contingency_tensors import (
    CandidateTrainingSummary,
    TrainingSchedule,
)
from piginot.contracts import canonical_json_bytes, strict_json_loads

_REVISION = "1" * 40


@pytest.fixture(scope="module")
def protocol() -> ArchitectureScreenProtocol:
    return load_committed_protocol()


@pytest.fixture()
def execution(
    protocol: ArchitectureScreenProtocol,
) -> tuple[S10SourceBindings, ContingencyExecutionResult]:
    models = build_candidate_models(
        protocol,
        boundary_tag_names=protocol.boundary_tag_names,
    )
    normalization = CaseNormalizationStatistics(
        parameter_names=(*protocol.raw_parameter_names, *protocol.derived_parameter_names),
        mean=(0.0,) * 11,
        standard_deviation=(1.0,) * 11,
        fit_case_ids=protocol.fit_case_ids,
        training_snapshot_hash=protocol.training_snapshot_hash,
    )
    schedule = TrainingSchedule(
        protocol_id=protocol.protocol_id,
        sampler_seed=protocol.sampler_seed,
        case_ids=protocol.fit_case_ids,
        case_indices=torch.zeros(
            (protocol.optimizer_steps, protocol.case_batch_size),
            dtype=torch.int64,
        ),
        query_indices=torch.zeros(
            (
                protocol.optimizer_steps,
                protocol.case_batch_size,
                protocol.query_points_per_case,
            ),
            dtype=torch.int64,
        ),
        qoi_query_count=64,
    )
    summaries = tuple(
        CandidateTrainingSummary(
            candidate.candidate_id,
            protocol.optimizer_steps,
            schedule.content_hash,
            1.0,
            0.3,
            0.4,
            0.3,
        )
        for candidate in protocol.candidates
    )
    metrics = tuple(
        ArchitectureScreenMetrics(
            candidate.candidate_id,
            -0.20 if index == 0 else -0.19,
            0.20 if index == 0 else 0.19,
            0.40 if index == 0 else 0.41,
            0.10,
            0.20,
            sum(parameter.numel() for parameter in model.parameters()),
            1.0,
        )
        for index, (candidate, model) in enumerate(
            zip(protocol.candidates, models, strict=True)
        )
    )
    bindings = S10SourceBindings(
        report_content_hash="2" * 64,
        public_case_manifest_hash=protocol.public_case_manifest_hash,
        request_plan_hash="3" * 64,
        execution_profile_hash="4" * 64,
        ledger_content_hash="5" * 64,
        active_case_count=344,
        successful_case_count=344,
    )
    states = tuple(
        CandidateModelStateEvidence(
            candidate.candidate_id,
            "0" * 64,
            model_state_hash(model),
        )
        for candidate, model in zip(protocol.candidates, models, strict=True)
    )
    decision = complete_train_screen(
        protocol,
        metrics,
        source_bindings_hash=bindings.content_hash,
        training_schedule_hash=schedule.content_hash,
        normalization_hash=normalization.content_hash,
        implementation_revision=_REVISION,
        model_states=states,
    )
    assert decision.selection.fallback_to_baseline
    return bindings, ContingencyExecutionResult(
        normalization=normalization,
        training_schedule=schedule,
        training_summaries=summaries,
        train_screen_decision=decision,
        development_confirmation=None,
        trained_models=models,
    )


def test_persistence_writes_only_safetensors_and_canonical_bound_evidence(
    tmp_path: Path,
    protocol: ArchitectureScreenProtocol,
    execution: tuple[S10SourceBindings, ContingencyExecutionResult],
) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    bindings, result = execution
    output_root = tmp_path / "run"
    reservation = reserve_output_root(
        output_root,
        protocol_content_hash=protocol.content_hash,
        implementation_revision=_REVISION,
    )

    evidence = persist_contingency_result(
        reservation,
        protocol=protocol,
        source_bindings=bindings,
        result=result,
    )

    evidence_path = output_root / CONTINGENCY_EVIDENCE_NAME
    assert evidence_path.read_bytes() == evidence.to_bytes()
    assert strict_json_loads(evidence_path.read_bytes())["final_candidate_id"] == (
        result.final_candidate_id
    )
    assert evidence.protocol_content_hash == protocol.content_hash
    assert evidence.source_bindings == bindings
    assert evidence.training_schedule.schedule_hash == (
        result.training_schedule.content_hash
    )
    assert evidence.normalization == result.normalization
    assert evidence.training_summaries == result.training_summaries
    assert evidence.train_screen_decision == result.train_screen_decision
    assert evidence.development_confirmation is None
    assert evidence.implementation_revision == _REVISION
    assert len(evidence.model_files) == len(protocol.candidates)

    for model_file, model in zip(evidence.model_files, result.trained_models, strict=True):
        path = output_root / model_file.file_name
        assert path.suffix == ".safetensors"
        assert path.read_bytes()[:2] != b"\x80\x04"
        assert model_file.byte_count == path.stat().st_size
        assert model_file.file_blake2b256 == hashlib.blake2b(
            path.read_bytes(), digest_size=32
        ).hexdigest()
        assert model_file.model_state_hash == model_state_hash(model)
        loaded = safetensors.load_file(str(path), device="cpu")
        assert set(loaded) == set(model.state_dict())
        assert all(torch.equal(loaded[name], value) for name, value in model.state_dict().items())

    assert set(path.name for path in output_root.iterdir()) == {
        RUN_RESERVATION_NAME,
        CONTINGENCY_EVIDENCE_NAME,
        *(item.file_name for item in evidence.model_files),
    }


def test_persistence_refuses_model_drift_before_writing_weights(
    tmp_path: Path,
    protocol: ArchitectureScreenProtocol,
    execution: tuple[S10SourceBindings, ContingencyExecutionResult],
) -> None:
    bindings, result = execution
    reservation = reserve_output_root(
        tmp_path / "run",
        protocol_content_hash=protocol.content_hash,
        implementation_revision=_REVISION,
    )
    with torch.no_grad():
        next(result.trained_models[0].parameters()).add_(1.0)

    with pytest.raises(ValueError, match="model state"):
        persist_contingency_result(
            reservation,
            protocol=protocol,
            source_bindings=bindings,
            result=result,
        )

    assert [path.name for path in reservation.output_root.iterdir()] == [
        RUN_RESERVATION_NAME
    ]


def test_output_is_create_once_and_cannot_be_resumed_or_overwritten(
    tmp_path: Path,
    protocol: ArchitectureScreenProtocol,
    execution: tuple[S10SourceBindings, ContingencyExecutionResult],
) -> None:
    bindings, result = execution
    output_root = tmp_path / "run"
    reservation = reserve_output_root(
        output_root,
        protocol_content_hash=protocol.content_hash,
        implementation_revision=_REVISION,
    )
    evidence = persist_contingency_result(
        reservation,
        protocol=protocol,
        source_bindings=bindings,
        result=result,
    )
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output_root.iterdir()
    }

    with pytest.raises(FileExistsError, match="empty|already"):
        reserve_output_root(
            output_root,
            protocol_content_hash=protocol.content_hash,
            implementation_revision=_REVISION,
        )
    with pytest.raises(FileExistsError, match="already"):
        persist_contingency_result(
            reservation,
            protocol=protocol,
            source_bindings=bindings,
            result=result,
        )

    assert evidence.to_bytes() == canonical_json_bytes(evidence)
    assert before == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output_root.iterdir()
    }


def test_reservation_rejects_relative_nonempty_and_revision_drift(
    tmp_path: Path,
    protocol: ArchitectureScreenProtocol,
) -> None:
    with pytest.raises(ValueError, match="absolute"):
        reserve_output_root(
            Path("relative"),
            protocol_content_hash=protocol.content_hash,
            implementation_revision=_REVISION,
        )
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "foreign.txt").write_text("foreign", encoding="utf-8")
    with pytest.raises(FileExistsError, match="empty"):
        reserve_output_root(
            nonempty,
            protocol_content_hash=protocol.content_hash,
            implementation_revision=_REVISION,
        )
    with pytest.raises(ValueError, match="revision"):
        reserve_output_root(
            tmp_path / "bad-revision",
            protocol_content_hash=protocol.content_hash,
            implementation_revision="not-a-revision",
        )


def test_schedule_and_model_file_evidence_reject_invalid_canonical_shapes(
    protocol: ArchitectureScreenProtocol,
    execution: tuple[S10SourceBindings, ContingencyExecutionResult],
) -> None:
    _, result = execution
    schedule = TrainingScheduleEvidence.from_schedule(result.training_schedule)
    assert schedule.schedule_hash == result.training_schedule.content_hash

    with pytest.raises(ValueError, match="sorted and unique"):
        replace(schedule, case_ids=("case-b", "case-a"))
    with pytest.raises(ValueError, match="smaller"):
        replace(schedule, qoi_query_count=schedule.query_points_per_case)
    with pytest.raises(ValueError, match="optimizer_steps"):
        replace(schedule, optimizer_steps=0)
    with pytest.raises(TypeError, match="TrainingSchedule"):
        TrainingScheduleEvidence.from_schedule(object())  # type: ignore[arg-type]

    candidate = protocol.candidates[0]
    model_file = ModelFileEvidence(
        candidate_id=candidate.candidate_id,
        file_name=f"{candidate.candidate_id}.safetensors",
        byte_count=1,
        file_blake2b256="a" * 64,
        operator_module_digest=candidate.operator_module_digest,
        initial_state_hash="b" * 64,
        model_state_hash="c" * 64,
    )
    assert model_file.file_name.endswith(".safetensors")
    with pytest.raises(ValueError, match="safe candidate"):
        replace(model_file, candidate_id="../escape", file_name="../escape.safetensors")
    with pytest.raises(ValueError, match="byte_count"):
        replace(model_file, byte_count=0)
    with pytest.raises(ValueError, match="operator_module_digest"):
        replace(model_file, operator_module_digest="sha256:" + "d" * 64)


def test_nonfallback_persistence_requires_and_preserves_exact_development_marker(
    tmp_path: Path,
    protocol: ArchitectureScreenProtocol,
    execution: tuple[S10SourceBindings, ContingencyExecutionResult],
) -> None:
    bindings, fallback_result = execution
    baseline = fallback_result.train_screen_decision.metrics[0]
    selected = replace(
        fallback_result.train_screen_decision.metrics[1],
        median_signed_peak_error=-0.15,
        median_absolute_peak_error=0.17,
        p90_absolute_peak_error=0.39,
    )
    metrics = (
        baseline,
        selected,
        *fallback_result.train_screen_decision.metrics[2:],
    )
    decision = complete_train_screen(
        protocol,
        metrics,
        source_bindings_hash=bindings.content_hash,
        training_schedule_hash=fallback_result.training_schedule.content_hash,
        normalization_hash=fallback_result.normalization.content_hash,
        implementation_revision=_REVISION,
        model_states=fallback_result.train_screen_decision.model_states,
    )
    assert not decision.selection.fallback_to_baseline
    confirmation = DevelopmentConfirmation(
        train_screen_decision_hash=decision.content_hash,
        baseline_metrics=baseline,
        selected_metrics=selected,
        final_candidate_id=selected.candidate_id,
        fallback_to_baseline=False,
        development_case_count=64,
    )
    result = replace(
        fallback_result,
        train_screen_decision=decision,
        development_confirmation=confirmation,
    )
    output_root = tmp_path / "run"
    reservation = reserve_output_root(
        output_root,
        protocol_content_hash=protocol.content_hash,
        implementation_revision=_REVISION,
    )
    marker = output_root / "DEVELOPMENT-ACCESS.json"
    marker.write_bytes(decision.to_bytes() + b"\n")

    evidence = persist_contingency_result(
        reservation,
        protocol=protocol,
        source_bindings=bindings,
        result=result,
    )

    assert marker.read_bytes() == decision.to_bytes() + b"\n"
    assert evidence.development_confirmation == confirmation
    assert set(path.name for path in output_root.iterdir()) == {
        RUN_RESERVATION_NAME,
        "DEVELOPMENT-ACCESS.json",
        CONTINGENCY_EVIDENCE_NAME,
        *(item.file_name for item in evidence.model_files),
    }
    with pytest.raises(ValueError, match="requires DEVELOPMENT"):
        replace(evidence, development_confirmation=None)
    with pytest.raises(ValueError, match="final candidate"):
        replace(evidence, final_candidate_id=BASELINE_CANDIDATE_ID)
    with pytest.raises(ValueError, match="DEVELOPMENT evidence"):
        replace(
            evidence,
            development_confirmation=replace(
                confirmation,
                train_screen_decision_hash="f" * 64,
            ),
        )


def test_nonfallback_persistence_refuses_changed_development_marker(
    tmp_path: Path,
    protocol: ArchitectureScreenProtocol,
    execution: tuple[S10SourceBindings, ContingencyExecutionResult],
) -> None:
    bindings, result = execution
    reservation = reserve_output_root(
        tmp_path / "run",
        protocol_content_hash=protocol.content_hash,
        implementation_revision=_REVISION,
    )
    (reservation.output_root / "DEVELOPMENT-ACCESS.json").write_bytes(b"changed\n")

    with pytest.raises((FileExistsError, RuntimeError), match="DEVELOPMENT|foreign"):
        persist_contingency_result(
            reservation,
            protocol=protocol,
            source_bindings=bindings,
            result=result,
        )


def test_evidence_rejects_cross_binding_and_candidate_state_tampering(
    tmp_path: Path,
    protocol: ArchitectureScreenProtocol,
    execution: tuple[S10SourceBindings, ContingencyExecutionResult],
) -> None:
    bindings, result = execution
    evidence = persist_contingency_result(
        reserve_output_root(
            tmp_path / "run",
            protocol_content_hash=protocol.content_hash,
            implementation_revision=_REVISION,
        ),
        protocol=protocol,
        source_bindings=bindings,
        result=result,
    )

    with pytest.raises(ValueError, match="protocol"):
        replace(evidence, protocol_content_hash="f" * 64)
    with pytest.raises(ValueError, match="source"):
        replace(
            evidence,
            source_bindings=replace(
                evidence.source_bindings,
                report_content_hash="f" * 64,
            ),
        )
    with pytest.raises(ValueError, match="normalization"):
        replace(
            evidence,
            normalization=replace(
                evidence.normalization,
                mean=(1.0, *evidence.normalization.mean[1:]),
            ),
        )
    with pytest.raises(ValueError, match="revision"):
        replace(evidence, implementation_revision="2" * 40)
    with pytest.raises(ValueError, match="final candidate"):
        replace(evidence, final_candidate_id=evidence.model_files[1].candidate_id)
    with pytest.raises(ValueError, match="schedule"):
        replace(
            evidence,
            training_summaries=(
                replace(evidence.training_summaries[0], schedule_hash="e" * 64),
                *evidence.training_summaries[1:],
            ),
        )
    with pytest.raises(ValueError, match="final state"):
        replace(
            evidence,
            model_files=(
                replace(evidence.model_files[0], model_state_hash="c" * 64),
                *evidence.model_files[1:],
            ),
        )
    with pytest.raises(ValueError, match="model.*state|initial"):
        replace(
            evidence,
            model_files=(
                replace(evidence.model_files[0], initial_state_hash="d" * 64),
                *evidence.model_files[1:],
            ),
        )
    with pytest.raises(ValueError, match="source"):
        replace(
            evidence,
            source_bindings=replace(bindings, report_content_hash="c" * 64),
        )
    with pytest.raises(ValueError, match="normalization"):
        replace(
            evidence,
            normalization=replace(
                evidence.normalization,
                mean=(0.1, *evidence.normalization.mean[1:]),
            ),
        )
    with pytest.raises(ValueError, match="final state"):
        replace(
            evidence,
            model_files=(
                replace(evidence.model_files[0], model_state_hash="b" * 64),
                *evidence.model_files[1:],
            ),
        )


def test_artifact_leaf_contracts_fail_closed(
    tmp_path: Path,
    protocol: ArchitectureScreenProtocol,
) -> None:
    schedule = TrainingSchedule(
        protocol_id=protocol.protocol_id,
        sampler_seed=protocol.sampler_seed,
        case_ids=protocol.fit_case_ids,
        case_indices=torch.zeros((1, 1), dtype=torch.int64),
        query_indices=torch.zeros((1, 1, 4), dtype=torch.int64),
        qoi_query_count=1,
    )
    schedule_evidence = TrainingScheduleEvidence.from_schedule(schedule)

    with pytest.raises(TypeError, match="OutputReservation"):
        OutputReservation(
            output_root=tmp_path / "run",
            protocol_content_hash=protocol.content_hash,
            implementation_revision=_REVISION,
            _token=object(),
        )
    with pytest.raises(TypeError, match="schedule"):
        TrainingScheduleEvidence.from_schedule(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="query batch"):
        replace(schedule_evidence, qoi_query_count=4)
    with pytest.raises(ValueError, match="sampler_seed"):
        replace(schedule_evidence, sampler_seed=-1)
    with pytest.raises(ValueError, match="safe candidate"):
        ModelFileEvidence(
            candidate_id="../bad",
            file_name="../bad.safetensors",
            byte_count=1,
            file_blake2b256="a" * 64,
            operator_module_digest="blake2b256:" + "a" * 64,
            initial_state_hash="a" * 64,
            model_state_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="operator_module_digest"):
        ModelFileEvidence(
            candidate_id="candidate",
            file_name="candidate.safetensors",
            byte_count=1,
            file_blake2b256="a" * 64,
            operator_module_digest="sha256:" + "a" * 64,
            initial_state_hash="a" * 64,
            model_state_hash="a" * 64,
        )


def test_nonfallback_requires_exact_development_marker_before_persistence(
    tmp_path: Path,
    protocol: ArchitectureScreenProtocol,
    execution: tuple[S10SourceBindings, ContingencyExecutionResult],
) -> None:
    bindings, fallback_result = execution
    baseline = fallback_result.train_screen_decision.metrics[0]
    selected = replace(
        fallback_result.train_screen_decision.metrics[1],
        median_signed_peak_error=-0.15,
        median_absolute_peak_error=0.17,
        p90_absolute_peak_error=0.39,
    )
    decision = complete_train_screen(
        protocol,
        (baseline, selected, *fallback_result.train_screen_decision.metrics[2:]),
        source_bindings_hash=bindings.content_hash,
        training_schedule_hash=fallback_result.training_schedule.content_hash,
        normalization_hash=fallback_result.normalization.content_hash,
        implementation_revision=_REVISION,
        model_states=fallback_result.train_screen_decision.model_states,
    )
    result = replace(
        fallback_result,
        train_screen_decision=decision,
        development_confirmation=DevelopmentConfirmation(
            train_screen_decision_hash=decision.content_hash,
            baseline_metrics=baseline,
            selected_metrics=selected,
            final_candidate_id=selected.candidate_id,
            fallback_to_baseline=False,
            development_case_count=64,
        ),
    )
    reservation = reserve_output_root(
        tmp_path / "run",
        protocol_content_hash=protocol.content_hash,
        implementation_revision=_REVISION,
    )

    with pytest.raises(RuntimeError, match="DEVELOPMENT"):
        persist_contingency_result(
            reservation,
            protocol=protocol,
            source_bindings=bindings,
            result=result,
        )

    (reservation.output_root / DEVELOPMENT_ACCESS_NAME).write_bytes(b"wrong\n")
    with pytest.raises(RuntimeError, match="DEVELOPMENT"):
        persist_contingency_result(
            reservation,
            protocol=protocol,
            source_bindings=bindings,
            result=result,
        )
