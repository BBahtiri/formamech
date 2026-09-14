"""Fail-closed phase orchestrator for the preregistered S11 contingency."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar

from campaigns.dogbone_v1.s11_contingency import (
    BASELINE_CANDIDATE_ID,
    ArchitectureCandidate,
    ArchitectureScreenMetrics,
    ArchitectureScreenProtocol,
)
from campaigns.dogbone_v1.s11_contingency_data import (
    CaseNormalizationStatistics,
    OperatorCaseTensors,
    fit_case_normalization,
    normalize_case_values,
    prepare_operator_case,
)
from campaigns.dogbone_v1.s11_contingency_models import (
    build_candidate_models,
    model_state_hash,
)
from campaigns.dogbone_v1.s11_contingency_run import (
    CandidateModelStateEvidence,
    TrainScreenDecision,
    complete_train_screen,
    validate_train_screen_decision,
)
from campaigns.dogbone_v1.s11_contingency_source import (
    S10SourceBindings,
    SelectiveS10Source,
    VerifiedS10Case,
)
from campaigns.dogbone_v1.s11_contingency_training import (
    ArchitectureCaseTensors,
    CandidateTrainingSummary,
    TrainingSchedule,
    build_training_schedule,
    evaluate_development_candidate,
    evaluate_train_screen_candidate,
    train_candidate_model,
)
from piginot.doe import EvidenceRole
from piginot.operators import PiGinotOperator, module_digest

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
ExecutionProgress = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class DevelopmentConfirmation:
    """One-shot DEVELOPMENT comparison of the baseline and TRAIN winner."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    train_screen_decision_hash: str
    baseline_metrics: ArchitectureScreenMetrics
    selected_metrics: ArchitectureScreenMetrics
    final_candidate_id: str
    fallback_to_baseline: bool
    development_case_count: int

    def __post_init__(self) -> None:
        _digest(self.train_screen_decision_hash, "train_screen_decision_hash")
        if type(self.baseline_metrics) is not ArchitectureScreenMetrics:
            raise TypeError("baseline_metrics must be exact ArchitectureScreenMetrics")
        if type(self.selected_metrics) is not ArchitectureScreenMetrics:
            raise TypeError("selected_metrics must be exact ArchitectureScreenMetrics")
        if self.baseline_metrics.candidate_id != BASELINE_CANDIDATE_ID:
            raise ValueError("baseline_metrics must identify the baseline candidate")
        if self.selected_metrics.candidate_id == BASELINE_CANDIDATE_ID:
            raise ValueError(
                "selected_metrics must identify the non-baseline TRAIN winner"
            )
        if type(self.fallback_to_baseline) is not bool:
            raise TypeError("fallback_to_baseline must be boolean")
        expected_final = (
            BASELINE_CANDIDATE_ID
            if self.fallback_to_baseline
            else self.selected_metrics.candidate_id
        )
        if self.final_candidate_id != expected_final:
            raise ValueError("final_candidate_id and fallback_to_baseline disagree")
        if (
            type(self.development_case_count) is not int
            or self.development_case_count != 64
        ):
            raise ValueError("DEVELOPMENT confirmation requires exactly 64 cases")


@dataclass(frozen=True, slots=True, eq=False)
class ContingencyExecutionResult:
    """In-memory result; durable artifacts are written by the outer runner."""

    normalization: CaseNormalizationStatistics
    training_schedule: TrainingSchedule
    training_summaries: tuple[CandidateTrainingSummary, ...]
    train_screen_decision: TrainScreenDecision
    development_confirmation: DevelopmentConfirmation | None
    trained_models: tuple[PiGinotOperator, ...]

    def __post_init__(self) -> None:
        if type(self.normalization) is not CaseNormalizationStatistics:
            raise TypeError("normalization must be exact CaseNormalizationStatistics")
        if type(self.training_schedule) is not TrainingSchedule:
            raise TypeError("training_schedule must be exact TrainingSchedule")
        if type(self.train_screen_decision) is not TrainScreenDecision:
            raise TypeError("train_screen_decision must be exact TrainScreenDecision")
        if type(self.training_summaries) is not tuple or any(
            type(item) is not CandidateTrainingSummary
            for item in self.training_summaries
        ):
            raise TypeError("training_summaries must contain exact summaries")
        if type(self.trained_models) is not tuple or any(
            type(model) is not PiGinotOperator for model in self.trained_models
        ):
            raise TypeError("trained_models must contain exact PiGinotOperator values")
        if (
            self.development_confirmation is not None
            and type(self.development_confirmation) is not DevelopmentConfirmation
        ):
            raise TypeError(
                "development_confirmation must be exact DevelopmentConfirmation or None"
            )
        candidate_ids = tuple(
            evidence.candidate_id
            for evidence in self.train_screen_decision.model_states
        )
        if (
            tuple(item.candidate_id for item in self.training_summaries)
            != candidate_ids
        ):
            raise ValueError("training summaries and decision candidate order differ")
        if len(self.trained_models) != len(candidate_ids):
            raise ValueError("trained model population differs from the decision")
        if any(
            summary.schedule_hash != self.training_schedule.content_hash
            for summary in self.training_summaries
        ):
            raise ValueError("training summary schedule differs from the result")
        if (
            self.training_schedule.content_hash
            != self.train_screen_decision.training_schedule_hash
            or self.normalization.content_hash
            != self.train_screen_decision.normalization_hash
        ):
            raise ValueError("result identities differ from the TRAIN decision")
        if self.train_screen_decision.selection.fallback_to_baseline != (
            self.development_confirmation is None
        ):
            raise ValueError("DEVELOPMENT confirmation does not match TRAIN selection")
        if (
            self.development_confirmation is not None
            and self.development_confirmation.train_screen_decision_hash
            != self.train_screen_decision.content_hash
        ):
            raise ValueError("DEVELOPMENT confirmation is bound to another decision")
        for model, evidence in zip(
            self.trained_models,
            self.train_screen_decision.model_states,
            strict=True,
        ):
            if model_state_hash(model) != evidence.final_state_hash:
                raise ValueError("returned model state differs from the TRAIN decision")

    @property
    def final_candidate_id(self) -> str:
        if self.development_confirmation is None:
            return BASELINE_CANDIDATE_ID
        return self.development_confirmation.final_candidate_id


def execute_architecture_contingency(
    protocol: ArchitectureScreenProtocol,
    source: SelectiveS10Source,
    *,
    implementation_revision: str,
    progress: Callable[[str], None] | None = None,
) -> ContingencyExecutionResult:
    """Run TRAIN fit/screen and, only after selection, one DEVELOPMENT check."""

    _validate_entry(protocol, source, implementation_revision)
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    _emit_progress(progress, "phase=TRAIN_FIT event=load_start")
    fit_raw = _materialize_source_cases(
        source.iter_fit(),
        expected_case_ids=protocol.fit_case_ids,
        expected_role=EvidenceRole.TRAIN,
        label="fit",
    )
    _emit_progress(
        progress, f"phase=TRAIN_FIT event=load_complete cases={len(fit_raw)}"
    )
    normalization = fit_case_normalization(
        fit_raw,
        fit_case_ids=protocol.fit_case_ids,
        training_snapshot_hash=protocol.training_snapshot_hash,
    )
    if type(normalization) is not CaseNormalizationStatistics:
        raise TypeError("normalizer must return exact CaseNormalizationStatistics")
    if (
        normalization.fit_case_ids != protocol.fit_case_ids
        or normalization.training_snapshot_hash != protocol.training_snapshot_hash
    ):
        raise ValueError("normalization was not fitted on the registered TRAIN fit set")
    fit_cases = _normalize_population(
        fit_raw,
        normalization,
        expected_case_ids=protocol.fit_case_ids,
        expected_role=EvidenceRole.TRAIN,
        label="fit",
    )
    schedule = build_training_schedule(protocol, fit_cases)
    schedule_hash = schedule.content_hash
    models = build_candidate_models(
        protocol,
        boundary_tag_names=fit_cases[0].boundary_tag_names,
    )
    _validate_fresh_models(protocol, models)
    initial_hashes = tuple(model_state_hash(model) for model in models)

    summaries: list[CandidateTrainingSummary] = []
    elapsed: list[float] = []
    for candidate, model in zip(protocol.candidates, models, strict=True):
        _emit_progress(
            progress,
            "phase=TRAIN_FIT "
            f"candidate={candidate.candidate_id} event=train_start "
            f"steps={protocol.optimizer_steps}",
        )
        started = time.perf_counter()
        summary = train_candidate_model(
            protocol,
            candidate,
            model,
            fit_cases,
            schedule,
            progress=_candidate_progress_callback(
                progress,
                candidate.candidate_id,
            ),
        )
        elapsed.append(time.perf_counter() - started)
        _validate_training_summary(protocol, candidate, summary, schedule_hash)
        _require_schedule_unchanged(schedule, schedule_hash)
        summaries.append(summary)
        _emit_progress(
            progress,
            "phase=TRAIN_FIT "
            f"candidate={candidate.candidate_id} event=train_complete "
            f"loss={summary.final_total_loss:.17g}",
        )
    final_hashes = tuple(model_state_hash(model) for model in models)

    # This is the first point at which the frozen TRAIN-screen labels are read.
    _emit_progress(progress, "phase=TRAIN_SCREEN event=load_start")
    screen_raw = _materialize_source_cases(
        source.iter_train_screen(),
        expected_case_ids=protocol.screen_case_ids,
        expected_role=EvidenceRole.TRAIN,
        label="TRAIN screen",
    )
    _emit_progress(
        progress,
        f"phase=TRAIN_SCREEN event=load_complete cases={len(screen_raw)}",
    )
    screen_cases = _normalize_population(
        screen_raw,
        normalization,
        expected_case_ids=protocol.screen_case_ids,
        expected_role=EvidenceRole.TRAIN,
        label="TRAIN screen",
    )
    metrics: list[ArchitectureScreenMetrics] = []
    for index, (candidate, model) in enumerate(
        zip(protocol.candidates, models, strict=True)
    ):
        _require_model_unchanged(model, final_hashes[index])
        metric = evaluate_train_screen_candidate(
            protocol,
            candidate,
            model,
            screen_cases,
            elapsed_seconds=elapsed[index],
        )
        _validate_metric(candidate, metric)
        _require_model_unchanged(model, final_hashes[index])
        _require_schedule_unchanged(schedule, schedule_hash)
        metrics.append(metric)

    states = tuple(
        CandidateModelStateEvidence(
            candidate.candidate_id,
            initial_hashes[index],
            final_hashes[index],
        )
        for index, candidate in enumerate(protocol.candidates)
    )
    decision = complete_train_screen(
        protocol,
        tuple(metrics),
        source_bindings_hash=source.bindings.content_hash,
        training_schedule_hash=schedule_hash,
        normalization_hash=normalization.content_hash,
        implementation_revision=implementation_revision,
        model_states=states,
    )
    validate_train_screen_decision(protocol, decision)
    _emit_progress(
        progress,
        "phase=TRAIN_SCREEN event=decision_complete "
        f"selected={decision.selection.selected_candidate_id} "
        f"fallback={str(decision.selection.fallback_to_baseline).lower()}",
    )

    confirmation: DevelopmentConfirmation | None = None
    if not decision.selection.fallback_to_baseline:
        _emit_progress(progress, "phase=DEVELOPMENT event=load_start")
        confirmation = _confirm_on_development(
            protocol,
            source,
            decision,
            normalization,
            models,
            final_hashes,
            schedule,
            schedule_hash,
        )
        _emit_progress(
            progress,
            "phase=DEVELOPMENT event=complete "
            f"cases={confirmation.development_case_count} "
            f"selected={confirmation.final_candidate_id} "
            f"fallback={str(confirmation.fallback_to_baseline).lower()}",
        )
    else:
        _emit_progress(
            progress,
            "phase=DEVELOPMENT event=closed reason=train_fallback",
        )
    return ContingencyExecutionResult(
        normalization=normalization,
        training_schedule=schedule,
        training_summaries=tuple(summaries),
        train_screen_decision=decision,
        development_confirmation=confirmation,
        trained_models=models,
    )


def _confirm_on_development(
    protocol: ArchitectureScreenProtocol,
    source: SelectiveS10Source,
    decision: TrainScreenDecision,
    normalization: CaseNormalizationStatistics,
    models: tuple[PiGinotOperator, ...],
    final_hashes: tuple[str, ...],
    schedule: TrainingSchedule,
    schedule_hash: str,
) -> DevelopmentConfirmation:
    selected_id = decision.selection.selected_candidate_id
    candidate_by_id = {
        candidate.candidate_id: (index, candidate)
        for index, candidate in enumerate(protocol.candidates)
    }
    selected_index, selected_candidate = candidate_by_id[selected_id]
    development_raw = _materialize_source_cases(
        source._open_development(decision),
        expected_case_ids=protocol.development_case_ids,
        expected_role=EvidenceRole.DEVELOPMENT,
        label="DEVELOPMENT",
    )
    development_cases = _normalize_population(
        development_raw,
        normalization,
        expected_case_ids=protocol.development_case_ids,
        expected_role=EvidenceRole.DEVELOPMENT,
        label="DEVELOPMENT",
    )
    comparison = (
        (0, protocol.candidates[0]),
        (selected_index, selected_candidate),
    )
    development_metrics: list[ArchitectureScreenMetrics] = []
    for index, candidate in comparison:
        model = models[index]
        _require_model_unchanged(model, final_hashes[index])
        metric = evaluate_development_candidate(
            protocol,
            candidate,
            model,
            development_cases,
            elapsed_seconds=decision.metrics[index].elapsed_seconds,
        )
        _validate_metric(candidate, metric)
        _require_model_unchanged(model, final_hashes[index])
        _require_schedule_unchanged(schedule, schedule_hash)
        development_metrics.append(metric)
    baseline, selected = development_metrics
    selected_passes = _passes_registered_guardrails(protocol, baseline, selected)
    return DevelopmentConfirmation(
        train_screen_decision_hash=decision.content_hash,
        baseline_metrics=baseline,
        selected_metrics=selected,
        final_candidate_id=(selected_id if selected_passes else BASELINE_CANDIDATE_ID),
        fallback_to_baseline=not selected_passes,
        development_case_count=len(development_cases),
    )


def _validate_entry(
    protocol: ArchitectureScreenProtocol,
    source: SelectiveS10Source,
    implementation_revision: str,
) -> None:
    if type(protocol) is not ArchitectureScreenProtocol:
        raise TypeError("protocol must be an exact ArchitectureScreenProtocol")
    if type(source) is not SelectiveS10Source:
        raise TypeError("source must be an exact production SelectiveS10Source")
    if (
        not isinstance(implementation_revision, str)
        or _REVISION_PATTERN.fullmatch(implementation_revision) is None
    ):
        raise ValueError("implementation_revision must be a lowercase Git revision")
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels were already opened")
    access_record = getattr(source, "_development_access_record", None)
    if access_record is not None and access_record.exists():
        raise RuntimeError("DEVELOPMENT labels were already opened")
    if type(source.bindings) is not S10SourceBindings:
        raise TypeError("source bindings must be exact S10SourceBindings")
    if source.bindings.public_case_manifest_hash != protocol.public_case_manifest_hash:
        raise ValueError("source and protocol public manifests differ")
    if getattr(source, "_protocol", None) != protocol:
        raise ValueError("source and orchestrator protocols differ")


def _materialize_source_cases(
    stream: Iterable[VerifiedS10Case],
    *,
    expected_case_ids: tuple[str, ...],
    expected_role: EvidenceRole,
    label: str,
) -> tuple[OperatorCaseTensors, ...]:
    prepared = tuple(
        prepare_operator_case(item.case, item.request, item.result) for item in stream
    )
    if len(prepared) != len(expected_case_ids):
        raise ValueError(
            f"{label} source must contain exactly {len(expected_case_ids)} cases"
        )
    if tuple(item.case_id for item in prepared) != expected_case_ids:
        raise ValueError(f"{label} source case order or identity changed")
    if any(item.role is not expected_role for item in prepared):
        raise ValueError(f"{label} source role changed")
    return prepared


def _normalize_population(
    cases: tuple[OperatorCaseTensors, ...],
    normalization: CaseNormalizationStatistics,
    *,
    expected_case_ids: tuple[str, ...],
    expected_role: EvidenceRole,
    label: str,
) -> tuple[ArchitectureCaseTensors, ...]:
    normalized = tuple(normalize_case_values(case, normalization) for case in cases)
    if any(type(case) is not ArchitectureCaseTensors for case in normalized):
        raise TypeError(f"{label} normalization returned a noncanonical tensor case")
    if tuple(case.case_id for case in normalized) != expected_case_ids:
        raise ValueError(f"{label} normalized case identity changed")
    if any(
        case.role is not expected_role
        or case.normalization_hash != normalization.content_hash
        for case in normalized
    ):
        raise ValueError(f"{label} role or normalization identity changed")
    return normalized


def _validate_fresh_models(
    protocol: ArchitectureScreenProtocol,
    models: tuple[PiGinotOperator, ...],
) -> None:
    if type(models) is not tuple or len(models) != len(protocol.candidates):
        raise ValueError("model builder must return the exact candidate population")
    if any(type(model) is not PiGinotOperator for model in models):
        raise TypeError("candidate models must be exact PiGinotOperator values")
    if len({id(model) for model in models}) != len(models):
        raise ValueError("every candidate requires a fresh model instance")
    for candidate, model in zip(protocol.candidates, models, strict=True):
        if module_digest(model) != candidate.operator_module_digest:
            raise ValueError(f"candidate {candidate.candidate_id} model schema changed")


def _validate_training_summary(
    protocol: ArchitectureScreenProtocol,
    candidate: ArchitectureCandidate,
    summary: CandidateTrainingSummary,
    schedule_hash: str,
) -> None:
    if type(summary) is not CandidateTrainingSummary:
        raise TypeError("trainer must return an exact CandidateTrainingSummary")
    if summary.candidate_id != candidate.candidate_id:
        raise ValueError("training summary candidate identity changed")
    if summary.optimizer_steps != protocol.optimizer_steps:
        raise ValueError("candidate did not complete the registered optimizer budget")
    if summary.schedule_hash != schedule_hash:
        raise ValueError("candidate did not use the shared registered schedule")


def _validate_metric(
    candidate: ArchitectureCandidate,
    metric: ArchitectureScreenMetrics,
) -> None:
    if type(metric) is not ArchitectureScreenMetrics:
        raise TypeError("evaluator must return exact ArchitectureScreenMetrics")
    if metric.candidate_id != candidate.candidate_id:
        raise ValueError("metric candidate identity changed")


def _passes_registered_guardrails(
    protocol: ArchitectureScreenProtocol,
    baseline: ArchitectureScreenMetrics,
    candidate: ArchitectureScreenMetrics,
) -> bool:
    baseline_signed = abs(baseline.median_signed_peak_error)
    if baseline_signed == 0.0:
        return False
    field_limit = 1.0 + protocol.maximum_field_relative_regression
    return (
        abs(candidate.median_signed_peak_error)
        <= (1.0 - protocol.minimum_signed_bias_reduction) * baseline_signed
        and candidate.median_absolute_peak_error
        <= (1.0 - protocol.minimum_median_peak_reduction)
        * baseline.median_absolute_peak_error
        and candidate.p90_absolute_peak_error <= baseline.p90_absolute_peak_error
        and candidate.median_u_relative_l2
        <= field_limit * baseline.median_u_relative_l2
        and candidate.median_p_relative_l2
        <= field_limit * baseline.median_p_relative_l2
    )


def _require_model_unchanged(model: PiGinotOperator, expected_hash: str) -> None:
    if model_state_hash(model) != expected_hash:
        raise RuntimeError("candidate model state changed after training")


def _require_schedule_unchanged(
    schedule: TrainingSchedule,
    expected_hash: str,
) -> None:
    if schedule.content_hash != expected_hash:
        raise RuntimeError("shared training schedule changed during execution")


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a lowercase 256-bit digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a lowercase 256-bit digest") from exc
    if value != value.lower():
        raise ValueError(f"{label} must be a lowercase 256-bit digest")
    return value


def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _candidate_progress_callback(
    progress: Callable[[str], None] | None,
    candidate_id: str,
) -> Callable[[int, float], None] | None:
    if progress is None:
        return None

    def emit(step: int, loss: float) -> None:
        _emit_progress(
            progress,
            "phase=TRAIN_FIT "
            f"candidate={candidate_id} event=train_progress "
            f"step={step} loss={loss:.17g}",
        )

    return emit


__all__ = (
    "ContingencyExecutionResult",
    "DevelopmentConfirmation",
    "execute_architecture_contingency",
)
