"""Create-once, canonical artifacts for the S11 architecture contingency."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, cast

import torch
from safetensors.torch import save as save_safetensors

from campaigns.dogbone_v1.s11_contingency import (
    BASELINE_CANDIDATE_ID,
    ArchitectureScreenProtocol,
)
from campaigns.dogbone_v1.s11_contingency_data import CaseNormalizationStatistics
from campaigns.dogbone_v1.s11_contingency_execute import (
    ContingencyExecutionResult,
    DevelopmentConfirmation,
)
from campaigns.dogbone_v1.s11_contingency_models import model_state_hash
from campaigns.dogbone_v1.s11_contingency_run import TrainScreenDecision
from campaigns.dogbone_v1.s11_contingency_source import S10SourceBindings
from campaigns.dogbone_v1.s11_contingency_tensors import (
    CandidateTrainingSummary,
    TrainingSchedule,
)
from piginot.contracts import canonical_json_bytes, hash_spec

RUN_RESERVATION_NAME = "RUN-RESERVATION.json"
CONTINGENCY_EVIDENCE_NAME = "S11-CONTINGENCY-EVIDENCE.json"
DEVELOPMENT_ACCESS_NAME = "DEVELOPMENT-ACCESS.json"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_FILE_STEM_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_RESERVATION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class TrainingScheduleEvidence:
    """Canonical summary bound to the complete tensor schedule by its digest."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    protocol_id: str
    sampler_seed: int
    case_ids: tuple[str, ...]
    optimizer_steps: int
    case_batch_size: int
    query_points_per_case: int
    qoi_query_count: int
    schedule_hash: str

    def __post_init__(self) -> None:
        if type(self.protocol_id) is not str or not self.protocol_id:
            raise ValueError("protocol_id must be non-empty text")
        if type(self.sampler_seed) is not int or self.sampler_seed < 0:
            raise ValueError("sampler_seed must be a non-negative integer")
        if type(self.case_ids) is not tuple or not self.case_ids:
            raise ValueError("case_ids must be a non-empty tuple")
        if self.case_ids != tuple(sorted(set(self.case_ids))):
            raise ValueError("case_ids must be sorted and unique")
        for value, label in (
            (self.optimizer_steps, "optimizer_steps"),
            (self.case_batch_size, "case_batch_size"),
            (self.query_points_per_case, "query_points_per_case"),
            (self.qoi_query_count, "qoi_query_count"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.qoi_query_count >= self.query_points_per_case:
            raise ValueError("qoi_query_count must be smaller than the query batch")
        _digest(self.schedule_hash, "schedule_hash")

    @classmethod
    def from_schedule(cls, schedule: TrainingSchedule) -> TrainingScheduleEvidence:
        if type(schedule) is not TrainingSchedule:
            raise TypeError("schedule must be an exact TrainingSchedule")
        return cls(
            protocol_id=schedule.protocol_id,
            sampler_seed=schedule.sampler_seed,
            case_ids=schedule.case_ids,
            optimizer_steps=schedule.optimizer_steps,
            case_batch_size=schedule.case_batch_size,
            query_points_per_case=schedule.query_points_per_case,
            qoi_query_count=schedule.qoi_query_count,
            schedule_hash=schedule.content_hash,
        )


@dataclass(frozen=True, slots=True)
class ModelFileEvidence:
    """Identity and safe-serialization evidence for one trained candidate."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    candidate_id: str
    file_name: str
    byte_count: int
    file_blake2b256: str
    operator_module_digest: str
    initial_state_hash: str
    model_state_hash: str

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not str or not self.candidate_id:
            raise ValueError("candidate_id must be non-empty text")
        expected_name = f"{self.candidate_id}.safetensors"
        if (
            type(self.file_name) is not str
            or self.file_name != expected_name
            or _FILE_STEM_PATTERN.fullmatch(self.candidate_id) is None
        ):
            raise ValueError("file_name must be the safe candidate .safetensors name")
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise ValueError("byte_count must be a positive integer")
        _digest(self.file_blake2b256, "file_blake2b256")
        if type(
            self.operator_module_digest
        ) is not str or not self.operator_module_digest.startswith("blake2b256:"):
            raise ValueError("operator_module_digest must use blake2b256")
        _digest(self.initial_state_hash, "initial_state_hash")
        _digest(self.model_state_hash, "model_state_hash")


@dataclass(frozen=True, slots=True)
class ContingencyEvidence:
    """Canonical evidence envelope for one complete architecture screen."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    protocol_content_hash: str
    source_bindings: S10SourceBindings
    training_schedule: TrainingScheduleEvidence
    normalization: CaseNormalizationStatistics
    training_summaries: tuple[CandidateTrainingSummary, ...]
    train_screen_decision: TrainScreenDecision
    development_confirmation: DevelopmentConfirmation | None
    final_candidate_id: str
    model_files: tuple[ModelFileEvidence, ...]
    implementation_revision: str
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        _digest(self.protocol_content_hash, "protocol_content_hash")
        if type(self.source_bindings) is not S10SourceBindings:
            raise TypeError("source_bindings must be exact S10SourceBindings")
        if type(self.training_schedule) is not TrainingScheduleEvidence:
            raise TypeError("training_schedule must be exact TrainingScheduleEvidence")
        if type(self.normalization) is not CaseNormalizationStatistics:
            raise TypeError("normalization must be exact CaseNormalizationStatistics")
        if type(self.training_summaries) is not tuple or any(
            type(item) is not CandidateTrainingSummary
            for item in self.training_summaries
        ):
            raise TypeError("training_summaries must contain exact summaries")
        if type(self.train_screen_decision) is not TrainScreenDecision:
            raise TypeError("train_screen_decision must be exact TrainScreenDecision")
        if (
            self.development_confirmation is not None
            and type(self.development_confirmation) is not DevelopmentConfirmation
        ):
            raise TypeError("development_confirmation has the wrong type")
        if type(self.final_candidate_id) is not str or not self.final_candidate_id:
            raise ValueError("final_candidate_id must be non-empty text")
        if type(self.model_files) is not tuple or not self.model_files:
            raise ValueError("model_files must be a non-empty tuple")
        if any(type(item) is not ModelFileEvidence for item in self.model_files):
            raise TypeError("model_files must contain exact ModelFileEvidence")
        _revision(self.implementation_revision)
        if self.schema_version != "1.0":
            raise ValueError("unsupported contingency evidence schema")
        if (
            self.train_screen_decision.protocol_content_hash
            != self.protocol_content_hash
        ):
            raise ValueError("evidence protocol does not match TRAIN decision")
        if (
            self.train_screen_decision.source_bindings_hash
            != self.source_bindings.content_hash
        ):
            raise ValueError("evidence source binding does not match TRAIN decision")
        if (
            self.training_schedule.schedule_hash
            != self.train_screen_decision.training_schedule_hash
        ):
            raise ValueError("evidence schedule does not match TRAIN decision")
        if (
            self.normalization.content_hash
            != self.train_screen_decision.normalization_hash
        ):
            raise ValueError("evidence normalization does not match TRAIN decision")
        if (
            self.implementation_revision
            != self.train_screen_decision.implementation_revision
        ):
            raise ValueError("evidence revision does not match TRAIN decision")
        candidate_ids = tuple(
            item.candidate_id for item in self.train_screen_decision.model_states
        )
        if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("decision model states must identify unique candidates")
        if (
            tuple(item.candidate_id for item in self.train_screen_decision.metrics)
            != candidate_ids
        ):
            raise ValueError("TRAIN metrics do not match decision candidate order")
        if (
            tuple(item.candidate_id for item in self.training_summaries)
            != candidate_ids
        ):
            raise ValueError("training summaries do not match decision candidates")
        if any(
            item.schedule_hash != self.training_schedule.schedule_hash
            or item.optimizer_steps != self.training_schedule.optimizer_steps
            for item in self.training_summaries
        ):
            raise ValueError("training summary schedule does not match evidence")
        if tuple(item.candidate_id for item in self.model_files) != candidate_ids:
            raise ValueError("model files do not match decision candidates")
        for state, model_file in zip(
            self.train_screen_decision.model_states,
            self.model_files,
            strict=True,
        ):
            if state.initial_state_hash != model_file.initial_state_hash:
                raise ValueError("model file initial state does not match decision")
            if state.final_state_hash != model_file.model_state_hash:
                raise ValueError("model file final state does not match decision")
        selection = self.train_screen_decision.selection
        if selection.fallback_to_baseline:
            if self.development_confirmation is not None:
                raise ValueError("TRAIN fallback must not contain DEVELOPMENT evidence")
            if self.final_candidate_id != BASELINE_CANDIDATE_ID:
                raise ValueError("TRAIN fallback final candidate must be the baseline")
        else:
            confirmation = self.development_confirmation
            if confirmation is None:
                raise ValueError(
                    "nonfallback TRAIN selection requires DEVELOPMENT evidence"
                )
            if (
                confirmation.train_screen_decision_hash
                != self.train_screen_decision.content_hash
            ):
                raise ValueError("DEVELOPMENT evidence does not match TRAIN decision")
            if (
                confirmation.selected_metrics.candidate_id
                != selection.selected_candidate_id
            ):
                raise ValueError("DEVELOPMENT candidate does not match TRAIN selection")
            if self.final_candidate_id != confirmation.final_candidate_id:
                raise ValueError("final candidate does not match DEVELOPMENT evidence")

    def to_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self))

    @property
    def content_hash(self) -> str:
        return cast(str, hash_spec(self, "content"))


@dataclass(frozen=True, slots=True)
class _ReservationRecord:
    """On-disk marker making an output directory non-resumable."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    protocol_content_hash: str
    implementation_revision: str
    status: str = "RESERVED"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        _digest(self.protocol_content_hash, "protocol_content_hash")
        _revision(self.implementation_revision)
        if self.status != "RESERVED" or self.schema_version != "1.0":
            raise ValueError("invalid output reservation record")

    def to_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self))


@dataclass(frozen=True, slots=True)
class OutputReservation:
    """In-process authority to fill one newly reserved output directory once."""

    output_root: Path
    protocol_content_hash: str
    implementation_revision: str
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _RESERVATION_TOKEN:
            raise TypeError("OutputReservation must be created by reserve_output_root")


@dataclass(frozen=True, slots=True)
class _PreparedModel:
    evidence: ModelFileEvidence
    payload: bytes = field(repr=False)


def reserve_output_root(
    output_root: Path,
    *,
    protocol_content_hash: str,
    implementation_revision: str,
) -> OutputReservation:
    """Atomically reserve a previously nonexistent absolute directory."""

    root = _absolute_path(output_root, "output_root")
    record = _ReservationRecord(protocol_content_hash, implementation_revision)
    if root.exists():
        state = (
            "non-empty" if root.is_dir() and any(root.iterdir()) else "already existing"
        )
        raise FileExistsError(
            f"output_root must be new and empty; found {state}: {root}"
        )
    parent = root.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("output_root parent must be a real directory")
    root.mkdir(mode=0o700)
    _write_once(root / RUN_RESERVATION_NAME, record.to_bytes())
    return OutputReservation(
        output_root=root,
        protocol_content_hash=protocol_content_hash,
        implementation_revision=implementation_revision,
        _token=_RESERVATION_TOKEN,
    )


def persist_contingency_result(
    reservation: OutputReservation,
    *,
    protocol: ArchitectureScreenProtocol,
    source_bindings: S10SourceBindings,
    result: ContingencyExecutionResult,
) -> ContingencyEvidence:
    """Validate the full result, then write safe model and canonical evidence bytes."""

    _validate_persistence_inputs(reservation, protocol, source_bindings, result)
    prepared = _prepare_models(protocol, source_bindings, result)
    schedule_evidence = TrainingScheduleEvidence.from_schedule(result.training_schedule)
    evidence = ContingencyEvidence(
        protocol_content_hash=protocol.content_hash,
        source_bindings=source_bindings,
        training_schedule=schedule_evidence,
        normalization=result.normalization,
        training_summaries=result.training_summaries,
        train_screen_decision=result.train_screen_decision,
        development_confirmation=result.development_confirmation,
        final_candidate_id=result.final_candidate_id,
        model_files=tuple(item.evidence for item in prepared),
        implementation_revision=reservation.implementation_revision,
    )
    evidence_bytes = evidence.to_bytes()
    for item in prepared:
        _write_once(reservation.output_root / item.evidence.file_name, item.payload)
    _write_once(reservation.output_root / CONTINGENCY_EVIDENCE_NAME, evidence_bytes)
    return evidence


def _validate_persistence_inputs(
    reservation: OutputReservation,
    protocol: ArchitectureScreenProtocol,
    source_bindings: S10SourceBindings,
    result: ContingencyExecutionResult,
) -> None:
    if (
        type(reservation) is not OutputReservation
        or reservation._token is not _RESERVATION_TOKEN
    ):
        raise TypeError("reservation must be an authentic OutputReservation")
    if type(protocol) is not ArchitectureScreenProtocol:
        raise TypeError("protocol must be exact ArchitectureScreenProtocol")
    if type(source_bindings) is not S10SourceBindings:
        raise TypeError("source_bindings must be exact S10SourceBindings")
    if type(result) is not ContingencyExecutionResult:
        raise TypeError("result must be exact ContingencyExecutionResult")
    root = reservation.output_root
    marker = root / RUN_RESERVATION_NAME
    expected_marker = _ReservationRecord(
        reservation.protocol_content_hash,
        reservation.implementation_revision,
    ).to_bytes()
    if not root.is_dir() or marker.read_bytes() != expected_marker:
        raise RuntimeError("output reservation is missing or changed")
    _validate_output_contents(root, marker, result)
    if reservation.protocol_content_hash != protocol.content_hash:
        raise ValueError("reservation protocol differs from the loaded protocol")
    decision = result.train_screen_decision
    schedule = result.training_schedule
    if decision.protocol_content_hash != protocol.content_hash:
        raise ValueError("TRAIN decision is bound to another protocol")
    if decision.source_bindings_hash != source_bindings.content_hash:
        raise ValueError("TRAIN decision is bound to another S10 source")
    if decision.training_schedule_hash != schedule.content_hash:
        raise ValueError("TRAIN decision is bound to another schedule")
    if decision.normalization_hash != result.normalization.content_hash:
        raise ValueError("TRAIN decision is bound to another normalization")
    if decision.implementation_revision != reservation.implementation_revision:
        raise ValueError("TRAIN decision implementation revision drifted")
    candidate_ids = tuple(candidate.candidate_id for candidate in protocol.candidates)
    if tuple(item.candidate_id for item in result.training_summaries) != candidate_ids:
        raise ValueError("training summaries do not cover the registered candidates")
    if tuple(item.candidate_id for item in decision.metrics) != candidate_ids:
        raise ValueError("TRAIN metrics do not cover the registered candidates")
    if tuple(item.candidate_id for item in decision.model_states) != candidate_ids:
        raise ValueError(
            "model-state evidence does not cover the registered candidates"
        )
    if len(result.trained_models) != len(candidate_ids):
        raise ValueError("trained model population changed")


def _validate_output_contents(
    root: Path,
    reservation_marker: Path,
    result: ContingencyExecutionResult,
) -> None:
    allowed = {reservation_marker.name}
    development_marker = root / DEVELOPMENT_ACCESS_NAME
    if result.development_confirmation is None:
        if development_marker.exists():
            raise RuntimeError("DEVELOPMENT marker exists for a TRAIN fallback result")
    else:
        expected = result.train_screen_decision.to_bytes() + b"\n"
        if (
            not development_marker.is_file()
            or development_marker.is_symlink()
            or development_marker.read_bytes() != expected
        ):
            raise RuntimeError("DEVELOPMENT marker does not match the TRAIN decision")
        allowed.add(development_marker.name)
    found = {path.name for path in root.iterdir()}
    if found != allowed:
        raise FileExistsError(
            "output reservation was already used or contains foreign files"
        )


def _prepare_models(
    protocol: ArchitectureScreenProtocol,
    source_bindings: S10SourceBindings,
    result: ContingencyExecutionResult,
) -> tuple[_PreparedModel, ...]:
    prepared: list[_PreparedModel] = []
    for candidate, state, model in zip(
        protocol.candidates,
        result.train_screen_decision.model_states,
        result.trained_models,
        strict=True,
    ):
        current_state = model_state_hash(model)
        if current_state != state.final_state_hash:
            raise ValueError(f"candidate {candidate.candidate_id} model state drifted")
        tensors = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        }
        _validate_tensor_payload(tensors)
        payload = save_safetensors(
            tensors,
            metadata={
                "candidate_id": candidate.candidate_id,
                "protocol_content_hash": protocol.content_hash,
                "source_bindings_hash": source_bindings.content_hash,
                "initial_state_hash": state.initial_state_hash,
                "final_state_hash": current_state,
                "operator_module_digest": candidate.operator_module_digest,
            },
        )
        file_name = f"{candidate.candidate_id}.safetensors"
        prepared.append(
            _PreparedModel(
                evidence=ModelFileEvidence(
                    candidate_id=candidate.candidate_id,
                    file_name=file_name,
                    byte_count=len(payload),
                    file_blake2b256=hashlib.blake2b(
                        payload, digest_size=32
                    ).hexdigest(),
                    operator_module_digest=candidate.operator_module_digest,
                    initial_state_hash=state.initial_state_hash,
                    model_state_hash=current_state,
                ),
                payload=payload,
            )
        )
    return tuple(prepared)


def _validate_tensor_payload(tensors: dict[str, torch.Tensor]) -> None:
    if not tensors:
        raise ValueError("candidate state dictionary must not be empty")
    for name, tensor in tensors.items():
        if type(name) is not str or not name:
            raise ValueError("candidate state keys must be non-empty text")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("candidate state values must be tensors")
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("candidate state tensors must be contiguous CPU values")
        if tensor.layout is not torch.strided:
            raise ValueError("candidate state tensors must use strided layout")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("candidate state tensors must be finite")


def _absolute_path(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{label} must be a pathlib.Path")
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    return path.resolve(strict=False)


def _write_once(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 256-bit digest")
    return value


def _revision(value: object) -> str:
    if type(value) is not str or _REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError("implementation_revision must be a lowercase Git revision")
    return value


__all__ = (
    "CONTINGENCY_EVIDENCE_NAME",
    "ContingencyEvidence",
    "DEVELOPMENT_ACCESS_NAME",
    "ModelFileEvidence",
    "OutputReservation",
    "RUN_RESERVATION_NAME",
    "TrainingScheduleEvidence",
    "persist_contingency_result",
    "reserve_output_root",
)
