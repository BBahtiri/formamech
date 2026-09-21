"""Fail-closed initialization of S13 from the selected trained S11 model."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import rfc8785

from campaigns.dogbone_v1.s11_contingency import (
    ArchitectureScreenMetrics,
    ArchitectureScreenProtocol,
)
from campaigns.dogbone_v1.s11_contingency_data import CaseNormalizationStatistics
from campaigns.dogbone_v1.s11_contingency_models import (
    build_candidate_models,
    model_state_hash,
)
from campaigns.dogbone_v1.s11_contingency_run import load_committed_protocol
from piginot.contracts import strict_json_loads
from piginot.operators import (
    PiGinotOperator,
    load_operator_state_dict_with_metadata,
    module_digest,
)

_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE_PATH = (
    _ROOT / "artifacts" / "campaigns" / "dogbone-v1" / "S11-CONTINGENCY-EVIDENCE.json"
)
_EVIDENCE_FILE_BLAKE2B256 = (
    "260eca6c345ee1edea17b9098859baaa6de4a35e27de65c6ccdead0cad831f36"
)
_EVIDENCE_TYPE = "piginot.campaigns.ContingencyEvidence"
_MODEL_FILE_TYPE = "piginot.campaigns.ModelFileEvidence"
_MODEL_STATE_TYPE = "piginot.campaigns.CandidateModelStateEvidence"
_NORMALIZATION_TYPE = "piginot.campaigns.CaseNormalizationStatistics"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class SelectedS11CheckpointEvidence:
    """Pinned identities needed to accept the selected S11 checkpoint."""

    evidence_file_blake2b256: str
    protocol_content_hash: str
    source_bindings_hash: str
    candidate_id: str
    file_name: str
    byte_count: int
    file_blake2b256: str
    operator_module_digest: str
    initial_state_hash: str
    final_state_hash: str
    normalization: CaseNormalizationStatistics
    baseline_metrics: ArchitectureScreenMetrics

    def __post_init__(self) -> None:
        for name in (
            "evidence_file_blake2b256",
            "protocol_content_hash",
            "source_bindings_hash",
            "file_blake2b256",
            "initial_state_hash",
            "final_state_hash",
        ):
            _digest(getattr(self, name), name)
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be non-empty text")
        if self.file_name != f"{self.candidate_id}.safetensors":
            raise ValueError("selected checkpoint file name is not canonical")
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise ValueError("byte_count must be a positive integer")
        if not isinstance(
            self.operator_module_digest, str
        ) or not self.operator_module_digest.startswith("blake2b256:"):
            raise ValueError("operator_module_digest must use blake2b256")
        _digest(
            self.operator_module_digest.removeprefix("blake2b256:"),
            "operator_module_digest",
        )
        if type(self.normalization) is not CaseNormalizationStatistics:
            raise TypeError("normalization must be exact CaseNormalizationStatistics")
        if type(self.baseline_metrics) is not ArchitectureScreenMetrics:
            raise TypeError("baseline_metrics must be exact ArchitectureScreenMetrics")
        if self.baseline_metrics.candidate_id != self.candidate_id:
            raise ValueError("baseline metrics must describe the selected candidate")

    @property
    def expected_metadata(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "protocol_content_hash": self.protocol_content_hash,
            "source_bindings_hash": self.source_bindings_hash,
            "initial_state_hash": self.initial_state_hash,
            "final_state_hash": self.final_state_hash,
            "operator_module_digest": self.operator_module_digest,
        }


@dataclass(frozen=True, slots=True)
class SelectedS11Initialization:
    """Loaded model and normalization bound to the pinned S11 evidence."""

    candidate_id: str
    model: PiGinotOperator
    normalization: CaseNormalizationStatistics
    source_bindings_hash: str
    evidence_file_blake2b256: str
    checkpoint_file_blake2b256: str
    checkpoint_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be non-empty text")
        if type(self.model) is not PiGinotOperator:
            raise TypeError("model must be an exact PiGinotOperator")
        if type(self.normalization) is not CaseNormalizationStatistics:
            raise TypeError("normalization must be exact CaseNormalizationStatistics")
        _digest(self.source_bindings_hash, "source_bindings_hash")
        _digest(self.evidence_file_blake2b256, "evidence_file_blake2b256")
        _digest(self.checkpoint_file_blake2b256, "checkpoint_file_blake2b256")
        if (
            not isinstance(self.checkpoint_path, Path)
            or not self.checkpoint_path.is_absolute()
        ):
            raise ValueError("checkpoint_path must be an absolute pathlib.Path")


def load_selected_s11_checkpoint_evidence() -> SelectedS11CheckpointEvidence:
    """Parse the exact tracked S11 evidence and retain only selected-model facts."""

    raw = _EVIDENCE_PATH.read_bytes()
    file_digest = hashlib.blake2b(raw, digest_size=32).hexdigest()
    if file_digest != _EVIDENCE_FILE_BLAKE2B256:
        raise ValueError("tracked S11 evidence file digest changed")
    document = strict_json_loads(raw)
    root = _mapping(document, "S11 contingency evidence")
    if rfc8785.dumps(cast(Any, root)) != raw:
        raise ValueError("S11 contingency evidence must be exact canonical JSON")
    if root.get("$type") != _EVIDENCE_TYPE:
        raise ValueError("S11 contingency evidence type changed")

    protocol = load_committed_protocol()
    protocol_hash = _text(root.get("protocol_content_hash"), "protocol_content_hash")
    if protocol_hash != protocol.content_hash:
        raise ValueError("S11 evidence differs from the committed protocol")
    candidate_id = _text(root.get("final_candidate_id"), "final_candidate_id")
    decision = _mapping(root.get("train_screen_decision"), "TRAIN decision")
    selection = _mapping(decision.get("selection"), "TRAIN selection")
    if (
        selection.get("selected_candidate_id") != candidate_id
        or selection.get("fallback_to_baseline") is not True
    ):
        raise ValueError("S11 selected checkpoint is not the registered fallback")

    model_file = _unique_candidate_record(
        root.get("model_files"), candidate_id, _MODEL_FILE_TYPE, "model file"
    )
    state = _unique_candidate_record(
        decision.get("model_states"),
        candidate_id,
        _MODEL_STATE_TYPE,
        "model state",
    )
    candidate = next(
        (item for item in protocol.candidates if item.candidate_id == candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError("selected S11 candidate is absent from the protocol")
    operator_digest = _text(
        model_file.get("operator_module_digest"), "operator_module_digest"
    )
    if operator_digest != candidate.operator_module_digest:
        raise ValueError("S11 model file operator digest differs from the protocol")
    initial_state_hash = _text(
        model_file.get("initial_state_hash"), "initial_state_hash"
    )
    final_state_hash = _text(model_file.get("model_state_hash"), "model_state_hash")
    if (
        state.get("initial_state_hash") != initial_state_hash
        or state.get("final_state_hash") != final_state_hash
    ):
        raise ValueError("S11 model-file and TRAIN-decision state hashes differ")

    normalization = _normalization(root.get("normalization"))
    if normalization.content_hash != decision.get("normalization_hash"):
        raise ValueError("S11 normalization differs from the TRAIN decision")
    if (
        normalization.fit_case_ids != protocol.fit_case_ids
        or normalization.training_snapshot_hash != protocol.training_snapshot_hash
    ):
        raise ValueError("S11 normalization population differs from the protocol")
    baseline_metrics = _architecture_metrics(
        _unique_candidate_record(
            decision.get("metrics"),
            candidate_id,
            "piginot.campaigns.ArchitectureScreenMetrics",
            "TRAIN-screen metric",
        )
    )

    return SelectedS11CheckpointEvidence(
        evidence_file_blake2b256=file_digest,
        protocol_content_hash=protocol_hash,
        source_bindings_hash=_text(
            decision.get("source_bindings_hash"), "source_bindings_hash"
        ),
        candidate_id=candidate_id,
        file_name=_text(model_file.get("file_name"), "file_name"),
        byte_count=_positive_int(model_file.get("byte_count"), "byte_count"),
        file_blake2b256=_text(model_file.get("file_blake2b256"), "file_blake2b256"),
        operator_module_digest=operator_digest,
        initial_state_hash=initial_state_hash,
        final_state_hash=final_state_hash,
        normalization=normalization,
        baseline_metrics=baseline_metrics,
    )


def load_selected_s11_initialization(
    checkpoint_root: Path,
) -> SelectedS11Initialization:
    """Load the selected S11 model after verifying every tracked identity."""

    return _load_selected_s11_checkpoint_core(
        checkpoint_root,
        protocol=load_committed_protocol(),
        evidence=load_selected_s11_checkpoint_evidence(),
    )


def _load_selected_s11_checkpoint_core(
    checkpoint_root: Path,
    *,
    protocol: ArchitectureScreenProtocol,
    evidence: SelectedS11CheckpointEvidence,
) -> SelectedS11Initialization:
    if not isinstance(checkpoint_root, Path):
        raise TypeError("checkpoint_root must be a pathlib.Path")
    if not checkpoint_root.is_absolute():
        raise ValueError("checkpoint_root must be absolute")
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise ValueError("checkpoint_root must be an existing non-symlink directory")
    if type(protocol) is not ArchitectureScreenProtocol:
        raise TypeError("protocol must be an exact ArchitectureScreenProtocol")
    if type(evidence) is not SelectedS11CheckpointEvidence:
        raise TypeError("evidence must be exact SelectedS11CheckpointEvidence")
    if evidence.protocol_content_hash != protocol.content_hash:
        raise ValueError("checkpoint evidence differs from the protocol")

    candidate_index = next(
        (
            index
            for index, candidate in enumerate(protocol.candidates)
            if candidate.candidate_id == evidence.candidate_id
        ),
        None,
    )
    if candidate_index is None:
        raise ValueError("selected checkpoint candidate is absent from the protocol")
    candidate = protocol.candidates[candidate_index]
    if candidate.operator_module_digest != evidence.operator_module_digest:
        raise ValueError("selected checkpoint operator digest differs from protocol")

    root = checkpoint_root.resolve(strict=True)
    checkpoint_path = root / evidence.file_name
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise FileNotFoundError("selected S11 safetensors file is missing or unsafe")
    checkpoint_path = checkpoint_path.resolve(strict=True)
    if checkpoint_path.parent != root:
        raise ValueError("selected S11 checkpoint escaped its root")
    if checkpoint_path.stat().st_size != evidence.byte_count:
        raise ValueError("selected S11 checkpoint byte count differs from evidence")
    file_digest = _file_blake2b256(checkpoint_path)
    if file_digest != evidence.file_blake2b256:
        raise ValueError("selected S11 checkpoint file digest differs from evidence")

    models = build_candidate_models(
        protocol,
        boundary_tag_names=protocol.boundary_tag_names,
    )
    model = models[candidate_index]
    if module_digest(model) != evidence.operator_module_digest:
        raise ValueError("runtime selected-model schema differs from S11 evidence")
    if model_state_hash(model) != evidence.initial_state_hash:
        raise ValueError(
            "runtime selected-model initialization differs from S11 evidence"
        )
    state = load_operator_state_dict_with_metadata(
        checkpoint_path,
        model=model,
        expected_metadata=evidence.expected_metadata,
    )
    if _file_blake2b256(checkpoint_path) != evidence.file_blake2b256:
        raise ValueError("selected S11 checkpoint changed while it was being loaded")
    model.load_state_dict(state, strict=True)
    if model_state_hash(model) != evidence.final_state_hash:
        raise ValueError("loaded selected-model state differs from S11 evidence")
    return SelectedS11Initialization(
        candidate_id=evidence.candidate_id,
        model=model,
        normalization=evidence.normalization,
        source_bindings_hash=evidence.source_bindings_hash,
        evidence_file_blake2b256=evidence.evidence_file_blake2b256,
        checkpoint_file_blake2b256=file_digest,
        checkpoint_path=checkpoint_path,
    )


def _normalization(value: object) -> CaseNormalizationStatistics:
    record = _mapping(value, "normalization")
    expected_fields = {
        "$type",
        "fit_case_ids",
        "mean",
        "parameter_names",
        "schema_version",
        "standard_deviation",
        "stress_reference_mpa",
        "training_snapshot_hash",
    }
    if set(record) != expected_fields or record.get("$type") != _NORMALIZATION_TYPE:
        raise ValueError("S11 normalization schema/type changed")
    return CaseNormalizationStatistics(
        parameter_names=tuple(
            _text(item, "normalization parameter name")
            for item in _list(record.get("parameter_names"), "parameter_names")
        ),
        mean=tuple(
            _number(item, "normalization mean")
            for item in _list(record.get("mean"), "mean")
        ),
        standard_deviation=tuple(
            _number(item, "normalization standard deviation")
            for item in _list(record.get("standard_deviation"), "standard_deviation")
        ),
        fit_case_ids=tuple(
            _text(item, "normalization fit case ID")
            for item in _list(record.get("fit_case_ids"), "fit_case_ids")
        ),
        training_snapshot_hash=_text(
            record.get("training_snapshot_hash"), "training_snapshot_hash"
        ),
        stress_reference_mpa=_number(
            record.get("stress_reference_mpa"), "stress_reference_mpa"
        ),
        schema_version=_text(record.get("schema_version"), "schema_version"),
    )


def _architecture_metrics(value: dict[str, object]) -> ArchitectureScreenMetrics:
    return ArchitectureScreenMetrics(
        candidate_id=_text(value.get("candidate_id"), "metric candidate_id"),
        median_signed_peak_error=_number(
            value.get("median_signed_peak_error"),
            "median_signed_peak_error",
        ),
        median_absolute_peak_error=_number(
            value.get("median_absolute_peak_error"),
            "median_absolute_peak_error",
        ),
        p90_absolute_peak_error=_number(
            value.get("p90_absolute_peak_error"),
            "p90_absolute_peak_error",
        ),
        median_u_relative_l2=_number(
            value.get("median_u_relative_l2"),
            "median_u_relative_l2",
        ),
        median_p_relative_l2=_number(
            value.get("median_p_relative_l2"),
            "median_p_relative_l2",
        ),
        trainable_parameter_count=_positive_int(
            value.get("trainable_parameter_count"),
            "trainable_parameter_count",
        ),
        elapsed_seconds=_number(value.get("elapsed_seconds"), "elapsed_seconds"),
    )


def _unique_candidate_record(
    value: object,
    candidate_id: str,
    expected_type: str,
    label: str,
) -> dict[str, object]:
    matches = tuple(
        _mapping(item, label)
        for item in _list(value, label)
        if isinstance(item, dict) and item.get("candidate_id") == candidate_id
    )
    if len(matches) != 1 or matches[0].get("$type") != expected_type:
        raise ValueError(f"S11 evidence must contain one selected {label}")
    return matches[0]


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{label} must be a JSON array")
    return cast(list[object], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 256-bit digest")
    return value


def _file_blake2b256(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=32)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "SelectedS11CheckpointEvidence",
    "SelectedS11Initialization",
    "load_selected_s11_checkpoint_evidence",
    "load_selected_s11_initialization",
)
