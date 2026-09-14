"""Immutable tensor and schedule contracts for the S11 contingency."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

import torch

from campaigns.dogbone_v1.s11_contingency import (
    DERIVED_PARAMETER_NAMES,
    RAW_PARAMETER_NAMES,
)
from piginot.doe import EvidenceRole, PublishedCaseStatus

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_ALL_PARAMETER_NAMES = (*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES)


class _Digest(Protocol):
    def update(self, value: bytes) -> object: ...


@dataclass(frozen=True, slots=True, eq=False)
class ArchitectureCaseTensors:
    """Owned, normalized tensors for one active TRAIN or DEVELOPMENT case."""

    case_id: str
    leakage_group_id: str
    role: EvidenceRole
    status: PublishedCaseStatus
    parameter_names: tuple[str, ...]
    normalized_parameters: torch.Tensor
    normalization_hash: str
    boundary_tag_names: tuple[str, ...]
    boundary_xy: torch.Tensor
    boundary_tags: torch.Tensor
    boundary_normals: torch.Tensor
    boundary_weights: torch.Tensor
    boundary_mask: torch.Tensor
    query_xy: torch.Tensor
    target_u: torch.Tensor
    target_P: torch.Tensor
    qoi_indices: torch.Tensor

    def __post_init__(self) -> None:
        _canonical_text(self.case_id, "case_id")
        _canonical_text(self.leakage_group_id, "leakage_group_id")
        if type(self.role) is not EvidenceRole or self.role not in {
            EvidenceRole.TRAIN,
            EvidenceRole.DEVELOPMENT,
        }:
            raise ValueError(
                "architecture-screen tensors must have role TRAIN or DEVELOPMENT"
            )
        if (
            type(self.status) is not PublishedCaseStatus
            or self.status is not PublishedCaseStatus.ACTIVE
        ):
            raise ValueError("architecture-screen tensors must have status ACTIVE")
        names = _canonical_names(self.parameter_names, "parameter_names")
        if names != _ALL_PARAMETER_NAMES:
            raise ValueError(
                "parameter_names must use the fixed raw-plus-derived order"
            )
        tag_names = _canonical_names(self.boundary_tag_names, "boundary_tag_names")
        digest = _canonical_text(self.normalization_hash, "normalization_hash")
        if _HASH_PATTERN.fullmatch(digest) is None:
            raise ValueError("normalization_hash must be a lowercase 256-bit digest")

        parameters = _owned_float64(self.normalized_parameters, "normalized_parameters")
        boundary_xy = _owned_float64(self.boundary_xy, "boundary_xy")
        boundary_tags = _owned_float64(self.boundary_tags, "boundary_tags")
        boundary_normals = _owned_float64(self.boundary_normals, "boundary_normals")
        boundary_weights = _owned_float64(self.boundary_weights, "boundary_weights")
        boundary_mask = _owned_bool(self.boundary_mask, "boundary_mask")
        query_xy = _owned_float64(self.query_xy, "query_xy")
        target_u = _owned_float64(self.target_u, "target_u")
        target_P = _owned_float64(self.target_P, "target_P")
        qoi_indices = _owned_indices(self.qoi_indices, "qoi_indices")

        if parameters.ndim != 1 or parameters.shape[0] != len(names):
            raise ValueError("normalized_parameters must align with parameter_names")
        if boundary_xy.ndim != 2 or boundary_xy.shape[1] != 2:
            raise ValueError("boundary_xy must have shape (boundary_points, 2)")
        boundary_count = boundary_xy.shape[0]
        if boundary_count == 0:
            raise ValueError("boundary cloud must be non-empty")
        if boundary_tags.shape != (boundary_count, len(tag_names)):
            raise ValueError("boundary_tags must align with coordinates and tag names")
        if boundary_normals.shape != (boundary_count, 2):
            raise ValueError("boundary_normals must align with boundary_xy")
        if boundary_weights.shape != (boundary_count,):
            raise ValueError("boundary_weights must align with boundary_xy")
        if boundary_mask.shape != (boundary_count,):
            raise ValueError("boundary_mask must align with boundary_xy")
        if not bool(boundary_mask.any().item()):
            raise ValueError("boundary_mask must expose a visible boundary point")
        if bool((boundary_weights < 0.0).any().item()):
            raise ValueError("boundary_weights must be non-negative")
        if not bool((boundary_weights[boundary_mask] > 0.0).all().item()):
            raise ValueError("every visible boundary point needs positive weight")
        visible_tags = boundary_tags[boundary_mask]
        if not bool(((visible_tags == 0.0) | (visible_tags == 1.0)).all().item()):
            raise ValueError("visible boundary_tags must be one-hot")
        if not bool((visible_tags.sum(dim=1) == 1.0).all().item()):
            raise ValueError("each visible boundary point needs exactly one tag")
        normal_lengths = torch.linalg.vector_norm(
            boundary_normals[boundary_mask], dim=1
        )
        if not bool(
            torch.allclose(
                normal_lengths,
                torch.ones_like(normal_lengths),
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError("visible boundary_normals must be unit vectors")
        visible_xy = boundary_xy[boundary_mask]
        if bool(((visible_xy.amax(dim=0) - visible_xy.amin(dim=0)) <= 0.0).any()):
            raise ValueError("visible boundary cloud must span both coordinate axes")

        if query_xy.ndim != 2 or query_xy.shape[1] != 2 or query_xy.shape[0] == 0:
            raise ValueError("query_xy must have shape (vertices, 2) and be non-empty")
        query_count = query_xy.shape[0]
        if target_u.shape != (query_count, 2):
            raise ValueError("target_u must have shape (vertices, 2)")
        if target_P.shape != (query_count, 2, 2):
            raise ValueError("target_P must have shape (vertices, 2, 2)")
        if qoi_indices.ndim != 1 or qoi_indices.numel() == 0:
            raise ValueError("qoi_indices must be a non-empty vector")
        if not bool((qoi_indices[1:] > qoi_indices[:-1]).all().item()):
            raise ValueError("qoi_indices must be sorted and unique")
        if int(qoi_indices[0]) < 0 or int(qoi_indices[-1]) >= query_count:
            raise ValueError("qoi_indices must lie within query vertex bounds")

        for name, value in (
            ("normalized_parameters", parameters),
            ("boundary_xy", boundary_xy),
            ("boundary_tags", boundary_tags),
            ("boundary_normals", boundary_normals),
            ("boundary_weights", boundary_weights),
            ("boundary_mask", boundary_mask),
            ("query_xy", query_xy),
            ("target_u", target_u),
            ("target_P", target_P),
            ("qoi_indices", qoi_indices),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "boundary_tag_names", tag_names)

    def parameters_for(self, parameter_names: tuple[str, ...]) -> torch.Tensor:
        """Return the registered raw or raw-plus-derived candidate vector."""

        requested = tuple(parameter_names)
        if requested == RAW_PARAMETER_NAMES:
            return self.normalized_parameters[: len(RAW_PARAMETER_NAMES)].clone()
        if requested == _ALL_PARAMETER_NAMES:
            return self.normalized_parameters.clone()
        raise ValueError("candidate parameter_names are not registered")


@dataclass(frozen=True, slots=True, eq=False)
class TrainingSchedule:
    """Materialized candidate-independent case and query schedule."""

    protocol_id: str
    sampler_seed: int
    case_ids: tuple[str, ...]
    case_indices: torch.Tensor
    query_indices: torch.Tensor
    qoi_query_count: int

    def __post_init__(self) -> None:
        _canonical_text(self.protocol_id, "protocol_id")
        if type(self.sampler_seed) is not int or self.sampler_seed < 0:
            raise ValueError("sampler_seed must be a non-negative integer")
        case_ids = _canonical_names(self.case_ids, "case_ids")
        if tuple(sorted(case_ids)) != case_ids:
            raise ValueError("case_ids must be sorted")
        case_indices = _owned_indices(self.case_indices, "case_indices")
        query_indices = _owned_indices(self.query_indices, "query_indices")
        if case_indices.ndim != 2 or 0 in case_indices.shape:
            raise ValueError("case_indices must have shape (steps, case_batch)")
        if (
            query_indices.ndim != 3
            or query_indices.shape[:2] != case_indices.shape
            or query_indices.shape[2] == 0
        ):
            raise ValueError(
                "query_indices must have shape (steps, case_batch, query_batch)"
            )
        if int(case_indices.min()) < 0 or int(case_indices.max()) >= len(case_ids):
            raise ValueError("case_indices exceed the registered fit population")
        if int(query_indices.min()) < 0:
            raise ValueError("query_indices must be non-negative")
        if (
            type(self.qoi_query_count) is not int
            or self.qoi_query_count <= 0
            or self.qoi_query_count >= query_indices.shape[2]
        ):
            raise ValueError("qoi_query_count must split generic and QoI queries")
        object.__setattr__(self, "case_ids", case_ids)
        object.__setattr__(self, "case_indices", case_indices)
        object.__setattr__(self, "query_indices", query_indices)

    @property
    def optimizer_steps(self) -> int:
        return int(self.case_indices.shape[0])

    @property
    def case_batch_size(self) -> int:
        return int(self.case_indices.shape[1])

    @property
    def query_points_per_case(self) -> int:
        return int(self.query_indices.shape[2])

    @property
    def content_hash(self) -> str:
        digest = hashlib.blake2b(digest_size=32)
        _update_hash(digest, self.protocol_id.encode("utf-8"))
        _update_hash(digest, str(self.sampler_seed).encode("ascii"))
        for case_id in self.case_ids:
            _update_hash(digest, case_id.encode("utf-8"))
        for tensor in (self.case_indices, self.query_indices):
            _update_hash(digest, str(tuple(tensor.shape)).encode("ascii"))
            _update_hash(digest, tensor.contiguous().numpy().tobytes())
        _update_hash(digest, str(self.qoi_query_count).encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateTrainingSummary:
    """Deterministic scalar trace from one completed optimizer budget."""

    candidate_id: str
    optimizer_steps: int
    schedule_hash: str
    final_total_loss: float
    final_u_mse: float
    final_P_mse: float
    final_peak_P_mse: float

    def __post_init__(self) -> None:
        _canonical_text(self.candidate_id, "candidate_id")
        if type(self.optimizer_steps) is not int or self.optimizer_steps <= 0:
            raise ValueError("optimizer_steps must be positive")
        if _HASH_PATTERN.fullmatch(self.schedule_hash) is None:
            raise ValueError("schedule_hash must be a lowercase 256-bit digest")
        for name in (
            "final_total_loss",
            "final_u_mse",
            "final_P_mse",
            "final_peak_P_mse",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


def _owned_float64(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout is not torch.strided or value.device.type != "cpu":
        raise ValueError(f"{name} must be a dense CPU tensor")
    if value.dtype is not torch.float64:
        raise ValueError(f"{name} must use torch.float64")
    if value.requires_grad:
        raise ValueError(f"{name} must not require gradients")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return value.detach().clone().contiguous()


def _owned_bool(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if (
        value.layout is not torch.strided
        or value.device.type != "cpu"
        or value.dtype is not torch.bool
    ):
        raise ValueError(f"{name} must be a dense boolean CPU tensor")
    return value.detach().clone().contiguous()


def _owned_indices(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if (
        value.layout is not torch.strided
        or value.device.type != "cpu"
        or value.dtype is not torch.int64
    ):
        raise ValueError(f"{name} must be a dense int64 CPU tensor")
    return value.detach().clone().contiguous()


def _canonical_text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be canonical non-empty text")
    return value


def _canonical_names(values: object, label: str) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise ValueError(f"{label} must be a non-empty tuple")
    result = tuple(_canonical_text(value, f"{label} entry") for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    return result


def _update_hash(digest: _Digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


__all__ = (
    "ArchitectureCaseTensors",
    "CandidateTrainingSummary",
    "TrainingSchedule",
)
