"""Selective, leakage-safe tensor preparation for the S11 contingency screen."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, cast

import torch

from campaigns.dogbone_v1.s11_contingency import (
    BOUNDARY_POINT_COUNT,
    BOUNDARY_TAG_NAMES,
    DERIVED_PARAMETER_NAMES,
    RAW_PARAMETER_NAMES,
    STRESS_REFERENCE_MPA,
)
from campaigns.dogbone_v1.s11_contingency_tensors import ArchitectureCaseTensors
from campaigns.dogbone_v1.strain_pilot import prescribed_displacement_mm
from piginot.contracts import hash_spec
from piginot.doe import EvidenceRole, PublishedCase, PublishedCaseStatus
from piginot.fem import FemRequest, FemResult

_ALL_PARAMETER_NAMES = (*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES)
_RAW_SOURCE_NAMES = {
    "L_total": "L_total",
    "W_grip": "W_grip",
    "W_gauge": "W_gauge",
    "R_fillet": "R_fillet",
    "E": "elastic_modulus",
    "nu": "poisson_ratio",
    "nominal_axial_strain": "nominal_axial_strain",
}
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


def _float_tensor(value: object, label: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{label} cannot be represented as a float64 tensor") from exc
    tensor = tensor.detach().clone().contiguous()
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{label} must be non-empty and finite")
    return tensor


def _clone_float_tensor(
    value: torch.Tensor,
    label: str,
    *,
    shape: tuple[int | None, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if (
        value.device.type != "cpu"
        or value.dtype is not torch.float64
        or value.layout is not torch.strided
        or value.ndim != len(shape)
    ):
        raise ValueError(f"{label} must be a dense CPU float64 tensor")
    for observed, expected in zip(value.shape, shape, strict=True):
        if expected is not None and observed != expected:
            raise ValueError(f"{label} has an invalid shape")
    if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{label} must be non-empty and finite")
    return value.detach().clone().contiguous()


@dataclass(frozen=True, slots=True)
class OperatorCaseTensors:
    """One verified public FEM case projected onto mesh-vertex operator tensors."""

    case_id: str
    leakage_group_id: str
    role: EvidenceRole
    status: PublishedCaseStatus
    raw_parameters: torch.Tensor
    derived_parameters: torch.Tensor
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
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be non-empty text")
        if not isinstance(self.leakage_group_id, str) or not self.leakage_group_id:
            raise ValueError("leakage_group_id must be non-empty text")
        role = EvidenceRole(self.role)
        status = PublishedCaseStatus(self.status)
        if status is not PublishedCaseStatus.ACTIVE:
            raise ValueError("operator cases must be ACTIVE public cases")
        raw = _clone_float_tensor(
            self.raw_parameters,
            "raw_parameters",
            shape=(len(RAW_PARAMETER_NAMES),),
        )
        derived = _clone_float_tensor(
            self.derived_parameters,
            "derived_parameters",
            shape=(len(DERIVED_PARAMETER_NAMES),),
        )
        boundary_xy = _clone_float_tensor(
            self.boundary_xy,
            "boundary_xy",
            shape=(BOUNDARY_POINT_COUNT, 2),
        )
        boundary_tags = _clone_float_tensor(
            self.boundary_tags,
            "boundary_tags",
            shape=(BOUNDARY_POINT_COUNT, len(BOUNDARY_TAG_NAMES)),
        )
        boundary_normals = _clone_float_tensor(
            self.boundary_normals,
            "boundary_normals",
            shape=(BOUNDARY_POINT_COUNT, 2),
        )
        boundary_weights = _clone_float_tensor(
            self.boundary_weights,
            "boundary_weights",
            shape=(BOUNDARY_POINT_COUNT,),
        )
        if bool((boundary_weights <= 0.0).any().item()):
            raise ValueError("boundary_weights must be positive")
        if not isinstance(self.boundary_mask, torch.Tensor):
            raise TypeError("boundary_mask must be a torch.Tensor")
        if (
            self.boundary_mask.device.type != "cpu"
            or self.boundary_mask.dtype is not torch.bool
            or self.boundary_mask.shape != (BOUNDARY_POINT_COUNT,)
            or not bool(self.boundary_mask.all().item())
        ):
            raise ValueError("boundary_mask must expose all 320 boundary points")
        boundary_mask = self.boundary_mask.detach().clone().contiguous()
        if not bool(
            torch.all((boundary_tags == 0.0) | (boundary_tags == 1.0)).item()
        ) or not bool((boundary_tags.sum(dim=1) == 1.0).all().item()):
            raise ValueError("boundary_tags must be one-hot in reference tag order")
        normal_lengths = torch.linalg.vector_norm(boundary_normals, dim=1)
        if not bool(
            torch.allclose(
                normal_lengths,
                torch.ones_like(normal_lengths),
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ):
            raise ValueError("boundary_normals must be unit vectors")

        query_xy = _clone_float_tensor(
            self.query_xy,
            "query_xy",
            shape=(None, 2),
        )
        target_u = _clone_float_tensor(
            self.target_u,
            "target_u",
            shape=(query_xy.shape[0], 2),
        )
        target_p = _clone_float_tensor(
            self.target_P,
            "target_P",
            shape=(query_xy.shape[0], 2, 2),
        )
        if not isinstance(self.qoi_indices, torch.Tensor):
            raise TypeError("qoi_indices must be a torch.Tensor")
        if (
            self.qoi_indices.device.type != "cpu"
            or self.qoi_indices.dtype is not torch.int64
            or self.qoi_indices.ndim != 1
            or self.qoi_indices.numel() == 0
        ):
            raise ValueError("qoi_indices must be a non-empty CPU int64 vector")
        qoi = self.qoi_indices.detach().clone().contiguous()
        qoi_values = tuple(int(value) for value in qoi.tolist())
        if (
            qoi_values != tuple(sorted(set(qoi_values)))
            or qoi_values[0] < 0
            or qoi_values[-1] >= query_xy.shape[0]
        ):
            raise ValueError("qoi_indices must be sorted unique query indices")

        object.__setattr__(self, "role", role)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "raw_parameters", raw)
        object.__setattr__(self, "derived_parameters", derived)
        object.__setattr__(self, "boundary_xy", boundary_xy)
        object.__setattr__(self, "boundary_tags", boundary_tags)
        object.__setattr__(self, "boundary_normals", boundary_normals)
        object.__setattr__(self, "boundary_weights", boundary_weights)
        object.__setattr__(self, "boundary_mask", boundary_mask)
        object.__setattr__(self, "query_xy", query_xy)
        object.__setattr__(self, "target_u", target_u)
        object.__setattr__(self, "target_P", target_p)
        object.__setattr__(self, "qoi_indices", qoi)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return _ALL_PARAMETER_NAMES

    @property
    def parameters(self) -> torch.Tensor:
        return torch.cat((self.raw_parameters, self.derived_parameters)).clone()


@dataclass(frozen=True, slots=True)
class CaseNormalizationStatistics:
    """Frozen z-score statistics whose provenance is an exact TRAIN fit set."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    parameter_names: tuple[str, ...]
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    fit_case_ids: tuple[str, ...]
    training_snapshot_hash: str
    stress_reference_mpa: float = STRESS_REFERENCE_MPA
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if tuple(self.parameter_names) != _ALL_PARAMETER_NAMES:
            raise ValueError("parameter_names must remain in registered protocol order")
        mean = _finite_tuple(self.mean, "mean")
        deviation = _finite_tuple(self.standard_deviation, "standard_deviation")
        if len(mean) != len(_ALL_PARAMETER_NAMES) or len(deviation) != len(mean):
            raise ValueError("normalization statistics width changed")
        if any(value <= 0.0 for value in deviation):
            raise ValueError("standard deviations must be positive")
        fit_ids = tuple(self.fit_case_ids)
        if not fit_ids or fit_ids != tuple(sorted(set(fit_ids))):
            raise ValueError("fit_case_ids must be non-empty sorted unique IDs")
        if (
            not isinstance(self.training_snapshot_hash, str)
            or _DIGEST_PATTERN.fullmatch(self.training_snapshot_hash) is None
        ):
            raise ValueError(
                "training_snapshot_hash must be a lowercase 256-bit digest"
            )
        if self.stress_reference_mpa != STRESS_REFERENCE_MPA:
            raise ValueError("stress reference must remain fixed at 760 MPa")
        if self.schema_version != "1.0":
            raise ValueError("unsupported normalization schema")
        object.__setattr__(self, "parameter_names", _ALL_PARAMETER_NAMES)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", deviation)
        object.__setattr__(self, "fit_case_ids", fit_ids)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_spec(self, "content"))


def prepare_operator_case(
    case: PublishedCase,
    request: FemRequest,
    result: FemResult,
) -> OperatorCaseTensors:
    """Project one already-verified S10 P2 result onto its mesh vertices."""

    if type(case) is not PublishedCase:
        raise TypeError("case must be an exact PublishedCase")
    if type(request) is not FemRequest:
        raise TypeError("request must be an exact FemRequest")
    if type(result) is not FemResult:
        raise TypeError("result must be an exact FemResult")
    if case.status is not PublishedCaseStatus.ACTIVE:
        raise ValueError("operator tensors require an ACTIVE public case")
    if request.request_id != case.case_id or result.request_id != case.case_id:
        raise ValueError("case, request, and result identities do not match")
    if result.mesh_id != request.mesh.mesh_id:
        raise ValueError("result mesh identity does not match the request")
    if request.config.element_order != 2 or result.element_order != 2:
        raise ValueError("S11 contingency data requires the verified P2 FEM recipe")

    vertex_count = len(request.mesh.nodes)
    if len(result.displacement) < vertex_count:
        raise ValueError("P2 displacement omitted mesh-vertex values")
    if len(result.recovered_first_piola_stress) < vertex_count:
        raise ValueError("P2 recovered stress omitted mesh-vertex values")
    if len(result.displacement_dof_coordinates) < vertex_count:
        raise ValueError("P2 displacement coordinates omitted mesh vertices")
    result_vertex_coordinates = _float_tensor(
        result.displacement_dof_coordinates[:vertex_count],
        "P2 vertex coordinates",
    )
    query_xy = _float_tensor(request.mesh.nodes, "mesh vertex coordinates")
    if result_vertex_coordinates.shape != query_xy.shape or not torch.equal(
        result_vertex_coordinates,
        query_xy,
    ):
        raise ValueError("P2 vertex coordinates do not align with mesh vertices")

    target_u = _float_tensor(
        result.displacement[:vertex_count],
        "P2 vertex displacement",
    )
    if target_u.shape != (vertex_count, 2):
        raise ValueError("P2 vertex displacement must have shape (vertices, 2)")
    recovered = _float_tensor(
        result.recovered_first_piola_stress[:vertex_count],
        "P2 recovered stress",
    )
    if recovered.shape != (vertex_count, 3, 3):
        raise ValueError("P2 recovered stress must have shape (vertices, 3, 3)")

    if set(request.mesh.semantic_tags) != set(BOUNDARY_TAG_NAMES):
        raise ValueError("mesh semantic tags do not match the DogBone reference order")
    # Local import keeps the runner free to consume this projection layer without
    # an import cycle while retaining its single boundary-sampling implementation.
    from campaigns.dogbone_v1.s11_contingency_run import (
        boundary_cloud_tensors,
        qoi_vertex_indices,
    )

    cloud = boundary_cloud_tensors(
        request.mesh,
        tag_names=BOUNDARY_TAG_NAMES,
        sample_count=BOUNDARY_POINT_COUNT,
    )
    qoi = qoi_vertex_indices(
        request.mesh,
        semantic_tag="right_arc",
        excluded_tags=("right_grip",),
    )
    raw = tuple(
        _coordinate(case.primitive_coordinates, _RAW_SOURCE_NAMES[name], name)
        for name in RAW_PARAMETER_NAMES
    )
    derived = tuple(
        _coordinate(case.derived_coordinates, name, name)
        for name in DERIVED_PARAMETER_NAMES
    )
    return OperatorCaseTensors(
        case_id=case.case_id,
        leakage_group_id=case.leakage_group_id,
        role=case.role,
        status=case.status,
        raw_parameters=torch.tensor(raw, dtype=torch.float64),
        derived_parameters=torch.tensor(derived, dtype=torch.float64),
        boundary_xy=cloud.xy,
        boundary_tags=cloud.features[:, 2:],
        boundary_normals=cloud.features[:, :2],
        boundary_weights=cloud.weights,
        boundary_mask=cloud.mask,
        query_xy=query_xy,
        target_u=target_u,
        target_P=recovered[:, :2, :2],
        qoi_indices=torch.tensor(qoi, dtype=torch.int64),
    )


def fit_case_normalization(
    cases: tuple[OperatorCaseTensors, ...],
    *,
    fit_case_ids: tuple[str, ...],
    training_snapshot_hash: str,
) -> CaseNormalizationStatistics:
    """Fit parameter statistics on exactly the caller-declared ACTIVE TRAIN set."""

    if type(cases) is not tuple or not cases:
        raise ValueError("cases must be a non-empty tuple")
    if any(type(case) is not OperatorCaseTensors for case in cases):
        raise TypeError("cases must contain exact OperatorCaseTensors")
    expected_ids = tuple(fit_case_ids)
    if expected_ids != tuple(sorted(set(expected_ids))) or not expected_ids:
        raise ValueError("fit_case_ids must be non-empty sorted unique IDs")
    ordered = tuple(sorted(cases, key=lambda item: item.case_id))
    if tuple(case.case_id for case in ordered) != expected_ids:
        raise ValueError("cases must match the exact fit_case_ids population")
    if any(
        case.role is not EvidenceRole.TRAIN
        or case.status is not PublishedCaseStatus.ACTIVE
        for case in ordered
    ):
        raise ValueError("normalization statistics require only ACTIVE TRAIN cases")
    rows = tuple(
        tuple(float(value) for value in case.parameters.tolist()) for case in ordered
    )
    width = len(_ALL_PARAMETER_NAMES)
    mean = tuple(
        math.fsum(row[index] for row in rows) / len(rows) for index in range(width)
    )
    deviation = tuple(
        _population_deviation(rows, index, mean[index]) for index in range(width)
    )
    return CaseNormalizationStatistics(
        parameter_names=_ALL_PARAMETER_NAMES,
        mean=mean,
        standard_deviation=deviation,
        fit_case_ids=expected_ids,
        training_snapshot_hash=training_snapshot_hash,
    )


def normalize_case_values(
    case: OperatorCaseTensors,
    statistics: CaseNormalizationStatistics,
) -> ArchitectureCaseTensors:
    """Apply frozen TRAIN statistics and fixed output scales to a new case value."""

    if type(case) is not OperatorCaseTensors:
        raise TypeError("case must be an exact OperatorCaseTensors")
    if type(statistics) is not CaseNormalizationStatistics:
        raise TypeError("statistics must be exact CaseNormalizationStatistics")
    if statistics.parameter_names != case.parameter_names:
        raise ValueError("case and normalization parameter order differ")
    mean = torch.tensor(statistics.mean, dtype=torch.float64)
    deviation = torch.tensor(statistics.standard_deviation, dtype=torch.float64)
    normalized_parameters = (case.parameters - mean) / deviation
    values = dict(zip(RAW_PARAMETER_NAMES, case.raw_parameters.tolist(), strict=True))
    displacement_scale = prescribed_displacement_mm(
        values["L_total"],
        values["nominal_axial_strain"],
    )
    return ArchitectureCaseTensors(
        case_id=case.case_id,
        leakage_group_id=case.leakage_group_id,
        role=case.role,
        status=case.status,
        parameter_names=case.parameter_names,
        normalized_parameters=normalized_parameters,
        normalization_hash=statistics.content_hash,
        boundary_tag_names=BOUNDARY_TAG_NAMES,
        boundary_xy=case.boundary_xy.clone(),
        boundary_tags=case.boundary_tags.clone(),
        boundary_normals=case.boundary_normals.clone(),
        boundary_weights=case.boundary_weights.clone(),
        boundary_mask=case.boundary_mask.clone(),
        query_xy=case.query_xy.clone(),
        target_u=case.target_u.clone() / displacement_scale,
        target_P=case.target_P.clone() / statistics.stress_reference_mpa,
        qoi_indices=case.qoi_indices.clone(),
    )


def _coordinate(
    coordinates: Mapping[str, float],
    source_name: str,
    protocol_name: str,
) -> float:
    try:
        value = coordinates[source_name]
    except KeyError as exc:
        raise ValueError(f"case omitted protocol parameter {protocol_name!r}") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"protocol parameter {protocol_name!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"protocol parameter {protocol_name!r} must be finite")
    return number


def _finite_tuple(values: object, label: str) -> tuple[float, ...]:
    if type(values) is not tuple:
        raise ValueError(f"{label} must be a tuple")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must contain finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} must contain finite numbers")
        result.append(number)
    return tuple(result)


def _population_deviation(
    rows: tuple[tuple[float, ...], ...],
    index: int,
    mean: float,
) -> float:
    variance = math.fsum((row[index] - mean) ** 2 for row in rows) / len(rows)
    deviation = math.sqrt(variance)
    return 1.0 if deviation == 0.0 else deviation


__all__ = (
    "BOUNDARY_POINT_COUNT",
    "BOUNDARY_TAG_NAMES",
    "STRESS_REFERENCE_MPA",
    "CaseNormalizationStatistics",
    "OperatorCaseTensors",
    "fit_case_normalization",
    "normalize_case_values",
    "prepare_operator_case",
)
