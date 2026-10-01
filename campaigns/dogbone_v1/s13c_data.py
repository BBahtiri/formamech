"""Geometry-only TRAIN materialization and energy objective for S13c.

Energy-form training needs no FEM labels: each registered TRAIN-fit case
contributes its morphed S10 mesh (quadrature strata), its material, its
prescribed loaded-end displacement and its normalised conditioning vector.
FEM result bytes are never resolved here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from campaigns.dogbone_v1.s11_contingency_data import CaseNormalizationStatistics
from campaigns.dogbone_v1.s11_contingency_source import (
    SelectiveS10Source,
    VerifiedS10Request,
)
from campaigns.dogbone_v1.s13_objective import _physics_material
from campaigns.dogbone_v1.s13c_protocol import (
    DOGBONE_S13C_CONDITIONING_PARAMETERS,
    DogboneS13cProtocol,
)
from campaigns.dogbone_v1.strain_pilot import prescribed_displacement_mm
from piginot.contracts import MaterialSpec
from piginot.doe import EvidenceRole, PublishedCaseStatus
from piginot.fem import FemRequest
from piginot.objectives import (
    PotentialEnergyConfig,
    PotentialEnergyTerms,
    TriangleDomain,
    plane_stress_potential_energy,
    stratified_triangle_quadrature,
)
from piginot.operators import (
    AxisAlignedTensionLifting,
    EnergyFormOperator,
    QueryFrame,
)
from piginot.training import TrainingPopulation

_SOURCE_NAMES = {
    "L_total": "L_total",
    "W_grip": "W_grip",
    "W_gauge": "W_gauge",
    "R_fillet": "R_fillet",
    "nominal_axial_strain": "nominal_axial_strain",
}
_GEOMETRY_TOLERANCE_MM = 1.0e-9


@dataclass(frozen=True, slots=True)
class DogboneS13cGeometryCase:
    """One registered TRAIN case reduced to what the energy objective needs."""

    case_id: str
    leakage_group_id: str
    request: FemRequest
    material: MaterialSpec
    domain: TriangleDomain
    conditioning: torch.Tensor
    half_length_mm: float
    half_grip_width_mm: float
    prescribed_displacement_mm: float

    def __post_init__(self) -> None:
        if type(self.request) is not FemRequest:
            raise TypeError("request must be an exact FemRequest")
        if self.request.request_id != self.case_id:
            raise ValueError("request and case identities differ")
        if type(self.domain) is not TriangleDomain:
            raise TypeError("domain must be a TriangleDomain")
        if (
            not isinstance(self.conditioning, torch.Tensor)
            or self.conditioning.shape != (len(DOGBONE_S13C_CONDITIONING_PARAMETERS),)
            or not bool(torch.isfinite(self.conditioning).all().item())
        ):
            raise ValueError("conditioning must be a finite five-parameter vector")
        for label in (
            "half_length_mm",
            "half_grip_width_mm",
            "prescribed_displacement_mm",
        ):
            value = getattr(self, label)
            if not isinstance(value, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be a finite positive float")

    @property
    def dtype(self) -> torch.dtype:
        dtype: torch.dtype = self.domain.vertices.dtype
        return dtype

    def frame(self) -> QueryFrame:
        return QueryFrame(
            torch.zeros((1, 2), dtype=self.dtype),
            torch.tensor(
                [[self.half_length_mm, self.half_grip_width_mm]], dtype=self.dtype
            ),
        )

    def lifting(self) -> AxisAlignedTensionLifting:
        return AxisAlignedTensionLifting(
            torch.tensor(
                [[self.half_length_mm, self.half_grip_width_mm]], dtype=self.dtype
            ),
            torch.tensor([self.prescribed_displacement_mm], dtype=self.dtype),
        )

    def parameters(self) -> torch.Tensor:
        return self.conditioning.to(self.dtype).unsqueeze(0)


def conditioning_vector(
    primitive_coordinates: Mapping[str, float],
    normalization: CaseNormalizationStatistics,
) -> torch.Tensor:
    """Z-score the five conditioning primitives with frozen TRAIN statistics."""

    if type(normalization) is not CaseNormalizationStatistics:
        raise TypeError("normalization must be exact CaseNormalizationStatistics")
    index = {
        name: position for position, name in enumerate(normalization.parameter_names)
    }
    values: list[float] = []
    for name in DOGBONE_S13C_CONDITIONING_PARAMETERS:
        raw = primitive_coordinates.get(_SOURCE_NAMES[name])
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"case omitted conditioning parameter {name!r}")
        position = index[name]
        values.append(
            (float(raw) - normalization.mean[position])
            / normalization.standard_deviation[position]
        )
    return torch.tensor(values, dtype=torch.float64)


def geometry_case(
    verified: VerifiedS10Request,
    normalization: CaseNormalizationStatistics,
    *,
    dtype: torch.dtype,
) -> DogboneS13cGeometryCase:
    """Reduce one verified TRAIN request to energy-objective geometry."""

    if type(verified) is not VerifiedS10Request:
        raise TypeError("verified must be an exact VerifiedS10Request")
    case = verified.case
    request = verified.request
    if case.role is not EvidenceRole.TRAIN or case.status is not (
        PublishedCaseStatus.ACTIVE
    ):
        raise PermissionError(f"{case.role.value} evidence cannot enter S13c")
    coordinates = case.primitive_coordinates
    length = float(coordinates["L_total"])
    grip_width = float(coordinates["W_grip"])
    strain = float(coordinates["nominal_axial_strain"])
    nodes = torch.tensor(request.mesh.nodes, dtype=torch.float64)
    cells = torch.tensor(request.mesh.cells, dtype=torch.int64)
    half_length = 0.5 * length
    half_grip_width = 0.5 * grip_width
    if (
        abs(float(nodes[:, 0].min())) > _GEOMETRY_TOLERANCE_MM
        or abs(float(nodes[:, 1].min())) > _GEOMETRY_TOLERANCE_MM
        or abs(float(nodes[:, 0].max()) - half_length) > _GEOMETRY_TOLERANCE_MM
        or abs(float(nodes[:, 1].max()) - half_grip_width) > _GEOMETRY_TOLERANCE_MM
    ):
        raise ValueError(f"{case.case_id} mesh extent differs from the quarter model")
    displacement = _loaded_end_displacement(request)
    expected = prescribed_displacement_mm(length, strain)
    if not math.isclose(displacement, expected, rel_tol=1.0e-12, abs_tol=0.0):
        raise ValueError(f"{case.case_id} loaded-end displacement differs from strain")
    return DogboneS13cGeometryCase(
        case_id=case.case_id,
        leakage_group_id=case.leakage_group_id,
        request=request,
        material=_physics_material(request.material),
        domain=TriangleDomain(nodes.to(dtype), cells),
        conditioning=conditioning_vector(coordinates, normalization),
        half_length_mm=half_length,
        half_grip_width_mm=half_grip_width,
        prescribed_displacement_mm=displacement,
    )


def _loaded_end_displacement(request: FemRequest) -> float:
    kinds: dict[str, tuple[str, tuple[str, ...]]] = {
        condition.boundary_tag: (condition.kind, condition.components)
        for condition in request.boundary_conditions
    }
    if kinds != {
        "left_symmetry": ("symmetry", ("x",)),
        "bottom": ("symmetry", ("y",)),
        "right_grip": ("dirichlet", ("x",)),
    }:
        raise ValueError("boundary conditions differ from the quarter-model lifting")
    (loaded,) = (
        condition
        for condition in request.boundary_conditions
        if condition.boundary_tag == "right_grip"
    )
    (value,) = loaded.values
    if value.unit.value != "mm":
        raise ValueError("loaded-end displacement must be in mm")
    return float(value.value)


@dataclass(frozen=True, slots=True)
class DogboneS13cTrainingData:
    """Verified geometry and identities of the registered TRAIN-fit population."""

    cases: tuple[DogboneS13cGeometryCase, ...]
    population: TrainingPopulation
    source_bindings_hash: str

    def __post_init__(self) -> None:
        if tuple(case.case_id for case in self.cases) != self.population.sample_ids:
            raise ValueError("materialized cases differ from the training population")

    def objective(self, config: PotentialEnergyConfig) -> DogboneS13cEnergyObjective:
        return DogboneS13cEnergyObjective(self.cases, config)


def materialize_dogbone_s13c_training_data(
    source: SelectiveS10Source,
    protocol: DogboneS13cProtocol,
    normalization: CaseNormalizationStatistics,
    *,
    dtype: torch.dtype = torch.float32,
) -> DogboneS13cTrainingData:
    """Resolve only registered TRAIN-fit geometry; FEM labels stay unread."""

    if type(source) is not SelectiveS10Source:
        raise TypeError("source must be an exact verified SelectiveS10Source")
    if type(protocol) is not DogboneS13cProtocol:
        raise TypeError("protocol must be exact DogboneS13cProtocol")
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels must remain closed during S13c")
    if source.bindings.content_hash != protocol.source_bindings_hash:
        raise ValueError("S10 source bindings differ from the S13c protocol")
    if source.fit_case_ids != protocol.fit_case_ids:
        raise ValueError("S10 TRAIN-fit population differs from the protocol")
    if normalization.content_hash != protocol.normalization_hash:
        raise ValueError("normalization differs from the S13c protocol")
    if source.train_snapshot.content_hash != protocol.dataset_snapshot_hash:
        raise ValueError("S10 TRAIN snapshot differs from the S13c protocol")
    cases = tuple(
        geometry_case(verified, normalization, dtype=dtype)
        for verified in source.iter_train_geometry(protocol.fit_case_ids)
    )
    population = TrainingPopulation(
        dataset_snapshot_hash=protocol.dataset_snapshot_hash,
        public_manifest_hash=protocol.public_manifest_hash,
        role=EvidenceRole.TRAIN,
        sample_ids=protocol.fit_case_ids,
    )
    if len({case.leakage_group_id for case in cases}) != len(cases):
        raise ValueError("TRAIN-fit leakage groups must be unique")
    return DogboneS13cTrainingData(
        cases=cases,
        population=population,
        source_bindings_hash=source.bindings.content_hash,
    )


class DogboneS13cEnergyObjective:
    """Potential energy of one registered case at fresh stratified points."""

    __slots__ = ("_cases", "_config")

    def __init__(
        self,
        cases: tuple[DogboneS13cGeometryCase, ...],
        config: PotentialEnergyConfig,
    ) -> None:
        if type(cases) is not tuple or not cases:
            raise ValueError("cases must be a non-empty tuple")
        if type(config) is not PotentialEnergyConfig:
            raise TypeError("config must be a PotentialEnergyConfig")
        self._cases: Mapping[str, DogboneS13cGeometryCase] = MappingProxyType(
            {case.case_id: case for case in cases}
        )
        self._config = config

    def __call__(
        self,
        model: torch.nn.Module,
        sample_id: str,
        step: int,
    ) -> PotentialEnergyTerms:
        if type(model) is not EnergyFormOperator:
            raise TypeError("model must be an exact EnergyFormOperator")
        case = self._cases.get(sample_id)
        if case is None:
            raise PermissionError(f"{sample_id} is not a registered TRAIN-fit case")
        return case_energy(model, case, self._config)


def case_energy(
    model: EnergyFormOperator,
    case: DogboneS13cGeometryCase,
    config: PotentialEnergyConfig,
    *,
    generator: torch.Generator | None = None,
    create_graph: bool = True,
) -> PotentialEnergyTerms:
    """Evaluate one case's energy terms at freshly drawn stratified points."""

    points, weights = stratified_triangle_quadrature(
        case.domain,
        config.points_per_cell,
        generator=generator,
    )
    _, gradient = model.displacement_and_gradient(
        query_xy=points.unsqueeze(0),
        parameters=case.parameters(),
        frame=case.frame(),
        lifting=case.lifting(),
        create_graph=create_graph,
    )
    return plane_stress_potential_energy(
        gradient[0],
        weights,
        case.material,
        config=config,
        reference_area=case.domain.area,
    )


__all__ = (
    "DogboneS13cEnergyObjective",
    "DogboneS13cGeometryCase",
    "DogboneS13cTrainingData",
    "case_energy",
    "conditioning_vector",
    "geometry_case",
    "materialize_dogbone_s13c_training_data",
)
