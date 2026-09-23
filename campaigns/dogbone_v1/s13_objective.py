"""TRAIN-only DogBone adapter from S10/S11 tensors to S12 objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import torch

from campaigns.dogbone_v1.s11_contingency import (
    BOUNDARY_TAG_NAMES,
    STRESS_REFERENCE_MPA,
)
from campaigns.dogbone_v1.s11_contingency_run import _outward_unit_normal
from campaigns.dogbone_v1.s11_contingency_tensors import ArchitectureCaseTensors
from piginot.contracts import MaterialSpec, Unit, hash_spec
from piginot.datasets import CollocationBatch, SemanticBCClass
from piginot.doe import EvidenceRole
from piginot.fem import FemRequest
from piginot.objectives import (
    CoordinateConversion,
    NondimensionalScales,
    ObjectiveResiduals,
    ObjectiveTerm,
    angular_momentum_residual,
    constitutive_residual,
    dirichlet_residual,
    displacement_gradient,
    strong_equilibrium_residual,
    traction_free_residual,
)
from piginot.operators import BoundaryConditionMode, PiGinotOperator
from piginot.physics._validation import DETERMINANT_MARGIN

_FEM_MATERIAL_MODEL = "finite_strain_plane_stress_neo_hookean"
_KERNEL_MATERIAL_MODEL = "compressible_neo_hookean"
_COORDINATE_CONVERSION = CoordinateConversion((1.0, 1.0))
_DIRICHLET_KINDS = frozenset(("dirichlet", "essential", "symmetry", "mixed"))
_COMPONENT_INDEX = {"x": 0, "y": 1}
S13_ORIENTATION_PENALTY_POLICY_ID = "s13-invalid-det-constitutive-row-v2"
S13_ORIENTATION_PENALTY_GAIN = 100.0
S13_ORIENTATION_RECOVERY_BUFFER = 0.1

DOGBONE_S13_OBJECTIVE_TERMS = (
    ObjectiveTerm.DATA_U,
    ObjectiveTerm.DATA_P,
    ObjectiveTerm.EQUILIBRIUM_STRONG,
    ObjectiveTerm.CONSTITUTIVE,
    ObjectiveTerm.ANGULAR,
    ObjectiveTerm.DIRICHLET,
    ObjectiveTerm.TRACTION_FREE,
)


@dataclass(frozen=True, slots=True)
class DogboneS13TrainingCase:
    """One exact active TRAIN case with supervised and collocation material."""

    tensors: ArchitectureCaseTensors
    request: FemRequest
    collocation: CollocationBatch

    def __post_init__(self) -> None:
        if type(self.tensors) is not ArchitectureCaseTensors:
            raise TypeError("tensors must be exact ArchitectureCaseTensors")
        if self.tensors.role is not EvidenceRole.TRAIN:
            raise PermissionError(
                f"{self.tensors.role.value} evidence cannot enter S13 training"
            )
        if type(self.request) is not FemRequest:
            raise TypeError("request must be an exact FemRequest")
        if type(self.collocation) is not CollocationBatch:
            raise TypeError("collocation must be an exact CollocationBatch")
        case_id = self.tensors.case_id
        stream = self.collocation.stream
        if self.request.request_id != case_id or stream.case_id != case_id:
            raise ValueError("tensor, request and collocation case identities differ")
        if stream.role is not EvidenceRole.TRAIN or any(
            point.role is not EvidenceRole.TRAIN for point in self.collocation.points
        ):
            raise PermissionError("non-TRAIN collocation cannot enter S13 training")
        if (
            stream.mesh_id != self.request.mesh.mesh_id
            or stream.mesh_content_hash != hash_spec(self.request.mesh, "content")
        ):
            raise ValueError("collocation mesh differs from its FEM request")
        if self.request.loads:
            raise ValueError(
                "DogBone S13 requires the registered zero-body-force request"
            )
        _physics_material(self.request.material)
        _scales(self.request)
        _point_masks(self.request, self.collocation)


class DogboneS13ObjectiveBuilder:
    """Build all seven graph-connected strong-HYBRID residuals."""

    __slots__ = ("_cases", "_sample_ids")

    def __init__(self, cases: tuple[DogboneS13TrainingCase, ...]) -> None:
        if type(cases) is not tuple or not cases:
            raise ValueError("cases must be a non-empty tuple")
        if any(type(case) is not DogboneS13TrainingCase for case in cases):
            raise TypeError("cases must contain exact DogboneS13TrainingCase values")
        sample_ids = tuple(case.tensors.case_id for case in cases)
        if sample_ids != tuple(sorted(set(sample_ids))):
            raise ValueError("training case IDs must be sorted and unique")
        first = cases[0]
        for case in cases[1:]:
            if case.tensors.boundary_tag_names != first.tensors.boundary_tag_names:
                raise ValueError("training cases must share the boundary-tag order")
            for name in (
                "boundary_xy",
                "boundary_tags",
                "boundary_normals",
                "boundary_weights",
                "boundary_mask",
                "query_xy",
                "target_u",
                "target_P",
            ):
                if (
                    getattr(case.tensors, name).shape
                    != getattr(first.tensors, name).shape
                ):
                    raise ValueError("training cases must share operator tensor shapes")
            if len(case.collocation.points) != len(first.collocation.points):
                raise ValueError(
                    "training cases must share the collocation point count"
                )
        self._cases: Mapping[str, DogboneS13TrainingCase] = MappingProxyType(
            {case.tensors.case_id: case for case in cases}
        )
        self._sample_ids = sample_ids

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self._sample_ids

    def __call__(
        self,
        model: torch.nn.Module,
        sample_ids: tuple[str, ...],
        step: int,
    ) -> ObjectiveResiduals:
        operator = _validated_operator(model)
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("step must be a positive integer")
        if type(sample_ids) is not tuple or not sample_ids:
            raise ValueError("sample_ids must be a non-empty tuple")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids must be unique within one batch")
        unknown = tuple(
            sample_id for sample_id in sample_ids if sample_id not in self._cases
        )
        if unknown:
            raise ValueError(f"unknown TRAIN sample ID(s): {', '.join(unknown)}")
        selected = tuple(self._cases[sample_id] for sample_id in sample_ids)
        _validate_model_inputs(operator, selected)

        tensors = tuple(case.tensors for case in selected)
        parameters = torch.stack(
            [case.parameters_for(operator.config.parameter_names) for case in tensors]
        )
        coordinate_rows = tuple(
            torch.tensor(
                [point.coordinates for point in case.collocation.points],
                dtype=parameters.dtype,
                device=parameters.device,
            )
            .unsqueeze(0)
            .requires_grad_(True)
            for case in selected
        )
        collocation_xy = torch.cat(coordinate_rows, dim=0)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            encoding = operator.encode_boundary(
                boundary_xy=torch.stack([case.boundary_xy for case in tensors]),
                boundary_tags=torch.stack([case.boundary_tags for case in tensors]),
                boundary_normals=torch.stack(
                    [case.boundary_normals for case in tensors]
                ),
                boundary_weights=torch.stack(
                    [case.boundary_weights for case in tensors]
                ),
                boundary_mask=torch.stack([case.boundary_mask for case in tensors]),
            )
            data_output = operator.decode_queries(
                encoding=encoding,
                query_xy=torch.stack([case.query_xy for case in tensors]),
                parameters=parameters,
            )
            physics_output = operator.decode_queries(
                encoding=encoding,
                query_xy=collocation_xy,
                parameters=parameters,
            )
            physics = _physics_residuals(
                selected,
                coordinate_rows,
                physics_output.u,
                physics_output.P,
            )

        values = {
            ObjectiveTerm.DATA_U: data_output.u
            - torch.stack([case.target_u for case in tensors]),
            ObjectiveTerm.DATA_P: data_output.P
            - torch.stack([case.target_P for case in tensors]),
            **physics,
        }
        if tuple(values) != DOGBONE_S13_OBJECTIVE_TERMS:
            raise RuntimeError("DogBone S13 objective term order changed")
        return ObjectiveResiduals(values)


def _physics_residuals(
    cases: tuple[DogboneS13TrainingCase, ...],
    coordinate_rows: tuple[torch.Tensor, ...],
    normalized_u: torch.Tensor,
    normalized_p: torch.Tensor,
) -> dict[ObjectiveTerm, torch.Tensor]:
    equilibrium: list[torch.Tensor] = []
    constitutive: list[torch.Tensor] = []
    angular: list[torch.Tensor] = []
    dirichlet: list[torch.Tensor] = []
    traction_free: list[torch.Tensor] = []
    for index, (case, coordinates) in enumerate(
        zip(cases, coordinate_rows, strict=True)
    ):
        scales = _scales(case.request)
        displacement = normalized_u[index : index + 1] * scales.displacement_mm
        stress = normalized_p[index : index + 1] * scales.stress_mpa
        gradient = displacement_gradient(
            displacement,
            coordinates,
            _COORDINATE_CONVERSION,
        )
        identity = torch.eye(2, dtype=gradient.dtype, device=gradient.device)
        deformation = identity + gradient
        interior_mask, free_mask, prescribed, component_mask, free_normals = (
            _point_masks(case.request, case.collocation)
        )
        interior_mask = interior_mask.to(device=stress.device)
        free_mask = free_mask.to(device=stress.device)
        prescribed = prescribed.to(dtype=stress.dtype, device=stress.device)
        component_mask = component_mask.to(device=stress.device)
        free_normals = free_normals.to(dtype=stress.dtype, device=stress.device)

        strong = strong_equilibrium_residual(
            stress,
            coordinates,
            torch.zeros_like(coordinates),
            scales,
            _COORDINATE_CONVERSION,
        )
        equilibrium.append(strong[:, interior_mask].reshape(-1, 2))
        constitutive.append(
            _constitutive_with_orientation_penalty(
                stress[:, interior_mask],
                gradient[:, interior_mask],
                _physics_material(case.request.material),
                scales,
            ).reshape(-1, 4)
        )
        angular.append(
            angular_momentum_residual(
                stress[:, interior_mask],
                deformation[:, interior_mask],
                scales,
            ).reshape(-1, 1)
        )
        selected_components = component_mask.any(dim=-1)[0]
        dirichlet.append(
            dirichlet_residual(
                displacement[:, selected_components],
                prescribed[:, selected_components],
                component_mask[:, selected_components],
                scales,
            ).reshape(-1)
        )
        traction_free.append(
            traction_free_residual(
                stress[:, free_mask],
                free_normals,
                scales,
            ).reshape(-1, 2)
        )
    return {
        ObjectiveTerm.EQUILIBRIUM_STRONG: torch.cat(equilibrium, dim=0),
        ObjectiveTerm.CONSTITUTIVE: torch.cat(constitutive, dim=0),
        ObjectiveTerm.ANGULAR: torch.cat(angular, dim=0),
        ObjectiveTerm.DIRICHLET: torch.cat(dirichlet, dim=0),
        ObjectiveTerm.TRACTION_FREE: torch.cat(traction_free, dim=0),
    }


def _constitutive_with_orientation_penalty(
    stress: torch.Tensor,
    gradient: torch.Tensor,
    material: MaterialSpec,
    scales: NondimensionalScales,
) -> torch.Tensor:
    """Keep each folded point in the loss without extending the S4 material law."""

    if stress.shape != gradient.shape or stress.shape[-2:] != (2, 2):
        raise ValueError("stress and gradient must contain aligned 2x2 fields")
    deformation = torch.eye(2, dtype=gradient.dtype, device=gradient.device) + gradient
    determinant = (
        deformation[..., 0, 0] * deformation[..., 1, 1]
        - deformation[..., 0, 1] * deformation[..., 1, 0]
    )
    if not bool(torch.isfinite(determinant).all().item()):
        raise ValueError("predicted deformation determinant must be finite")
    admissible = determinant > DETERMINANT_MARGIN
    if bool(admissible.all().item()):
        return constitutive_residual(stress, gradient, material, scales)

    flat_stress = stress.reshape(-1, 2, 2)
    flat_gradient = gradient.reshape(-1, 2, 2)
    flat_determinant = determinant.reshape(-1)
    valid_indices = torch.nonzero(admissible.reshape(-1)).reshape(-1)
    invalid_indices = torch.nonzero(~admissible.reshape(-1)).reshape(-1)
    residual = torch.zeros_like(flat_stress)
    if valid_indices.numel():
        valid_residual = constitutive_residual(
            flat_stress.index_select(0, valid_indices),
            flat_gradient.index_select(0, valid_indices),
            material,
            scales,
        )
        residual = residual.index_copy(0, valid_indices, valid_residual)
    violation = (
        S13_ORIENTATION_RECOVERY_BUFFER
        + DETERMINANT_MARGIN
        - flat_determinant.index_select(0, invalid_indices)
    ) * S13_ORIENTATION_PENALTY_GAIN
    penalty = violation[:, None, None].expand(-1, 2, 2)
    residual = residual.index_copy(0, invalid_indices, penalty)
    return residual.reshape(stress.shape)


def _validated_operator(model: torch.nn.Module) -> PiGinotOperator:
    if type(model) is not PiGinotOperator:
        raise TypeError("DogBone S13 requires an exact PiGinotOperator")
    if model.config.boundary_condition_mode is not BoundaryConditionMode.SOFT:
        raise ValueError("DogBone S13 uses the selected soft-Dirichlet architecture")
    parameters = tuple(model.parameters())
    if any(
        parameter.device.type != "cpu" or parameter.dtype is not torch.float64
        for parameter in parameters
    ):
        raise ValueError("DogBone S13 requires deterministic CPU float64 parameters")
    return model


def _validate_model_inputs(
    model: PiGinotOperator,
    cases: tuple[DogboneS13TrainingCase, ...],
) -> None:
    if model.config.boundary_tag_names != BOUNDARY_TAG_NAMES:
        raise ValueError("model boundary-tag order differs from DogBone S13")
    for case in cases:
        if case.tensors.boundary_tag_names != model.config.boundary_tag_names:
            raise ValueError("case boundary-tag order differs from the model")
        case.tensors.parameters_for(model.config.parameter_names)


def _physics_material(material: MaterialSpec) -> MaterialSpec:
    if type(material) is not MaterialSpec:
        raise TypeError("request material must be an exact MaterialSpec")
    if material.model != _FEM_MATERIAL_MODEL:
        raise ValueError(f"DogBone S13 material model must be {_FEM_MATERIAL_MODEL!r}")
    elastic_modulus = material.properties.get("elastic_modulus")
    if (
        elastic_modulus is None
        or elastic_modulus.unit is not Unit.MPA
        or elastic_modulus.value != STRESS_REFERENCE_MPA
    ):
        raise ValueError("DogBone S13 stress scale must match the 760 MPa material")
    return replace(material, model=_KERNEL_MATERIAL_MODEL)


def _scales(request: FemRequest) -> NondimensionalScales:
    x_coordinates = tuple(node[0] for node in request.mesh.nodes)
    length = max(x_coordinates) - min(x_coordinates)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("DogBone axial mesh extent must be finite and positive")
    prescribed_values: list[float] = []
    for condition in request.boundary_conditions:
        if (
            condition.boundary_tag == "right_grip"
            and condition.kind.casefold() in _DIRICHLET_KINDS
        ):
            for component, value in zip(
                condition.components,
                condition.values,
                strict=True,
            ):
                if component == "x":
                    if value.unit is not Unit.MM:
                        raise ValueError("right-grip displacement must use mm")
                    prescribed_values.append(float(value.value))
    if len(prescribed_values) != 1 or prescribed_values[0] <= 0.0:
        raise ValueError("DogBone S13 requires one positive right-grip x displacement")
    return NondimensionalScales(
        length_mm=length,
        displacement_mm=prescribed_values[0],
        stress_mpa=STRESS_REFERENCE_MPA,
    )


def _point_masks(
    request: FemRequest,
    collocation: CollocationBatch,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    point_count = len(collocation.points)
    interior = torch.tensor(
        [
            point.semantic_class is SemanticBCClass.INTERIOR
            for point in collocation.points
        ],
        dtype=torch.bool,
    )
    free = torch.tensor(
        [point.semantic_class is SemanticBCClass.FREE for point in collocation.points],
        dtype=torch.bool,
    )
    if not bool(interior.any().item()):
        raise ValueError("DogBone S13 collocation requires interior points")
    if not bool(free.any().item()):
        raise ValueError("DogBone S13 collocation requires traction-free points")

    prescribed = torch.zeros((1, point_count, 2), dtype=torch.float64)
    component_mask = torch.zeros((1, point_count, 2), dtype=torch.bool)
    for point_index, point in enumerate(collocation.points):
        for condition in request.boundary_conditions:
            if (
                condition.boundary_tag not in point.semantic_tags
                or condition.kind.casefold() not in _DIRICHLET_KINDS
            ):
                continue
            values = (
                tuple(0.0 for _ in condition.components)
                if not condition.values
                else tuple(_millimetres(value) for value in condition.values)
            )
            for component, value in zip(
                condition.components,
                values,
                strict=True,
            ):
                try:
                    component_index = _COMPONENT_INDEX[component]
                except KeyError as exc:
                    raise ValueError(
                        f"unsupported DogBone boundary component {component!r}"
                    ) from exc
                if component_mask[0, point_index, component_index] and not math.isclose(
                    float(prescribed[0, point_index, component_index]),
                    value,
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    raise ValueError("conflicting prescribed displacement values")
                component_mask[0, point_index, component_index] = True
                prescribed[0, point_index, component_index] = value
    if not bool(component_mask.any().item()):
        raise ValueError("DogBone S13 collocation requires Dirichlet components")

    normals: list[tuple[float, float]] = []
    for point, selected in zip(collocation.points, free.tolist(), strict=True):
        if not selected:
            continue
        facet_index = point.source_entity_index
        left_index, right_index = request.mesh.boundary_facets[facet_index]
        left = request.mesh.nodes[left_index]
        right = request.mesh.nodes[right_index]
        normals.append(
            _outward_unit_normal(
                request.mesh,
                facet_index,
                right[0] - left[0],
                right[1] - left[1],
            )
        )
    free_normals = torch.tensor(normals, dtype=torch.float64).unsqueeze(0)
    return interior, free, prescribed, component_mask, free_normals


def _millimetres(value: object) -> float:
    unit = getattr(value, "unit", None)
    number = getattr(value, "value", None)
    if (
        unit is not Unit.MM
        or isinstance(number, bool)
        or not isinstance(number, (int, float))
    ):
        raise ValueError("prescribed displacement values must use canonical mm")
    result = float(number)
    if not math.isfinite(result):
        raise ValueError("prescribed displacement values must be finite")
    return result


__all__ = (
    "DOGBONE_S13_OBJECTIVE_TERMS",
    "DogboneS13ObjectiveBuilder",
    "DogboneS13TrainingCase",
)
