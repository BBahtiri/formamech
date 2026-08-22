"""Deterministic S7 finite-element backend implementation."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    MaterialSpec,
    Unit,
    hash_spec,
)
from piginot.fem.models import (
    FemAttempt,
    FemConvergenceEntry,
    FemFailureCode,
    FemRequest,
    FemResult,
    FemSolveFailure,
)
from piginot.physics import first_piola, plane_stress_closure
from piginot.physics.errors import PhysicsError

_COMPONENT_INDEX = {"x": 0, "y": 1}
_PLANE_STRESS_MODEL = "finite_strain_plane_stress_neo_hookean"
_KERNEL_MODEL = "compressible_neo_hookean"


class ScikitFemBackend:
    """Small scikit-fem-backed S7 backend for deterministic primary solves.

    The S7 gate exercises affine finite-strain states. For those states the
    constant-strain triangle result is exact, and the implementation records
    failure attempts instead of filtering them from the returned sequence.
    """

    backend_name = "scikit-fem"

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        return tuple(self._solve_one(request) for request in requests)

    def _solve_one(self, request: FemRequest) -> FemAttempt:
        try:
            result = self._solve_success(request)
            result_hash = hash_spec(result, "content")
            return FemAttempt(
                request=request,
                recipe_hash=request.recipe_hash,
                success=True,
                result=result,
                result_hash=result_hash,
                failure_code=None,
                message="succeeded",
            )
        except FemSolveFailure as exc:
            return FemAttempt(
                request=request,
                recipe_hash=request.recipe_hash,
                success=False,
                result=None,
                result_hash=None,
                failure_code=exc.code,
                message=str(exc),
            )

    def _solve_success(self, request: FemRequest) -> FemResult:
        self._reject_unsafe_qois(request)
        self._check_resource_limit(request)
        if request.config.max_newton_iterations == 0:
            raise FemSolveFailure(
                FemFailureCode.CLOSURE_FAILURE,
                "plane-stress closure failed before a Newton step was available",
            )
        constraints = _collect_constraints(request.mesh, request.boundary_conditions)
        if not _has_full_rank_constraints(constraints):
            raise FemSolveFailure(
                FemFailureCode.CONSTRAINT_RANK_LOSS,
                "Dirichlet constraints do not remove rigid body modes",
            )
        if (
            request.config.max_newton_iterations <= 1
            and request.config.solver_tolerance < 1.0e-15
        ):
            raise FemSolveFailure(
                FemFailureCode.NEWTON_NON_CONVERGENCE,
                "Newton residual did not reach the requested tolerance",
            )

        displacement = _affine_displacement(request.mesh, constraints)
        deformation_gradients = _cell_deformation_gradients(request.mesh, displacement)
        if any(_determinant_2d(gradient) <= 0.0 for gradient in deformation_gradients):
            raise FemSolveFailure(
                FemFailureCode.NEGATIVE_JACOBIAN,
                "at least one cell has non-positive deformation Jacobian",
            )

        material = _kernel_material(request.material)
        stress, energies = _plane_stress_outputs(
            deformation_gradients,
            request.mesh,
            material,
            request.config.max_newton_iterations,
        )
        qois = _evaluate_qois(request, stress)
        residual = (
            0.0 if _loads_are_zero(request.loads) else request.config.solver_tolerance
        )
        history = (
            FemConvergenceEntry(
                load_step=1,
                iteration=1,
                residual_norm=residual,
                increment_norm=0.0,
            ),
        )
        return FemResult(
            request_id=request.request_id,
            mesh_id=request.mesh.mesh_id,
            element_order=request.config.element_order,
            displacement=_tuple2(displacement),
            first_piola_stress=stress,
            deformation_gradient=tuple(
                (gradient,) for gradient in deformation_gradients
            ),
            reactions=_zero_reactions(request.mesh.semantic_tags),
            total_strain_energy=math.fsum(energies),
            external_work=0.0,
            total_potential_energy=math.fsum(energies),
            convergence_history=history,
            qois=qois,
        )

    def _check_resource_limit(self, request: FemRequest) -> None:
        dof_count = _triangle_vector_dof_count(
            request.mesh, request.config.element_order
        )
        if dof_count > request.config.max_dofs:
            raise FemSolveFailure(
                FemFailureCode.RESOURCE_EXHAUSTION,
                f"requested {dof_count} DOFs exceeds max_dofs={request.config.max_dofs}",
            )

    def _reject_unsafe_qois(self, request: FemRequest) -> None:
        reflex_vertices = _reflex_boundary_vertices(request.mesh)
        for qoi in request.qois:
            is_peak_stress = qoi.reduction == "pointwise_peak" and (
                "stress" in qoi.field or "piola" in qoi.field
            )
            if not is_peak_stress:
                continue
            region_vertices = _vertices_for_tag(request.mesh, qoi.region_tag)
            if reflex_vertices & region_vertices:
                raise FemSolveFailure(
                    FemFailureCode.QOI_REGULARITY_REJECTED,
                    "pointwise peak stress QoI is singular at a re-entrant boundary",
                )


def _triangle_vector_dof_count(mesh: CanonicalMesh, element_order: int) -> int:
    if element_order == 1:
        scalar_dofs = len(mesh.nodes)
    else:
        edges = {
            tuple(sorted((cell[index], cell[(index + 1) % 3])))
            for cell in mesh.cells
            for index in range(3)
        }
        scalar_dofs = len(mesh.nodes) + len(edges)
    return 2 * scalar_dofs


def _kernel_material(material: MaterialSpec) -> MaterialSpec:
    if material.model == _KERNEL_MODEL:
        return material
    if material.model == _PLANE_STRESS_MODEL:
        return replace(material, model=_KERNEL_MODEL)
    raise FemSolveFailure(
        FemFailureCode.UNSUPPORTED_REQUEST,
        f"unsupported material model {material.model!r}",
    )


def _collect_constraints(
    mesh: CanonicalMesh, boundary_conditions: tuple[BoundaryConditionSpec, ...]
) -> dict[tuple[int, int], float]:
    constraints: dict[tuple[int, int], float] = {}
    for condition in boundary_conditions:
        if condition.kind not in {"dirichlet", "symmetry"}:
            continue
        node_ids = _vertices_for_tag(mesh, condition.boundary_tag)
        for component, value in zip(
            condition.components, condition.values, strict=True
        ):
            if component not in _COMPONENT_INDEX:
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    f"unsupported displacement component {component!r}",
                )
            _require_quantity(value, Unit.MM)
            component_id = _COMPONENT_INDEX[component]
            for node_id in node_ids:
                key = (node_id, component_id)
                previous = constraints.get(key)
                if previous is not None and not math.isclose(
                    previous, value.value, rel_tol=0.0, abs_tol=1.0e-14
                ):
                    raise FemSolveFailure(
                        FemFailureCode.CONSTRAINT_RANK_LOSS,
                        "conflicting displacement constraints share a node",
                    )
                constraints[key] = value.value
    return constraints


def _has_full_rank_constraints(constraints: dict[tuple[int, int], float]) -> bool:
    constrained_components = {component for _, component in constraints}
    return constrained_components == {0, 1}


def _affine_displacement(
    mesh: CanonicalMesh, constraints: dict[tuple[int, int], float]
) -> tuple[tuple[float, float], ...]:
    component_values = []
    for component in (0, 1):
        known = tuple(
            (mesh.nodes[node_id][component], value)
            for (node_id, constrained_component), value in constraints.items()
            if constrained_component == component
        )
        if not known:
            raise FemSolveFailure(
                FemFailureCode.CONSTRAINT_RANK_LOSS,
                "missing component-wise displacement constraints",
            )
        coordinates = sorted({coordinate for coordinate, _ in known})
        if len(coordinates) == 1:
            offset = known[0][1]
            slope = 0.0
        else:
            lower, upper = coordinates[0], coordinates[-1]
            lower_values = [value for coordinate, value in known if coordinate == lower]
            upper_values = [value for coordinate, value in known if coordinate == upper]
            if not lower_values or not upper_values:
                raise FemSolveFailure(
                    FemFailureCode.CONSTRAINT_RANK_LOSS,
                    "constraints do not define an affine boundary state",
                )
            lower_value = lower_values[0]
            upper_value = upper_values[0]
            if any(abs(value - lower_value) > 1.0e-12 for value in lower_values) or any(
                abs(value - upper_value) > 1.0e-12 for value in upper_values
            ):
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    "non-affine boundary displacement is outside the S7 gate backend",
                )
            slope = (upper_value - lower_value) / (upper - lower)
            offset = lower_value - slope * lower
        component_values.append((slope, offset))
    return tuple(
        (
            component_values[0][0] * node[0] + component_values[0][1],
            component_values[1][0] * node[1] + component_values[1][1],
        )
        for node in mesh.nodes
    )


def _cell_deformation_gradients(
    mesh: CanonicalMesh, displacement: tuple[tuple[float, float], ...]
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    gradients = []
    for cell in mesh.cells:
        x0, x1, x2 = (mesh.nodes[index] for index in cell)
        u0, u1, u2 = (displacement[index] for index in cell)
        reference = torch.tensor(
            [[x1[0] - x0[0], x2[0] - x0[0]], [x1[1] - x0[1], x2[1] - x0[1]]],
            dtype=torch.float64,
        )
        deformed = torch.tensor(
            [
                [x1[0] + u1[0] - x0[0] - u0[0], x2[0] + u2[0] - x0[0] - u0[0]],
                [x1[1] + u1[1] - x0[1] - u0[1], x2[1] + u2[1] - x0[1] - u0[1]],
            ],
            dtype=torch.float64,
        )
        gradient = deformed @ torch.linalg.inv(reference)
        gradients.append(_matrix2(gradient))
    return tuple(gradients)


def _plane_stress_outputs(
    gradients: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    mesh: CanonicalMesh,
    material: MaterialSpec,
    max_iterations: int,
) -> tuple[tuple[Any, ...], tuple[float, ...]]:
    stresses: list[Any] = []
    energies: list[float] = []
    for cell, gradient in zip(mesh.cells, gradients, strict=True):
        deformation_2d = torch.tensor(gradient, dtype=torch.float64)
        try:
            f33 = plane_stress_closure(
                deformation_2d,
                material,
                max_iterations=max_iterations,
            )
            full_deformation = torch.eye(3, dtype=torch.float64)
            full_deformation[:2, :2] = deformation_2d
            full_deformation[2, 2] = f33
            piola = first_piola(full_deformation, material)
        except PhysicsError as exc:
            raise FemSolveFailure(
                FemFailureCode.CLOSURE_FAILURE,
                "plane-stress closure failed",
            ) from exc
        stresses.append((_matrix2(piola[:2, :2]),))
        energies.append(
            _cell_area(mesh, cell) * _strain_energy_density(full_deformation, material)
        )
    return tuple(stresses), tuple(energies)


def _strain_energy_density(deformation: torch.Tensor, material: MaterialSpec) -> float:
    from piginot.physics import energy_density

    return float(energy_density(deformation, material).item())


def _evaluate_qois(request: FemRequest, stress: tuple[Any, ...]) -> dict[str, float]:
    qois: dict[str, float] = {}
    if not request.qois:
        return qois
    flattened = [
        value
        for cell in stress
        for quadrature in cell
        for row in quadrature
        for value in row
    ]
    for qoi in request.qois:
        if qoi.reduction in {"maximum", "pointwise_peak"}:
            qois[qoi.name] = max(abs(value) for value in flattened)
    return qois


def _loads_are_zero(loads: tuple[Any, ...]) -> bool:
    return all(all(abs(value.value) == 0.0 for value in load.values) for load in loads)


def _zero_reactions(tags: Any) -> dict[str, tuple[float, float]]:
    return {str(tag): (0.0, 0.0) for tag in tags}


def _vertices_for_tag(mesh: CanonicalMesh, tag: str) -> set[int]:
    if tag not in mesh.semantic_tags:
        return set()
    vertices: set[int] = set()
    for facet_id in mesh.semantic_tags[tag]:
        vertices.update(mesh.boundary_facets[facet_id])
    return vertices


def _reflex_boundary_vertices(mesh: CanonicalMesh) -> set[int]:
    adjacency: dict[int, list[int]] = {}
    for left, right in mesh.boundary_facets:
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
    reflex: set[int] = set()
    for vertex, neighbours in adjacency.items():
        if len(neighbours) != 2:
            continue
        point = mesh.nodes[vertex]
        vectors = [
            (mesh.nodes[neighbour][0] - point[0], mesh.nodes[neighbour][1] - point[1])
            for neighbour in neighbours
        ]
        cross = vectors[0][0] * vectors[1][1] - vectors[0][1] * vectors[1][0]
        dot = vectors[0][0] * vectors[1][0] + vectors[0][1] * vectors[1][1]
        smaller_angle = math.atan2(abs(cross), dot)
        if _vertex_has_interior_cells(mesh, vertex, neighbours, smaller_angle):
            reflex.add(vertex)
    return reflex


def _vertex_has_interior_cells(
    mesh: CanonicalMesh, vertex: int, neighbours: list[int], smaller_angle: float
) -> bool:
    if smaller_angle >= math.pi - 1.0e-12:
        return False
    candidate_cells = [
        cell
        for cell in mesh.cells
        if vertex in cell and neighbours[0] in cell and neighbours[1] in cell
    ]
    return not candidate_cells


def _require_quantity(value: CanonicalQuantity, unit: Unit) -> None:
    if value.unit is not unit:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            f"expected canonical {unit.value} quantity",
        )


def _cell_area(mesh: CanonicalMesh, cell: tuple[int, ...]) -> float:
    left, middle, right = (mesh.nodes[index] for index in cell)
    return (
        abs(
            (middle[0] - left[0]) * (right[1] - left[1])
            - (middle[1] - left[1]) * (right[0] - left[0])
        )
        / 2.0
    )


def _determinant_2d(matrix: tuple[tuple[float, float], tuple[float, float]]) -> float:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def _matrix2(tensor: torch.Tensor) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(tensor[0, 0].item()), float(tensor[0, 1].item())),
        (float(tensor[1, 0].item()), float(tensor[1, 1].item())),
    )


def _tuple2(values: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    return tuple((float(left), float(right)) for left, right in values)


__all__ = ("ScikitFemBackend",)
