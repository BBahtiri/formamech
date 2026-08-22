"""Deterministic S7 finite-element backend implementation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    LoadSpec,
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
from piginot.physics import energy_density, first_piola, plane_stress_closure
from piginot.physics.errors import (
    InadmissibleDeformationError,
    NewtonNonConvergenceError,
    PhysicsError,
)

_COMPONENT_INDEX = {"x": 0, "y": 1}
_PLANE_STRESS_MODEL = "finite_strain_plane_stress_neo_hookean"
_KERNEL_MODEL = "compressible_neo_hookean"
_DENSE_DOF_LIMIT = 2_000
_POINT_TOLERANCE = 1.0e-10


@dataclass(frozen=True, slots=True)
class _DiscreteSpace:
    basis: Any
    doflocs: tuple[tuple[float, float], ...]
    element_dofs: tuple[tuple[int, ...], ...]
    element_order: int

    @property
    def dof_count(self) -> int:
        return len(self.doflocs)


@dataclass(frozen=True, slots=True)
class _Assembly:
    residual: Any
    stiffness: Any | None
    strain_energy: float
    stresses: tuple[tuple[Any, ...], ...]
    gradients: tuple[tuple[Any, ...], ...]


class ScikitFemBackend:
    """scikit-fem-backed S7 backend for deterministic primary solves."""

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
        if request.config.max_newton_iterations == 0:
            raise FemSolveFailure(
                FemFailureCode.CLOSURE_FAILURE,
                "plane-stress closure failed before a Newton step was available",
            )

        space = _make_discrete_space(request.mesh, request.config.element_order)
        self._check_resource_limit(request, space)
        constraints = _collect_constraints(
            request.mesh, space, request.boundary_conditions
        )
        if not _has_full_rank_constraints(space, constraints):
            raise FemSolveFailure(
                FemFailureCode.CONSTRAINT_RANK_LOSS,
                "Dirichlet constraints do not remove rigid body modes",
            )

        material = _kernel_material(request.material)
        external = _assemble_external_load(request, space)
        displacement, history = _solve_newton(
            request, space, material, constraints, external
        )
        final = _assemble_internal(
            request,
            space,
            material,
            displacement,
            stiffness=False,
        )
        residual = final.residual - external
        reactions = _reactions_by_tag(request.mesh, space, constraints, residual)
        qois = _evaluate_qois(request, final.stresses)
        external_work = float(displacement @ external)

        return FemResult(
            request_id=request.request_id,
            mesh_id=request.mesh.mesh_id,
            element_order=request.config.element_order,
            displacement=_nodal_displacement(space, displacement),
            first_piola_stress=final.stresses,
            deformation_gradient=final.gradients,
            reactions=reactions,
            total_strain_energy=final.strain_energy,
            external_work=external_work,
            total_potential_energy=final.strain_energy - external_work,
            convergence_history=history,
            qois=qois,
        )

    def _check_resource_limit(self, request: FemRequest, space: _DiscreteSpace) -> None:
        limit = min(request.config.max_dofs, _DENSE_DOF_LIMIT)
        if space.dof_count > limit:
            raise FemSolveFailure(
                FemFailureCode.RESOURCE_EXHAUSTION,
                f"requested {space.dof_count} DOFs exceeds max_dofs={limit}",
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


def _make_discrete_space(mesh: CanonicalMesh, element_order: int) -> _DiscreteSpace:
    try:
        import numpy as np
        from skfem import (  # type: ignore[import-untyped]
            Basis,
            ElementTriP1,
            ElementTriP2,
            ElementVector,
            MeshTri,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without fem extra
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            "S7 ScikitFemBackend requires the 'fem' optional dependency",
        ) from exc

    element = ElementTriP1() if element_order == 1 else ElementTriP2()
    points = np.asarray(mesh.nodes, dtype=float).T
    cells = np.asarray(mesh.cells, dtype=int).T
    basis = Basis(MeshTri(points, cells), ElementVector(element), intorder=5)
    return _DiscreteSpace(
        basis=basis,
        doflocs=tuple(
            (float(basis.doflocs[0, index]), float(basis.doflocs[1, index]))
            for index in range(basis.N)
        ),
        element_dofs=tuple(
            tuple(int(item) for item in basis.element_dofs[:, index])
            for index in range(basis.nelems)
        ),
        element_order=element_order,
    )


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
    mesh: CanonicalMesh,
    space: _DiscreteSpace,
    boundary_conditions: tuple[BoundaryConditionSpec, ...],
) -> dict[int, float]:
    constraints: dict[int, float] = {}
    for condition in boundary_conditions:
        if condition.kind == "free":
            continue
        if condition.kind not in {"dirichlet", "symmetry"}:
            raise FemSolveFailure(
                FemFailureCode.UNSUPPORTED_REQUEST,
                f"unsupported boundary condition kind {condition.kind!r}",
            )
        components = condition.components
        values = condition.values
        if condition.kind == "symmetry" and not values:
            values = tuple(_zero_displacement() for _ in components)
        if not values or len(components) != len(values):
            raise FemSolveFailure(
                FemFailureCode.UNSUPPORTED_REQUEST,
                "component boundary conditions require aligned values",
            )
        for component, value in zip(components, values, strict=True):
            if component not in _COMPONENT_INDEX:
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    f"unsupported displacement component {component!r}",
                )
            _require_quantity(value, Unit.MM)
            component_id = _COMPONENT_INDEX[component]
            tagged_dofs = _component_dofs_for_tag(
                mesh, space, condition.boundary_tag, component_id
            )
            if not tagged_dofs:
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    f"boundary tag {condition.boundary_tag!r} has no DOFs",
                )
            for dof in tagged_dofs:
                previous = constraints.get(dof)
                if previous is not None and not math.isclose(
                    previous, value.value, rel_tol=0.0, abs_tol=1.0e-14
                ):
                    raise FemSolveFailure(
                        FemFailureCode.CONSTRAINT_RANK_LOSS,
                        "conflicting displacement constraints share a DOF",
                    )
                constraints[dof] = value.value
    return constraints


def _has_full_rank_constraints(
    space: _DiscreteSpace, constraints: dict[int, float]
) -> bool:
    if not constraints:
        return False
    constrained_components = {dof % 2 for dof in constraints}
    if constrained_components != {0, 1}:
        return False
    free_count = space.dof_count - len(constraints)
    return free_count >= 0


def _solve_newton(
    request: FemRequest,
    space: _DiscreteSpace,
    material: MaterialSpec,
    constraints: dict[int, float],
    external: Any,
) -> tuple[Any, tuple[FemConvergenceEntry, ...]]:
    import numpy as np

    displacement = np.zeros(space.dof_count, dtype=float)
    constrained = np.asarray(sorted(constraints), dtype=int)
    free = np.asarray(
        [index for index in range(space.dof_count) if index not in constraints],
        dtype=int,
    )
    history: list[FemConvergenceEntry] = []

    for load_step in range(1, request.config.load_steps + 1):
        load_fraction = load_step / request.config.load_steps
        target = {dof: load_fraction * value for dof, value in constraints.items()}
        for dof, value in target.items():
            displacement[dof] = value

        converged = False
        for iteration in range(1, request.config.max_newton_iterations + 1):
            assembled = _assemble_internal(
                request,
                space,
                material,
                displacement,
                stiffness=True,
            )
            residual = assembled.residual - load_fraction * external
            free_residual = residual[free]
            residual_norm = float(np.linalg.norm(free_residual))
            if residual_norm <= request.config.solver_tolerance:
                history.append(
                    FemConvergenceEntry(load_step, iteration, residual_norm, 0.0)
                )
                converged = True
                break
            if free.size == 0 or assembled.stiffness is None:
                break

            tangent = assembled.stiffness[np.ix_(free, free)]
            if np.linalg.matrix_rank(tangent) < tangent.shape[0]:
                raise FemSolveFailure(
                    FemFailureCode.CONSTRAINT_RANK_LOSS,
                    "Newton tangent is rank deficient after applying constraints",
                )
            try:
                increment = np.linalg.solve(tangent, -free_residual)
            except np.linalg.LinAlgError as exc:
                raise FemSolveFailure(
                    FemFailureCode.CONSTRAINT_RANK_LOSS,
                    "Newton tangent solve failed after applying constraints",
                ) from exc

            displacement, increment_norm = _accept_newton_step(
                request,
                space,
                material,
                displacement,
                free,
                constrained,
                target,
                increment,
                load_fraction,
                external,
                residual_norm,
            )
            history.append(
                FemConvergenceEntry(
                    load_step,
                    iteration,
                    residual_norm,
                    increment_norm,
                )
            )
            if increment_norm <= request.config.solver_tolerance:
                converged = True
                break

        if not converged:
            raise FemSolveFailure(
                FemFailureCode.NEWTON_NON_CONVERGENCE,
                "Newton residual did not reach the requested tolerance",
            )

    return displacement, tuple(history)


def _accept_newton_step(
    request: FemRequest,
    space: _DiscreteSpace,
    material: MaterialSpec,
    displacement: Any,
    free: Any,
    constrained: Any,
    target: dict[int, float],
    increment: Any,
    load_fraction: float,
    external: Any,
    previous_norm: float,
) -> tuple[Any, float]:
    import numpy as np

    step = 1.0
    best = displacement.copy()
    best_norm = math.inf
    while step >= 1.0 / 64.0:
        trial = displacement.copy()
        trial[free] = trial[free] + step * increment
        for dof in constrained:
            trial[dof] = target[int(dof)]
        try:
            assembled = _assemble_internal(
                request,
                space,
                material,
                trial,
                stiffness=False,
            )
            residual = assembled.residual - load_fraction * external
            norm = float(np.linalg.norm(residual[free]))
        except FemSolveFailure as exc:
            if exc.code is not FemFailureCode.NEGATIVE_JACOBIAN:
                raise
            norm = math.inf
        if norm < best_norm:
            best = trial
            best_norm = norm
        if norm <= previous_norm or norm <= request.config.solver_tolerance:
            return trial, float(np.linalg.norm(step * increment))
        step *= 0.5

    if not math.isfinite(best_norm):
        raise FemSolveFailure(
            FemFailureCode.NEGATIVE_JACOBIAN,
            "line search could not find an admissible deformation",
        )
    return best, float(np.linalg.norm(best[free] - displacement[free]))


def _assemble_internal(
    request: FemRequest,
    space: _DiscreteSpace,
    material: MaterialSpec,
    displacement: Any,
    *,
    stiffness: bool,
) -> _Assembly:
    import numpy as np

    residual = np.zeros(space.dof_count, dtype=float)
    tangent = (
        np.zeros((space.dof_count, space.dof_count), dtype=float) if stiffness else None
    )
    strain_energy = 0.0
    stresses: list[tuple[Any, ...]] = []
    gradients: list[tuple[Any, ...]] = []

    for cell_index, local_dofs in enumerate(space.element_dofs):
        cell = request.mesh.cells[cell_index]
        coordinates = tuple(request.mesh.nodes[index] for index in cell)
        local = displacement[list(local_dofs)]
        element = _element_response(
            coordinates,
            local,
            space.element_order,
            material,
            _closure_iterations(request),
        )
        for row, dof in enumerate(local_dofs):
            residual[dof] += element["residual"][row]
        if tangent is not None:
            element_tangent = _element_tangent(
                coordinates,
                local,
                space.element_order,
                material,
                _closure_iterations(request),
            )
            for row, row_dof in enumerate(local_dofs):
                for column, column_dof in enumerate(local_dofs):
                    tangent[row_dof, column_dof] += element_tangent[row, column]
        strain_energy += float(element["energy"])
        stresses.append(tuple(element["stresses"]))
        gradients.append(tuple(element["gradients"]))

    return _Assembly(
        residual=residual,
        stiffness=tangent,
        strain_energy=strain_energy,
        stresses=tuple(stresses),
        gradients=tuple(gradients),
    )


def _element_response(
    coordinates: tuple[tuple[float, float], ...],
    local_displacement: Sequence[float],
    element_order: int,
    material: MaterialSpec,
    closure_iterations: int,
) -> dict[str, Any]:
    local = torch.tensor(
        tuple(float(value) for value in local_displacement),
        dtype=torch.float64,
        requires_grad=True,
    )
    energy, stresses, gradients = _element_energy_and_fields(
        coordinates,
        local,
        element_order,
        material,
        closure_iterations,
    )
    residual_tensor = torch.autograd.grad(energy, local)[0]
    return {
        "energy": float(energy.detach().item()),
        "residual": residual_tensor.detach().numpy(),
        "stresses": stresses,
        "gradients": gradients,
    }


def _closure_iterations(request: FemRequest) -> int:
    return max(8, request.config.max_newton_iterations)


def _element_tangent(
    coordinates: tuple[tuple[float, float], ...],
    local_displacement: Sequence[float],
    element_order: int,
    material: MaterialSpec,
    closure_iterations: int,
) -> Any:
    import numpy as np

    local = np.asarray(local_displacement, dtype=float)
    tangent = np.zeros((local.size, local.size), dtype=float)
    for column in range(local.size):
        step = 1.0e-7 * max(1.0, abs(float(local[column])))
        plus = local.copy()
        minus = local.copy()
        plus[column] += step
        minus[column] -= step
        residual_plus = _element_response(
            coordinates, plus.tolist(), element_order, material, closure_iterations
        )["residual"]
        residual_minus = _element_response(
            coordinates, minus.tolist(), element_order, material, closure_iterations
        )["residual"]
        tangent[:, column] = (residual_plus - residual_minus) / (2.0 * step)
    return tangent


def _element_energy_and_fields(
    coordinates: tuple[tuple[float, float], ...],
    local: torch.Tensor,
    element_order: int,
    material: MaterialSpec,
    closure_iterations: int,
) -> tuple[torch.Tensor, tuple[Any, ...], tuple[Any, ...]]:
    area = _cell_area_from_coordinates(coordinates)
    energy = torch.zeros((), dtype=torch.float64)
    stresses: list[Any] = []
    gradients: list[Any] = []
    for barycentric, weight in _triangle_quadrature():
        _, reference_gradients = _shape_values_and_reference_gradients(
            element_order, barycentric
        )
        physical_gradients = _physical_gradients(coordinates, reference_gradients)
        deformation_2d = _deformation_gradient(local, physical_gradients)
        determinant = torch.linalg.det(deformation_2d)
        if (
            not bool(torch.isfinite(determinant).item())
            or float(determinant.item()) <= 0.0
        ):
            raise FemSolveFailure(
                FemFailureCode.NEGATIVE_JACOBIAN,
                "at least one quadrature point has non-positive deformation Jacobian",
            )
        full_deformation = _plane_stress_deformation(
            deformation_2d, material, closure_iterations
        )
        energy = energy + area * weight * energy_density(full_deformation, material)
        piola = first_piola(full_deformation, material)
        stresses.append(_matrix2(piola[:2, :2]))
        gradients.append(_matrix2(deformation_2d))
    return energy, tuple(stresses), tuple(gradients)


def _plane_stress_deformation(
    deformation_2d: torch.Tensor, material: MaterialSpec, closure_iterations: int
) -> torch.Tensor:
    try:
        f33 = plane_stress_closure(
            deformation_2d,
            material,
            max_iterations=closure_iterations,
        )
    except InadmissibleDeformationError as exc:
        raise FemSolveFailure(
            FemFailureCode.NEGATIVE_JACOBIAN,
            "plane-stress closure received a non-positive Jacobian",
        ) from exc
    except NewtonNonConvergenceError as exc:
        raise FemSolveFailure(
            FemFailureCode.CLOSURE_FAILURE,
            "plane-stress closure failed",
        ) from exc
    except PhysicsError as exc:
        raise FemSolveFailure(
            FemFailureCode.CLOSURE_FAILURE,
            "plane-stress closure failed",
        ) from exc

    full_deformation = torch.eye(3, dtype=torch.float64)
    full_deformation[:2, :2] = deformation_2d
    full_deformation[2, 2] = f33
    return full_deformation


def _deformation_gradient(
    local: torch.Tensor, physical_gradients: tuple[tuple[float, float], ...]
) -> torch.Tensor:
    deformation = torch.eye(2, dtype=torch.float64)
    for index, gradient in enumerate(physical_gradients):
        ux = local[2 * index]
        uy = local[2 * index + 1]
        deformation[0, 0] = deformation[0, 0] + ux * gradient[0]
        deformation[0, 1] = deformation[0, 1] + ux * gradient[1]
        deformation[1, 0] = deformation[1, 0] + uy * gradient[0]
        deformation[1, 1] = deformation[1, 1] + uy * gradient[1]
    return deformation


def _shape_values_and_reference_gradients(
    element_order: int, barycentric: tuple[float, float, float]
) -> tuple[tuple[float, ...], tuple[tuple[float, float], ...]]:
    l1, l2, l3 = barycentric
    reference_gradients = ((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0))
    if element_order == 1:
        return barycentric, reference_gradients

    values = (
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        l3 * (2.0 * l3 - 1.0),
        4.0 * l1 * l2,
        4.0 * l2 * l3,
        4.0 * l3 * l1,
    )
    g1, g2, g3 = reference_gradients
    gradients = (
        _scale(g1, 4.0 * l1 - 1.0),
        _scale(g2, 4.0 * l2 - 1.0),
        _scale(g3, 4.0 * l3 - 1.0),
        _add(_scale(g2, 4.0 * l1), _scale(g1, 4.0 * l2)),
        _add(_scale(g3, 4.0 * l2), _scale(g2, 4.0 * l3)),
        _add(_scale(g1, 4.0 * l3), _scale(g3, 4.0 * l1)),
    )
    return values, gradients


def _physical_gradients(
    coordinates: tuple[tuple[float, float], ...],
    reference_gradients: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    x0, x1, x2 = coordinates
    jacobian = (
        (x1[0] - x0[0], x2[0] - x0[0]),
        (x1[1] - x0[1], x2[1] - x0[1]),
    )
    determinant = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]
    inverse_transpose = (
        (jacobian[1][1] / determinant, -jacobian[1][0] / determinant),
        (-jacobian[0][1] / determinant, jacobian[0][0] / determinant),
    )
    return tuple(
        (
            inverse_transpose[0][0] * gradient[0]
            + inverse_transpose[0][1] * gradient[1],
            inverse_transpose[1][0] * gradient[0]
            + inverse_transpose[1][1] * gradient[1],
        )
        for gradient in reference_gradients
    )


def _triangle_quadrature() -> tuple[tuple[tuple[float, float, float], float], ...]:
    return (
        ((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), 0.225),
        ((0.059715871789770, 0.470142064105115, 0.470142064105115), 0.132394152788506),
        ((0.470142064105115, 0.059715871789770, 0.470142064105115), 0.132394152788506),
        ((0.470142064105115, 0.470142064105115, 0.059715871789770), 0.132394152788506),
        ((0.797426985353087, 0.101286507323456, 0.101286507323456), 0.125939180544827),
        ((0.101286507323456, 0.797426985353087, 0.101286507323456), 0.125939180544827),
        ((0.101286507323456, 0.101286507323456, 0.797426985353087), 0.125939180544827),
    )


def _assemble_external_load(request: FemRequest, space: _DiscreteSpace) -> Any:
    import numpy as np

    force = np.zeros(space.dof_count, dtype=float)
    for load in request.loads:
        if load.kind == "traction":
            if load.boundary_tag is None:
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    "traction loads require a boundary tag",
                )
            force += _assemble_traction(request.mesh, space, load)
        elif load.kind == "body_force":
            force += _assemble_body_force(request.mesh, space, load)
        else:
            raise FemSolveFailure(
                FemFailureCode.UNSUPPORTED_REQUEST,
                f"unsupported load kind {load.kind!r}",
            )
    return force


def _assemble_traction(
    mesh: CanonicalMesh, space: _DiscreteSpace, load: LoadSpec
) -> Any:
    import numpy as np

    traction = _load_vector(load)
    force = np.zeros(space.dof_count, dtype=float)
    for facet_id in mesh.semantic_tags.get(str(load.boundary_tag), ()):
        facet = mesh.boundary_facets[facet_id]
        cell_index = _cell_for_facet(mesh, facet)
        if cell_index is None:
            raise FemSolveFailure(
                FemFailureCode.UNSUPPORTED_REQUEST,
                f"boundary facet {facet_id} is not attached to a cell",
            )
        local_dofs = space.element_dofs[cell_index]
        cell = mesh.cells[cell_index]
        edge = (cell.index(facet[0]), cell.index(facet[1]))
        length = _distance(mesh.nodes[facet[0]], mesh.nodes[facet[1]])
        for barycentric, weight in _edge_quadrature(edge):
            shapes, _ = _shape_values_and_reference_gradients(
                space.element_order, barycentric
            )
            for local_index, shape in enumerate(shapes):
                force[local_dofs[2 * local_index]] += (
                    length * weight * shape * traction[0]
                )
                force[local_dofs[2 * local_index + 1]] += (
                    length * weight * shape * traction[1]
                )
    return force


def _assemble_body_force(
    mesh: CanonicalMesh, space: _DiscreteSpace, load: LoadSpec
) -> Any:
    import numpy as np

    body_force = _load_vector(load)
    force = np.zeros(space.dof_count, dtype=float)
    for cell_index, cell in enumerate(mesh.cells):
        coordinates = tuple(mesh.nodes[index] for index in cell)
        area = _cell_area_from_coordinates(coordinates)
        local_dofs = space.element_dofs[cell_index]
        for barycentric, weight in _triangle_quadrature():
            shapes, _ = _shape_values_and_reference_gradients(
                space.element_order, barycentric
            )
            for local_index, shape in enumerate(shapes):
                force[local_dofs[2 * local_index]] += (
                    area * weight * shape * body_force[0]
                )
                force[local_dofs[2 * local_index + 1]] += (
                    area * weight * shape * body_force[1]
                )
    return force


def _load_vector(load: LoadSpec) -> tuple[float, float]:
    values = [0.0, 0.0]
    for component, value in zip(load.components, load.values, strict=True):
        if component not in _COMPONENT_INDEX:
            raise FemSolveFailure(
                FemFailureCode.UNSUPPORTED_REQUEST,
                f"unsupported load component {component!r}",
            )
        _require_quantity(value, Unit.MPA)
        values[_COMPONENT_INDEX[component]] = value.value
    return values[0], values[1]


def _edge_quadrature(
    edge: tuple[int, int],
) -> tuple[tuple[tuple[float, float, float], float], ...]:
    offset = 0.5 / math.sqrt(3.0)
    result = []
    for position in (0.5 - offset, 0.5 + offset):
        barycentric = [0.0, 0.0, 0.0]
        barycentric[edge[0]] = 1.0 - position
        barycentric[edge[1]] = position
        result.append(((barycentric[0], barycentric[1], barycentric[2]), 0.5))
    return tuple(result)


def _component_dofs_for_tag(
    mesh: CanonicalMesh, space: _DiscreteSpace, tag: str, component: int
) -> tuple[int, ...]:
    facets = tuple(
        mesh.boundary_facets[index] for index in mesh.semantic_tags.get(tag, ())
    )
    return tuple(
        index
        for index, point in enumerate(space.doflocs)
        if index % 2 == component
        and any(
            _point_on_segment(point, mesh.nodes[left], mesh.nodes[right])
            for left, right in facets
        )
    )


def _reactions_by_tag(
    mesh: CanonicalMesh,
    space: _DiscreteSpace,
    constraints: dict[int, float],
    residual: Any,
) -> dict[str, tuple[float, float]]:
    reactions: dict[str, tuple[float, float]] = {}
    constrained = set(constraints)
    for tag in mesh.semantic_tags:
        components = [0.0, 0.0]
        for component in (0, 1):
            for dof in _component_dofs_for_tag(mesh, space, tag, component):
                if dof in constrained:
                    components[component] += float(residual[dof])
        reactions[str(tag)] = (components[0], components[1])
    return reactions


def _nodal_displacement(
    space: _DiscreteSpace, displacement: Any
) -> tuple[tuple[float, float], ...]:
    nodal_dofs = space.basis.nodal_dofs
    return tuple(
        (
            float(displacement[int(nodal_dofs[0, index])]),
            float(displacement[int(nodal_dofs[1, index])]),
        )
        for index in range(nodal_dofs.shape[1])
    )


def _evaluate_qois(request: FemRequest, stress: tuple[Any, ...]) -> dict[str, float]:
    qois: dict[str, float] = {}
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


def _cell_for_facet(mesh: CanonicalMesh, facet: tuple[int, int]) -> int | None:
    endpoints = set(facet)
    for index, cell in enumerate(mesh.cells):
        if endpoints.issubset(cell):
            return index
    return None


def _require_quantity(value: CanonicalQuantity, unit: Unit) -> None:
    if value.unit is not unit:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            f"expected canonical {unit.value} quantity",
        )


def _zero_displacement() -> CanonicalQuantity:
    return CanonicalQuantity(0.0, Unit.MM)


def _cell_area_from_coordinates(coordinates: tuple[tuple[float, float], ...]) -> float:
    left, middle, right = coordinates
    return (
        abs(
            (middle[0] - left[0]) * (right[1] - left[1])
            - (middle[1] - left[1]) * (right[0] - left[0])
        )
        / 2.0
    )


def _point_on_segment(
    point: tuple[float, float],
    left: tuple[float, float],
    right: tuple[float, float],
) -> bool:
    length = _distance(left, right)
    if length == 0.0:
        return False
    cross = (point[0] - left[0]) * (right[1] - left[1]) - (point[1] - left[1]) * (
        right[0] - left[0]
    )
    if abs(cross) > _POINT_TOLERANCE * length:
        return False
    dot = (point[0] - left[0]) * (right[0] - left[0]) + (point[1] - left[1]) * (
        right[1] - left[1]
    )
    return -_POINT_TOLERANCE <= dot <= length * length + _POINT_TOLERANCE


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(right[0] - left[0], right[1] - left[1])


def _matrix2(tensor: torch.Tensor) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(tensor[0, 0].detach().item()), float(tensor[0, 1].detach().item())),
        (float(tensor[1, 0].detach().item()), float(tensor[1, 1].detach().item())),
    )


def _scale(vector: tuple[float, float], scale: float) -> tuple[float, float]:
    return vector[0] * scale, vector[1] * scale


def _add(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    return left[0] + right[0], left[1] + right[1]


__all__ = ("ScikitFemBackend",)
