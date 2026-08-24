"""Deterministic scikit-fem finite-strain plane-stress backend."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import bmat, csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import (  # type: ignore[import-untyped]
    MatrixRankWarning,
    spsolve,
)
from skfem import (  # type: ignore[import-untyped]
    Basis,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    MeshTri,
)

from piginot.contracts import MaterialSpec, Unit, canonical_json_bytes, hash_spec
from piginot.fem.assembly import AssemblyState, assemble_state, recover_stress
from piginot.fem.binding import (
    BoundConstraints,
    assemble_dead_loads,
    bind_constraints,
    canonical_facet_map,
    facets_for_tag,
)
from piginot.fem.body_forces import BodyForceProvider
from piginot.fem.field_providers import (
    DirichletFieldProvider,
    TractionFieldProvider,
    immutable_provider_registry,
)
from piginot.fem.models import (
    FemAttempt,
    FemConvergenceEntry,
    FemFailureCode,
    FemRequest,
    FemResult,
    FemSolveFailure,
)
from piginot.fem.regularity import peak_stress_screen
from piginot.store import ObjectStore

FloatArray = NDArray[np.float64]
_PLANE_STRESS_MODEL = "finite_strain_plane_stress_neo_hookean"
_KERNEL_MODEL = "compressible_neo_hookean"
_QUADRATURE_ORDER = 5
_MIN_LINE_SEARCH_STEP = 2.0**-20


class ScikitFemBackend:
    """Solve S7 requests using sparse Total-Lagrangian FEM assembly."""

    backend_name = "scikit-fem"

    def __init__(
        self,
        *,
        body_force_providers: tuple[BodyForceProvider, ...] = (),
        dirichlet_field_providers: tuple[DirichletFieldProvider, ...] = (),
        traction_field_providers: tuple[TractionFieldProvider, ...] = (),
        store: ObjectStore | None = None,
    ) -> None:
        self._body_force_providers = immutable_provider_registry(
            body_force_providers,
            label="body-force",
        )
        self._dirichlet_field_providers = immutable_provider_registry(
            dirichlet_field_providers,
            label="Dirichlet-field",
        )
        self._traction_field_providers = immutable_provider_registry(
            traction_field_providers,
            label="traction-field",
        )
        self._store = store

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        """Return one visible outcome for every supplied request."""

        return tuple(self._solve_one(request) for request in requests)

    def _solve_one(self, request: FemRequest) -> FemAttempt:
        try:
            result = self._solve_success(request)
            store_hash = None
            if self._store is not None:
                payload = canonical_json_bytes({"domain": "content", "payload": result})
                store_hash = self._store.put(
                    payload,
                    recipe_hash=request.recipe_hash,
                )
            return FemAttempt(
                request=request,
                recipe_hash=request.recipe_hash,
                success=True,
                result=result,
                result_hash=hash_spec(result, "content"),
                store_content_hash=store_hash,
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
                store_content_hash=None,
                failure_code=exc.code,
                message=str(exc),
            )

    def _solve_success(self, request: FemRequest) -> FemResult:
        _validate_qois(request)
        peak_outcome = peak_stress_screen(request)
        if peak_outcome is not None:
            raise FemSolveFailure(*peak_outcome)

        skmesh, basis, scalar_basis = _build_basis(request)
        if basis.N > request.config.max_dofs:
            raise FemSolveFailure(
                FemFailureCode.RESOURCE_EXHAUSTION,
                f"requested {basis.N} DOFs exceeds max_dofs={request.config.max_dofs}",
            )
        constraints = bind_constraints(
            basis,
            skmesh,
            request.mesh,
            request.boundary_conditions,
            provider_ids=request.boundary_field_provider_ids,
            providers=self._dirichlet_field_providers,
            rigid_mode_constraint=request.rigid_mode_constraint,
        )
        material = _kernel_material(request.material)
        external_force = assemble_dead_loads(
            basis,
            skmesh,
            request.mesh,
            request.loads,
            thickness=1.0,
            provider_ids=request.body_force_provider_ids,
            providers=self._body_force_providers,
            traction_provider_ids=request.traction_field_provider_ids,
            traction_providers=self._traction_field_providers,
        )
        displacement, state, history, residual = _solve_newton(
            basis,
            constraints,
            external_force,
            material,
            request,
        )
        stress = _matrix3_field(state.first_piola)
        recovered = recover_stress(scalar_basis, state.first_piola)
        external_work = float(external_force @ displacement)
        return FemResult(
            request_id=request.request_id,
            mesh_id=request.mesh.mesh_id,
            element_order=request.config.element_order,
            displacement_dof_coordinates=_dof_coordinates(scalar_basis),
            displacement=_displacement_values(basis, displacement),
            quadrature_points=_quadrature_points(basis),
            quadrature_weights=_quadrature_weights(basis),
            first_piola_stress=stress,
            deformation_gradient=_matrix3_field(state.deformation_gradient),
            recovered_first_piola_stress=_matrix3_dofs(recovered),
            reactions=_reaction_result(basis, constraints, residual),
            rigid_mode_multipliers=_rigid_mode_multipliers(residual, constraints),
            total_strain_energy=state.strain_energy,
            external_work=external_work,
            total_potential_energy=state.strain_energy - external_work,
            convergence_history=history,
            qois=_evaluate_qois(
                request,
                recovered,
                scalar_basis,
                skmesh,
            ),
        )


def _build_basis(request: FemRequest) -> tuple[Any, Any, Any]:
    points = np.asarray(request.mesh.nodes, dtype=np.float64).T
    cells = np.asarray(request.mesh.cells, dtype=np.int32).T
    skmesh = MeshTri(points, cells, sort_t=False)
    scalar = ElementTriP1() if request.config.element_order == 1 else ElementTriP2()
    basis = Basis(skmesh, ElementVector(scalar), intorder=_QUADRATURE_ORDER)
    scalar_basis = Basis(skmesh, scalar, intorder=_QUADRATURE_ORDER)
    return skmesh, basis, scalar_basis


def _kernel_material(material: MaterialSpec) -> MaterialSpec:
    if material.model == _KERNEL_MODEL:
        return material
    if material.model == _PLANE_STRESS_MODEL:
        return replace(material, model=_KERNEL_MODEL)
    raise FemSolveFailure(
        FemFailureCode.UNSUPPORTED_REQUEST,
        f"unsupported material model {material.model!r}",
    )


def _validate_qois(request: FemRequest) -> None:
    for qoi in request.qois:
        if qoi.field not in {"first_piola", "first_piola_stress"}:
            raise FemSolveFailure(
                FemFailureCode.UNSUPPORTED_REQUEST,
                f"unsupported QoI field {qoi.field!r}",
            )
        if qoi.reduction not in {"maximum", "pointwise_peak"}:
            raise FemSolveFailure(
                FemFailureCode.UNSUPPORTED_REQUEST,
                f"unsupported QoI reduction {qoi.reduction!r}",
            )
        if qoi.unit is not Unit.MPA:
            raise FemSolveFailure(
                FemFailureCode.UNSUPPORTED_REQUEST,
                "first-Piola stress QoIs must use canonical MPa",
            )


def _solve_newton(
    basis: Any,
    constraints: BoundConstraints,
    external_force: FloatArray,
    material: MaterialSpec,
    request: FemRequest,
) -> tuple[
    FloatArray,
    AssemblyState,
    tuple[FemConvergenceEntry, ...],
    FloatArray,
]:
    displacement = np.zeros(basis.N, dtype=np.float64)
    history: list[FemConvergenceEntry] = []
    final_state: AssemblyState | None = None
    final_residual: FloatArray | None = None

    for load_step in range(1, request.config.load_steps + 1):
        load_factor = load_step / request.config.load_steps
        displacement = displacement.copy()
        displacement[constraints.constrained] = (
            load_factor * constraints.target_values[constraints.constrained]
        )

        converged = False
        for iteration in range(1, request.config.max_newton_iterations + 1):
            state = assemble_state(
                basis,
                displacement,
                material,
                closure_iterations=request.config.plane_stress_max_iterations,
                thickness=1.0,
            )
            residual = state.internal_force - load_factor * external_force
            residual_norm = _constrained_residual_norm(
                residual,
                displacement,
                constraints,
                load_factor,
            )
            if residual_norm <= request.config.solver_tolerance:
                history.append(
                    FemConvergenceEntry(
                        load_step=load_step,
                        iteration=iteration,
                        residual_norm=residual_norm,
                        increment_norm=0.0,
                    )
                )
                final_state = state
                final_residual = residual
                converged = True
                break

            increment = _solve_increment(
                state,
                residual,
                displacement,
                constraints,
                load_factor,
            )
            displacement, increment_norm = _line_search(
                basis,
                displacement,
                increment,
                constraints,
                external_force,
                load_factor,
                material,
                request,
                residual_norm,
            )
            history.append(
                FemConvergenceEntry(
                    load_step=load_step,
                    iteration=iteration,
                    residual_norm=residual_norm,
                    increment_norm=increment_norm,
                )
            )

        if not converged:
            raise FemSolveFailure(
                FemFailureCode.NEWTON_NON_CONVERGENCE,
                f"load step {load_step} did not converge in "
                f"{request.config.max_newton_iterations} Newton iterations",
            )

    if final_state is None or final_residual is None:
        raise FemSolveFailure(
            FemFailureCode.NEWTON_NON_CONVERGENCE,
            "no load step produced a converged state",
        )
    return displacement, final_state, tuple(history), final_residual


def _solve_increment(
    state: AssemblyState,
    residual: FloatArray,
    displacement: FloatArray,
    constraints: BoundConstraints,
    load_factor: float,
) -> FloatArray:
    free = constraints.free
    tangent = state.tangent[free][:, free]
    global_matrix = constraints.global_matrix[:, free]
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            if global_matrix.shape[0]:
                saddle_point = bmat(
                    (
                        (tangent, csr_matrix(global_matrix.T)),
                        (
                            csr_matrix(global_matrix),
                            csr_matrix(
                                (global_matrix.shape[0], global_matrix.shape[0])
                            ),
                        ),
                    ),
                    format="csr",
                )
                constraint_residual = (
                    constraints.global_matrix @ displacement
                    - load_factor * constraints.global_targets
                )
                right_hand_side = -np.concatenate((residual[free], constraint_residual))
                solution = np.asarray(
                    spsolve(saddle_point, right_hand_side), dtype=np.float64
                )
                increment = solution[: free.size]
            else:
                increment = np.asarray(
                    spsolve(tangent, -residual[free]), dtype=np.float64
                )
        except (MatrixRankWarning, ValueError) as exc:
            raise FemSolveFailure(
                FemFailureCode.CONSTRAINT_RANK_LOSS,
                "constrained Newton tangent is singular",
            ) from exc
    if not bool(np.isfinite(increment).all()):
        raise FemSolveFailure(
            FemFailureCode.CONSTRAINT_RANK_LOSS,
            "constrained Newton tangent produced a non-finite increment",
        )
    return increment


def _line_search(
    basis: Any,
    displacement: FloatArray,
    increment: FloatArray,
    constraints: BoundConstraints,
    external_force: FloatArray,
    load_factor: float,
    material: MaterialSpec,
    request: FemRequest,
    previous_norm: float,
) -> tuple[FloatArray, float]:
    step = 1.0
    while step >= _MIN_LINE_SEARCH_STEP:
        candidate = displacement.copy()
        candidate[constraints.free] += step * increment
        candidate[constraints.constrained] = (
            load_factor * constraints.target_values[constraints.constrained]
        )
        try:
            state = assemble_state(
                basis,
                candidate,
                material,
                closure_iterations=request.config.plane_stress_max_iterations,
                thickness=1.0,
            )
        except FemSolveFailure as exc:
            if exc.code is not FemFailureCode.NEGATIVE_JACOBIAN:
                raise
            step *= 0.5
            continue
        candidate_residual = state.internal_force - load_factor * external_force
        candidate_norm = _constrained_residual_norm(
            candidate_residual,
            candidate,
            constraints,
            load_factor,
        )
        if candidate_norm < previous_norm:
            return candidate, float(step * np.linalg.norm(increment))
        step *= 0.5
    raise FemSolveFailure(
        FemFailureCode.NEWTON_NON_CONVERGENCE,
        "Newton line search did not reduce the free residual",
    )


def _constrained_residual_norm(
    residual: FloatArray,
    displacement: FloatArray,
    constraints: BoundConstraints,
    load_factor: float,
) -> float:
    free_residual = residual[constraints.free]
    global_matrix = constraints.global_matrix[:, constraints.free]
    if not global_matrix.shape[0]:
        return float(np.linalg.norm(free_residual))
    gram = global_matrix @ global_matrix.T
    coefficients = np.linalg.solve(gram, global_matrix @ free_residual)
    projected = free_residual - global_matrix.T @ coefficients
    constraint_residual = (
        constraints.global_matrix @ displacement
        - load_factor * constraints.global_targets
    )
    return float(
        np.hypot(np.linalg.norm(projected), np.linalg.norm(constraint_residual))
    )


def _rigid_mode_multipliers(
    residual: FloatArray,
    constraints: BoundConstraints,
) -> tuple[float, ...]:
    global_matrix = constraints.global_matrix[:, constraints.free]
    if not global_matrix.shape[0]:
        return ()
    gram = global_matrix @ global_matrix.T
    values = -np.linalg.solve(gram, global_matrix @ residual[constraints.free])
    return tuple(float(value) for value in values)


def _displacement_values(
    basis: Any, displacement: FloatArray
) -> tuple[tuple[float, float], ...]:
    x_dofs, y_dofs = (np.asarray(values) for values in basis.split_indices())
    if x_dofs.shape != y_dofs.shape:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            "vector element components have incompatible DOF layouts",
        )
    return tuple(
        (float(displacement[x_dof]), float(displacement[y_dof]))
        for x_dof, y_dof in zip(x_dofs, y_dofs, strict=True)
    )


def _reaction_result(
    basis: Any,
    constraints: BoundConstraints,
    residual: FloatArray,
) -> dict[str, tuple[float, float]]:
    x_dofs = set(int(value) for value in basis.split_indices()[0])
    result: dict[str, tuple[float, float]] = {}
    for name, dofs in constraints.by_name.items():
        reaction = [0.0, 0.0]
        for dof in dofs:
            component = 0 if int(dof) in x_dofs else 1
            reaction[component] += float(residual[dof])
        result[name] = (reaction[0], reaction[1])
    return result


def _dof_coordinates(basis: Any) -> tuple[tuple[float, float], ...]:
    return tuple(
        (float(basis.doflocs[0, dof]), float(basis.doflocs[1, dof]))
        for dof in range(basis.N)
    )


def _quadrature_points(basis: Any) -> tuple[tuple[tuple[float, float], ...], ...]:
    coordinates = np.transpose(
        np.asarray(basis.global_coordinates(), dtype=np.float64), (1, 2, 0)
    )
    return tuple(
        tuple((float(point[0]), float(point[1])) for point in cell)
        for cell in coordinates
    )


def _quadrature_weights(basis: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(weight) for weight in cell)
        for cell in np.asarray(basis.dx, dtype=np.float64)
    )


def _matrix3_field(values: FloatArray) -> tuple[Any, ...]:
    return tuple(
        tuple(
            tuple(
                tuple(float(point[row, column]) for column in range(3))
                for row in range(3)
            )
            for point in cell
        )
        for cell in values
    )


def _matrix3_dofs(values: FloatArray) -> tuple[Any, ...]:
    return tuple(
        tuple(
            tuple(float(point[row, column]) for column in range(3)) for row in range(3)
        )
        for point in values
    )


def _evaluate_qois(
    request: FemRequest,
    recovered_stress: FloatArray,
    scalar_basis: Any,
    skmesh: Any,
) -> dict[str, float]:
    result: dict[str, float] = {}
    facet_map = canonical_facet_map(skmesh)
    for qoi in request.qois:
        facets = facets_for_tag(request.mesh, facet_map, qoi.region_tag)
        dofs = np.asarray(
            scalar_basis.get_dofs(facets=facets).all(),
            dtype=np.int64,
        )
        values = recovered_stress[dofs].reshape(-1)
        result[qoi.name] = float(np.max(np.abs(values)))
    return result


__all__ = ("ScikitFemBackend",)
