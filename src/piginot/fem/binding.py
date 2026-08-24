"""Declarative S2 boundary-condition and dead-load binding for S7."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skfem import FacetBasis, LinearForm, asm  # type: ignore[import-untyped]
from skfem.helpers import dot  # type: ignore[import-untyped]

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    LoadSpec,
    Unit,
)
from piginot.fem.body_forces import BodyForceProvider
from piginot.fem.field_providers import (
    DirichletFieldProvider,
    TractionFieldProvider,
)
from piginot.fem.models import FemFailureCode, FemSolveFailure

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
_COMPONENT_DOF_NAME = {"x": "u^1", "y": "u^2"}
_COMPONENT_INDEX = {"x": 0, "y": 1}


@dataclass(frozen=True, slots=True)
class BoundConstraints:
    """Essential DOFs, target values, and reaction ownership."""

    constrained: IntArray
    free: IntArray
    target_values: FloatArray
    by_name: dict[str, IntArray]
    global_matrix: FloatArray
    global_targets: FloatArray


@LinearForm  # type: ignore[untyped-decorator]
def _dead_traction(test: Any, data: Any) -> Any:
    return dot(data.traction, test)


@LinearForm  # type: ignore[untyped-decorator]
def _registered_body_force(test: Any, data: Any) -> Any:
    return dot(data.body_force, test)


@LinearForm  # type: ignore[untyped-decorator]
def _mean_x_constraint(test: Any, _data: Any) -> Any:
    return test[0]


@LinearForm  # type: ignore[untyped-decorator]
def _mean_y_constraint(test: Any, _data: Any) -> Any:
    return test[1]


@LinearForm  # type: ignore[untyped-decorator]
def _mean_rotation_constraint(test: Any, data: Any) -> Any:
    return data.x[0] * test[1] - data.x[1] * test[0]


def canonical_facet_map(skmesh: Any) -> dict[tuple[int, int], int]:
    return {
        _facet_key(int(skmesh.facets[0, index]), int(skmesh.facets[1, index])): index
        for index in range(skmesh.facets.shape[1])
    }


def facets_for_tag(
    mesh: CanonicalMesh,
    facet_map: dict[tuple[int, int], int],
    tag: str,
) -> IntArray:
    if tag not in mesh.semantic_tags:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            f"unknown boundary tag {tag!r}",
        )
    tagged_facets = mesh.semantic_tags[tag]
    if not tagged_facets:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            f"boundary tag {tag!r} has no facets",
        )
    return np.asarray(
        [
            facet_map[_facet_key(*mesh.boundary_facets[index])]
            for index in tagged_facets
        ],
        dtype=np.int64,
    )


def bind_constraints(
    basis: Any,
    skmesh: Any,
    mesh: CanonicalMesh,
    conditions: tuple[BoundaryConditionSpec, ...],
    *,
    provider_ids: Mapping[str, str],
    providers: Mapping[str, DirichletFieldProvider],
    rigid_mode_constraint: str | None = None,
) -> BoundConstraints:
    facet_map = canonical_facet_map(skmesh)
    target_by_dof: dict[int, float] = {}
    owner_by_dof: dict[int, str] = {}
    by_name: dict[str, IntArray] = {}

    condition_names = {condition.name for condition in conditions}
    dangling = sorted(set(provider_ids) - condition_names)
    if dangling:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            f"boundary-field provider references unknown condition {dangling[0]!r}",
        )

    for condition in conditions:
        facets = facets_for_tag(mesh, facet_map, condition.boundary_tag)
        provider_id = provider_ids.get(condition.name)
        provider = providers.get(provider_id) if provider_id is not None else None
        if provider_id is not None:
            if condition.kind != "dirichlet":
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    "boundary-field providers require a Dirichlet condition",
                )
            if condition.values:
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    "boundary-field conditions cannot also carry constant values",
                )
            if provider is None or provider.provider_id != provider_id:
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    f"boundary-field provider for {condition.name!r} is not registered",
                )
        if condition.kind == "free":
            if condition.components or condition.values:
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    "free boundaries cannot carry components or values",
                )
            continue
        if condition.kind not in {"dirichlet", "symmetry"}:
            raise FemSolveFailure(
                FemFailureCode.UNSUPPORTED_REQUEST,
                f"unsupported boundary condition kind {condition.kind!r}",
            )
        components, values = _condition_components(condition, mesh, facets, skmesh)
        owned: list[int] = []
        for component, value in zip(components, values, strict=True):
            dofs = np.asarray(
                basis.get_dofs(facets=facets).all([_component_name(component)]),
                dtype=np.int64,
            )
            prescribed_values = (
                np.full(dofs.shape, value, dtype=np.float64)
                if provider is None
                else _evaluate_dirichlet_field(provider, basis, dofs, component)
            )
            for dof, prescribed_value in zip(dofs, prescribed_values, strict=True):
                index = int(dof)
                previous = target_by_dof.get(index)
                if previous is not None and not math.isclose(
                    previous,
                    float(prescribed_value),
                    rel_tol=0.0,
                    abs_tol=1.0e-14,
                ):
                    raise FemSolveFailure(
                        FemFailureCode.CONSTRAINT_RANK_LOSS,
                        "conflicting essential constraints share a DOF",
                    )
                if index in owner_by_dof and owner_by_dof[index] != condition.name:
                    raise FemSolveFailure(
                        FemFailureCode.CONSTRAINT_RANK_LOSS,
                        "duplicate essential constraints share a DOF",
                    )
                target_by_dof[index] = float(prescribed_value)
                owner_by_dof[index] = condition.name
                owned.append(index)
        owned_array = np.asarray(sorted(set(owned)), dtype=np.int64)
        by_name[condition.name] = owned_array

    constrained = np.asarray(sorted(target_by_dof), dtype=np.int64)
    target = np.zeros(basis.N, dtype=np.float64)
    if constrained.size:
        target[constrained] = [target_by_dof[int(dof)] for dof in constrained]
    free = np.setdiff1d(np.arange(basis.N, dtype=np.int64), constrained)
    global_matrix = _global_rigid_mode_matrix(
        basis,
        rigid_mode_constraint,
        constrained,
    )
    global_targets = np.zeros(global_matrix.shape[0], dtype=np.float64)
    _require_rigid_mode_rank(basis, constrained, global_matrix)
    return BoundConstraints(
        constrained,
        free,
        target,
        by_name,
        global_matrix,
        global_targets,
    )


def _global_rigid_mode_matrix(
    basis: Any,
    strategy: str | None,
    constrained: IntArray,
) -> FloatArray:
    if strategy is None:
        return np.empty((0, basis.N), dtype=np.float64)
    if strategy != "zero_mean_translation_rotation":
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            f"unsupported rigid-mode constraint {strategy!r}",
        )
    if constrained.size:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            "global rigid-mode constraints cannot be mixed with essential boundaries",
        )
    rows = np.vstack(
        (
            np.asarray(asm(_mean_x_constraint, basis), dtype=np.float64),
            np.asarray(asm(_mean_y_constraint, basis), dtype=np.float64),
            np.asarray(asm(_mean_rotation_constraint, basis), dtype=np.float64),
        )
    )
    row_norms = np.linalg.norm(rows, axis=1)
    if not bool(np.isfinite(rows).all()) or bool(np.any(row_norms <= 0.0)):
        raise FemSolveFailure(
            FemFailureCode.CONSTRAINT_RANK_LOSS,
            "global rigid-mode constraints are rank deficient",
        )
    return np.asarray(rows / row_norms[:, None], dtype=np.float64)


def assemble_dead_loads(
    basis: Any,
    skmesh: Any,
    mesh: CanonicalMesh,
    loads: tuple[LoadSpec, ...],
    *,
    thickness: float,
    provider_ids: Mapping[str, str],
    providers: Mapping[str, BodyForceProvider],
    traction_provider_ids: Mapping[str, str],
    traction_providers: Mapping[str, TractionFieldProvider],
) -> FloatArray:
    result = np.zeros(basis.N, dtype=np.float64)
    facet_map = canonical_facet_map(skmesh)
    load_names = {load.name for load in loads}
    dangling = sorted(set(traction_provider_ids) - load_names)
    if dangling:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            f"traction-field provider references unknown load {dangling[0]!r}",
        )
    for load in loads:
        traction_provider_id = traction_provider_ids.get(load.name)
        if traction_provider_id is not None and (
            load.kind != "traction" or load.boundary_tag is None
        ):
            raise FemSolveFailure(
                FemFailureCode.UNSUPPORTED_REQUEST,
                "traction-field providers require a traction load with a boundary",
            )
        if load.kind == "body_force":
            if load.boundary_tag is not None or load.components or load.values:
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    "registered body-force loads cannot carry boundary/value fields",
                )
            provider_id = provider_ids.get(load.name)
            provider = providers.get(provider_id) if provider_id is not None else None
            if provider is None or provider.provider_id != provider_id:
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    f"body-force provider for {load.name!r} is not registered",
                )
            points = np.asarray(basis.global_coordinates(), dtype=np.float64)
            try:
                body_force = np.asarray(provider.evaluate(points), dtype=np.float64)
            except Exception as exc:
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    f"body-force provider for {load.name!r} failed",
                ) from exc
            if body_force.shape != points.shape or not bool(
                np.isfinite(body_force).all()
            ):
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    "body-force provider returned an invalid canonical field",
                )
            assembled = asm(
                _registered_body_force,
                basis,
                body_force=body_force,
            )
            result = result + np.asarray(assembled, dtype=np.float64) * thickness
            continue
        if load.kind != "traction" or load.boundary_tag is None:
            raise FemSolveFailure(
                FemFailureCode.UNSUPPORTED_REQUEST,
                f"unsupported load kind {load.kind!r}",
            )
        facets = facets_for_tag(mesh, facet_map, load.boundary_tag)
        facet_basis = FacetBasis(
            skmesh,
            basis.elem,
            facets=facets,
            dofs=basis.dofs,
        )
        traction = _traction_field(
            load,
            facet_basis,
            provider_id=traction_provider_id,
            providers=traction_providers,
        )
        assembled = asm(
            _dead_traction,
            facet_basis,
            traction=traction,
        )
        result = result + np.asarray(assembled, dtype=np.float64) * thickness
    return result


def _evaluate_dirichlet_field(
    provider: DirichletFieldProvider,
    basis: Any,
    dofs: IntArray,
    component: str,
) -> FloatArray:
    points = np.asarray(basis.doflocs[:, dofs], dtype=np.float64)
    try:
        field = np.asarray(provider.evaluate(points), dtype=np.float64)
    except Exception as exc:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            "boundary-field provider failed",
        ) from exc
    if field.shape != points.shape or not bool(np.isfinite(field).all()):
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            "boundary-field provider returned an invalid canonical field",
        )
    return np.asarray(field[_COMPONENT_INDEX[component]], dtype=np.float64)


def _traction_field(
    load: LoadSpec,
    facet_basis: Any,
    *,
    provider_id: str | None,
    providers: Mapping[str, TractionFieldProvider],
) -> FloatArray:
    if provider_id is None:
        constant_traction = np.zeros(2, dtype=np.float64)
        for component, value in zip(load.components, load.values, strict=True):
            if value.unit is not Unit.MPA:
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    "dead nominal traction must use canonical MPa",
                )
            constant_traction[_traction_component_index(component)] = value.value
        return constant_traction[:, None, None]

    if load.values:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            "traction-field loads cannot also carry constant values",
        )
    provider = providers.get(provider_id)
    if provider is None or provider.provider_id != provider_id:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            f"traction-field provider for {load.name!r} is not registered",
        )
    points = np.asarray(facet_basis.global_coordinates(), dtype=np.float64)
    normals = np.asarray(facet_basis.normals, dtype=np.float64)
    try:
        evaluated = np.asarray(provider.evaluate(points, normals), dtype=np.float64)
    except Exception as exc:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            f"traction-field provider for {load.name!r} failed",
        ) from exc
    if evaluated.shape != points.shape or not bool(np.isfinite(evaluated).all()):
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            "traction-field provider returned an invalid canonical field",
        )
    components = load.components or tuple(_COMPONENT_INDEX)
    spatial_traction = np.zeros_like(evaluated)
    for component in components:
        index = _traction_component_index(component)
        spatial_traction[index] = evaluated[index]
    return spatial_traction


def _traction_component_index(component: str) -> int:
    try:
        return _COMPONENT_INDEX[component]
    except KeyError as exc:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            f"unsupported traction component {component!r}",
        ) from exc


def _condition_components(
    condition: BoundaryConditionSpec,
    mesh: CanonicalMesh,
    facets: IntArray,
    skmesh: Any,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    components: tuple[str, ...]
    values: tuple[float, ...]
    if condition.kind == "symmetry" and not condition.components:
        components = (_axis_aligned_normal_component(facets, skmesh),)
        values = (0.0,)
    else:
        components = condition.components
        if not components:
            raise FemSolveFailure(
                FemFailureCode.UNSUPPORTED_REQUEST,
                "essential boundary conditions require components",
            )
        if condition.values:
            if len(condition.values) != len(components):
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    "boundary components and values must align",
                )
            values = tuple(value.value for value in condition.values)
        else:
            values = tuple(0.0 for _ in components)
    for component in components:
        _component_name(component)
    for value in condition.values:
        if value.unit is not Unit.MM:
            raise FemSolveFailure(
                FemFailureCode.UNSUPPORTED_REQUEST,
                "essential displacement must use canonical mm",
            )
    if condition.kind == "symmetry" and any(abs(value) > 0.0 for value in values):
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            "symmetry displacement must be zero",
        )
    return components, values


def _axis_aligned_normal_component(facets: IntArray, skmesh: Any) -> str:
    vectors = (
        skmesh.p[:, skmesh.facets[1, facets]] - skmesh.p[:, skmesh.facets[0, facets]]
    )
    if bool(np.all(np.abs(vectors[0]) <= 1.0e-12)):
        return "x"
    if bool(np.all(np.abs(vectors[1]) <= 1.0e-12)):
        return "y"
    raise FemSolveFailure(
        FemFailureCode.UNSUPPORTED_REQUEST,
        "implicit symmetry is supported only on axis-aligned boundaries",
    )


def _facet_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left <= right else (right, left)


def _component_name(component: str) -> str:
    try:
        return _COMPONENT_DOF_NAME[component]
    except KeyError as exc:
        raise FemSolveFailure(
            FemFailureCode.UNSUPPORTED_REQUEST,
            f"unsupported displacement component {component!r}",
        ) from exc


def _require_rigid_mode_rank(
    basis: Any,
    constrained: IntArray,
    global_matrix: FloatArray,
) -> None:
    rows: list[tuple[float, float, float]] = []
    x_dofs = set(int(value) for value in basis.split_indices()[0])
    for dof in constrained:
        x = float(basis.doflocs[0, dof])
        y = float(basis.doflocs[1, dof])
        rows.append((1.0, 0.0, -y) if int(dof) in x_dofs else (0.0, 1.0, x))
    if global_matrix.shape[0]:
        rigid_modes = np.zeros((basis.N, 3), dtype=np.float64)
        for dof in range(basis.N):
            x = float(basis.doflocs[0, dof])
            y = float(basis.doflocs[1, dof])
            rigid_modes[dof] = (1.0, 0.0, -y) if dof in x_dofs else (0.0, 1.0, x)
        rows.extend(tuple(row) for row in global_matrix @ rigid_modes)
    rank = 0 if not rows else int(np.linalg.matrix_rank(np.asarray(rows)))
    if rank < 3:
        raise FemSolveFailure(
            FemFailureCode.CONSTRAINT_RANK_LOSS,
            "essential constraints do not remove all rigid-body modes",
        )


__all__ = (
    "BoundConstraints",
    "assemble_dead_loads",
    "bind_constraints",
    "canonical_facet_map",
    "facets_for_tag",
)
