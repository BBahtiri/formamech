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


@LinearForm  # type: ignore[untyped-decorator]
def _dead_traction(test: Any, data: Any) -> Any:
    return dot(data.traction, test)


@LinearForm  # type: ignore[untyped-decorator]
def _registered_body_force(test: Any, data: Any) -> Any:
    return dot(data.body_force, test)


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
    return np.asarray(
        [
            facet_map[_facet_key(*mesh.boundary_facets[index])]
            for index in mesh.semantic_tags[tag]
        ],
        dtype=np.int64,
    )


def bind_constraints(
    basis: Any,
    skmesh: Any,
    mesh: CanonicalMesh,
    conditions: tuple[BoundaryConditionSpec, ...],
) -> BoundConstraints:
    facet_map = canonical_facet_map(skmesh)
    target_by_dof: dict[int, float] = {}
    owner_by_dof: dict[int, str] = {}
    by_name: dict[str, IntArray] = {}

    for condition in conditions:
        facets = facets_for_tag(mesh, facet_map, condition.boundary_tag)
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
            for dof in dofs:
                index = int(dof)
                previous = target_by_dof.get(index)
                if previous is not None and not math.isclose(
                    previous, value, rel_tol=0.0, abs_tol=1.0e-14
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
                target_by_dof[index] = value
                owner_by_dof[index] = condition.name
                owned.append(index)
        owned_array = np.asarray(sorted(set(owned)), dtype=np.int64)
        by_name[condition.name] = owned_array

    constrained = np.asarray(sorted(target_by_dof), dtype=np.int64)
    target = np.zeros(basis.N, dtype=np.float64)
    if constrained.size:
        target[constrained] = [target_by_dof[int(dof)] for dof in constrained]
    free = np.setdiff1d(np.arange(basis.N, dtype=np.int64), constrained)
    _require_rigid_mode_rank(basis, constrained)
    return BoundConstraints(constrained, free, target, by_name)


def assemble_dead_loads(
    basis: Any,
    skmesh: Any,
    mesh: CanonicalMesh,
    loads: tuple[LoadSpec, ...],
    *,
    thickness: float,
    provider_ids: Mapping[str, str],
    providers: Mapping[str, BodyForceProvider],
) -> FloatArray:
    result = np.zeros(basis.N, dtype=np.float64)
    facet_map = canonical_facet_map(skmesh)
    for load in loads:
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
        traction = np.zeros(2, dtype=np.float64)
        for component, value in zip(load.components, load.values, strict=True):
            if value.unit is not Unit.MPA:
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    "dead nominal traction must use canonical MPa",
                )
            try:
                component_index = _COMPONENT_INDEX[component]
            except KeyError as exc:
                raise FemSolveFailure(
                    FemFailureCode.UNSUPPORTED_REQUEST,
                    f"unsupported traction component {component!r}",
                ) from exc
            traction[component_index] = value.value
        facets = facets_for_tag(mesh, facet_map, load.boundary_tag)
        facet_basis = FacetBasis(
            skmesh,
            basis.elem,
            facets=facets,
            dofs=basis.dofs,
        )
        assembled = asm(
            _dead_traction,
            facet_basis,
            traction=traction[:, None, None],
        )
        result = result + np.asarray(assembled, dtype=np.float64) * thickness
    return result


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


def _require_rigid_mode_rank(basis: Any, constrained: IntArray) -> None:
    rows: list[tuple[float, float, float]] = []
    x_dofs = set(int(value) for value in basis.split_indices()[0])
    for dof in constrained:
        x = float(basis.doflocs[0, dof])
        y = float(basis.doflocs[1, dof])
        rows.append((1.0, 0.0, -y) if int(dof) in x_dofs else (0.0, 1.0, x))
    if len(rows) < 3 or int(np.linalg.matrix_rank(np.asarray(rows))) < 3:
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
