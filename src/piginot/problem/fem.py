"""The reference FEM request of one case, built from its ProblemSpec.

For the two shipped families the request equals the campaign's S10 request
byte for byte (ADR 0106 decision 6). A nominal-strain value follows the rule
of the S22 recipes: the strain times the largest coordinate of the region's
nodes along the component.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    ExecutionProfile,
    MaterialSpec,
)
from piginot.contracts.problem_spec import (
    COMPONENTS,
    ProblemSpec,
    ProblemSpecError,
    region_nodes,
)
from piginot.contracts.units import Dimension, Unit

if TYPE_CHECKING:
    from piginot.fem.models import FemRequest

FEM_MATERIAL_MODELS: Final[Mapping[str, str]] = {
    "compressible_neo_hookean": "finite_strain_plane_stress_neo_hookean",
}
"""The FEM solver's name of a closed-form model; a formula model keeps its
name (the solver resolves it through :mod:`piginot.materials`)."""


def _quantity(inputs: Mapping[str, object], name: str) -> CanonicalQuantity:
    value = inputs[name]
    if not isinstance(value, CanonicalQuantity):
        raise TypeError(f"input {name!r} must be a CanonicalQuantity")
    return value


def fem_request(
    spec: ProblemSpec,
    inputs: Mapping[str, object],
    mesh: CanonicalMesh,
    request_id: str,
    execution_profile: ExecutionProfile,
) -> FemRequest:
    """The FEM request for ``inputs`` (canonical quantities) on ``mesh``.

    Needs the ``verify`` extra (the FEM package); importing this module does not.
    """

    from piginot.fem.models import FemRequest, FemSolverConfig

    if spec.material.energy is not None:
        from piginot.materials import register

        register(spec.material.energy)
    values = spec.values(inputs)
    material = MaterialSpec(
        name=spec.fem.material_name,
        model=FEM_MATERIAL_MODELS.get(spec.material.model, spec.material.model),
        properties={
            prop: _quantity(inputs, name)
            for prop, name in spec.material.parameters.items()
        },
    )
    conditions: list[BoundaryConditionSpec] = []
    for condition in spec.boundary_conditions:
        if condition.kind == "traction_free":
            continue
        if condition.kind == "neumann":
            raise ProblemSpecError(
                "unsupported",
                f"{condition.name}: traction loads are not supported in this version",
            )
        if condition.region not in mesh.semantic_tags:
            raise ProblemSpecError(
                "region", f"the mesh has no region {condition.region!r}"
            )
        prescribed: list[CanonicalQuantity] = []
        for component, value in zip(
            condition.components, condition.values, strict=False
        ):
            if value.nominal_strain is not None:
                axis = COMPONENTS.index(component)
                end = max(
                    float(mesh.nodes[index][axis])
                    for index in region_nodes(mesh, condition.region)
                )
                amount = values[value.nominal_strain] * end
            else:
                assert value.expression is not None
                amount = value.expression.evaluate(values)
            prescribed.append(CanonicalQuantity(amount, Unit.MM, Dimension.LENGTH))
        conditions.append(
            BoundaryConditionSpec(
                name=condition.name,
                boundary_tag=condition.region,
                kind=condition.kind,
                components=condition.components,
                values=tuple(prescribed),
            )
        )
    settings = spec.fem
    return FemRequest(
        request_id=request_id,
        mesh=mesh,
        material=material,
        boundary_conditions=tuple(conditions),
        loads=(),
        qois=(),
        config=FemSolverConfig(
            element_order=settings.element_order,
            solver_tolerance=settings.solver_tolerance,
            max_newton_iterations=settings.max_newton_iterations,
            plane_stress_max_iterations=settings.plane_stress_max_iterations,
            load_steps=settings.load_steps,
            max_dofs=settings.max_dofs,
        ),
        execution_profile=execution_profile,
    )


__all__ = ("FEM_MATERIAL_MODELS", "fem_request")
