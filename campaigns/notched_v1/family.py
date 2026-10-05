"""The S17 centre-notched plate family (ADR 0025).

The S1 centre-notched plate: a 50 mm strip of width ``W`` with a C2-blended
double-edge notch of depth ``d`` and root radius ``r`` at mid-length, loaded
in displacement-controlled tension. The quarter model is ``[0, 25] x [0, W/2]``
with symmetry on ``x = 0`` and ``y = 0``; the notch root lies on ``x = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    DerivedExpression,
    DerivedQuantitySpec,
    Dimension,
    GeometryFamilySpec,
    MaterialSpec,
    ParameterSpec,
    Unit,
    hash_spec,
)
from piginot.fem import FemSolverConfig
from piginot.morphing import (
    BlendedNotchMap,
    BlendedNotchMeshSpec,
    BlendedNotchParameters,
    FamilyMembershipPredicate,
    MorphValidityCriteria,
    build_blended_notch_reference_mesh,
)

RECIPE_ID = "centre-notched-quarter-v1"
CAMPAIGN_ID = "notched-v1"
HALF_LENGTH_MM = 25.0
GEOMETRY = ("width", "notch_radius", "notch_depth")
INPUTS = (*GEOMETRY, "nominal_axial_strain")
BOUNDS = {
    "width": (16.0, 30.0),
    "notch_radius": (2.0, 8.0),
    "notch_depth": (1.0, 5.0),
    "nominal_axial_strain": (0.10, 0.20),
}
ELASTIC_MODULUS_MPA = 760.0
POISSON_RATIO = 0.23
LOWER_TAG = "bottom"
LOADED_END_TAG = "right_end"
NOTCH_TAG = "notch"
FLAT_TAG = "top"
LEFT_TAG = "left_symmetry"

_REFERENCE = BlendedNotchParameters(
    half_length_mm=HALF_LENGTH_MM,
    width_mm=23.0,
    notch_radius_mm=5.0,
    notch_depth_mm=3.0,
)
_MESH_SPEC = BlendedNotchMeshSpec(
    notch_columns=24,
    outer_columns=24,
    transverse_rows=16,
    lower_tag=LOWER_TAG,
    loaded_end_tag=LOADED_END_TAG,
    notch_profile_tag=NOTCH_TAG,
    flat_profile_tag=FLAT_TAG,
    left_end_tag=LEFT_TAG,
)
MORPH_MAP = BlendedNotchMap(
    reference=_REFERENCE,
    width_parameter="width",
    notch_radius_parameter="notch_radius",
    notch_depth_parameter="notch_depth",
)
VALIDITY = MorphValidityCriteria(
    jacobian_margin=0.01,
    minimum_feature_size_mm=0.05,
    minimum_angle_degrees=5.0,
    maximum_aspect_ratio=20.0,
)


def _length(value: float) -> CanonicalQuantity:
    return CanonicalQuantity(value, Unit.MM, Dimension.LENGTH)


def _ratio(value: float) -> CanonicalQuantity:
    return CanonicalQuantity(value, Unit.ONE, Dimension.DIMENSIONLESS)


def _derived(
    name: str, expression: str, lower: float, upper: float, description: str
) -> DerivedQuantitySpec:
    return DerivedQuantitySpec(
        name=name,
        expression=DerivedExpression.parse(expression),
        lower_bound=_ratio(lower),
        upper_bound=_ratio(upper),
        unit=Unit.ONE,
        description=description,
    )


def _family_spec(reference: CanonicalMesh) -> GeometryFamilySpec:
    return GeometryFamilySpec(
        name=RECIPE_ID,
        parameters=tuple(
            ParameterSpec(name, _length(BOUNDS[name][0]), _length(BOUNDS[name][1]))
            for name in GEOMETRY
        ),
        derived_quantities=(
            _derived(
                "net_section_factor",
                "width / (width - 2 * notch_depth)",
                30.0 / 28.0,
                16.0 / 6.0,
                "Gross width over the net ligament width at the notch root.",
            ),
            _derived(
                "notch_sharpness",
                "notch_depth / notch_radius",
                1.0 / 8.0,
                5.0 / 2.0,
                "Notch depth over the root radius of curvature.",
            ),
        ),
        topology_signature=reference.topology_signature,
        semantic_tags=tuple(reference.semantic_tags),
    )


def _structural_hash(mesh: CanonicalMesh) -> str:
    return hash_spec(
        {
            "schema": "canonical-triangle-mesh-v1",
            "nodes": mesh.nodes,
            "cells": mesh.cells,
            "cell_type": mesh.cell_type,
            "boundary_facets": mesh.boundary_facets,
            "semantic_tags": mesh.semantic_tags,
            "region_ids": mesh.region_ids,
            "unit": mesh.unit,
            "orientation": mesh.orientation,
        },
        domain="content",
    )


@dataclass(frozen=True, slots=True)
class NotchedCompiledFamily:
    """Runtime family whose canonical identity binds the structural mesh ID."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    recipe_id: str
    reference_mesh: CanonicalMesh = field(repr=False, metadata={"canonical": False})
    mesh_spec: BlendedNotchMeshSpec
    morph_map: BlendedNotchMap
    membership: FamilyMembershipPredicate
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if type(self.reference_mesh) is not CanonicalMesh:
            raise ValueError("reference_mesh must be a CanonicalMesh")
        if self.reference_mesh.mesh_id != self.membership.reference_mesh_id:
            raise ValueError("reference mesh identity does not match membership")
        if self.reference_mesh.mesh_id != _structural_hash(self.reference_mesh):
            raise ValueError("reference mesh structural identity does not match")
        if (
            self.reference_mesh.topology_signature
            != self.membership.reference_topology_signature
        ):
            raise ValueError("reference mesh topology does not match membership")


def compiled_family() -> NotchedCompiledFamily:
    """Compile the family; creates no campaign secret."""

    reference = build_blended_notch_reference_mesh(
        _REFERENCE,
        _MESH_SPEC,
        source_provenance={
            "generator": "blended-notch-structured-reference-v1",
            "recipe_id": RECIPE_ID,
        },
    )
    membership = FamilyMembershipPredicate.compile(
        _family_spec(reference), reference, VALIDITY
    )
    return NotchedCompiledFamily(
        recipe_id=RECIPE_ID,
        reference_mesh=reference,
        mesh_spec=_MESH_SPEC,
        morph_map=MORPH_MAP,
        membership=membership,
    )


def parameter_point(
    width_mm: float, notch_radius_mm: float, notch_depth_mm: float
) -> dict[str, CanonicalQuantity]:
    """One unitized geometry point in family vocabulary order."""

    return {
        "width": _length(width_mm),
        "notch_radius": _length(notch_radius_mm),
        "notch_depth": _length(notch_depth_mm),
    }


def prescribed_displacement_mm(nominal_strain: float) -> float:
    """Loaded-end displacement of the quarter model for a nominal strain."""

    return nominal_strain * HALF_LENGTH_MM


def material(
    elastic_modulus: float = ELASTIC_MODULUS_MPA, poisson_ratio: float = POISSON_RATIO
) -> MaterialSpec:
    return MaterialSpec(
        "notched-v1-reference-material",
        "finite_strain_plane_stress_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(elastic_modulus, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(poisson_ratio, Unit.ONE),
        },
    )


def boundary_conditions(nominal_strain: float) -> tuple[BoundaryConditionSpec, ...]:
    return (
        BoundaryConditionSpec("left_x_symmetry", LEFT_TAG, "symmetry", ("x",)),
        BoundaryConditionSpec("bottom_y_symmetry", LOWER_TAG, "symmetry", ("y",)),
        BoundaryConditionSpec(
            "right_end_displacement",
            LOADED_END_TAG,
            "dirichlet",
            ("x",),
            (CanonicalQuantity(prescribed_displacement_mm(nominal_strain), Unit.MM),),
        ),
    )


def solver_config(*, load_steps: int = 64) -> FemSolverConfig:
    return FemSolverConfig(
        element_order=2,
        solver_tolerance=1.0e-10,
        max_newton_iterations=20,
        plane_stress_max_iterations=32,
        load_steps=load_steps,
        max_dofs=100_000,
    )


__all__ = (
    "BOUNDS",
    "CAMPAIGN_ID",
    "GEOMETRY",
    "HALF_LENGTH_MM",
    "INPUTS",
    "MORPH_MAP",
    "RECIPE_ID",
    "NotchedCompiledFamily",
    "boundary_conditions",
    "compiled_family",
    "material",
    "parameter_point",
    "prescribed_displacement_mm",
    "solver_config",
)
