"""Pinned proposal for the first S9 production-family definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from piginot.contracts import (
    CanonicalMesh,
    CanonicalQuantity,
    DerivedExpression,
    DerivedQuantitySpec,
    Dimension,
    GeometryFamilySpec,
    ParameterSpec,
    Unit,
)
from piginot.morphing import (
    CircularTransitionMap,
    CircularTransitionMeshSpec,
    CircularTransitionParameters,
    FamilyMembershipPredicate,
    MorphValidityCriteria,
    build_circular_transition_reference_mesh,
)

REFERENCE_REPOSITORY_URL = "https://github.com/BBahtiri/PI-GINOT-DogBone"
REFERENCE_REPOSITORY_COMMIT = "9f6794141beca9d3e66433465c3b07039a7e41d2"
RECIPE_ID = "dogbone-quarter-no-hole-v1"
_REFERENCE = CircularTransitionParameters(
    total_length_mm=54.0,
    outer_width_mm=20.0,
    inner_width_mm=10.0,
    transition_radius_mm=12.0,
)
_MESH_SPEC = CircularTransitionMeshSpec(
    inner_columns=36,
    transition_columns=20,
    transverse_rows=16,
    lower_tag="bottom",
    loaded_end_tag="right_grip",
    transition_profile_tag="right_arc",
    inner_profile_tag="gauge_top",
    left_end_tag="left_symmetry",
)
_MORPH_MAP = CircularTransitionMap(
    reference=_REFERENCE,
    total_length_parameter="L_total",
    outer_width_parameter="W_grip",
    inner_width_parameter="W_gauge",
    transition_radius_parameter="R_fillet",
    minimum_inner_length_mm=2.0,
)
_VALIDITY = MorphValidityCriteria(
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
    name: str,
    expression: str,
    lower: float,
    upper: float,
    description: str,
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
        parameters=(
            ParameterSpec("L_total", _length(40.0), _length(70.0)),
            ParameterSpec("W_grip", _length(16.0), _length(26.0)),
            ParameterSpec("W_gauge", _length(6.0), _length(14.0)),
            ParameterSpec("R_fillet", _length(8.0), _length(20.0)),
        ),
        derived_quantities=(
            _derived(
                "arc_feasibility_ratio",
                "2 * R_fillet / (W_grip - W_gauge)",
                1.35,
                20.0,
                "Transition radius divided by the half-width change, with a mesh-quality margin.",
            ),
            _derived(
                "fillet_to_gauge_ratio",
                "R_fillet / W_gauge",
                8.0 / 14.0,
                20.0 / 6.0,
                "Transition-radius to reduced-width ratio.",
            ),
            _derived(
                "grip_to_gauge_ratio",
                "W_grip / W_gauge",
                16.0 / 14.0,
                26.0 / 6.0,
                "Outer-width to reduced-width ratio.",
            ),
            _derived(
                "length_to_gauge_ratio",
                "L_total / W_gauge",
                40.0 / 14.0,
                70.0 / 6.0,
                "Total-length to reduced-width ratio.",
            ),
        ),
        topology_signature=reference.topology_signature,
        semantic_tags=tuple(reference.semantic_tags),
    )


@dataclass(frozen=True, slots=True)
class CompiledFamily:
    """Public canonical preimage bound by the later CP-ENVELOPE."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    recipe_id: str
    source_repository_url: str
    source_repository_commit: str
    reference_mesh: CanonicalMesh
    mesh_spec: CircularTransitionMeshSpec
    morph_map: CircularTransitionMap
    membership: FamilyMembershipPredicate
    schema_version: str = "1.0"


def compiled_family() -> CompiledFamily:
    """Compile the exact proposed family without creating campaign secrets."""

    reference = build_circular_transition_reference_mesh(
        _REFERENCE,
        _MESH_SPEC,
        source_provenance={
            "generator": "circular-transition-structured-reference-v1",
            "recipe_id": RECIPE_ID,
            "source_repository": REFERENCE_REPOSITORY_URL,
            "source_commit": REFERENCE_REPOSITORY_COMMIT,
        },
    )
    membership = FamilyMembershipPredicate.compile(
        _family_spec(reference), reference, _VALIDITY
    )
    return CompiledFamily(
        recipe_id=RECIPE_ID,
        source_repository_url=REFERENCE_REPOSITORY_URL,
        source_repository_commit=REFERENCE_REPOSITORY_COMMIT,
        reference_mesh=reference,
        mesh_spec=_MESH_SPEC,
        morph_map=_MORPH_MAP,
        membership=membership,
    )


def parameter_point(
    total_length_mm: float,
    outer_width_mm: float,
    inner_width_mm: float,
    transition_radius_mm: float,
) -> dict[str, CanonicalQuantity]:
    """Create one explicitly unitized parameter point in family vocabulary order."""

    return {
        "L_total": _length(total_length_mm),
        "W_grip": _length(outer_width_mm),
        "W_gauge": _length(inner_width_mm),
        "R_fillet": _length(transition_radius_mm),
    }


__all__ = (
    "CompiledFamily",
    "REFERENCE_REPOSITORY_COMMIT",
    "REFERENCE_REPOSITORY_URL",
    "RECIPE_ID",
    "compiled_family",
    "parameter_point",
)
