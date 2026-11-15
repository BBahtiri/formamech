"""Pinned public family templates and fixed-policy morphing, without campaign data."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache

from piginot.contracts import (
    CanonicalMesh,
    CanonicalQuantity,
    Dimension,
    ProblemSpec,
    Unit,
    canonical_json_bytes,
)
from piginot.morphing import (
    BlendedNotchMap,
    BlendedNotchMeshSpec,
    BlendedNotchParameters,
    CircularTransitionMap,
    CircularTransitionMeshSpec,
    CircularTransitionParameters,
    MorphValidityCriteria,
    build_blended_notch_reference_mesh,
    build_circular_transition_reference_mesh,
)
from piginot.morphing.validation import validate_deformation
from piginot.problem.expressions import value as expression_value
from piginot_campaigns.adapters import GeometryTemplate, load_geometry_templates

MorphMap = CircularTransitionMap | BlendedNotchMap

_VALIDITY = MorphValidityCriteria(
    jacobian_margin=0.01,
    minimum_feature_size_mm=0.05,
    minimum_angle_degrees=5.0,
    maximum_aspect_ratio=20.0,
)
_DOGBONE_REFERENCE = CircularTransitionParameters(54.0, 20.0, 10.0, 12.0)
_DOGBONE_MESH = CircularTransitionMeshSpec(
    inner_columns=36,
    transition_columns=20,
    transverse_rows=16,
    lower_tag="bottom",
    loaded_end_tag="right_grip",
    transition_profile_tag="right_arc",
    inner_profile_tag="gauge_top",
    left_end_tag="left_symmetry",
)
_DOGBONE_MORPH = CircularTransitionMap(
    reference=_DOGBONE_REFERENCE,
    total_length_parameter="L_total",
    outer_width_parameter="W_grip",
    inner_width_parameter="W_gauge",
    transition_radius_parameter="R_fillet",
    minimum_inner_length_mm=2.0,
)
_NOTCHED_REFERENCE = BlendedNotchParameters(25.0, 23.0, 5.0, 3.0)
_NOTCHED_MESH = BlendedNotchMeshSpec(
    notch_columns=24,
    outer_columns=24,
    transverse_rows=16,
    lower_tag="bottom",
    loaded_end_tag="right_end",
    notch_profile_tag="notch",
    flat_profile_tag="top",
    left_end_tag="left_symmetry",
)
_NOTCHED_MORPH = BlendedNotchMap(
    reference=_NOTCHED_REFERENCE,
    width_parameter="width",
    notch_radius_parameter="notch_radius",
    notch_depth_parameter="notch_depth",
)


def _level(level: int) -> None:
    if isinstance(level, bool) or not isinstance(level, int) or level not in (1, 2, 4):
        raise ValueError("geometry level must be one of the declared levels 1, 2, 4")


def validity_criteria() -> MorphValidityCriteria:
    return _VALIDITY


@lru_cache(maxsize=12)
def _template(
    adapter_id: str, resolution: tuple[int, int, int], level: int
) -> tuple[CanonicalMesh, MorphMap]:
    _level(level)
    if adapter_id == "dogbone-v1":
        dogbone_spec = replace(
            _DOGBONE_MESH,
            inner_columns=resolution[0] * level,
            transition_columns=resolution[1] * level,
            transverse_rows=resolution[2] * level,
        )
        return build_circular_transition_reference_mesh(
            _DOGBONE_REFERENCE,
            dogbone_spec,
            source_provenance={
                "generator": "circular-transition-structured-reference-v1",
                "recipe_id": "dogbone-quarter-no-hole-v1",
                "source_repository": "https://github.com/BBahtiri/PI-GINOT-DogBone",
                "source_commit": "9f6794141beca9d3e66433465c3b07039a7e41d2",
            },
        ), _DOGBONE_MORPH
    if adapter_id == "notched-v1":
        notched_spec = replace(
            _NOTCHED_MESH,
            notch_columns=resolution[0] * level,
            outer_columns=resolution[1] * level,
            transverse_rows=resolution[2] * level,
        )
        return build_blended_notch_reference_mesh(
            _NOTCHED_REFERENCE,
            notched_spec,
            source_provenance={
                "generator": "blended-notch-structured-reference-v1",
                "recipe_id": "centre-notched-quarter-v1",
            },
        ), _NOTCHED_MORPH
    raise ValueError(f"unsupported public adapter {adapter_id!r}")


def _registered_template(adapter_id: str, spec: ProblemSpec) -> GeometryTemplate:
    for template in load_geometry_templates(adapter_id):
        if (
            spec.geometry.reference_mesh_id == template.identity.reference_mesh_id
            and spec.geometry.morph_map_sha256 == template.identity.morph_map_sha256
        ):
            return template
    raise ValueError(
        "ProblemSpec geometry identities are not registered by this adapter"
    )


def export_geometry(
    adapter_id: str, spec: ProblemSpec
) -> tuple[CanonicalMesh, MorphMap]:
    """The exact registered reference template and typed executable morph map."""

    template = _registered_template(adapter_id, spec)
    registered = template.identity
    reference, morph = _template(adapter_id, template.resolution, 1)
    if (
        reference.mesh_id != registered.reference_mesh_id
        or hashlib.sha256(canonical_json_bytes(morph)).hexdigest()
        != registered.morph_map_sha256
    ):
        raise ValueError("installed geometry code differs from the registered template")
    declared = {p.name for p in spec.parameters if p.role == "geometry"}
    if declared != set(morph.parameter_names):
        raise ValueError(
            "geometry parameter vocabulary differs from the registered morph"
        )
    if any(spec.parameter(name).unit is not Unit.MM for name in declared):
        raise ValueError("registered geometry parameters must use millimetres")
    if adapter_id == "notched-v1" and not any(
        constant.name == "half_length"
        and constant.value == 25.0
        and constant.unit is Unit.MM
        for constant in spec.constants
    ):
        raise ValueError("the registered notched morph requires half_length = 25 mm")
    spec.check_mesh(reference)
    return reference, morph


def _values(spec: ProblemSpec, values: Mapping[str, float]) -> dict[str, float]:
    result = {}
    for parameter in spec.parameters:
        raw = values.get(parameter.name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"parameter {parameter.name!r} must be a finite number")
        number = float(raw)
        if not math.isfinite(number) or not parameter.lo <= number <= parameter.hi:
            raise ValueError(
                f"parameter {parameter.name!r} is outside its declared range"
            )
        result[parameter.name] = number
    for constraint in spec.constraints:
        try:
            value = expression_value(constraint.expression, spec.scope(result))
        except (ArithmeticError, KeyError) as exc:
            raise ValueError(
                f"parameter constraint {constraint.name!r} cannot be evaluated"
            ) from exc
        if not math.isfinite(value) or not constraint.lo <= value <= constraint.hi:
            raise ValueError(
                f"parameter constraint {constraint.name!r} is outside its declared range"
            )
    return result


def geometry_values(
    adapter_id: str, spec: ProblemSpec, values: Mapping[str, float], level: int = 1
) -> CanonicalMesh:
    """Morph one declared level; invalid parameter/mesh states remain refusals."""

    _level(level)
    _, morph = export_geometry(adapter_id, spec)
    point = _values(spec, values)
    theta = {
        name: CanonicalQuantity(point[name], Unit.MM, Dimension.LENGTH)
        for name in morph.parameter_names
    }
    template = _registered_template(adapter_id, spec)
    reference, _ = _template(adapter_id, template.resolution, level)
    mesh, report = validate_deformation(reference, morph.bind(theta), _VALIDITY)
    if mesh is None:
        raise ValueError(f"geometry level {level} is invalid: {report.reason_codes}")
    return mesh
