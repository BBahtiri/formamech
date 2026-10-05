"""Exact codecs for the family, envelope and model specifications of a bundle.

Every decoder rebuilds the typed object and requires its canonical JSON to equal
the stored bytes, so a bundle cannot carry a specification that decodes to
something other than what was hashed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import rfc8785

from piginot.contracts import (
    CanonicalMesh,
    Unit,
    canonical_json_bytes,
    hash_spec,
    strict_json_loads,
)
from piginot.doe.envelope import ExecutableEnvelope
from piginot.morphing.blended_notch import BlendedNotchMap, BlendedNotchParameters
from piginot.morphing.circular_transition import (
    CircularTransitionMap,
    CircularTransitionParameters,
)
from piginot.morphing.membership import (
    FamilyMembershipPredicate,
    PrimitiveBound,
    _bound_from_payload,
)


class SpecDecodeError(ValueError):
    """A stored specification does not decode to exactly its own bytes."""


def _object(payload: bytes | str, label: str) -> dict[str, Any]:
    document = strict_json_loads(payload)
    if not isinstance(document, dict):
        raise SpecDecodeError(f"{label} must be a JSON object")
    return document


def _exact(value: object, payload: bytes | str, label: str) -> None:
    expected = payload.encode() if isinstance(payload, str) else payload
    if canonical_json_bytes(value) != expected:
        raise SpecDecodeError(f"{label} does not round-trip to its stored bytes")


def mesh_structural_id(mesh: CanonicalMesh) -> str:
    """The content identity a canonical mesh carries as ``mesh_id``."""

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


def decode_mesh(payload: bytes) -> CanonicalMesh:
    """A canonical mesh from its canonical JSON, whose identity is its content."""

    document = _object(payload, "mesh")
    if document.get("$type") != "piginot.contracts.CanonicalMesh":
        raise SpecDecodeError("not a canonical mesh")
    mesh = CanonicalMesh(
        mesh_id=document["mesh_id"],
        nodes=tuple(tuple(float(v) for v in node) for node in document["nodes"]),  # type: ignore[misc]
        cells=tuple(tuple(int(v) for v in cell) for cell in document["cells"]),
        cell_type=document["cell_type"],
        boundary_facets=tuple((int(a), int(b)) for a, b in document["boundary_facets"]),
        semantic_tags={
            name: tuple(int(i) for i in facets)
            for name, facets in document["semantic_tags"].items()
        },
        region_ids=tuple(int(i) for i in document["region_ids"]),
        unit=Unit(document["unit"]),
        orientation=document["orientation"],
        topology_signature=document["topology_signature"],
        quality_metrics={k: float(v) for k, v in document["quality_metrics"].items()},
        source_provenance=dict(document["source_provenance"]),
    )
    _exact(mesh, payload, "mesh")
    if mesh_structural_id(mesh) != mesh.mesh_id:
        raise SpecDecodeError("mesh identity differs from its content")
    return mesh


def _circular_transition(document: Mapping[str, Any]) -> CircularTransitionMap:
    reference = document["reference"]
    return CircularTransitionMap(
        reference=CircularTransitionParameters(
            total_length_mm=float(reference["total_length_mm"]),
            outer_width_mm=float(reference["outer_width_mm"]),
            inner_width_mm=float(reference["inner_width_mm"]),
            transition_radius_mm=float(reference["transition_radius_mm"]),
        ),
        total_length_parameter=document["total_length_parameter"],
        outer_width_parameter=document["outer_width_parameter"],
        inner_width_parameter=document["inner_width_parameter"],
        transition_radius_parameter=document["transition_radius_parameter"],
        minimum_inner_length_mm=float(document["minimum_inner_length_mm"]),
    )


def _blended_notch(document: Mapping[str, Any]) -> BlendedNotchMap:
    reference = document["reference"]
    return BlendedNotchMap(
        reference=BlendedNotchParameters(
            half_length_mm=float(reference["half_length_mm"]),
            width_mm=float(reference["width_mm"]),
            notch_radius_mm=float(reference["notch_radius_mm"]),
            notch_depth_mm=float(reference["notch_depth_mm"]),
        ),
        width_parameter=document["width_parameter"],
        notch_radius_parameter=document["notch_radius_parameter"],
        notch_depth_parameter=document["notch_depth_parameter"],
    )


MORPH_MAP_DECODERS: Mapping[str, Callable[[Mapping[str, Any]], object]] = {
    "piginot.morphing.BlendedNotchMap": _blended_notch,
    "piginot.morphing.CircularTransitionMap": _circular_transition,
}


def decode_morph_map(payload: bytes) -> object:
    """A registered morph map from its canonical JSON."""

    document = _object(payload, "morph map")
    decoder = MORPH_MAP_DECODERS.get(str(document.get("$type")))
    if decoder is None:
        raise SpecDecodeError(f"unsupported morph map {document.get('$type')!r}")
    morph_map = decoder(document)
    _exact(morph_map, payload, "morph map")
    return morph_map


def decode_membership(payload: bytes) -> FamilyMembershipPredicate:
    membership = FamilyMembershipPredicate.from_bytes(payload)
    _exact(membership, payload, "membership predicate")
    return membership


def decode_envelope(payload: bytes) -> ExecutableEnvelope:
    """An executable S9 envelope from its canonical JSON."""

    document = _object(payload, "envelope")
    if document.get("$type") != "piginot.doe.ExecutableEnvelope":
        raise SpecDecodeError("not an executable envelope")

    def bounds(key: str) -> tuple[PrimitiveBound, ...]:
        return tuple(
            cast(PrimitiveBound, _bound_from_payload(item, derived=False))
            for item in document[key]
        )

    envelope = ExecutableEnvelope(
        envelope_id=document["envelope_id"],
        compiled_family_hash=document["compiled_family_hash"],
        geometry_membership=decode_membership(
            rfc8785.dumps(document["geometry_membership"])
        ),
        geometry_membership_hash=document["geometry_membership_hash"],
        material_bounds=bounds("material_bounds"),
        load_bounds=bounds("load_bounds"),
        refusal_policy_id=document["refusal_policy_id"],
        schema_version=document["schema_version"],
    )
    _exact(envelope, payload, "envelope")
    return envelope


# ------------------------------------------------------------------ model spec

SUPPORTED_DTYPES = ("float64",)
SUPPORTED_LIFTINGS = ("axis-aligned-tension",)
SUPPORTED_QOI_FIELDS = ("P_served", "sigma")
_COMPONENTS = ("x", "y")


@dataclass(frozen=True, slots=True)
class ScaledParameter:
    """``scale * product(parameters)`` of request values, left to right."""

    parameters: tuple[str, ...]
    scale: float

    def evaluate(self, values: Mapping[str, float]) -> float:
        result = self.scale
        for name in self.parameters:
            result *= values[name]
        return result

    def document(self) -> dict[str, Any]:
        return {"parameters": list(self.parameters), "scale": self.scale}


@dataclass(frozen=True, slots=True)
class ConditioningInput:
    """One operator input: ``(request[source] - mean) / standard_deviation``."""

    name: str
    source: str
    mean: float
    standard_deviation: float

    def __post_init__(self) -> None:
        if not self.name or not self.source:
            raise SpecDecodeError("conditioning inputs need a name and a source")
        if not math.isfinite(self.mean) or not (
            math.isfinite(self.standard_deviation) and self.standard_deviation > 0.0
        ):
            raise SpecDecodeError(f"{self.name} has an invalid normalisation")


@dataclass(frozen=True, slots=True)
class QoiSpec:
    """The largest Frobenius norm of a served field over tagged vertices."""

    name: str
    field: str
    tag: str
    exclude_tags: tuple[str, ...]
    unit: str

    def __post_init__(self) -> None:
        if self.field not in SUPPORTED_QOI_FIELDS:
            raise SpecDecodeError(f"unsupported QoI field {self.field!r}")


@dataclass(frozen=True, slots=True)
class ConfidenceSpec:
    """The frozen indicator and its calibrated error quantile curve.

    ``values`` are quantiles of the pointwise in-plane stress error divided by
    ``reference_stress_mpa``; ``upper_edges`` bound the indicator bins.
    """

    variant: str
    length_scale_mm: float
    reference_stress_mpa: float
    quantile: float
    upper_edges: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.variant != "patch-mean":
            raise SpecDecodeError(f"unsupported confidence variant {self.variant!r}")
        if not 0.0 < self.quantile < 1.0:
            raise SpecDecodeError("the confidence quantile must lie in (0, 1)")
        if not self.values or len(self.values) != len(self.upper_edges):
            raise SpecDecodeError("confidence edges and values must align")
        for label, sequence in (("edges", self.upper_edges), ("values", self.values)):
            if any(b < a for a, b in zip(sequence, sequence[1:], strict=False)):
                raise SpecDecodeError(f"confidence {label} must be non-decreasing")
        for value in (self.length_scale_mm, self.reference_stress_mpa):
            if not math.isfinite(value) or value <= 0.0:
                raise SpecDecodeError("confidence scales must be positive")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Everything between a request and the operator call, and what is served."""

    architecture_id: str
    operator_config: Mapping[str, Any]
    conditioning: tuple[ConditioningInput, ...]
    frame_origin: tuple[ScaledParameter, ScaledParameter]
    frame_extent: tuple[ScaledParameter, ScaledParameter]
    lifting_kind: str
    lifting_displacement: ScaledParameter
    material_name: str
    material_model: str
    material_parameters: Mapping[str, str]
    boundary_conditions: Mapping[str, tuple[str, ...]]
    served_stress: str
    dtype: str
    query_chunk: int
    qois: tuple[QoiSpec, ...]
    confidence: ConfidenceSpec
    units: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.conditioning:
            raise SpecDecodeError("the operator needs at least one input")
        names = [item.name for item in self.conditioning]
        if len(set(names)) != len(names):
            raise SpecDecodeError("conditioning names must be unique")
        if list(self.operator_config.get("parameter_names", ())) != names:
            raise SpecDecodeError("conditioning order differs from the operator")
        if self.lifting_kind not in SUPPORTED_LIFTINGS:
            raise SpecDecodeError(f"unsupported lifting {self.lifting_kind!r}")
        if self.dtype not in SUPPORTED_DTYPES:
            raise SpecDecodeError(f"unsupported dtype {self.dtype!r}")
        if isinstance(self.query_chunk, bool) or self.query_chunk <= 0:
            raise SpecDecodeError("query_chunk must be a positive integer")
        for tag, components in self.boundary_conditions.items():
            if not tag or any(c not in _COMPONENTS for c in components):
                raise SpecDecodeError(f"boundary condition on {tag!r} is invalid")
        if not self.served_stress:
            raise SpecDecodeError("the served-stress authority must be declared")
        if len({q.name for q in self.qois}) != len(self.qois):
            raise SpecDecodeError("QoI names must be unique")

    @property
    def inputs(self) -> tuple[str, ...]:
        """Every request input the model reads, sorted."""

        names = {item.source for item in self.conditioning}
        names.update(self.material_parameters.values())
        for scaled in (
            *self.frame_origin,
            *self.frame_extent,
            self.lifting_displacement,
        ):
            names.update(scaled.parameters)
        return tuple(sorted(names))

    def document(self) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "operator_config": dict(self.operator_config),
            "conditioning": [
                {
                    "name": item.name,
                    "source": item.source,
                    "mean": item.mean,
                    "standard_deviation": item.standard_deviation,
                }
                for item in self.conditioning
            ],
            "frame": {
                "origin": [item.document() for item in self.frame_origin],
                "extent": [item.document() for item in self.frame_extent],
            },
            "lifting": {
                "kind": self.lifting_kind,
                "displacement": self.lifting_displacement.document(),
            },
            "material": {
                "name": self.material_name,
                "model": self.material_model,
                "parameters": dict(self.material_parameters),
            },
            "boundary_conditions": {
                tag: list(components)
                for tag, components in self.boundary_conditions.items()
            },
            "served_stress": self.served_stress,
            "dtype": self.dtype,
            "query_chunk": self.query_chunk,
            "qois": [
                {
                    "name": q.name,
                    "field": q.field,
                    "tag": q.tag,
                    "exclude_tags": list(q.exclude_tags),
                    "unit": q.unit,
                }
                for q in self.qois
            ],
            "confidence": {
                "variant": self.confidence.variant,
                "length_scale_mm": self.confidence.length_scale_mm,
                "reference_stress_mpa": self.confidence.reference_stress_mpa,
                "quantile": self.confidence.quantile,
                "upper_edges": list(self.confidence.upper_edges),
                "values": list(self.confidence.values),
            },
            "units": dict(self.units),
        }

    def to_bytes(self) -> bytes:
        return rfc8785.dumps(self.document())


def _scaled(value: Mapping[str, Any]) -> ScaledParameter:
    return ScaledParameter(
        tuple(str(name) for name in value["parameters"]), float(value["scale"])
    )


def _pair(values: Any, label: str) -> tuple[ScaledParameter, ScaledParameter]:
    items = tuple(_scaled(item) for item in values)
    if len(items) != 2:
        raise SpecDecodeError(f"the frame {label} needs two entries")
    return items[0], items[1]


def decode_model_spec(payload: bytes) -> ModelSpec:
    """The model specification from its canonical JSON."""

    document = _object(payload, "model spec")
    try:
        confidence = document["confidence"]
        spec = ModelSpec(
            architecture_id=document["architecture_id"],
            operator_config=dict(document["operator_config"]),
            conditioning=tuple(
                ConditioningInput(
                    item["name"],
                    item["source"],
                    float(item["mean"]),
                    float(item["standard_deviation"]),
                )
                for item in document["conditioning"]
            ),
            frame_origin=_pair(document["frame"]["origin"], "origin"),
            frame_extent=_pair(document["frame"]["extent"], "extent"),
            lifting_kind=document["lifting"]["kind"],
            lifting_displacement=_scaled(document["lifting"]["displacement"]),
            material_name=document["material"]["name"],
            material_model=document["material"]["model"],
            material_parameters=dict(document["material"]["parameters"]),
            boundary_conditions={
                tag: tuple(components)
                for tag, components in document["boundary_conditions"].items()
            },
            served_stress=document["served_stress"],
            dtype=document["dtype"],
            query_chunk=document["query_chunk"],
            qois=tuple(
                QoiSpec(
                    q["name"], q["field"], q["tag"], tuple(q["exclude_tags"]), q["unit"]
                )
                for q in document["qois"]
            ),
            confidence=ConfidenceSpec(
                variant=confidence["variant"],
                length_scale_mm=float(confidence["length_scale_mm"]),
                reference_stress_mpa=float(confidence["reference_stress_mpa"]),
                quantile=float(confidence["quantile"]),
                upper_edges=tuple(float(v) for v in confidence["upper_edges"]),
                values=tuple(float(v) for v in confidence["values"]),
            ),
            units=dict(document["units"]),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise SpecDecodeError(f"model spec is incomplete: {exc}") from exc
    if spec.to_bytes() != payload:
        raise SpecDecodeError("model spec does not round-trip to its stored bytes")
    return spec


__all__ = (
    "MORPH_MAP_DECODERS",
    "SUPPORTED_DTYPES",
    "SUPPORTED_LIFTINGS",
    "SUPPORTED_QOI_FIELDS",
    "ConditioningInput",
    "ConfidenceSpec",
    "ModelSpec",
    "QoiSpec",
    "ScaledParameter",
    "SpecDecodeError",
    "decode_envelope",
    "decode_membership",
    "decode_mesh",
    "decode_model_spec",
    "decode_morph_map",
    "mesh_structural_id",
)
