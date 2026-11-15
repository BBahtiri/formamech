"""``load_bundle``, ``predict`` and ``inspect``: the bundle's inference closure.

Loading verifies every byte against the manifest, decodes every specification
exactly, rebuilds the operator from its weights and refuses a bundle whose
parts disagree. ``predict`` checks a request against the executable envelope
(S9 bounds, S6 morph validity, topology, tags and query containment) before
anything is evaluated, then serves the declared fields in float64. Every
result, accepted or refused, carries the bundle's limitation paragraph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import torch

from piginot.bundle.limitations import REQUIRED_LIMITATION_PARAGRAPH
from piginot.bundle.manifest import (
    BundleIntegrityError,
    BundleManifest,
    BundleReasonCode,
    read_json,
    verify_bundle,
)
from piginot.bundle.problem import MATERIAL_DOMAIN_ERRORS
from piginot.bundle.problem import PredictionReasonCode as PredictionReasonCode
from piginot.bundle.serving import (
    InadmissibleDeformation,
    ServingError,
    ServingModel,
    StressAuthorityMismatch,
    build_serving,
)
from piginot.bundle.specs import (
    ModelSpec,
    SpecDecodeError,
    decode_envelope,
    decode_mesh,
    decode_model_spec,
    decode_morph_map,
)
from piginot.contracts import CanonicalMesh, CanonicalQuantity, MaterialSpec
from piginot.doe.envelope import ExecutableEnvelope
from piginot.morphing import apply_morph
from piginot.operators.energy_operator import (
    AxisAlignedTensionLifting,
    QueryFrame,
    embed_plane_stress_deformation,
)
from piginot.physics.neo_hookean import first_piola
from piginot.physics.plane_stress import plane_stress_closure
from piginot.verification.confidence import (
    QuantileCurve,
    boundary_traction_residual,
    cell_quadrature_points,
    vertex_patch_mean,
)
from piginot.verification.envelope import (
    EnvelopeChecker,
    GeometryOutcome,
    RefusalDecision,
)
from piginot.verification.status import ClaimLevel

REFERENCE_MESH_PATH = "spec/reference-mesh.json"
MORPH_MAP_PATH = "spec/morph-map.json"
ENVELOPE_PATH = "spec/envelope.json"
MODEL_PATH = "spec/model.json"
WEIGHTS_PATH = "weights/model.safetensors"
VERIFICATION_PATH = "report/verification.json"
LIMITATIONS_PATH = "report/limitations.json"
BUNDLE_FILES = tuple(
    sorted(
        (
            ENVELOPE_PATH,
            LIMITATIONS_PATH,
            MODEL_PATH,
            MORPH_MAP_PATH,
            REFERENCE_MESH_PATH,
            VERIFICATION_PATH,
            WEIGHTS_PATH,
        )
    )
)
OUTSTANDING_STATUSES = ("FAIL", "ERROR", "NOT_RUN", "SKIPPED", "WAIVED")
FIELD_NAMES = ("u", "F_2D", "F33", "J_3D", "P_served", "sigma")
_AXES = {"x": 0, "y": 1}
_BARYCENTRIC_TOLERANCE = 1.0e-9


class Verdict(StrEnum):
    IN_DOMAIN = "IN_DOMAIN"
    REFUSED = "REFUSED"


def _refuse(code: BundleReasonCode, detail: str) -> BundleIntegrityError:
    return BundleIntegrityError((code,), detail)


# ------------------------------------------------------------------ loading


@dataclass(frozen=True, slots=True)
class Bundle:
    """A verified bundle ready to serve."""

    path: Path
    manifest: BundleManifest
    reference_mesh: CanonicalMesh
    morph_map: Any
    envelope: ExecutableEnvelope
    model: ModelSpec
    serving: ServingModel
    verification: Mapping[str, Any]
    limitations: Mapping[str, Any]
    checker: EnvelopeChecker = field(repr=False)

    @property
    def claim_level(self) -> ClaimLevel:
        return ClaimLevel(self.manifest.claim_level)

    @property
    def identities(self) -> dict[str, str]:
        files = self.manifest.files
        return {
            "bundle_name": self.manifest.name,
            "bundle_manifest_sha256": self.manifest.manifest_sha256,
            "compiled_family_hash": self.envelope.compiled_family_hash,
            "envelope_id": self.envelope.envelope_id,
            "model_spec_sha256": files[MODEL_PATH].sha256,
            "reference_mesh_id": self.reference_mesh.mesh_id,
            "weights_sha256": files[WEIGHTS_PATH].sha256,
            **(self.model.problem.identities if self.model.problem is not None else {}),
        }

    def geometry(self, inputs: Mapping[str, object]) -> GeometryOutcome:
        """Morph the reference mesh to the geometry inputs (S6 validity)."""

        theta = {
            bound.name: inputs[bound.name]
            for bound in self.envelope.geometry_membership.primitive_bounds
        }
        result = apply_morph(
            self.reference_mesh,
            self.morph_map,
            self.envelope.geometry_membership,
            theta,
        )
        return GeometryOutcome(
            result.mesh if result.accepted else None,
            tuple(reason.value for reason in result.reason_codes),
        )


def _check_reports(
    manifest: BundleManifest, contents: Mapping[str, bytes]
) -> tuple[dict[str, Any], dict[str, Any]]:
    verification = read_json(contents, VERIFICATION_PATH)
    limitations = read_json(contents, LIMITATIONS_PATH)
    if limitations.get("paragraph") != REQUIRED_LIMITATION_PARAGRAPH:
        raise _refuse(
            BundleReasonCode.LIMITATIONS_MISSING,
            "report/limitations.json lacks the required limitation paragraph",
        )
    items = limitations.get("items")
    if not isinstance(items, list) or not all(
        isinstance(item, dict) and item.get("id") and item.get("statement")
        for item in items
    ):
        raise _refuse(BundleReasonCode.REPORT_INVALID, "limitation items are malformed")
    try:
        claim = ClaimLevel(manifest.claim_level)
    except ValueError as exc:
        raise _refuse(BundleReasonCode.REPORT_INVALID, "unknown claim level") from exc
    if verification.get("claim_level") != claim.value:
        raise _refuse(
            BundleReasonCode.REPORT_INVALID,
            "manifest claim level differs from the verification report",
        )
    if claim is not ClaimLevel.NOT_VERIFIED and verification.get("verdict") != "PASS":
        raise _refuse(
            BundleReasonCode.REPORT_INVALID,
            "a verified claim needs a passing verification report",
        )
    outstanding = verification.get("outstanding_work")
    if not isinstance(outstanding, list) or not all(
        isinstance(item, dict)
        and item.get("id")
        and item.get("status") in OUTSTANDING_STATUSES
        and item.get("reason")
        for item in outstanding
    ):
        raise _refuse(
            BundleReasonCode.REPORT_INVALID,
            "outstanding work must list every non-passing item with its reason",
        )
    return verification, limitations


def _check_consistency(
    mesh: CanonicalMesh, envelope: ExecutableEnvelope, model: ModelSpec
) -> None:
    membership = envelope.geometry_membership
    problems: list[str] = []
    if membership.reference_mesh_id != mesh.mesh_id:
        problems.append("membership reference mesh differs from the bundled mesh")
    if membership.reference_topology_signature != mesh.topology_signature:
        problems.append("membership topology differs from the bundled mesh")
    bounded = {bound.name for bound in envelope.primitive_bounds}
    unbounded = sorted(set(model.inputs) - bounded)
    if unbounded:
        problems.append(
            f"the model reads inputs the envelope does not bound: {unbounded}"
        )
    tags = set(mesh.semantic_tags)
    if set(model.boundary_conditions) - tags:
        problems.append("boundary conditions name unknown tags")
    for qoi in model.qois:
        if qoi.tag not in tags or set(qoi.exclude_tags) - tags:
            problems.append(f"QoI {qoi.name} names unknown tags")
    if problems:
        raise _refuse(BundleReasonCode.SPEC_INCONSISTENT, "; ".join(problems))


def load_bundle(path: Path | str) -> Bundle:
    """Verify and load a bundle directory; refuses with stable reason codes."""

    directory = Path(path)
    manifest, contents = verify_bundle(directory)
    if tuple(sorted(manifest.files)) != BUNDLE_FILES:
        raise _refuse(
            BundleReasonCode.FILE_SET_INVALID,
            f"expected exactly {', '.join(BUNDLE_FILES)}",
        )
    try:
        mesh = decode_mesh(contents[REFERENCE_MESH_PATH])
        morph_map = decode_morph_map(contents[MORPH_MAP_PATH])
        envelope = decode_envelope(contents[ENVELOPE_PATH])
        model = decode_model_spec(contents[MODEL_PATH])
    except (SpecDecodeError, KeyError, TypeError, ValueError) as exc:
        raise _refuse(BundleReasonCode.SPEC_INVALID, str(exc)) from exc
    _check_consistency(mesh, envelope, model)
    if model.problem is not None:
        try:
            model.problem.check_bundle(mesh, morph_map, envelope)
        except ValueError as exc:
            raise _refuse(BundleReasonCode.SPEC_INCONSISTENT, str(exc)) from exc
    verification, limitations = _check_reports(manifest, contents)
    if (
        model.problem is not None
        and manifest.claim_level != ClaimLevel.NOT_VERIFIED.value
    ):
        raise _refuse(
            BundleReasonCode.REPORT_INVALID,
            "portable problem candidates must be NOT_VERIFIED",
        )
    try:
        serving = build_serving(
            model.architecture_id,
            model.operator_config,
            contents[WEIGHTS_PATH],
            served_stress=model.served_stress,
            **(
                {"problem_hash": model.problem.problem_hash}
                if model.problem is not None
                else {}
            ),
        )
    except StressAuthorityMismatch as exc:
        raise _refuse(BundleReasonCode.STRESS_AUTHORITY_MISMATCH, str(exc)) from exc
    except ServingError as exc:
        raise _refuse(BundleReasonCode.WEIGHTS_INVALID, str(exc)) from exc
    holder: dict[str, Bundle] = {}

    def geometry(inputs: Mapping[str, object]) -> GeometryOutcome:
        return holder["bundle"].geometry(inputs)

    bundle = Bundle(
        path=directory,
        manifest=manifest,
        reference_mesh=mesh,
        morph_map=morph_map,
        envelope=envelope,
        model=model,
        serving=serving,
        verification=MappingProxyType(verification),
        limitations=MappingProxyType(limitations),
        checker=EnvelopeChecker(envelope, reference_mesh=mesh, geometry=geometry),
    )
    holder["bundle"] = bundle
    return bundle


# ------------------------------------------------------------------ predict


@dataclass(frozen=True, slots=True)
class PointFields:
    """Served fields at a set of points, all float64."""

    coordinates: torch.Tensor
    u: torch.Tensor
    F_2D: torch.Tensor
    F33: torch.Tensor
    J_3D: torch.Tensor
    P_served: torch.Tensor
    P_material: torch.Tensor
    sigma: torch.Tensor
    confidence_indicator: torch.Tensor | None = None
    confidence_bound: torch.Tensor | None = None
    beyond_calibration: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class QoiValue:
    name: str
    value: float
    unit: str
    vertex: int
    location: tuple[float, float]


@dataclass(frozen=True, slots=True)
class Prediction:
    """The full answer to one request, accepted or refused."""

    verdict: Verdict
    reason_codes: tuple[str, ...]
    claim_level: ClaimLevel
    identities: Mapping[str, str]
    units: Mapping[str, str]
    stress_authority: str
    limitations: str
    mesh: CanonicalMesh | None = None
    vertices: PointFields | None = None
    queries: PointFields | None = None
    qois: tuple[QoiValue, ...] = ()
    confidence_quantile: float | None = None
    constitutive_gap: float | None = None

    @property
    def confidence_status(self) -> str | None:
        return "NOT_CALIBRATED" if "problem_hash" in self.identities else None

    @property
    def in_domain(self) -> bool:
        return self.verdict is Verdict.IN_DOMAIN

    def document(self) -> dict[str, Any]:
        """JSON-ready report of the prediction."""

        def fields(points: PointFields | None) -> dict[str, Any] | None:
            if points is None:
                return None
            result: dict[str, Any] = {
                "coordinates": points.coordinates.tolist(),
                **{name: getattr(points, name).tolist() for name in FIELD_NAMES},
            }
            if points.confidence_indicator is not None:
                assert points.confidence_bound is not None
                assert points.beyond_calibration is not None
                result["confidence"] = {
                    "indicator": points.confidence_indicator.tolist(),
                    "stress_error_bound": points.confidence_bound.tolist(),
                    "beyond_calibration": points.beyond_calibration.tolist(),
                }
            return result

        result = {
            "verdict": self.verdict.value,
            "reason_codes": list(self.reason_codes),
            "claim_level": self.claim_level.value,
            "identities": dict(self.identities),
            "units": dict(self.units),
            "stress_authority": self.stress_authority,
            "limitations": self.limitations,
            "mesh": None
            if self.mesh is None
            else {
                "mesh_id": self.mesh.mesh_id,
                "cells": [list(cell) for cell in self.mesh.cells],
            },
            "vertices": fields(self.vertices),
            "queries": fields(self.queries),
            "qois": [
                {
                    "name": q.name,
                    "value": q.value,
                    "unit": q.unit,
                    "vertex": q.vertex,
                    "location": list(q.location),
                }
                for q in self.qois
            ],
            "confidence_quantile": self.confidence_quantile,
            "constitutive_gap": self.constitutive_gap,
        }
        if self.confidence_status is not None:
            result["confidence_status"] = self.confidence_status
        return result


def _refused(bundle: Bundle, codes: Sequence[str]) -> Prediction:
    return Prediction(
        verdict=Verdict.REFUSED,
        reason_codes=tuple(sorted(set(codes))),
        claim_level=ClaimLevel.NOT_VERIFIED,
        identities=MappingProxyType(bundle.identities),
        units=MappingProxyType(dict(bundle.model.units)),
        stress_authority=bundle.serving.stress_authority,
        limitations=REQUIRED_LIMITATION_PARAGRAPH,
    )


def _decide(
    bundle: Bundle,
    inputs: Mapping[str, object],
    queries: Sequence[object] | None,
) -> tuple[RefusalDecision, CanonicalMesh | None]:
    """The envelope decision, morphing an accepted geometry once."""

    if isinstance(inputs, Mapping) and bundle.envelope.evaluate(inputs).accepted:
        if bundle.model.problem is not None and not bundle.model.problem.accepts(
            inputs
        ):
            return RefusalDecision(
                False, (PredictionReasonCode.PROBLEM_OUT_OF_DOMAIN.value,)
            ), None
        outcome = bundle.geometry(inputs)
        if outcome.mesh is not None and not outcome.reason_codes:
            decision = bundle.checker.check(inputs, mesh=outcome.mesh, queries=queries)
            return decision, outcome.mesh if decision.accepted else None
    return bundle.checker.check(inputs, queries=queries), None


def _tag_vertices(mesh: CanonicalMesh, tag: str) -> set[int]:
    vertices: set[int] = set()
    for index in mesh.semantic_tags[tag]:
        vertices.update(mesh.boundary_facets[index])
    return vertices


def _free_components(
    mesh: CanonicalMesh, constrained: Mapping[str, tuple[str, ...]]
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, ...], ...]]:
    free: list[tuple[int, ...] | None] = [None] * len(mesh.boundary_facets)
    for tag, indices in mesh.semantic_tags.items():
        fixed = {_AXES[name] for name in constrained.get(tag, ())}
        for index in indices:
            free[index] = tuple(c for c in (0, 1) if c not in fixed)
    if any(entry is None for entry in free):
        raise ValueError("every boundary facet needs a semantic tag")
    facets = tuple((int(a), int(b)) for a, b in mesh.boundary_facets)
    return facets, tuple(entry for entry in free if entry is not None)


def containing_cells(
    vertices: torch.Tensor, cells: torch.Tensor, points: torch.Tensor
) -> list[tuple[int, ...]]:
    """Every cell whose closure holds each point (nearest cell as a fallback)."""

    corners = vertices[cells]
    origin = corners[:, 0]
    first = corners[:, 1] - origin
    second = corners[:, 2] - origin
    determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    result: list[tuple[int, ...]] = []
    for start in range(0, points.shape[0], 256):
        block = points[start : start + 256, None, :] - origin[None]
        b1 = (block[..., 0] * second[:, 1] - block[..., 1] * second[:, 0]) / determinant
        b2 = (first[:, 0] * block[..., 1] - first[:, 1] * block[..., 0]) / determinant
        smallest = torch.minimum(torch.minimum(1.0 - b1 - b2, b1), b2)
        for row in smallest:
            inside = torch.nonzero(row >= -_BARYCENTRIC_TOLERANCE).flatten()
            if inside.numel() == 0:
                inside = torch.argmax(row).reshape(1)
            result.append(tuple(int(i) for i in inside))
    return result


class _Request:
    """The tensors one accepted request needs."""

    def __init__(
        self, bundle: Bundle, inputs: Mapping[str, object], mesh: CanonicalMesh
    ):
        spec = bundle.model
        values: dict[str, float] = {}
        for name in spec.inputs:
            quantity = inputs[name]
            assert isinstance(quantity, CanonicalQuantity)
            values[name] = float(quantity.value)
        dtype = torch.float64
        self.parameters = torch.tensor(
            [
                (values[item.source] - item.mean) / item.standard_deviation
                for item in spec.conditioning
            ],
            dtype=dtype,
        ).unsqueeze(0)
        if spec.problem is not None:
            self.frame, self.lifting, self.material, self.elastic_modulus = (
                spec.problem.request_context(values)
            )
        else:
            assert spec.frame_origin is not None and spec.frame_extent is not None
            assert spec.lifting_displacement is not None
            origin = [item.evaluate(values) for item in spec.frame_origin]
            extent = [item.evaluate(values) for item in spec.frame_extent]
            self.frame = QueryFrame(
                torch.tensor([origin], dtype=dtype), torch.tensor([extent], dtype=dtype)
            )
            self.lifting = AxisAlignedTensionLifting(
                torch.tensor([extent], dtype=dtype),
                torch.tensor([spec.lifting_displacement.evaluate(values)], dtype=dtype),
            )
            self.material = MaterialSpec(
                name=spec.material_name,
                model=spec.material_model,
                properties={
                    prop: cast(CanonicalQuantity, inputs[source])
                    for prop, source in spec.material_parameters.items()
                },
            )
            self.elastic_modulus = float(
                self.material.properties["elastic_modulus"].value
            )
        self.vertices = torch.tensor(mesh.nodes, dtype=dtype)
        self.cells = torch.tensor(mesh.cells, dtype=torch.int64)


def _fields(bundle: Bundle, request: _Request, points: torch.Tensor) -> PointFields:
    served = bundle.serving.serve(
        points,
        parameters=request.parameters,
        frame=request.frame,
        lifting=request.lifting,
        material=request.material,
        chunk=bundle.model.query_chunk,
    )
    deformation = torch.eye(2, dtype=points.dtype) + served.gradient
    if bundle.model.problem is not None:
        from piginot.physics import first_piola as material_piola
        from piginot.physics import plane_stress_closure as material_closure
    else:
        material_piola, material_closure = first_piola, plane_stress_closure
    f33 = material_closure(deformation, request.material)
    deformation_3d = embed_plane_stress_deformation(deformation, f33)
    determinant = torch.linalg.det(deformation_3d)
    material_stress = material_piola(deformation_3d, request.material)[..., :2, :2]
    sigma = served.stress @ deformation.transpose(-2, -1) / determinant[:, None, None]
    return PointFields(
        coordinates=points,
        u=served.u,
        F_2D=deformation,
        F33=f33,
        J_3D=determinant,
        P_served=served.stress,
        P_material=material_stress,
        sigma=sigma,
    )


def _vertex_indicator(
    bundle: Bundle, request: _Request, mesh: CanonicalMesh, vertex_stress: torch.Tensor
) -> torch.Tensor:
    spec = bundle.model.confidence
    assert spec is not None
    points = cell_quadrature_points(request.vertices, request.cells)
    residual = bundle.serving.residual(
        points.reshape(-1, 2),
        parameters=request.parameters,
        frame=request.frame,
        lifting=request.lifting,
        material=request.material,
        length_scale=spec.length_scale_mm,
    )
    cell_mean = residual.reshape(points.shape[:2]).mean(dim=1)
    patch = vertex_patch_mean(request.vertices.shape[0], request.cells, cell_mean)
    facets, free = _free_components(mesh, bundle.model.boundary_conditions)
    traction = boundary_traction_residual(
        vertex_stress,
        request.vertices,
        request.cells,
        facets,
        free,
        elastic_modulus=request.elastic_modulus,
    )
    return patch + traction


def _with_confidence(
    points: PointFields, indicator: torch.Tensor, bundle: Bundle
) -> PointFields:
    spec = bundle.model.confidence
    assert spec is not None
    curve = QuantileCurve(spec.quantile, spec.upper_edges, spec.values)
    return PointFields(
        coordinates=points.coordinates,
        u=points.u,
        F_2D=points.F_2D,
        F33=points.F33,
        J_3D=points.J_3D,
        P_served=points.P_served,
        P_material=points.P_material,
        sigma=points.sigma,
        confidence_indicator=indicator,
        confidence_bound=curve.predict(indicator) * spec.reference_stress_mpa,
        beyond_calibration=indicator > spec.upper_edges[-1],
    )


def _qois(
    bundle: Bundle, mesh: CanonicalMesh, vertices: PointFields
) -> tuple[QoiValue, ...]:
    values: list[QoiValue] = []
    for qoi in bundle.model.qois:
        selected = _tag_vertices(mesh, qoi.tag)
        for tag in qoi.exclude_tags:
            selected -= _tag_vertices(mesh, tag)
        if not selected:
            raise ValueError(f"QoI {qoi.name} has no vertices")
        index = torch.tensor(sorted(selected), dtype=torch.int64)
        tensor = getattr(vertices, qoi.field)
        norms = torch.linalg.vector_norm(tensor[index], dim=(-2, -1))
        best = int(index[int(torch.argmax(norms))])
        x, y = (float(v) for v in vertices.coordinates[best])
        values.append(QoiValue(qoi.name, float(norms.max()), qoi.unit, best, (x, y)))
    return tuple(values)


def predict(
    bundle: Bundle,
    inputs: Mapping[str, object],
    queries: Sequence[object] | None = None,
    *,
    confidence: bool = True,
) -> Prediction:
    """Serve one request, or refuse it with sorted stable reason codes."""

    decision, mesh = _decide(bundle, inputs, queries)
    if not decision.accepted or mesh is None:
        return _refused(bundle, decision.reason_codes)
    request = _Request(bundle, inputs, mesh)
    try:
        vertices = _fields(bundle, request, request.vertices)
        query_fields = (
            None
            if queries is None
            else _fields(
                bundle,
                request,
                torch.tensor([list(point) for point in queries], dtype=torch.float64),  # type: ignore[call-overload]
            )
        )
    except InadmissibleDeformation:
        return _refused(bundle, (PredictionReasonCode.INADMISSIBLE_DEFORMATION.value,))
    except MATERIAL_DOMAIN_ERRORS:
        if bundle.model.problem is None:
            raise
        return _refused(bundle, (PredictionReasonCode.MATERIAL_DOMAIN_FAILURE.value,))
    if not all(
        bool(torch.isfinite(getattr(points, name)).all())
        for points in (vertices, query_fields)
        if points is not None
        for name in FIELD_NAMES
    ):
        return _refused(bundle, (PredictionReasonCode.INADMISSIBLE_DEFORMATION.value,))
    if confidence and bundle.model.confidence is not None:
        indicator = _vertex_indicator(bundle, request, mesh, vertices.P_served)
        vertices = _with_confidence(vertices, indicator, bundle)
        if query_fields is not None:
            owners = containing_cells(
                request.vertices, request.cells, query_fields.coordinates
            )
            query_indicator = torch.stack(
                [
                    indicator[request.cells[list(cells)].flatten()].max()
                    for cells in owners
                ]
            )
            query_fields = _with_confidence(query_fields, query_indicator, bundle)
    gap = max(
        float(
            torch.linalg.vector_norm(
                points.P_served - points.P_material, dim=(-2, -1)
            ).max()
        )
        for points in (vertices, query_fields)
        if points is not None
    )
    return Prediction(
        verdict=Verdict.IN_DOMAIN,
        reason_codes=(),
        claim_level=bundle.claim_level,
        identities=MappingProxyType({**bundle.identities, "mesh_id": mesh.mesh_id}),
        units=MappingProxyType(dict(bundle.model.units)),
        stress_authority=bundle.serving.stress_authority,
        limitations=REQUIRED_LIMITATION_PARAGRAPH,
        mesh=mesh,
        vertices=vertices,
        queries=query_fields,
        qois=_qois(bundle, mesh, vertices),
        confidence_quantile=bundle.model.confidence.quantile
        if confidence and bundle.model.confidence is not None
        else None,
        constitutive_gap=gap / request.elastic_modulus,
    )


# ------------------------------------------------------------------ inspect


def _bound(bound: Any) -> dict[str, Any]:
    return {
        "name": bound.name,
        "lower": bound.lower_bound.value,
        "upper": bound.upper_bound.value,
        "unit": bound.lower_bound.unit.value,
    }


def inspect(bundle: Bundle) -> dict[str, Any]:
    """What the bundle serves, what it was verified for, and what is left."""

    verification = bundle.verification
    outstanding = list(verification.get("outstanding_work", []))
    return {
        "format": bundle.manifest.format,
        "name": bundle.manifest.name,
        "created_utc": bundle.manifest.created_utc,
        "claim_level": bundle.claim_level.value,
        "identities": bundle.identities,
        "build_identities": dict(sorted(bundle.manifest.identities.items())),
        "files": {
            path: {"sha256": entry.sha256, "bytes": entry.bytes}
            for path, entry in sorted(bundle.manifest.files.items())
        },
        "envelope": {
            "envelope_id": bundle.envelope.envelope_id,
            "inputs": [_bound(b) for b in bundle.envelope.primitive_bounds],
            "derived_bounds": len(bundle.envelope.geometry_membership.derived_bounds),
        },
        "model": {
            "architecture_id": bundle.model.architecture_id,
            "parameter_count": bundle.serving.parameter_count,
            "served_stress": bundle.serving.stress_authority,
            "inputs": list(bundle.model.inputs),
            "qois": [q.name for q in bundle.model.qois],
            "confidence": None
            if bundle.model.confidence is None
            else {
                "variant": bundle.model.confidence.variant,
                "quantile": bundle.model.confidence.quantile,
            },
            "units": dict(bundle.model.units),
        },
        "verification": {
            "gate_id": verification.get("gate_id"),
            "verdict": verification.get("verdict"),
            "claim_level": verification.get("claim_level"),
            "outstanding_work": outstanding,
        },
        "limitations": {
            "paragraph": bundle.limitations["paragraph"],
            "items": list(bundle.limitations["items"]),
        },
    }


__all__ = (
    "BUNDLE_FILES",
    "ENVELOPE_PATH",
    "FIELD_NAMES",
    "LIMITATIONS_PATH",
    "MODEL_PATH",
    "MORPH_MAP_PATH",
    "OUTSTANDING_STATUSES",
    "REFERENCE_MESH_PATH",
    "VERIFICATION_PATH",
    "WEIGHTS_PATH",
    "Bundle",
    "PointFields",
    "Prediction",
    "PredictionReasonCode",
    "QoiValue",
    "Verdict",
    "containing_cells",
    "inspect",
    "load_bundle",
    "predict",
)
