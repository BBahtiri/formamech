"""Unit-explicit extraction of immutable S7 result artifacts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from piginot.contracts import Dimension, Unit, hash_spec
from piginot.datasets.request_plan import PublicFemRequestPlan
from piginot.datasets.snapshots import (
    DatasetSnapshotValidationError,
    RoleDatasetSnapshot,
    validate_snapshot_attempt,
)
from piginot.datasets.units import AreaUnit
from piginot.doe import EvidenceRole
from piginot.fem import FemAttempt
from piginot.store import ObjectStore

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class FemExtractionError(ValueError):
    """A FEM attempt cannot be converted into trusted S10 artifacts."""


class FieldLocation(StrEnum):
    """Physical support on which a field artifact is represented."""

    DISPLACEMENT_DOF = "DISPLACEMENT_DOF"
    QUADRATURE = "QUADRATURE"
    RECOVERED_DOF = "RECOVERED_DOF"
    BOUNDARY = "BOUNDARY"
    GLOBAL = "GLOBAL"


@dataclass(frozen=True, slots=True)
class ExtractedArtifact:
    """One immutable numeric artifact with units and exact S7 provenance."""

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    case_id: str
    name: str
    location: FieldLocation
    unit: Unit
    dimension: Dimension
    values: tuple[object, ...]
    fem_result_hash: str
    coordinates: tuple[object, ...] = ()
    quadrature_weights: tuple[object, ...] = ()
    coordinate_unit: Unit | None = None
    quadrature_weight_unit: AreaUnit | None = None

    def __post_init__(self) -> None:
        if not self.case_id or not self.name:
            raise FemExtractionError("artifact case_id and name must be non-empty")
        location = FieldLocation(self.location)
        object.__setattr__(self, "location", location)
        if not isinstance(self.unit, Unit) or not isinstance(self.dimension, Dimension):
            raise FemExtractionError("artifact unit and dimension must be explicit")
        if not self.unit.supports_dimension(self.dimension):
            raise FemExtractionError("artifact unit does not support its dimension")
        if not self.values:
            raise FemExtractionError("artifact values must not be empty")
        if (
            not isinstance(self.fem_result_hash, str)
            or _DIGEST.fullmatch(self.fem_result_hash) is None
        ):
            raise FemExtractionError(
                "artifact fem_result_hash must be a lowercase digest"
            )
        values = tuple(self.values)
        coordinates = tuple(self.coordinates)
        weights = tuple(self.quadrature_weights)
        _finite_payload(values, "artifact values")
        _finite_payload(coordinates, "artifact coordinates")
        _finite_payload(weights, "artifact quadrature weights")
        if coordinates:
            if self.coordinate_unit is not Unit.MM:
                raise FemExtractionError(
                    "coordinate_unit must be canonical mm when coordinates exist"
                )
        elif self.coordinate_unit is not None:
            raise FemExtractionError(
                "coordinate_unit must be absent when coordinates are absent"
            )
        if weights:
            if not coordinates:
                raise FemExtractionError(
                    "quadrature weights require physical coordinates"
                )
            if self.quadrature_weight_unit is not AreaUnit.SQUARE_MILLIMETRE:
                raise FemExtractionError(
                    "quadrature_weight_unit must be canonical mm^2"
                )
        elif self.quadrature_weight_unit is not None:
            raise FemExtractionError(
                "quadrature_weight_unit must be absent when weights are absent"
            )
        if location is FieldLocation.QUADRATURE:
            _validate_quadrature_layout(values, coordinates, weights)
        elif weights:
            raise FemExtractionError(
                "quadrature weights are valid only for QUADRATURE artifacts"
            )
        if location in {
            FieldLocation.DISPLACEMENT_DOF,
            FieldLocation.RECOVERED_DOF,
        } and (not coordinates or len(values) != len(coordinates)):
            raise FemExtractionError(
                "DOF layout requires one physical coordinate per value"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "quadrature_weights", weights)

    @property
    def content_hash(self) -> str:
        return hash_spec(self, "content")


@dataclass(frozen=True, slots=True)
class ExtractedFemArtifacts:
    """Complete S10 field, reaction, energy, and QoI view of one FEM result."""

    __canonical_namespace__: ClassVar[str] = "piginot.datasets"

    case_id: str
    leakage_group_id: str
    role: EvidenceRole
    snapshot_content_hash: str
    request_plan_hash: str
    store_content_hash: str
    fem_result_hash: str
    fields: tuple[ExtractedArtifact, ...]
    reactions: tuple[ExtractedArtifact, ...]
    energies: tuple[ExtractedArtifact, ...]
    qois: tuple[ExtractedArtifact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise FemExtractionError("bundle case_id must be non-empty")
        if not isinstance(self.leakage_group_id, str) or not self.leakage_group_id:
            raise FemExtractionError("bundle leakage_group_id must be non-empty")
        try:
            role = EvidenceRole(self.role)
        except (TypeError, ValueError) as exc:
            raise FemExtractionError("bundle role must be an evidence role") from exc
        if role is EvidenceRole.HOLDOUT:
            raise FemExtractionError("bundle role must be a public evidence role")
        object.__setattr__(self, "role", role)
        for value, label in (
            (self.snapshot_content_hash, "snapshot_content_hash"),
            (self.request_plan_hash, "request_plan_hash"),
            (self.store_content_hash, "store_content_hash"),
            (self.fem_result_hash, "fem_result_hash"),
        ):
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise FemExtractionError(f"bundle {label} must be a digest")
        collections = (
            tuple(self.fields),
            tuple(self.reactions),
            tuple(self.energies),
            tuple(self.qois),
        )
        if not collections[0]:
            raise FemExtractionError("bundle fields must not be empty")
        flattened = tuple(item for group in collections for item in group)
        if any(type(item) is not ExtractedArtifact for item in flattened):
            raise FemExtractionError(
                "bundle collections must contain ExtractedArtifact values"
            )
        if any(
            item.case_id != self.case_id or item.fem_result_hash != self.fem_result_hash
            for item in flattened
        ):
            raise FemExtractionError(
                "every artifact must match the bundle case and result provenance"
            )
        names = tuple(item.name for item in flattened)
        if len(names) != len(set(names)):
            raise FemExtractionError("bundle artifact names must be unique")
        for name, group in zip(
            ("fields", "reactions", "energies", "qois"),
            collections,
            strict=True,
        ):
            object.__setattr__(self, name, group)

    @property
    def content_hash(self) -> str:
        return hash_spec(self, "content")


def extract_fem_artifacts(
    snapshot: RoleDatasetSnapshot,
    request_plan: PublicFemRequestPlan,
    attempt: FemAttempt,
    store: ObjectStore,
) -> ExtractedFemArtifacts:
    """Extract S10 artifacts from a snapshot-bound, store-verified FEM result."""

    try:
        record, result, result_hash = validate_snapshot_attempt(
            snapshot,
            request_plan,
            attempt,
            store,
        )
    except DatasetSnapshotValidationError as exc:
        raise FemExtractionError(str(exc)) from exc
    request = attempt.request
    case_id = request.request_id
    store_hash = attempt.store_content_hash
    if store_hash is None:
        raise FemExtractionError("trusted FEM attempt is missing ObjectStore content")

    fields = (
        ExtractedArtifact(
            case_id=case_id,
            name="deformation_gradient",
            location=FieldLocation.QUADRATURE,
            unit=Unit.ONE,
            dimension=Dimension.DIMENSIONLESS,
            coordinates=result.quadrature_points,
            quadrature_weights=result.quadrature_weights,
            coordinate_unit=Unit.MM,
            quadrature_weight_unit=AreaUnit.SQUARE_MILLIMETRE,
            values=result.deformation_gradient,
            fem_result_hash=result_hash,
        ),
        ExtractedArtifact(
            case_id=case_id,
            name="displacement",
            location=FieldLocation.DISPLACEMENT_DOF,
            unit=Unit.MM,
            dimension=Dimension.LENGTH,
            coordinates=result.displacement_dof_coordinates,
            coordinate_unit=Unit.MM,
            values=result.displacement,
            fem_result_hash=result_hash,
        ),
        ExtractedArtifact(
            case_id=case_id,
            name="first_piola_stress",
            location=FieldLocation.QUADRATURE,
            unit=Unit.MPA,
            dimension=Dimension.STRESS,
            coordinates=result.quadrature_points,
            quadrature_weights=result.quadrature_weights,
            coordinate_unit=Unit.MM,
            quadrature_weight_unit=AreaUnit.SQUARE_MILLIMETRE,
            values=result.first_piola_stress,
            fem_result_hash=result_hash,
        ),
        ExtractedArtifact(
            case_id=case_id,
            name="recovered_first_piola_stress",
            location=FieldLocation.RECOVERED_DOF,
            unit=Unit.MPA,
            dimension=Dimension.STRESS,
            coordinates=result.displacement_dof_coordinates,
            coordinate_unit=Unit.MM,
            values=result.recovered_first_piola_stress,
            fem_result_hash=result_hash,
        ),
    )
    reactions = tuple(
        ExtractedArtifact(
            case_id=case_id,
            name=f"reaction:{name}",
            location=FieldLocation.BOUNDARY,
            unit=Unit.N,
            dimension=Dimension.FORCE,
            values=tuple(values),
            fem_result_hash=result_hash,
        )
        for name, values in sorted(result.reactions.items())
    )
    energies = tuple(
        ExtractedArtifact(
            case_id=case_id,
            name=name,
            location=FieldLocation.GLOBAL,
            unit=Unit.N_MM,
            dimension=Dimension.ENERGY,
            values=(value,),
            fem_result_hash=result_hash,
        )
        for name, value in (
            ("external_work", result.external_work),
            ("total_potential_energy", result.total_potential_energy),
            ("total_strain_energy", result.total_strain_energy),
        )
    )
    qois = tuple(
        ExtractedArtifact(
            case_id=case_id,
            name=f"qoi:{spec.name}",
            location=FieldLocation.GLOBAL,
            unit=spec.unit,
            dimension=_qoi_dimension(spec.dimension),
            values=(result.qois[spec.name],),
            fem_result_hash=result_hash,
        )
        for spec in sorted(request.qois, key=lambda item: item.name)
    )
    return ExtractedFemArtifacts(
        case_id=case_id,
        leakage_group_id=record.leakage_group_id,
        role=record.role,
        snapshot_content_hash=snapshot.content_hash,
        request_plan_hash=request_plan.content_hash,
        store_content_hash=store_hash,
        fem_result_hash=result_hash,
        fields=fields,
        reactions=reactions,
        energies=energies,
        qois=qois,
    )


def _finite_payload(value: object, label: str) -> None:
    if isinstance(value, tuple):
        for item in value:
            _finite_payload(item, label)
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise FemExtractionError(f"{label} must contain only finite numbers")


def _validate_quadrature_layout(
    values: tuple[object, ...],
    coordinates: tuple[object, ...],
    weights: tuple[object, ...],
) -> None:
    if (
        not coordinates
        or not weights
        or not (len(values) == len(coordinates) == len(weights))
    ):
        raise FemExtractionError(
            "quadrature layout requires values, coordinates, and weights per cell"
        )
    for cell_values, cell_coordinates, cell_weights in zip(
        values,
        coordinates,
        weights,
        strict=True,
    ):
        if not all(
            isinstance(cell, tuple) and bool(cell)
            for cell in (cell_values, cell_coordinates, cell_weights)
        ):
            raise FemExtractionError(
                "quadrature layout requires non-empty tuple data in every cell"
            )
        if not (
            len(cell_values) == len(cell_coordinates) == len(cell_weights)  # type: ignore[arg-type]
        ):
            raise FemExtractionError(
                "quadrature layout requires one point and weight per value"
            )


def _qoi_dimension(value: Dimension | None) -> Dimension:
    if not isinstance(value, Dimension):
        raise FemExtractionError("request QoI dimension must be explicit")
    return value


__all__ = (
    "ExtractedArtifact",
    "ExtractedFemArtifacts",
    "FemExtractionError",
    "FieldLocation",
    "extract_fem_artifacts",
)
