"""Recipe-addressable adapters from exact MMS fields to FEM providers."""

from __future__ import annotations

from dataclasses import dataclass

from piginot.contracts import MaterialSpec, hash_spec
from piginot.fem.field_providers import FloatArray
from piginot.fem.manufactured import build_manufactured_solution


def _validate(case_name: str, material: MaterialSpec) -> None:
    if not isinstance(case_name, str) or not case_name:
        raise ValueError("case_name must be non-empty")
    if not isinstance(material, MaterialSpec):
        raise TypeError("material must be a MaterialSpec")
    build_manufactured_solution(case_name, material)


def _provider_id(kind: str, case_name: str, material: MaterialSpec) -> str:
    return hash_spec(
        {
            "provider_kind": kind,
            "case_name": case_name,
            "material": material,
        },
        "recipe",
    )


@dataclass(frozen=True, slots=True)
class ManufacturedDisplacementProvider:
    """Evaluate an exact manufactured displacement in canonical mm."""

    case_name: str
    material: MaterialSpec

    def __post_init__(self) -> None:
        _validate(self.case_name, self.material)

    @property
    def provider_id(self) -> str:
        return _provider_id(
            "manufactured-displacement-v1", self.case_name, self.material
        )

    def evaluate(self, reference_points: FloatArray) -> FloatArray:
        return build_manufactured_solution(self.case_name, self.material).displacement(
            reference_points
        )


@dataclass(frozen=True, slots=True)
class ManufacturedBodyForceProvider:
    """Evaluate exact manufactured reference-volume force density."""

    case_name: str
    material: MaterialSpec

    def __post_init__(self) -> None:
        _validate(self.case_name, self.material)

    @property
    def provider_id(self) -> str:
        return _provider_id("manufactured-body-force-v1", self.case_name, self.material)

    def evaluate(self, reference_points: FloatArray) -> FloatArray:
        return build_manufactured_solution(self.case_name, self.material).body_force(
            reference_points
        )


@dataclass(frozen=True, slots=True)
class ManufacturedTractionProvider:
    """Evaluate exact manufactured nominal traction on reference facets."""

    case_name: str
    material: MaterialSpec

    def __post_init__(self) -> None:
        _validate(self.case_name, self.material)

    @property
    def provider_id(self) -> str:
        return _provider_id("manufactured-traction-v1", self.case_name, self.material)

    def evaluate(
        self,
        reference_points: FloatArray,
        outward_reference_normals: FloatArray,
    ) -> FloatArray:
        return build_manufactured_solution(self.case_name, self.material).traction(
            reference_points,
            outward_reference_normals,
        )


__all__ = (
    "ManufacturedBodyForceProvider",
    "ManufacturedDisplacementProvider",
    "ManufacturedTractionProvider",
)
