"""Small public adapter descriptors; numerical builders are loaded separately."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING, cast

from piginot.compiler.adapters import (
    ENERGY_ARCHITECTURE,
    AdapterDescriptor,
    GeometryIdentity,
)
from piginot.contracts import ProblemSpec, canonical_json_bytes
from piginot_campaigns import __version__

if TYPE_CHECKING:
    from piginot.compiler.planning import ExecutionPlan
    from piginot.contracts import CanonicalMesh
    from piginot.doe import PublishedCase
    from piginot.morphing import MorphValidityCriteria
    from piginot_campaigns.geometry import MorphMap

_RESOURCES = {
    "dogbone-v1": "resources/dogbone-v1.json",
    "notched-v1": "resources/notched-v1.json",
}
_TEMPLATES = "resources/geometry-templates.json"
_FILES = ("__init__.py", "adapters.py", "geometry.py", _TEMPLATES, *_RESOURCES.values())
_MATERIALS = (
    "compressible_neo_hookean",
    "mooney_rivlin",
    "yeoh",
    "gent",
    "hgo",
    "hgo_single",
    "goh",
    "formula-isotropic",
    "formula-fibre",
)


def _read(name: str) -> bytes:
    return files("piginot_campaigns").joinpath(name).read_bytes()


def implementation_sha256() -> str:
    """Bind the observed installed implementation and public metadata bytes."""

    manifest = {
        "schema": "piginot.public-adapter-implementation-v1",
        "version": __version__,
        "files": {name: hashlib.sha256(_read(name)).hexdigest() for name in _FILES},
    }
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def load_problem(adapter_id: str) -> ProblemSpec:
    if adapter_id not in _RESOURCES:
        raise ValueError(f"unsupported public adapter {adapter_id!r}")
    return ProblemSpec.from_bytes(_read(_RESOURCES[adapter_id]))


@dataclass(frozen=True, slots=True)
class GeometryTemplate:
    name: str
    identity: GeometryIdentity
    resolution: tuple[int, int, int]


def load_geometry_templates(adapter_id: str) -> tuple[GeometryTemplate, ...]:
    """Pure metadata: exact registered templates, without loading mesh builders."""

    if adapter_id not in _RESOURCES:
        raise ValueError(f"unsupported public adapter {adapter_id!r}")
    entries = json.loads(_read(_TEMPLATES))[adapter_id]
    return tuple(
        GeometryTemplate(
            entry["name"],
            GeometryIdentity(entry["reference_mesh_id"], entry["morph_map_sha256"]),
            cast(tuple[int, int, int], tuple(entry["resolution"])),
        )
        for entry in entries
    )


@dataclass(frozen=True, slots=True)
class PublicAdapter:
    adapter_id: str

    def export_geometry(self, spec: ProblemSpec) -> tuple[CanonicalMesh, MorphMap]:
        from piginot_campaigns.geometry import export_geometry

        return export_geometry(self.adapter_id, spec)

    def validity_criteria(self) -> MorphValidityCriteria:
        from piginot_campaigns.geometry import validity_criteria

        return validity_criteria()

    def geometry_values(
        self, spec: ProblemSpec, values: Mapping[str, float], level: int = 1
    ) -> CanonicalMesh:
        from piginot_campaigns.geometry import geometry_values

        return geometry_values(self.adapter_id, spec, values, level)

    def geometry(
        self, spec: ProblemSpec, case: PublishedCase, level: int = 1
    ) -> CanonicalMesh:
        return self.geometry_values(spec, case.primitive_coordinates, level)

    def execute(self, plan: ExecutionPlan, *, resume: bool = False) -> int:
        from piginot.compiler.execution import execute_campaign

        return execute_campaign(plan, self, resume=resume)

    @property
    def descriptor(self) -> AdapterDescriptor:
        return AdapterDescriptor(
            adapter_id=self.adapter_id,
            version=__version__,
            implementation_sha256=implementation_sha256(),
            geometries=tuple(
                t.identity for t in load_geometry_templates(self.adapter_id)
            ),
            materials=_MATERIALS,
            liftings=("quarter", "adf"),
            architectures=(ENERGY_ARCHITECTURE,),
            reference_initializations=("cold", "neo-hookean-full-load-hgo"),
        )


DOGBONE = PublicAdapter("dogbone-v1")
NOTCHED = PublicAdapter("notched-v1")
