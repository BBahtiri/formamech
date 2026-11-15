"""Installed fixed-CAD adapter, explicitly bound to one geometry resource."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from piginot import __version__
from piginot.cad.resource import GeometryResource, load_geometry_resource
from piginot.compiler.adapters import (
    ENERGY_ARCHITECTURE,
    AdapterDescriptor,
    GeometryIdentity,
)
from piginot.contracts import CanonicalMesh, ProblemSpec, canonical_json_bytes

if TYPE_CHECKING:
    from piginot.compiler.planning import ExecutionPlan
    from piginot.doe import PublishedCase
    from piginot.morphing import FixedGeometryMap, MorphValidityCriteria

ADAPTER_ID = "cad-fixed-v1"
_FIXED_RECIPE = b'{"$type":"piginot.morphing.FixedGeometryMap"}'
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


def implementation_sha256() -> str:
    """Bind the installed resource parser, adapter and fixed geometry recipe."""
    package = Path(__file__).resolve().parents[1]
    files = ("cad/adapter.py", "cad/resource.py", "morphing/fixed.py")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": "piginot.fixed-cad-adapter-implementation-v1",
                "files": {
                    name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                    for name in files
                },
            }
        )
    ).hexdigest()


def morph_map_sha256() -> str:
    return hashlib.sha256(_FIXED_RECIPE).hexdigest()


@dataclass(frozen=True, slots=True)
class FixedCadAdapter:
    resource: GeometryResource

    @property
    def geometry_inputs(self) -> tuple[tuple[str, Path], ...]:
        return self.resource.geometry_inputs

    @property
    def geometry_input_sha256(self) -> tuple[tuple[str, str], ...]:
        return (
            ("geometry-resource", self.resource.sha256),
            ("geometry-source", self.resource.source_sha256),
            *(
                (f"geometry-mesh-{level}", self.resource.meshes[level].sha256)
                for level in (1, 2, 4)
            ),
        )

    @property
    def descriptor(self) -> AdapterDescriptor:
        return AdapterDescriptor(
            adapter_id=ADAPTER_ID,
            version=__version__,
            implementation_sha256=implementation_sha256(),
            geometries=(
                GeometryIdentity(self.resource.meshes[1].mesh_id, morph_map_sha256()),
            ),
            materials=_MATERIALS,
            liftings=("adf",),
            architectures=(ENERGY_ARCHITECTURE,),
            reference_initializations=("cold", "neo-hookean-full-load-hgo"),
        )

    def _check_spec(self, spec: ProblemSpec) -> None:
        geometry = spec.geometry
        if (
            geometry.reference_mesh_id != self.resource.meshes[1].mesh_id
            or geometry.morph_map_sha256 != morph_map_sha256()
        ):
            raise ValueError(
                "ProblemSpec geometry differs from the prepared CAD resource"
            )
        if any(parameter.role == "geometry" for parameter in spec.parameters):
            raise ValueError("a fixed CAD model accepts no geometry parameters")
        if spec.lifting.kind != "adf":
            raise ValueError("prepared CAD geometry requires its declared ADF lifting")
        for condition in spec.boundary_conditions:
            if condition.kind == "neumann":
                raise ValueError("Neumann loads are unsupported by this CAD adapter")
            if (
                condition.kind in ("dirichlet", "symmetry")
                and self.resource.boundary_kinds.get(condition.region) != "line"
            ):
                raise ValueError(
                    "essential CAD boundaries must be declared straight line groups"
                )

    def geometry_values(
        self,
        spec: ProblemSpec,
        values: Mapping[str, float],
        level: int = 1,
    ) -> CanonicalMesh:
        self._check_spec(spec)
        if set(values) != {parameter.name for parameter in spec.parameters}:
            raise ValueError(
                "CAD parameters must match the declared load/material vocabulary"
            )
        for parameter in spec.parameters:
            number = values[parameter.name]
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or not parameter.lo <= number <= parameter.hi
            ):
                raise ValueError(
                    f"parameter {parameter.name!r} is outside its declared range"
                )
        mesh = self.resource.mesh(level)
        return self._validated_mesh(spec, mesh)

    @staticmethod
    def _validated_mesh(spec: ProblemSpec, mesh: CanonicalMesh) -> CanonicalMesh:
        from piginot.geometry import CellType, MeshPayload, canonicalize_mesh

        validated = canonicalize_mesh(
            MeshPayload(
                nodes=mesh.nodes,
                cells=mesh.cells,
                cell_type=CellType.TRIANGLE,
                boundary_facets=mesh.boundary_facets,
                boundary_tags=mesh.semantic_tags,
                region_ids=mesh.region_ids,
                unit=mesh.unit,
                source_provenance=mesh.source_provenance,
            )
        )
        if (
            validated.mesh_id != mesh.mesh_id
            or validated.topology_signature != mesh.topology_signature
        ):
            raise ValueError(
                "prepared mesh differs from its validated geometry identity"
            )
        validation_spec = spec
        if validated.mesh_id != spec.geometry.reference_mesh_id:
            validation_spec = replace(
                spec,
                geometry=replace(spec.geometry, reference_mesh_id=validated.mesh_id),
            )
        validation_spec.check_mesh(validated)
        return validated

    def geometry(
        self, spec: ProblemSpec, case: PublishedCase, level: int = 1
    ) -> CanonicalMesh:
        return self.geometry_values(spec, case.primitive_coordinates, level)

    def export_geometry(
        self, spec: ProblemSpec
    ) -> tuple[CanonicalMesh, FixedGeometryMap]:
        from piginot.morphing import FixedGeometryMap

        self._check_spec(spec)
        return self._validated_mesh(spec, self.resource.mesh(1)), FixedGeometryMap()

    def validity_criteria(self) -> MorphValidityCriteria:
        from piginot.morphing import MorphValidityCriteria

        return MorphValidityCriteria()

    def execute(self, plan: ExecutionPlan, *, resume: bool = False) -> int:
        from piginot.compiler.execution import execute_campaign

        return execute_campaign(plan, self, resume=resume)


@dataclass(frozen=True, slots=True)
class FixedCadProvider:
    def bind_resource(self, path: Path, sha256: str) -> FixedCadAdapter:
        return FixedCadAdapter(load_geometry_resource(path, sha256))


FIXED_CAD = FixedCadProvider()
