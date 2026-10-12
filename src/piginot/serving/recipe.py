"""FEM recipes: a bundle's reference FEM problem, rebuilt for any input (S22).

A recipe is a sidecar JSON file bound to one bundle by its manifest hash, so
shipped bundles stay byte-identical. It holds the canonical form of the
family's S10 FEM request without the mesh and request ID, plus two rules:

- ``material_inputs``: which inputs set the material's properties;
- ``axial_strain_rule``: the one Dirichlet value that varies, equal to the
  nominal axial strain times the x-position of the tagged end of the served
  mesh (uniform nominal strain from the symmetry plane at x = 0).

For all 128 DEVELOPMENT cases of both families the rebuilt request equals the
campaign's S10 request byte for byte (S22 evidence).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    ExecutionProfile,
    LoadSpec,
    MaterialSpec,
    QoISpec,
    canonical_json_bytes,
)
from piginot.contracts.units import Dimension, Unit
from piginot.fem.models import FemRequest, FemSolverConfig

SCHEMA = "piginot.fem-recipe-v1"
_TYPES: Mapping[str, Any] = {
    "piginot.contracts.MaterialSpec": MaterialSpec,
    "piginot.contracts.BoundaryConditionSpec": BoundaryConditionSpec,
    "piginot.contracts.LoadSpec": LoadSpec,
    "piginot.contracts.QoISpec": QoISpec,
    "piginot.contracts.FemSolverConfig": FemSolverConfig,
    "piginot.contracts.ExecutionProfile": ExecutionProfile,
}
_QUANTITY = "piginot.contracts.CanonicalQuantity"
_VARYING = ("$type", "mesh", "request_id")


class RecipeError(ValueError):
    """A recipe that is malformed or bound to another bundle."""


def decode(value: Any) -> Any:
    """Rebuild contract objects from their canonical JSON form."""

    if isinstance(value, dict):
        if "$type" in value:
            kind = value["$type"]
            fields = {k: decode(v) for k, v in value.items() if k != "$type"}
            if kind == _QUANTITY:
                return CanonicalQuantity(
                    fields["value"],
                    Unit(fields["unit"]),
                    Dimension(fields["dimension"]),
                )
            if kind not in _TYPES:
                raise RecipeError(f"unsupported type in a recipe: {kind}")
            return _TYPES[kind](**fields)
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return tuple(decode(v) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class FemRecipe:
    """The FEM problem of one bundle, minus the mesh and the case's inputs."""

    bundle: str
    bundle_manifest_sha256: str
    template: Mapping[str, Any]
    material_inputs: Mapping[str, str]
    axial_strain_rule: Mapping[str, str]
    source: str

    def document(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "bundle": self.bundle,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "template": dict(self.template),
            "material_inputs": dict(self.material_inputs),
            "axial_strain_rule": dict(self.axial_strain_rule),
            "source": self.source,
        }

    @property
    def sha256(self) -> str:
        text = json.dumps(self.document(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def from_request(
        cls,
        request: FemRequest,
        *,
        bundle: str,
        bundle_manifest_sha256: str,
        material_inputs: Mapping[str, str],
        axial_strain_rule: Mapping[str, str],
        source: str,
    ) -> FemRecipe:
        canonical = json.loads(canonical_json_bytes(request))
        template = {k: v for k, v in canonical.items() if k not in _VARYING}
        recipe = cls(
            bundle,
            bundle_manifest_sha256,
            template,
            dict(material_inputs),
            dict(axial_strain_rule),
            source,
        )
        rebuilt = recipe._assemble(request.mesh, request.request_id)
        if canonical_json_bytes(rebuilt) != canonical_json_bytes(request):
            raise RecipeError("the template does not round-trip to its source request")
        return recipe

    @classmethod
    def load(cls, path: Path, *, bundle: str, manifest_sha256: str) -> FemRecipe:
        document = json.loads(path.read_text("utf-8"))
        if document.get("schema") != SCHEMA:
            raise RecipeError(f"{path} is not a {SCHEMA} document")
        if (
            document["bundle"] != bundle
            or document["bundle_manifest_sha256"] != manifest_sha256
        ):
            raise RecipeError(f"{path} belongs to another bundle")
        return cls(
            document["bundle"],
            document["bundle_manifest_sha256"],
            document["template"],
            document["material_inputs"],
            document["axial_strain_rule"],
            document["source"],
        )

    def _assemble(self, mesh: CanonicalMesh, request_id: str) -> FemRequest:
        fields = decode(dict(self.template))
        return FemRequest(mesh=mesh, request_id=request_id, **fields)

    def build_request(
        self,
        inputs: Mapping[str, CanonicalQuantity],
        mesh: CanonicalMesh,
        request_id: str,
    ) -> FemRequest:
        """The FEM request for ``inputs`` on the served ``mesh``."""

        request = self._assemble(mesh, request_id)
        properties = dict(request.material.properties)
        for prop, name in self.material_inputs.items():
            properties[prop] = inputs[name]
        material = replace(request.material, properties=properties)
        rule = self.axial_strain_rule
        tag = rule["boundary_tag"]
        if tag not in mesh.semantic_tags:
            raise RecipeError(f"the served mesh has no boundary tag {tag!r}")
        nodes = {
            index
            for facet in mesh.semantic_tags[tag]
            for index in mesh.boundary_facets[facet]
        }
        end = max(float(mesh.nodes[index][0]) for index in nodes)
        strain = float(inputs[rule["strain_input"]].value)
        displacement = CanonicalQuantity(strain * end, Unit.MM, Dimension.LENGTH)
        if all(
            c.name != rule["boundary_condition"] for c in request.boundary_conditions
        ):
            raise RecipeError(
                f"no boundary condition named {rule['boundary_condition']!r}"
            )
        conditions = tuple(
            replace(condition, values=(displacement,))
            if condition.name == rule["boundary_condition"]
            else condition
            for condition in request.boundary_conditions
        )
        return replace(request, material=material, boundary_conditions=conditions)
