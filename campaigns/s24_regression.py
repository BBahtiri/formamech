"""S24: the shipped families re-expressed as ProblemSpecs (ADR 0106 d. 4, 6).

For each family, from ``campaigns/<family>/problem_spec.json`` alone:

- the parameter ranges and constraints equal the bundle's envelope, the
  reference mesh and morph map are the bundle's, and the spec passes the mesh
  rules on the reference mesh;
- the material, QoIs and boundary conditions equal the bundle's model spec;
- on the served mesh of each of the 64 DEVELOPMENT cases (the S21 protocol's
  case list), the spec's frame and ``quarter`` lifting equal the bundle's bit
  for bit, and the FEM request built from the spec equals the campaign's S10
  request byte for byte (on the execution profile of the shipped recipe);
- the lifting evidence of the spec and of its ``adf`` variant, over every
  feasible box corner, the centre and 64 seeded interior points.

Writes ``artifacts/problem/S24-REGRESSION.json`` with ``--write``::

    python -m campaigns.s24_regression --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from campaigns import s21_certificate as s21
from piginot.contracts import CanonicalQuantity, ProblemSpec, canonical_json_bytes

REPOSITORY = s21.REPOSITORY
EVIDENCE = Path("artifacts/problem/S24-REGRESSION.json")
SPECS = {
    "dogbone-v1": Path("campaigns/dogbone_v1"),
    "notched-v1": Path("campaigns/notched_v1"),
}
INTERIOR_POINTS = 64


def spec_path(campaign: str, variant: str = "") -> Path:
    suffix = f"_{variant}" if variant else ""
    return SPECS[campaign] / f"problem_spec{suffix}.json"


def load_spec(campaign: str, variant: str = "") -> ProblemSpec:
    return ProblemSpec.load(REPOSITORY / spec_path(campaign, variant))


def _bounds(items: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    return {
        item["name"]: [
            item["lower_bound"]["value"],
            item["upper_bound"]["value"],
            item["lower_bound"]["unit"],
        ]
        for item in items
    }


def envelope_checks(spec: ProblemSpec, name: str) -> dict[str, bool]:
    """Ranges, constraints, mesh, morph map and model fields against the bundle."""

    bundle = s21._bundle(name)
    root = REPOSITORY / "bundles" / name
    envelope = json.loads((root / "spec" / "envelope.json").read_text("utf-8"))
    membership = envelope["geometry_membership"]
    expected = {
        **_bounds(membership["primitive_bounds"]),
        **_bounds(envelope["load_bounds"]),
        **_bounds(envelope["material_bounds"]),
    }
    ranges = {p.name: [p.lo, p.hi, p.unit.value] for p in spec.parameters}
    derived = {
        d["name"]: [
            json.loads(json.dumps(d["canonical_ast"])),
            d["lower_bound"]["value"],
            d["upper_bound"]["value"],
        ]
        for d in membership["derived_bounds"]
    }
    constraints = {
        c.name: [c.expression.canonical(), c.lo, c.hi] for c in spec.constraints
    }
    morph = hashlib.sha256((root / "spec" / "morph-map.json").read_bytes()).hexdigest()
    model = bundle.model
    spec.check_mesh(bundle.reference_mesh)
    return {
        "ranges": ranges == expected,
        "constraints": constraints == derived,
        "reference_mesh": spec.geometry.reference_mesh_id
        == bundle.reference_mesh.mesh_id
        == membership["reference_mesh_id"],
        "morph_map": spec.geometry.morph_map_sha256 == morph,
        "mesh_rules": True,
        "material": spec.material.model == model.material_model
        and dict(spec.material.parameters) == dict(model.material_parameters),
        "qois": [(q.name, q.field, q.region, q.exclude_regions) for q in spec.qois]
        == [(q.name, q.field, q.tag, q.exclude_tags) for q in model.qois],
        "dirichlet": {
            c.region: c.components for c in spec.boundary_conditions if c.is_dirichlet
        }
        == dict(model.boundary_conditions),
    }


def _recipe_profile(bundle: str) -> Any:
    from piginot.serving.recipe import decode

    recipe = json.loads(
        (REPOSITORY / "bundles" / "fem-recipes" / f"{bundle}.json").read_text("utf-8")
    )
    return decode(recipe["template"]["execution_profile"])


def development_cases(campaign: str, spec: ProblemSpec) -> dict[str, Any]:
    """Lifting, frame and FEM request of every DEVELOPMENT case."""

    import torch

    from piginot.bundle import predict
    from piginot.lifting import AxisAlignedTensionLifting
    from piginot.problem import build_lifting, fem_request, query_frame

    family = s21.runtime(campaign)
    bundle = s21._bundle(family.bundle)
    model = bundle.model
    profile = _recipe_profile(family.bundle)
    protocol = json.loads((REPOSITORY / s21.PROTOCOL_PATH).read_text("utf-8"))
    ids = protocol["families"][campaign]["case_ids"]
    from campaigns.s22_recipes import case_inputs

    lifting_equal = frame_equal = requests_equal = 0
    for case_id in ids:
        case, inputs = case_inputs(campaign, case_id)
        served = predict(bundle, inputs, confidence=False).mesh
        assert served is not None
        values = {
            name: float(quantity.value)
            for name, quantity in inputs.items()
            if isinstance(quantity, CanonicalQuantity)
        }
        points = torch.tensor([served.nodes], dtype=torch.float64)
        extent = [item.evaluate(values) for item in model.frame_extent]
        origin = [item.evaluate(values) for item in model.frame_origin]
        shipped = AxisAlignedTensionLifting(
            torch.tensor([extent], dtype=torch.float64),
            torch.tensor(
                [model.lifting_displacement.evaluate(values)], dtype=torch.float64
            ),
        )(points)
        scoped = spec.values(inputs)
        built = build_lifting(spec, [scoped])(points)
        if torch.equal(built.offset, shipped.offset) and torch.equal(
            built.scale, shipped.scale
        ):
            lifting_equal += 1
        frame = query_frame(spec, [scoped])
        if frame.origin.tolist() == [origin] and frame.extent.tolist() == [extent]:
            frame_equal += 1
        reference = family.s10_request(case, profile)
        request = fem_request(spec, inputs, served, case_id, profile)
        if canonical_json_bytes(request) == canonical_json_bytes(reference):
            requests_equal += 1
    return {
        "cases": len(ids),
        "lifting_bit_identical": lifting_equal,
        "frame_bit_identical": frame_equal,
        "fem_requests_byte_identical": requests_equal,
    }


def lifting_evidence(campaign: str, spec: ProblemSpec) -> dict[str, Any]:
    from piginot.problem import box_points, verify_lifting

    bundle = s21._bundle(s21.runtime(campaign).bundle)

    def mesh_for(values: Mapping[str, float]) -> Any:
        inputs = {
            p.name: CanonicalQuantity(values[p.name], p.unit, p.unit.dimension)
            for p in spec.parameters
            if p.role == "geometry"
        }
        return bundle.geometry(inputs).mesh

    evidence = verify_lifting(
        spec, mesh_for, box_points(spec, interior=INTERIOR_POINTS)
    )
    document = evidence.document()
    if evidence.passed:
        record = evidence.verification(spec)
        document["lifting_verification"] = {
            "verification_id": record.verification_id,
            "parameter_range_hash": record.parameter_range_hash,
            "verification_hash": record.verification_hash,
            "boundary_refinement_levels": list(record.boundary_refinement_levels),
            "max_boundary_residual": record.max_boundary_residual,
        }
    return document


def build(campaign: str) -> dict[str, Any]:
    spec = load_spec(campaign)
    adf = load_spec(campaign, "adf")
    return {
        "spec": spec_path(campaign).as_posix(),
        "problem_hash": spec.problem_hash,
        "adf_spec": spec_path(campaign, "adf").as_posix(),
        "adf_problem_hash": adf.problem_hash,
        "envelope": envelope_checks(spec, s21.runtime(campaign).bundle),
        "development": development_cases(campaign, spec),
        "lifting": {
            "quarter": lifting_evidence(campaign, spec),
            "adf": lifting_evidence(campaign, adf),
        },
    }


def passed(record: Mapping[str, Any]) -> bool:
    development = record["development"]
    cases = development["cases"]
    return (
        all(record["envelope"].values())
        and development["lifting_bit_identical"] == cases
        and development["frame_bit_identical"] == cases
        and development["fem_requests_byte_identical"] == cases
        and all(item["passed"] for item in record["lifting"].values())
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S24 family regression.")
    parser.add_argument("--write", action="store_true")
    options = parser.parse_args(arguments)
    families = {}
    for campaign in SPECS:
        record = build(campaign)
        record["passed"] = passed(record)
        families[campaign] = record
        print(campaign, json.dumps(record["development"]), record["passed"])
    if options.write:
        path = REPOSITORY / EVIDENCE
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"schema_version": "1.0", "step": "S24", "families": families}
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 0 if all(r["passed"] for r in families.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
