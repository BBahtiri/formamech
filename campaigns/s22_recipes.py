"""S22: write the FEM recipe sidecars of the shipped bundles, with evidence.

For each family the recipe is taken from the S10 request of the first
DEVELOPMENT case and must rebuild the S10 request of every DEVELOPMENT case
byte for byte from that case's inputs and the bundle's served mesh::

    python -m campaigns.s22_recipes --write
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from campaigns import s21_certificate as s21
from piginot.contracts import canonical_json_bytes
from piginot.serving.recipe import FemRecipe

REPOSITORY = s21.REPOSITORY
RECIPES = Path("bundles/fem-recipes")
EVIDENCE = Path("artifacts/serving/S22-RECIPES.json")
RULES = {
    "dogbone-v1": ("right_grip_displacement", "right_grip"),
    "notched-v1": ("right_end_displacement", "right_end"),
}
MATERIAL_INPUTS = {
    "elastic_modulus": "elastic_modulus",
    "poisson_ratio": "poisson_ratio",
}


def recipe_path(bundle: str) -> Path:
    return RECIPES / f"{bundle}.json"


def case_inputs(campaign: str, case_id: str) -> tuple[Any, dict[str, object]]:
    from campaigns.dogbone_v1.s14_candidate import S14Case

    family = s21.runtime(campaign)
    case = S14Case.from_published(s21._published(campaign)[case_id])
    values = {k: float(v) for k, v in case.primitive_coordinates.items()}
    inputs = family.theta(
        values,
        elastic_modulus=values["elastic_modulus"],
        poisson_ratio=values["poisson_ratio"],
    )
    return case, inputs


def build(campaign: str) -> tuple[FemRecipe, dict[str, Any]]:
    from piginot.bundle import predict

    family = s21.runtime(campaign)
    bundle = s21._bundle(family.bundle)
    ids = json.loads((REPOSITORY / s21.PROTOCOL_PATH).read_text("utf-8"))["families"][
        campaign
    ]["case_ids"]
    recipe: FemRecipe | None = None
    reproduced: list[str] = []
    for case_id in ids:
        case, inputs = case_inputs(campaign, case_id)
        reference = family.s10_request(case, family.profile())
        if recipe is None:
            condition, tag = RULES[campaign]
            recipe = FemRecipe.from_request(
                reference,
                bundle=family.bundle,
                bundle_manifest_sha256=bundle.manifest.manifest_sha256,
                material_inputs=MATERIAL_INPUTS,
                axial_strain_rule={
                    "boundary_condition": condition,
                    "boundary_tag": tag,
                    "strain_input": "nominal_axial_strain",
                },
                source=f"S10 request of {case_id} ({campaign})",
            )
        served = predict(bundle, inputs, confidence=False).mesh
        assert served is not None
        built = recipe.build_request(inputs, served, case_id)
        if canonical_json_bytes(built) == canonical_json_bytes(reference):
            reproduced.append(case_id)
    assert recipe is not None
    return recipe, {
        "bundle": family.bundle,
        "recipe_sha256": recipe.sha256,
        "cases": len(ids),
        "reproduced_byte_for_byte": len(reproduced),
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the S22 FEM recipes.")
    parser.add_argument("--write", action="store_true")
    options = parser.parse_args(arguments)
    evidence = {}
    for campaign in s21.FAMILIES:
        recipe, record = build(campaign)
        evidence[campaign] = record
        print(campaign, record)
        if options.write:
            path = REPOSITORY / recipe_path(recipe.bundle)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(recipe.document(), indent=2, sort_keys=True) + "\n"
            )
    if options.write:
        path = REPOSITORY / EVIDENCE
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"schema_version": "1.0", "step": "S22", "families": evidence}
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    complete = all(
        r["reproduced_byte_for_byte"] == r["cases"] for r in evidence.values()
    )
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
