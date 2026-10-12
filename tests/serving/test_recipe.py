"""S22 FEM recipes: bound to their bundle, reproduce the S10 request."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from campaigns import s22_recipes
from piginot.bundle import load_bundle, predict
from piginot.contracts import canonical_json_bytes
from piginot.serving.recipe import FemRecipe, RecipeError, decode

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = "notched-v1-cp-freeze"


def _recipe() -> FemRecipe:
    bundle = load_bundle(ROOT / "bundles" / BUNDLE)
    return FemRecipe.load(
        ROOT / s22_recipes.recipe_path(BUNDLE),
        bundle=BUNDLE,
        manifest_sha256=bundle.manifest.manifest_sha256,
    )


def test_a_recipe_rebuilds_an_s10_request_byte_for_byte() -> None:
    from campaigns import s21_certificate as s21

    recipe = _recipe()
    case_id = json.loads((ROOT / s21.PROTOCOL_PATH).read_text())["families"][
        "notched-v1"
    ]["case_ids"][5]
    case, inputs = s22_recipes.case_inputs("notched-v1", case_id)
    family = s21.runtime("notched-v1")
    served = predict(s21._bundle(BUNDLE), inputs, confidence=False).mesh
    assert served is not None
    built = recipe.build_request(inputs, served, case_id)
    # The recipe carries the registered S10 execution profile (Python 3.11.15
    # on the reference machine); the campaign builds its profile from the
    # running interpreter, so the comparison uses the recipe's profile.
    reference = family.s10_request(case, built.execution_profile)
    assert canonical_json_bytes(built) == canonical_json_bytes(reference)


def test_a_recipe_is_bound_to_its_bundle(tmp_path: Path) -> None:
    path = ROOT / s22_recipes.recipe_path(BUNDLE)
    with pytest.raises(RecipeError, match="another bundle"):
        FemRecipe.load(path, bundle=BUNDLE, manifest_sha256="0" * 64)
    with pytest.raises(RecipeError, match="another bundle"):
        FemRecipe.load(
            path, bundle="dogbone-v1-s13c-cp-freeze", manifest_sha256="0" * 64
        )
    other = tmp_path / "x.json"
    other.write_text(json.dumps({"schema": "other"}))
    with pytest.raises(RecipeError, match="is not a"):
        FemRecipe.load(other, bundle=BUNDLE, manifest_sha256="0" * 64)


def test_the_committed_recipes_match_a_rebuild() -> None:
    evidence = json.loads((ROOT / s22_recipes.EVIDENCE).read_text())
    for campaign, record in evidence["families"].items():
        assert record["reproduced_byte_for_byte"] == record["cases"] == 64
        document = json.loads(
            (ROOT / s22_recipes.recipe_path(record["bundle"])).read_text()
        )
        recipe = FemRecipe(
            document["bundle"],
            document["bundle_manifest_sha256"],
            document["template"],
            document["material_inputs"],
            document["axial_strain_rule"],
            document["source"],
        )
        assert recipe.sha256 == record["recipe_sha256"], campaign


def test_bad_rules_and_types_are_refused() -> None:
    recipe = _recipe()
    from campaigns import s21_certificate as s21

    case_id = "candidate-000020"
    _, inputs = s22_recipes.case_inputs("notched-v1", case_id)
    served = predict(s21._bundle(BUNDLE), inputs, confidence=False).mesh
    assert served is not None
    no_tag = FemRecipe(
        **{
            **_fields(recipe),
            "axial_strain_rule": {
                **recipe.axial_strain_rule,
                "boundary_tag": "nowhere",
            },
        }
    )
    with pytest.raises(RecipeError, match="no boundary tag"):
        no_tag.build_request(inputs, served, "x")
    no_bc = FemRecipe(
        **{
            **_fields(recipe),
            "axial_strain_rule": {
                **recipe.axial_strain_rule,
                "boundary_condition": "none",
            },
        }
    )
    with pytest.raises(RecipeError, match="no boundary condition"):
        no_bc.build_request(inputs, served, "x")
    with pytest.raises(RecipeError, match="unsupported type"):
        decode({"$type": "piginot.contracts.Unknown"})


def _fields(recipe: FemRecipe) -> dict[str, object]:
    return {
        "bundle": recipe.bundle,
        "bundle_manifest_sha256": recipe.bundle_manifest_sha256,
        "template": recipe.template,
        "material_inputs": recipe.material_inputs,
        "axial_strain_rule": recipe.axial_strain_rule,
        "source": recipe.source,
    }
