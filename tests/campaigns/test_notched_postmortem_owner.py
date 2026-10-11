"""S20 notched post-mortem, owner part: D7 and D8 logic on public inputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from campaigns.notched_v1 import postmortem as public
from campaigns.notched_v1 import postmortem_owner as owner

ROOT = public.REPOSITORY


def _manifest() -> dict:
    return json.loads((ROOT / owner.CAMPAIGN / "public-case-manifest.json").read_text())


def _fake_holdout(manifest: dict) -> list[dict[str, float]]:
    train = [c for c in manifest["cases"] if c["role"] == "TRAIN"][:64]
    return [{**c["primitive_coordinates"], **c["derived_coordinates"]} for c in train]


def test_mahalanobis_ignores_constant_axes_and_is_scale_free() -> None:
    rng = np.random.default_rng(0)
    train = np.column_stack(
        [rng.normal(size=200), np.full(200, 7.0), rng.normal(size=200)]
    )
    points = np.array([[0.0, 7.0, 0.0], [3.0, 7.0, 0.0]])
    distance, used = owner.mahalanobis(train, points, ["a", "fixed", "b"])
    assert used == ["a", "b"]
    assert distance[0] < 0.2 < 2.5 < distance[1]
    scaled, _ = owner.mahalanobis(
        train * [10.0, 1.0, 1.0], points * [10.0, 1.0, 1.0], ["a", "f", "b"]
    )
    assert scaled == pytest.approx(distance)


def test_d7_runs_on_the_recorded_cases() -> None:
    manifest = _manifest()
    document = owner.envelope_distance(
        manifest, _fake_holdout(manifest), owner._records(ROOT)
    )
    assert document["train_cases"] > 100
    assert set(document["distances"]) == {"DEVELOPMENT", "HOLDOUT"}
    assert len(document["distances"]["HOLDOUT"]) == 64
    for quantity in public.QUANTITIES:
        assert -1.0 <= document[quantity]["spearman_pooled"] <= 1.0


def test_d8_selects_the_largest_reference_terms_and_compares() -> None:
    development = owner._records(ROOT)["DEVELOPMENT"]
    ids = owner.refinement_cases(development)
    assert len(ids) == owner.D8_CASES
    terms = {
        c["case_id"]: c["components"]["p_relative_l2"]["discretisation"]
        for c in development
    }
    assert min(terms[i] for i in ids) >= sorted(terms.values())[-owner.D8_CASES]
    observed = {
        case_id: {
            q: next(c for c in development if c["case_id"] == case_id)["components"][q][
                "discretisation"
            ]
            / 3.0
            for q in public.QUANTITIES
        }
        for case_id in ids
    }
    loose = owner.compare_refinement(observed, development)
    assert loose["p_relative_l2"]["median_bound_over_observed"] == pytest.approx(3.0)
    assert loose["bound_within_2x"] is False
    tight = owner.compare_refinement(
        {k: {q: v * 2.0 for q, v in values.items()} for k, values in observed.items()},
        development,
    )
    assert tight["bound_within_2x"] is True


def test_verdicts_combine_public_and_owner_parts(tmp_path: Path) -> None:
    shutil.copytree(
        ROOT / "artifacts" / "campaigns" / "notched-v1", tmp_path / owner.CAMPAIGN
    )
    before = owner.verdicts(tmp_path)
    assert before["hypotheses"]["a"]["verdict"] == "NOT_RUN"
    d7 = {q: {"spearman_pooled": 0.5} for q in public.QUANTITIES}
    (tmp_path / owner.D7_PATH).write_text(json.dumps(d7))
    d8 = {
        "comparison": {
            **{q: {"median_bound_over_observed": 1.5} for q in public.QUANTITIES},
            "bound_within_2x": True,
        }
    }
    (tmp_path / owner.D8_PATH).write_text(json.dumps(d8))
    after = owner.verdicts(tmp_path)
    assert after["hypotheses"]["a"]["verdict"] == "SUPPORTED"
    assert after["hypotheses"]["e"]["evidence"]["D8_bound_within_2x"] is True
    (tmp_path / owner.D7_PATH).write_text(
        json.dumps({q: {"spearman_pooled": 0.1} for q in public.QUANTITIES})
    )
    assert owner.verdicts(tmp_path)["hypotheses"]["a"]["verdict"] == "NOT_SUPPORTED"
