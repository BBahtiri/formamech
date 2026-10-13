"""S23 H1 reveal on a synthetic family (ADR 0104, G23.reveal-once).

The family's S14 scoring is replaced by a fake; the commitment, the
preconditions, the consumption record, the coverage counts, the burn and the
refusal of a second reveal are the real code.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from campaigns import s23_calibration as s23
from campaigns import s23_reveal as reveal
from piginot.bounds import CalibrationRecord, MahalanobisGate, case_ids_sha256
from piginot.contracts import canonical_json_bytes
from piginot.doe.commitment import compute_holdout_commitment
from piginot.doe.secret_material import generate_campaign_secrets
from piginot.evaluation import BurnedCaseRegistry
from piginot.git_environment import isolated_git_environment

CAMPAIGN, BUNDLE = "plate-v1", "plate-bundle"
QUANTITIES = ["p_relative_l2", "absolute_peak_error", "u_relative_l2"]


@dataclass(frozen=True)
class FakeCase:
    case_id: str
    leakage_group_id: str
    coordinates: dict[str, float]


def _record(case: FakeCase) -> dict[str, object]:
    x = case.coordinates["x"]
    surrogate = {q: 0.01 * x for q in QUANTITIES}
    components = {
        q: {"discretisation": 0.005 * x, "projection": 0.001} for q in QUANTITIES
    }
    return {
        "case_id": case.case_id,
        "slices": ["all"],
        "reported_slices": [],
        "surrogate": surrogate,
        "terms": {},
        "components": components,
        "minimum_vertex_determinant": 1.0,
    }


def fake_adapter(campaign_id: str) -> reveal.FamilyAdapter:
    def holdout_cases(points: Any) -> tuple[FakeCase, ...]:
        return tuple(FakeCase("holdout-00", "holdout-00", dict(p)) for p in points)

    return reveal.FamilyAdapter(
        campaign_id,
        holdout_cases,
        lambda case, profile: case,
        lambda model, solved, normalization, directory, allowed_roles: (
            solved[0],
            None,
            None,
        ),
        lambda request: (None, 1.5),
        lambda case, request, result: (case, request, result),
        _record,
        lambda: None,
        lambda: None,
        frozenset(),
    )


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        env={
            **isolated_git_environment(),
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        },
    )


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    path.write_bytes(data)


@pytest.fixture
def family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path / "repo"
    rng = random.Random(7)
    points = [
        {"x": rng.uniform(0.0, 1.0), "y": rng.uniform(0.0, 1.0)} for _ in range(64)
    ]
    ids = [f"fresh-h1-{i:03d}" for i in range(64)]
    secrets = generate_campaign_secrets()
    commitment = compute_holdout_commitment(
        campaign_hash="c" * 64,
        campaign_secrets=secrets,
        holdout_coordinates=points,
        created_utc="2026-10-13T00:00:00Z",
    )
    _write(
        root / "artifacts/campaigns" / CAMPAIGN / "FRESH-HOLDOUTS.json",
        {
            "sets": {
                "H1": {
                    "case_ids": ids,
                    "commitment": json.loads(canonical_json_bytes(commitment)),
                },
                "H2": {"case_ids": [f"fresh-h2-{i:03d}" for i in range(64)]},
            }
        },
    )
    protocol = {
        "protocol_id": s23.PROTOCOL_ID,
        "alpha": 0.1,
        "quantities": QUANTITIES,
        "families": {
            CAMPAIGN: {
                "bundle": BUNDLE,
                "calibration_case_ids": ["dev-1"],
                "h1_case_ids": ids,
            }
        },
        "coverage_test": {
            "gated": [
                f"{CAMPAIGN}:{kind}:{q}"
                for kind in ("budget_total", "reference_terms")
                for q in QUANTITIES[:2]
            ],
            "per_check_level": 0.05 / 4,
        },
        "thresholds_accepted": "test",
    }
    _write(root / s23.PROTOCOL_PATH, canonical_json_bytes(protocol))
    protocol_sha = hashlib.sha256((root / s23.PROTOCOL_PATH).read_bytes()).hexdigest()
    calibration = CalibrationRecord(
        bundle=BUNDLE,
        bundle_manifest_sha256="d" * 64,
        campaign_id=CAMPAIGN,
        protocol_sha256=protocol_sha,
        alpha=0.1,
        n=64,
        k=59,
        calibration_case_ids_sha256=case_ids_sha256(["dev-1"]),
        bounds={
            "budget_total": {q: 0.0165 for q in QUANTITIES},
            "reference_terms": {q: 0.0055 for q in QUANTITIES},
        },
        # x and y uncorrelated, unit precision; radius 0.6 around (0.5, 0.5)
        gate=MahalanobisGate(("x", "y"), (0.5, 0.5), ((1.0, 0.0), (0.0, 1.0)), 0.6),
    )
    _write(
        root / s23.sidecar_path(BUNDLE), canonical_json_bytes(calibration.document())
    )
    _write(root / s23.evidence_path(CAMPAIGN), {"evidence": True})
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fixture")
    private = tmp_path / "private" / "fresh-holdouts-v1.json"
    _write(
        private,
        {
            "holdout_salt_hex": secrets.holdout_salt.hex(),
            "sets": {
                "H1": [
                    {"case_id": i, "coordinates": p}
                    for i, p in zip(ids, points, strict=True)
                ]
            },
        },
    )
    monkeypatch.setattr(reveal, "adapter", fake_adapter)
    monkeypatch.setattr(reveal, "_bundle_model", lambda bundle: None)
    monkeypatch.setattr(reveal, "_normalization", lambda campaign: None)
    return {"root": root, "private": private, "points": points, "ids": ids}


def test_a_reveal_scores_counts_burns_and_runs_once(
    family: dict[str, Any], tmp_path: Path
) -> None:
    root = family["root"]
    document = reveal.run(root, CAMPAIGN, family["private"], tmp_path / "run", 1)
    assert (root / reveal.consumed_path(CAMPAIGN)).exists()
    assert (root / reveal.reveal_path(CAMPAIGN)).exists()
    burned = BurnedCaseRegistry.from_document(
        json.loads((root / reveal.burned_path(CAMPAIGN)).read_text())
    )
    assert burned.case_ids == frozenset(family["ids"])
    assert [case["case_id"] for case in document["cases"]] == family["ids"]
    counts = document["coverage"]["counts"]
    inside = [
        p
        for p in family["points"]
        if ((p["x"] - 0.5) ** 2 + (p["y"] - 0.5) ** 2) ** 0.5 <= 0.6
    ]
    expected = sum(0.015 * p["x"] + 0.001 <= 0.0165 for p in inside)
    total = counts["budget_total:p_relative_l2"]
    assert total["m"] == len(inside) == document["coverage"]["in_calibration"]
    assert total["covered"] == expected
    assert total["gated"] and not counts["budget_total:u_relative_l2"]["gated"]
    assert total["pass"] is (expected >= total["acceptance_count"])
    with pytest.raises(reveal.RevealRefused, match="already revealed"):
        reveal.run(root, CAMPAIGN, family["private"], tmp_path / "run", 1)


def test_an_uncommitted_calibration_is_refused(
    family: dict[str, Any], tmp_path: Path
) -> None:
    root = family["root"]
    path = root / s23.evidence_path(CAMPAIGN)
    path.write_text('{"changed": true}')
    with pytest.raises(reveal.RevealRefused, match="uncommitted"):
        reveal.run(root, CAMPAIGN, family["private"], tmp_path / "run", 1)
    assert not (root / reveal.consumed_path(CAMPAIGN)).exists()


def test_tampered_coordinates_do_not_match_the_commitment(
    family: dict[str, Any], tmp_path: Path
) -> None:
    document = json.loads(family["private"].read_text())
    document["sets"]["H1"][0]["coordinates"]["x"] += 1e-9
    family["private"].write_text(json.dumps(document))
    with pytest.raises(ValueError, match="commitment"):
        reveal.run(family["root"], CAMPAIGN, family["private"], tmp_path / "run", 1)
    assert not (family["root"] / reveal.reveal_path(CAMPAIGN)).exists()


def test_the_run_root_must_be_outside_the_repository(family: dict[str, Any]) -> None:
    with pytest.raises(reveal.RevealRefused, match="outside"):
        reveal.run(family["root"], CAMPAIGN, family["private"], family["root"] / "r", 1)


def test_the_main_command_reports_a_refusal(
    family: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reveal, "REPOSITORY", family["root"])
    monkeypatch.setattr(s23, "FAMILIES", {CAMPAIGN: BUNDLE})
    arguments = ["--campaign", CAMPAIGN, "--reveal", str(family["private"])]
    assert (
        reveal.main([*arguments, "--run-root", str(tmp_path / "run"), "--workers", "1"])
        == 0
    )
    assert reveal.main([*arguments, "--run-root", str(tmp_path / "run")]) == 1


def test_an_interrupted_reveal_resumes_after_its_consumption_record(
    family: dict[str, Any], tmp_path: Path
) -> None:
    root = family["root"]
    protocol, digest, calibration = reveal.preconditions(root, CAMPAIGN)
    reveal.consume(root, CAMPAIGN, digest, calibration)
    first = (root / reveal.consumed_path(CAMPAIGN)).read_bytes()
    reveal.consume(root, CAMPAIGN, digest, calibration)
    assert (root / reveal.consumed_path(CAMPAIGN)).read_bytes() == first
    assert reveal.run(root, CAMPAIGN, family["private"], tmp_path / "run", 1)["cases"]
