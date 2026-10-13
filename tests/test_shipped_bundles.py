"""The bundles in bundles/ are the gated ones and serve their example requests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from piginot.bundle import load_bundle, predict
from piginot.cli import parse_request

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ROOT / "bundles"
RECORDED = {
    "dogbone-v1-s13c-cp-freeze": (
        "009db9d379d6f94ccda5395762e22c9c810c466a686626d500f1575d45b1bb4c",
        "HOLDOUT_VERIFIED",
    ),
    "notched-v1-cp-freeze": (
        "3749f4be4761a55582dd07f90e3cc5950b8231357162366d79dc47463dddcc99",
        "DEVELOPMENT_VERIFIED",
    ),
}
REQUESTS = (
    ("dogbone-v1-s13c-cp-freeze", "dogbone-in-domain.json", "IN_DOMAIN", []),
    ("notched-v1-cp-freeze", "notched-in-domain.json", "IN_DOMAIN", []),
    (
        "notched-v1-cp-freeze",
        "notched-out-of-envelope.json",
        "REFUSED",
        ["PARAMETER_OUT_OF_BOUNDS"],
    ),
)


def test_the_shipped_bundles_are_the_gated_ones() -> None:
    assert sorted(path.name for path in BUNDLES.iterdir() if path.is_dir()) == sorted(
        [*RECORDED, "requests", "fem-recipes", "calibration", "h1-results"]
    )
    # S22 FEM recipes, S23 calibrations and their H1 results: one sidecar per
    # gated bundle, bound to its manifest
    for folder in ("fem-recipes", "calibration", "h1-results"):
        sidecars = sorted((BUNDLES / folder).glob("*.json"))
        assert [p.stem for p in sidecars] == sorted(RECORDED)
        for path in sidecars:
            document = json.loads(path.read_text("utf-8"))
            assert document["bundle_manifest_sha256"] == RECORDED[path.stem][0]
    for name, (manifest, claim) in RECORDED.items():
        bundle = load_bundle(BUNDLES / name)
        assert bundle.manifest.manifest_sha256 == manifest
        assert bundle.claim_level.value == claim
    recorded = json.loads(
        (ROOT / "artifacts/campaigns/notched-v1/gates/S15.json").read_text("utf-8")
    )
    assert recorded["bundle_manifest_sha256"] == RECORDED["notched-v1-cp-freeze"][0]


@pytest.mark.parametrize(("name", "request_file", "verdict", "codes"), REQUESTS)
def test_example_requests_are_served_or_refused(
    name: str, request_file: str, verdict: str, codes: list[str]
) -> None:
    bundle = load_bundle(BUNDLES / name)
    document = json.loads((BUNDLES / "requests" / request_file).read_text("utf-8"))
    inputs, queries = parse_request(document)
    prediction = predict(bundle, inputs, queries, confidence=False)
    assert prediction.verdict.value == verdict
    assert list(prediction.reason_codes) == codes
    if verdict == "IN_DOMAIN":
        assert prediction.claim_level.value == RECORDED[name][1]
        assert prediction.document()["qois"]
