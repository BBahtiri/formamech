"""The shipped families as ProblemSpecs (S24, ADR 0106 decisions 4 and 6)."""

from __future__ import annotations

import json
import random
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import torch

from campaigns import s24_regression as regression
from piginot.bundle import Bundle, load_bundle
from piginot.contracts import CanonicalQuantity, ProblemSpec
from piginot.lifting import AdfLifting, boundary_points, boundary_residual
from piginot.problem import build_lifting, query_frame

REPOSITORY = Path(__file__).resolve().parents[2]
FAMILIES = {
    "dogbone-v1": "dogbone-v1-s13c-cp-freeze",
    "notched-v1": "notched-v1-cp-freeze",
}
DRAWS = 1000


@cache
def _bundle(name: str) -> Bundle:
    return load_bundle(REPOSITORY / "bundles" / name)


def _spec(campaign: str, variant: str = "") -> ProblemSpec:
    return ProblemSpec.load(REPOSITORY / regression.spec_path(campaign, variant))


def _draws(spec: ProblemSpec, count: int, seed: int) -> list[dict[str, float]]:
    generator = random.Random(seed)
    result: list[dict[str, float]] = []
    while len(result) < count:
        values = {p.name: generator.uniform(p.lo, p.hi) for p in spec.parameters}
        scoped = spec.scope(values)
        if all(c.lo <= c.expression.evaluate(scoped) <= c.hi for c in spec.constraints):
            result.append(values)
    return result


@pytest.mark.parametrize("campaign", FAMILIES)
def test_the_family_specs_read_and_pass_the_mesh_rules(campaign: str) -> None:
    bundle = _bundle(FAMILIES[campaign])
    for variant in ("", "adf"):
        spec = _spec(campaign, variant)
        spec.check_mesh(bundle.reference_mesh)
        path = REPOSITORY / regression.spec_path(campaign, variant)
        assert ProblemSpec.from_document(json.loads(path.read_text("utf-8"))) == spec


@pytest.mark.parametrize("campaign", FAMILIES)
def test_the_adf_variant_changes_only_the_lifting(campaign: str) -> None:
    base = _spec(campaign).document()
    adf = _spec(campaign, "adf").document()
    assert adf["lifting"]["kind"] == "adf" and base["lifting"]["kind"] == "quarter"
    assert adf["lifting"]["displacement_scale"] == base["lifting"]["displacement_scale"]
    assert adf["name"] == f"{base['name']}-adf"
    for document in (base, adf):
        del document["lifting"], document["name"]
    assert adf == base


@pytest.mark.parametrize("campaign", FAMILIES)
def test_the_spec_matches_the_shipped_bundle(campaign: str) -> None:
    checks = regression.envelope_checks(_spec(campaign), FAMILIES[campaign])
    assert all(checks.values()), checks


@pytest.mark.parametrize("campaign", FAMILIES)
def test_the_quarter_lifting_is_the_bundle_lifting_bit_for_bit(campaign: str) -> None:
    from piginot.lifting import AxisAlignedTensionLifting

    spec = _spec(campaign)
    bundle = _bundle(FAMILIES[campaign])
    model = bundle.model
    for values in _draws(spec, 8, seed=1):
        geometry = {
            p.name: CanonicalQuantity(values[p.name], p.unit, p.unit.dimension)
            for p in spec.parameters
            if p.role == "geometry"
        }
        mesh = bundle.geometry(geometry).mesh
        if mesh is None:
            continue
        points = torch.tensor([mesh.nodes], dtype=torch.float64)
        extent = [item.evaluate(values) for item in model.frame_extent]
        shipped = AxisAlignedTensionLifting(
            torch.tensor([extent], dtype=torch.float64),
            torch.tensor(
                [model.lifting_displacement.evaluate(values)], dtype=torch.float64
            ),
        )(points)
        built = build_lifting(spec, [values])(points)
        assert torch.equal(built.offset, shipped.offset)
        assert torch.equal(built.scale, shipped.scale)
        assert query_frame(spec, [values]).extent.tolist() == [extent]


@pytest.mark.parametrize("campaign", FAMILIES)
def test_both_liftings_meet_the_data_for_a_thousand_parameter_vectors(
    campaign: str,
) -> None:
    """G24.lifting-exact on the family geometry, without meshes."""

    adf_spec = _spec(campaign, "adf")
    values = _draws(adf_spec, DRAWS, seed=2)
    adf = build_lifting(adf_spec, values)
    assert isinstance(adf, AdfLifting)
    generator = torch.Generator().manual_seed(3)
    assert boundary_residual(adf, 33, generator=generator) <= 1e-12
    quarter = build_lifting(_spec(campaign), values)
    worst = 0.0
    for component, pieces in enumerate(adf.pieces):
        for piece in pieces:
            points = boundary_points(piece, 33)
            lifted = quarter(points)
            phi = torch.randn(
                lifted.offset.shape, generator=generator, dtype=torch.float64
            )
            u = lifted.offset + lifted.scale * phi
            error = (u[..., component] - piece.value[:, None]).abs()
            worst = max(worst, float((error / adf.displacement[:, None]).max()))
    assert worst <= 1e-12


def _evidence() -> dict[str, Any]:
    path = REPOSITORY / regression.EVIDENCE
    document: dict[str, Any] = json.loads(path.read_text("utf-8"))
    return document


@pytest.mark.parametrize("campaign", FAMILIES)
def test_the_committed_regression_evidence_is_current_and_passes(campaign: str) -> None:
    record = _evidence()["families"][campaign]
    assert record["problem_hash"] == _spec(campaign).problem_hash
    assert record["adf_problem_hash"] == _spec(campaign, "adf").problem_hash
    assert record["passed"] is True
    development = record["development"]
    assert development["cases"] == 64
    for key in (
        "lifting_bit_identical",
        "frame_bit_identical",
        "fem_requests_byte_identical",
    ):
        assert development[key] == 64
    for kind in ("quarter", "adf"):
        lifting = record["lifting"][kind]
        assert lifting["passed"] and lifting["problem_hash"] in (
            record["problem_hash"],
            record["adf_problem_hash"],
        )
        assert lifting["parameter_vectors"] >= 64
