from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import rfc8785
import torch

import piginot
from piginot.bundle import (
    BUNDLE_FILES,
    REQUIRED_LIMITATION_PARAGRAPH,
    Bundle,
    BundleIntegrityError,
    BundleReasonCode,
    PredictionReasonCode,
    Verdict,
    inspect,
    load_bundle,
    predict,
)
from piginot.bundle import inference as inference_module
from piginot.bundle import serving as serving_module
from piginot.bundle.inference import (
    LIMITATIONS_PATH,
    MODEL_PATH,
    VERIFICATION_PATH,
    WEIGHTS_PATH,
    containing_cells,
)
from piginot.bundle.serving import (
    ServedPoints,
    ServingError,
    build_energy_form,
    read_safetensors,
)
from piginot.contracts import CanonicalQuantity, Unit
from piginot.verification import ClaimLevel
from tests.bundle import helpers

G15_FIELDS = ("u", "P_served", "F_2D", "F33", "J_3D", "sigma")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Bundle:
    directory = tmp_path_factory.mktemp("bundle") / "synthetic"
    helpers.build(directory)
    return load_bundle(directory)


def test_load_and_inspect(bundle: Bundle) -> None:
    report = inspect(bundle)

    assert tuple(report["files"]) == BUNDLE_FILES
    assert report["claim_level"] == "HOLDOUT_VERIFIED"
    assert report["build_identities"] == {
        "origin": "tests",
        "piginot_version": piginot.__version__,
    }
    assert report["limitations"]["paragraph"] == REQUIRED_LIMITATION_PARAGRAPH
    assert report["verification"]["outstanding_work"][0]["status"] == "NOT_RUN"
    assert report["model"]["served_stress"] == bundle.serving.stress_authority
    assert report["model"]["parameter_count"] == bundle.serving.parameter_count
    assert len(report["envelope"]["inputs"]) == 7
    assert (
        bundle.identities["weights_sha256"]
        == bundle.manifest.files[WEIGHTS_PATH].sha256
    )


def test_predict_returns_every_g15_field(bundle: Bundle) -> None:
    queries = [[5.0, 2.0], [0.0, 0.0], [10.0, 3.0]]
    prediction = predict(bundle, helpers.request(), queries)

    assert prediction.verdict is Verdict.IN_DOMAIN and prediction.in_domain
    assert prediction.reason_codes == ()
    assert prediction.claim_level is ClaimLevel.HOLDOUT_VERIFIED
    assert prediction.limitations == REQUIRED_LIMITATION_PARAGRAPH
    assert prediction.identities["mesh_id"] == prediction.mesh.mesh_id  # type: ignore[union-attr]
    assert prediction.units["P_served"] == "MPa"
    vertices = prediction.vertices
    queried = prediction.queries
    assert vertices is not None and queried is not None
    count = len(prediction.mesh.nodes)  # type: ignore[union-attr]
    assert vertices.u.shape == (count, 2) and queried.u.shape == (3, 2)
    assert vertices.P_served.shape == (count, 2, 2)
    assert vertices.F33.shape == vertices.J_3D.shape == (count,)
    for points in (vertices, queried):
        determinant = torch.linalg.det(points.F_2D) * points.F33
        assert torch.allclose(points.J_3D, determinant, rtol=1e-12)
        sigma = (
            points.P_served @ points.F_2D.transpose(-2, -1) / points.J_3D[:, None, None]
        )
        assert torch.equal(points.sigma, sigma)
        assert points.confidence_bound is not None
        assert bool((points.confidence_bound > 0).all())
    assert queried.confidence_indicator is not None
    assert vertices.confidence_indicator is not None
    # The query at the origin is a mesh vertex; its indicator bounds that vertex.
    origin = int(torch.argmin(vertices.coordinates.norm(dim=1)))
    assert queried.confidence_indicator[1] >= vertices.confidence_indicator[origin]
    assert prediction.constitutive_gap is not None
    assert prediction.constitutive_gap < 1e-12
    (qoi,) = prediction.qois
    assert qoi.name == "peak" and qoi.unit == "MPa"
    assert qoi.value == pytest.approx(
        float(vertices.P_served[qoi.vertex].norm()), rel=0, abs=0
    )

    document = json.loads(json.dumps(prediction.document()))
    for name in G15_FIELDS:
        assert name in document["vertices"] and name in document["queries"]
    for key in (
        "verdict",
        "reason_codes",
        "claim_level",
        "identities",
        "units",
        "qois",
        "limitations",
        "stress_authority",
    ):
        assert key in document
    assert "confidence" in document["queries"]
    assert "P_material" not in document["vertices"]


def test_predict_without_confidence(bundle: Bundle) -> None:
    prediction = predict(bundle, helpers.request(), confidence=False)
    assert prediction.vertices is not None
    assert prediction.vertices.confidence_indicator is None
    assert prediction.queries is None
    assert "confidence" not in prediction.document()["vertices"]


def _quantity(value: float, unit: Unit) -> CanonicalQuantity:
    return CanonicalQuantity(value, unit)


@pytest.mark.parametrize(
    ("change", "queries", "code"),
    [
        ({"nominal_axial_strain": 0.25}, None, "PARAMETER_OUT_OF_BOUNDS"),
        ({"nominal_axial_strain": 0.0999999}, None, "PARAMETER_OUT_OF_BOUNDS"),
        ({"L_total": 39.0}, None, "PARAMETER_OUT_OF_BOUNDS"),
        ({}, [[1.0e3, 0.0]], "QUERY_OUTSIDE_DOMAIN"),
        ({}, [[float("nan"), 0.0]], "INVALID_QUERY"),
        ({}, [], "INVALID_QUERY"),
    ],
)
def test_out_of_envelope_requests_refuse_with_a_reason(
    bundle: Bundle, change: dict[str, float], queries: Any, code: str
) -> None:
    prediction = predict(bundle, helpers.request(**change), queries)

    assert prediction.verdict is Verdict.REFUSED
    assert code in prediction.reason_codes, prediction.reason_codes
    assert prediction.claim_level is ClaimLevel.NOT_VERIFIED
    assert prediction.vertices is None and prediction.qois == ()
    assert prediction.limitations == REQUIRED_LIMITATION_PARAGRAPH
    assert prediction.document()["limitations"] == REQUIRED_LIMITATION_PARAGRAPH


def test_the_s14_refusal_fixtures_refuse_through_predict(bundle: Bundle) -> None:
    from campaigns.dogbone_v1.s14_envelope import refusal_fixtures

    fixtures = [f for f in refusal_fixtures() if f.mesh_mutation is None]
    by_category: dict[str, Any] = {}
    for fixture in fixtures:
        by_category.setdefault(fixture.category, fixture)
    assert set(by_category) == {
        "primitive-bound",
        "derived-bound",
        "material-load",
        "query-containment",
        "guard-band",
    }
    for fixture in by_category.values():
        prediction = predict(
            bundle, fixture.request_theta(), fixture.queries, confidence=False
        )
        assert prediction.verdict is Verdict.REFUSED
        assert fixture.expected_code in prediction.reason_codes, fixture.fixture_id


def test_malformed_requests_refuse(bundle: Bundle) -> None:
    request = helpers.request()
    missing = {k: v for k, v in request.items() if k != "W_gauge"}
    extra = {**request, "thickness": _quantity(1.0, Unit.MM)}
    wrong_unit = {**request, "nominal_axial_strain": _quantity(0.15, Unit.MPA)}
    bare = {**request, "L_total": 54.0}
    for inputs, code in (
        (missing, "MISSING_PARAMETER"),
        (extra, "UNKNOWN_PARAMETER"),
        (wrong_unit, "PARAMETER_UNIT_MISMATCH"),
        (bare, "INVALID_PARAMETER_VALUE"),
        ("not a mapping", "INVALID_PARAMETER_VALUE"),
    ):
        prediction = predict(bundle, inputs)  # type: ignore[arg-type]
        assert prediction.reason_codes == (code,), (inputs, prediction.reason_codes)


def test_inadmissible_deformation_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "folded"
    helpers.build(directory, weights=helpers.tiny_weights(output_bias=(-400.0, 0.0)))
    prediction = predict(load_bundle(directory), helpers.request())
    assert prediction.reason_codes == (
        PredictionReasonCode.INADMISSIBLE_DEFORMATION.value,
    )


# ------------------------------------------------------------ G15.substitution

HEAD = "test-predicted-stress-head"
HEAD_ARCHITECTURE = "test-two-head-v1"
OFFSET = 5.0


class _TwoHead:
    """A stand-in architecture whose served stress comes from its own head."""

    architecture_id = HEAD_ARCHITECTURE
    stress_authority = HEAD

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.parameter_count = inner.parameter_count

    def serve(self, query_xy: torch.Tensor, **kwargs: Any) -> ServedPoints:
        served = self._inner.serve(query_xy, **kwargs)
        head = served.stress + OFFSET * torch.eye(2, dtype=served.stress.dtype)
        return ServedPoints(served.u, served.gradient, head)

    def residual(self, query_xy: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self._inner.residual(query_xy, **kwargs)


def _two_head(
    config: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    meta: Mapping[str, str],
) -> _TwoHead:
    return _TwoHead(build_energy_form(config, tensors, meta))


@pytest.fixture
def two_head_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Bundle:
    monkeypatch.setitem(serving_module.ARCHITECTURES, HEAD_ARCHITECTURE, _two_head)
    directory = tmp_path / "two-head"
    helpers.build(
        directory,
        model=helpers.tiny_model_spec(
            architecture_id=HEAD_ARCHITECTURE, served_stress=HEAD
        ),
    )
    return load_bundle(directory)


def _serves_declared_head(prediction: Any) -> bool:
    points = prediction.vertices
    identity = torch.eye(2, dtype=points.P_material.dtype)
    return (
        bool(
            torch.allclose(
                points.P_served, points.P_material + OFFSET * identity, atol=1e-9
            )
        )
        and prediction.stress_authority == HEAD
    )


def test_served_stress_is_the_declared_authority(two_head_bundle: Bundle) -> None:
    prediction = predict(two_head_bundle, helpers.request(), confidence=False)
    assert _serves_declared_head(prediction)
    assert prediction.constitutive_gap == pytest.approx(OFFSET * 2**0.5 / 760.0)


def test_the_serve_p_material_mutant_is_caught(
    two_head_bundle: Bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``SERVE_P_MATERIAL``: serving the material law instead of ``P_served``."""

    original = inference_module._fields

    def substituted(*args: Any, **kwargs: Any) -> Any:
        points = original(*args, **kwargs)
        return dataclasses.replace(points, P_served=points.P_material)

    monkeypatch.setattr(inference_module, "_fields", substituted)
    prediction = predict(two_head_bundle, helpers.request(), confidence=False)
    assert not _serves_declared_head(prediction)


def test_a_declared_authority_the_architecture_lacks_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "mismatch"
    helpers.write_raw(
        directory,
        {MODEL_PATH: helpers.tiny_model_spec(served_stress=HEAD).to_bytes()},
    )
    with pytest.raises(BundleIntegrityError) as error:
        load_bundle(directory)
    assert error.value.reason_codes == (BundleReasonCode.STRESS_AUTHORITY_MISMATCH,)


# ------------------------------------------------------------ load refusals


def _limitations(paragraph: str) -> bytes:
    return rfc8785.dumps(
        {"paragraph": paragraph, "items": [{"id": "a", "statement": "b"}]}
    )


@pytest.mark.parametrize(
    ("replace", "claim", "code"),
    [
        ({LIMITATIONS_PATH: _limitations("Trust me.")}, None, "LIMITATIONS_MISSING"),
        (
            {
                LIMITATIONS_PATH: rfc8785.dumps(
                    {"paragraph": REQUIRED_LIMITATION_PARAGRAPH, "items": [{}]}
                )
            },
            None,
            "REPORT_INVALID",
        ),
        ({}, "DEVELOPMENT_VERIFIED", "REPORT_INVALID"),
        ({}, "CERTIFIED", "REPORT_INVALID"),
        (
            {VERIFICATION_PATH: rfc8785.dumps(helpers.verification(verdict="FAIL"))},
            None,
            "REPORT_INVALID",
        ),
        (
            {
                VERIFICATION_PATH: rfc8785.dumps(
                    helpers.verification(
                        outstanding_work=[{"id": "x", "status": "PASS"}]
                    )
                )
            },
            None,
            "REPORT_INVALID",
        ),
        ({VERIFICATION_PATH: b"{"}, None, "REPORT_INVALID"),
        ({LIMITATIONS_PATH: b"[]"}, None, "REPORT_INVALID"),
        ({"extra/notes.txt": b"x"}, None, "FILE_SET_INVALID"),
        ({WEIGHTS_PATH: None}, None, "FILE_SET_INVALID"),
        ({MODEL_PATH: b"{}"}, None, "SPEC_INVALID"),
        (
            {WEIGHTS_PATH: helpers.tiny_weights(digest="blake2b256:" + "0" * 64)},
            None,
            "WEIGHTS_INVALID",
        ),
        ({WEIGHTS_PATH: b"\x00" * 4}, None, "WEIGHTS_INVALID"),
        (
            {MODEL_PATH: helpers.tiny_model_spec(architecture_id="unknown").to_bytes()},
            None,
            "WEIGHTS_INVALID",
        ),
        (
            {
                MODEL_PATH: helpers.tiny_model_spec(
                    material_parameters={
                        "elastic_modulus": "elastic_modulus",
                        "poisson_ratio": "thickness",
                    }
                ).to_bytes()
            },
            None,
            "SPEC_INCONSISTENT",
        ),
        (
            {
                MODEL_PATH: helpers.tiny_model_spec(
                    boundary_conditions={"top": ("x",)}
                ).to_bytes()
            },
            None,
            "SPEC_INCONSISTENT",
        ),
    ],
)
def test_inconsistent_bundles_are_refused(
    tmp_path: Path, replace: dict[str, bytes | None], claim: str | None, code: str
) -> None:
    directory = tmp_path / "b"
    helpers.write_raw(directory, replace, claim_level=claim or helpers.CLAIM)
    with pytest.raises(BundleIntegrityError) as error:
        load_bundle(directory)
    assert f"BUNDLE_{code}" in [c.value for c in error.value.reason_codes]


def test_build_refuses_existing_paths_and_leaves_nothing_on_failure(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileExistsError):
        helpers.build(tmp_path)
    target = tmp_path / "bad"
    with pytest.raises(BundleIntegrityError):
        helpers.build(
            target, verification=helpers.verification(claim_level="NOT_VERIFIED")
        )
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_safetensors_reader_refuses_malformed_headers() -> None:
    weights = helpers.tiny_weights()
    tensors, metadata = read_safetensors(weights)
    assert metadata["architecture_id"] and tensors
    for payload in (
        (10**9).to_bytes(8, "little") + b"{}",
        (2).to_bytes(8, "little") + b"no",
        (2).to_bytes(8, "little") + b"[]",
    ):
        with pytest.raises(ServingError):
            read_safetensors(payload)
    with pytest.raises(ServingError):
        build_energy_form({"parameter_names": []}, tensors, metadata)
    with pytest.raises(ServingError):
        build_energy_form(dict(helpers.tiny_model_spec().operator_config), {}, metadata)
    with pytest.raises(ServingError):
        build_energy_form(
            dict(helpers.tiny_model_spec().operator_config),
            tensors,
            {**metadata, "architecture_id": "other"},
        )


def test_containing_cells_handles_vertices_edges_and_misses() -> None:
    vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
    points = torch.tensor([[0.2, 0.2], [0.5, 0.5], [0.9, 0.9], [2.0, 2.0]])
    assert containing_cells(vertices, cells, points) == [(0,), (0, 1), (1,), (1,)]


def test_v1_registers_one_architecture_and_refuses_any_other() -> None:
    from piginot.operators import ENERGY_ARCHITECTURE_ID

    assert tuple(serving_module.ARCHITECTURES) == (ENERGY_ARCHITECTURE_ID,)
    with pytest.raises(serving_module.ServingError, match="unregistered architecture"):
        serving_module.build_serving(
            "pi-ginot-s11-boundary-cross-attention-v1",
            {},
            b"",
            served_stress="constitutive-plane-stress-neo-hookean",
        )
