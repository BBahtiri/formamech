"""The S24 ADF viability run glue (ADR 0106 decision 5); no training here."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from campaigns.dogbone_v1 import s24_adf
from campaigns.dogbone_v1.family import compiled_family
from campaigns.dogbone_v1.s10_requests import _load_public_sources
from campaigns.dogbone_v1.s11_contingency_source import (
    VerifiedS10Request,
    _rebuild_request,
)
from campaigns.dogbone_v1.s13c_data import case_energy, geometry_case
from campaigns.dogbone_v1.s13c_evaluate import _verify_binding
from campaigns.dogbone_v1.s13c_protocol import (
    build_dogbone_s13c_protocol,
    build_initial_operator,
    energy_config,
    load_s13c_normalization,
)
from piginot.contracts import ExecutionProfile
from piginot.lifting import AdfLifting
from tests.platform_support import requires_pinned_initial_weights

_PROFILE = ExecutionProfile(
    python_version="3.11.0",
    library_versions={"torch": "test"},
    blas_threads=1,
    device="cpu",
    dtype="float64",
    rng_algorithm="none",
    seed=0,
    container_digest=None,
)
_REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(scope="module")
def requests() -> tuple[VerifiedS10Request, ...]:
    protocol = build_dogbone_s13c_protocol()
    _, manifest = _load_public_sources()
    cases = {case.case_id: case for case in manifest.cases}
    family = compiled_family()
    chosen = (protocol.fit_case_ids[0], protocol.fit_case_ids[-1])
    return tuple(
        VerifiedS10Request(
            cases[case_id], _rebuild_request(cases[case_id], _PROFILE, family)
        )
        for case_id in chosen
    )


def _nodes(case: Any, tag: str) -> torch.Tensor:
    mesh = case.request.mesh
    ids = sorted(
        {n for facet in mesh.semantic_tags[tag] for n in mesh.boundary_facets[facet]}
    )
    return torch.tensor([[mesh.nodes[i] for i in ids]], dtype=torch.float64)


def test_the_adf_case_lifts_onto_the_s10_boundary_data(
    requests: tuple[VerifiedS10Request, ...],
) -> None:
    normalization = load_s13c_normalization()
    for verified in requests:
        case = s24_adf.adf_geometry_case(verified, normalization, dtype=torch.float64)
        base = geometry_case(verified, normalization, dtype=torch.float64)
        assert case.case_id == base.case_id
        assert torch.equal(case.parameters(), base.parameters())
        lifting = case.lifting()
        assert isinstance(lifting, AdfLifting)
        d = base.prescribed_displacement_mm
        for tag, component, value in (
            ("left_symmetry", 0, 0.0),
            ("right_grip", 0, d),
            ("bottom", 1, 0.0),
        ):
            lifted = lifting(_nodes(case, tag))
            phi = torch.randn(lifted.offset.shape, dtype=torch.float64)
            u = lifted.offset + lifted.scale * phi
            assert float((u[0, :, component] - value).abs().max()) <= 1e-12 * d


def test_the_adf_case_trains_in_float32_with_finite_gradients(
    requests: tuple[VerifiedS10Request, ...],
) -> None:
    case = s24_adf.adf_geometry_case(
        requests[0], load_s13c_normalization(), dtype=torch.float32
    )
    assert case.lifting().displacement.dtype is torch.float32
    model = build_initial_operator()
    terms = case_energy(
        model,
        case,
        energy_config(),
        generator=torch.Generator().manual_seed(0),
    )
    terms.total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)


def test_a_case_whose_scale_differs_from_s10_is_refused(
    requests: tuple[VerifiedS10Request, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s24_adf, "DISPLACEMENT_TOLERANCE", -1.0)
    with pytest.raises(ValueError, match="displacement scale"):
        s24_adf.adf_geometry_case(
            requests[0], load_s13c_normalization(), dtype=torch.float64
        )


def test_materialization_requires_a_verified_source() -> None:
    with pytest.raises(TypeError, match="SelectiveS10Source"):
        s24_adf.materialize_adf_training_data(
            object(),  # type: ignore[arg-type]
            build_dogbone_s13c_protocol(),
            load_s13c_normalization(),
        )


def test_the_spec_must_use_the_adf_lifting() -> None:
    with pytest.raises(ValueError, match="adf lifting"):
        s24_adf.load_spec.__wrapped__(s24_adf.BASE_SPEC_PATH)


def test_the_committed_protocol_is_the_rebuild_and_accepted() -> None:
    committed = json.loads((s24_adf.REPOSITORY / s24_adf.PROTOCOL_PATH).read_text())
    assert committed == s24_adf.build_protocol()
    assert committed["status"] == "accepted"
    viability = committed["viability"]
    assert viability["problem_spec_hash"] == s24_adf.load_spec().problem_hash
    assert viability["optimizer_steps"] == 2400
    assert s24_adf.accepted_protocol(require_committed=False) == committed


def test_a_changed_protocol_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = s24_adf.build_protocol()
    document["viability"]["optimizer_steps"] = 400
    path = tmp_path / "artifacts/problem/S24-PROTOCOL.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(s24_adf, "REPOSITORY", tmp_path)
    monkeypatch.setattr(s24_adf, "build_protocol", lambda: {"rebuilt": True})
    with pytest.raises(ValueError, match="differs from its rebuild"):
        s24_adf.accepted_protocol(require_committed=False)
    path.unlink()
    with pytest.raises(RuntimeError, match="not written"):
        s24_adf.accepted_protocol(require_committed=False)


def test_an_unaccepted_protocol_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = {"status": "proposed"}
    path = tmp_path / "artifacts/problem/S24-PROTOCOL.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(s24_adf, "REPOSITORY", tmp_path)
    monkeypatch.setattr(s24_adf, "build_protocol", lambda: document)
    with pytest.raises(RuntimeError, match="not accepted"):
        s24_adf.accepted_protocol(require_committed=False)


def test_training_requires_the_protocol_and_a_clean_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(**_: object) -> None:
        raise RuntimeError("must be committed before the run")

    monkeypatch.setattr(s24_adf, "accepted_protocol", refuse)
    with pytest.raises(RuntimeError, match="committed"):
        s24_adf.run_adf_trial(s10_run_root=tmp_path, run_root=tmp_path / "run")

    class _Dirty:
        head = _REVISION
        dirty_worktree = True

    monkeypatch.setattr(s24_adf, "accepted_protocol", lambda **_: {})
    monkeypatch.setattr(s24_adf, "_git_provenance", lambda: _Dirty())
    with pytest.raises(RuntimeError, match="clean revision"):
        s24_adf.run_adf_trial(s10_run_root=tmp_path, run_root=tmp_path / "run")


@requires_pinned_initial_weights
def test_the_binding_binds_the_spec_and_the_evaluator_checks_it() -> None:
    protocol = build_dogbone_s13c_protocol()
    binding = s24_adf.build_adf_binding(protocol, _REVISION)
    assert binding.problem_spec_hash == s24_adf.load_spec().problem_hash
    _verify_binding(binding, protocol, binding.problem_spec_hash)
    with pytest.raises(ValueError, match="problem_spec_hash"):
        _verify_binding(binding, protocol)


def test_the_screen_record_applies_the_guardrails() -> None:
    protocol = s24_adf.build_protocol()
    metrics = SimpleNamespace(
        median_absolute_peak_error=0.01,
        p90_absolute_peak_error=0.02,
        median_signed_peak_error=-0.005,
        p_relative_l2=0.012,
        u_relative_l2=0.001,
        invalid_collocation_points=0,
    )
    evaluation = SimpleNamespace(
        result=SimpleNamespace(metrics=metrics),
        run_binding=SimpleNamespace(
            problem_spec_hash="f" * 64, implementation_revision=_REVISION
        ),
        manifest=SimpleNamespace(
            run_hash="1" * 64,
            step=2400,
            manifest_hash="2" * 64,
            model_state_hash="3" * 64,
        ),
        diagnostics=[
            SimpleNamespace(relative_energy_gap=g) for g in (1e-4, 2e-4, 3e-4)
        ],
        minimum_collocation_determinant=0.98,
        failed_guardrails=(),
        development_opened=False,
    )
    record = s24_adf.screen_record(evaluation, protocol)
    assert record["passed"] and record["median_relative_energy_gap"] == 2e-4
    assert record["s13c_p_relative_l2"] == pytest.approx(0.00908661, rel=1e-6)
    failing = SimpleNamespace(
        **{**vars(evaluation), "failed_guardrails": ("u_relative_l2",)}
    )
    assert not s24_adf.screen_record(failing, protocol)["passed"]
