"""The S26 HGO viability glue (ADR 0108 decision 8); no training, no re-solve."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from campaigns.dogbone_v1 import material_run, s26_hgo
from campaigns.dogbone_v1.family import compiled_family
from campaigns.dogbone_v1.material_run import MaterialViability
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
from campaigns.dogbone_v1.s25_mooney_rivlin import encode_result, reference_entry
from piginot.contracts import ExecutionProfile, canonical_json_bytes
from piginot.contracts.units import Dimension
from piginot.physics import reference_modulus
from piginot.problem import admit_material
from tests.platform_support import requires_pinned_initial_weights

RUN = s26_hgo.RUN
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


def _same(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if hasattr(left, "__dataclass_fields__"):
        return type(left) is type(right) and all(
            _same(getattr(left, f.name), getattr(right, f.name)) for f in fields(left)
        )
    return left == right


@pytest.fixture(scope="module")
def requests() -> tuple[VerifiedS10Request, ...]:
    protocol = build_dogbone_s13c_protocol()
    _, manifest = _load_public_sources()
    cases = {case.case_id: case for case in manifest.cases}
    family = compiled_family()
    chosen = (protocol.fit_case_ids[0], protocol.screen_case_ids[0])
    return tuple(
        VerifiedS10Request(
            cases[case_id], _rebuild_request(cases[case_id], _PROFILE, family)
        )
        for case_id in chosen
    )


def test_the_material_is_hgo_on_the_shipped_matrix() -> None:
    material = RUN.physics_material()
    assert material.model == "hgo"
    values = {name: q.value for name, q in material.properties.items()}
    assert values["k1"] == 50.0 and values["k2"] == 2.0
    assert values["fibre_angle"] == 30.0
    assert material.properties["fibre_angle"].dimension is Dimension.ANGLE
    assert reference_modulus(material) == pytest.approx(760.0, rel=1e-13)
    assert RUN.fem_material().name == "dogbone-hgo-reference-material"


def test_the_spec_is_admitted_with_fem_and_only_the_material_differs() -> None:
    spec = RUN.load_spec()
    admission = admit_material(spec, fem=True)
    assert admission.passed, admission.document()
    base = json.loads(
        (material_run.REPOSITORY / "campaigns/dogbone_v1/problem_spec.json").read_text()
    )
    document = json.loads((material_run.REPOSITORY / RUN.spec_path).read_text())
    differing = {key for key in document if document[key] != base[key]}
    assert differing == {"name", "parameters", "material", "fem"}
    geometry = [p for p in document["parameters"] if p["role"] != "material"]
    assert geometry == [p for p in base["parameters"] if p["role"] != "material"]


def test_the_spec_must_use_the_declared_model() -> None:
    wrong = replace(
        RUN, model="goh", spec_path=Path("campaigns/dogbone_v1/problem_spec_hgo.json")
    )
    material_run._SPECS.clear()
    with pytest.raises(ValueError, match="goh"):
        wrong.load_spec()
    material_run._SPECS.clear()


def test_the_request_and_case_change_only_the_material(
    requests: tuple[VerifiedS10Request, ...],
) -> None:
    normalization = load_s13c_normalization()
    for verified in requests:
        request = RUN.request(verified.request)
        assert request.material == RUN.fem_material()
        assert replace(request, material=verified.request.material) == (
            verified.request
        )
        case = RUN.geometry_case(verified, normalization, dtype=torch.float64)
        base = geometry_case(verified, normalization, dtype=torch.float64)
        assert case.material == RUN.physics_material()
        for item in fields(base):
            if item.name != "material":
                assert _same(getattr(case, item.name), getattr(base, item.name))


def test_the_case_trains_in_float32_with_finite_gradients(
    requests: tuple[VerifiedS10Request, ...],
) -> None:
    case = RUN.geometry_case(
        requests[0], load_s13c_normalization(), dtype=torch.float32
    )
    model = build_initial_operator()
    terms = case_energy(
        model, case, energy_config(), generator=torch.Generator().manual_seed(0)
    )
    assert bool(torch.isfinite(terms.total))
    terms.total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)


def test_materialization_requires_a_verified_source() -> None:
    with pytest.raises(TypeError, match="SelectiveS10Source"):
        RUN.materialize(
            object(),  # type: ignore[arg-type]
            build_dogbone_s13c_protocol(),
            load_s13c_normalization(),
        )


def test_the_committed_protocol_is_the_rebuild_and_accepted() -> None:
    committed = json.loads((material_run.REPOSITORY / RUN.protocol_path).read_text())
    assert committed == RUN.build_protocol()
    assert committed["status"] == "accepted" and committed["step"] == "S26"
    viability = committed["viability"]
    assert viability["problem_spec_hash"] == RUN.load_spec().problem_hash
    assert viability["optimizer_steps"] == 2400
    assert viability["material"]["properties"]["fibre_angle"] == 30.0
    s13c = build_dogbone_s13c_protocol()
    assert viability["references"]["cases"] == list(s13c.screen_case_ids)
    assert viability["closed_evidence"] == ["DEVELOPMENT", "HOLDOUT", "H1", "H2"]
    assert RUN.accepted_protocol(require_committed=False) == committed


def test_a_changed_or_missing_protocol_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / RUN.protocol_path
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "accepted"}), encoding="utf-8")
    monkeypatch.setattr(material_run, "REPOSITORY", tmp_path)
    monkeypatch.setattr(
        MaterialViability, "build_protocol", lambda self, root=None: {"rebuilt": 1}
    )
    with pytest.raises(ValueError, match="differs from its rebuild"):
        RUN.accepted_protocol(require_committed=False)
    proposed = {"status": "proposed"}
    monkeypatch.setattr(
        MaterialViability, "build_protocol", lambda self, root=None: proposed
    )
    path.write_text(json.dumps(proposed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not accepted"):
        RUN.accepted_protocol(require_committed=False)
    monkeypatch.setattr(material_run, "_committed", lambda path: False)
    with pytest.raises(RuntimeError, match="committed before the run"):
        RUN.accepted_protocol()
    path.unlink()
    with pytest.raises(RuntimeError, match="not written"):
        RUN.accepted_protocol(require_committed=False)


class _Dirty:
    head = _REVISION
    dirty_worktree = True


def test_runs_require_the_protocol_and_a_clean_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: object, **_: object) -> None:
        raise RuntimeError("must be committed before the run")

    monkeypatch.setattr(MaterialViability, "accepted_protocol", refuse)
    with pytest.raises(RuntimeError, match="committed"):
        RUN.run_trial(s10_run_root=tmp_path, run_root=tmp_path / "run")
    with pytest.raises(RuntimeError, match="committed"):
        RUN.write_references(s10_run_root=tmp_path, output=tmp_path / "references")
    monkeypatch.setattr(MaterialViability, "accepted_protocol", lambda self, **_: {})
    monkeypatch.setattr(material_run, "_git_provenance", lambda: _Dirty())
    with pytest.raises(RuntimeError, match="clean revision"):
        RUN.run_trial(s10_run_root=tmp_path, run_root=tmp_path / "run")
    with pytest.raises(RuntimeError, match="clean revision"):
        RUN.write_references(s10_run_root=tmp_path, output=tmp_path / "references")


def test_references_are_written_outside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MaterialViability, "accepted_protocol", lambda self, **_: {})
    for output in (Path("relative"), material_run.REPOSITORY / "build" / "refs"):
        with pytest.raises(ValueError, match="outside the repository"):
            RUN.write_references(
                s10_run_root=tmp_path, output=output, require_committed=False
            )


@requires_pinned_initial_weights
def test_the_binding_binds_the_spec_and_the_evaluator_checks_it() -> None:
    protocol = build_dogbone_s13c_protocol()
    binding = RUN.build_binding(protocol, _REVISION)
    assert binding.problem_spec_hash == RUN.load_spec().problem_hash
    _verify_binding(binding, protocol, binding.problem_spec_hash)
    with pytest.raises(ValueError, match="problem_spec_hash"):
        _verify_binding(binding, protocol)


def test_the_reference_loader_checks_index_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from campaigns.dogbone_v1.s25_mooney_rivlin import solve_reference
    from tests.campaigns.test_dogbone_s25_mooney_rivlin import _plate_request

    request = _plate_request("plate-a")
    payload = solve_reference(request)
    monkeypatch.setattr(MaterialViability, "request", lambda self, item: item)
    (tmp_path / "plate-a.json").write_bytes(payload)
    index = {"plate-a": reference_entry(request, payload)}
    document = RUN.references_document(index, _REVISION)
    assert document["step"] == "S26"
    loader = RUN.reference_loader(tmp_path, document)
    verified = SimpleNamespace(case=SimpleNamespace(case_id="plate-a"), request=request)
    loaded, result = loader(verified)
    assert loaded == request and result.request_id == "plate-a"
    with pytest.raises(ValueError, match="not this protocol's"):
        RUN.reference_loader(tmp_path, {**document, "material_hash": "0" * 64})
    tampered = replace(result, total_strain_energy=result.total_strain_energy + 1.0)
    (tmp_path / "plate-a.json").write_bytes(encode_result(tampered))
    with pytest.raises(ValueError, match="differs from its index"):
        loader(verified)


def _evaluation(**changes: Any) -> SimpleNamespace:
    metrics = SimpleNamespace(
        median_absolute_peak_error=0.01,
        p90_absolute_peak_error=0.02,
        median_signed_peak_error=-0.005,
        p_relative_l2=0.012,
        u_relative_l2=0.001,
        invalid_collocation_points=0,
    )
    values: dict[str, Any] = {
        "result": SimpleNamespace(metrics=metrics),
        "run_binding": SimpleNamespace(
            problem_spec_hash="f" * 64, implementation_revision=_REVISION
        ),
        "manifest": SimpleNamespace(
            run_hash="1" * 64,
            step=2400,
            manifest_hash="2" * 64,
            model_state_hash="3" * 64,
        ),
        "diagnostics": [SimpleNamespace(relative_energy_gap=g) for g in (3e-4, 1e-4)],
        "minimum_collocation_determinant": 0.98,
        "failed_guardrails": (),
        "development_opened": False,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_the_screen_record_applies_the_guardrails() -> None:
    protocol = RUN.build_protocol()
    references = {"cases": {}}
    record = RUN.screen_record(_evaluation(), protocol, references)
    assert record["passed"] and record["step"] == "S26"
    assert (
        record["references_sha256"]
        == __import__("hashlib").sha256(canonical_json_bytes(references)).hexdigest()
    )
    failing = _evaluation(failed_guardrails=("u_relative_l2",))
    assert not RUN.screen_record(failing, protocol, references)["passed"]
    empty = RUN.screen_record(_evaluation(diagnostics=[]), protocol, references)
    assert empty["median_relative_energy_gap"] is None


def test_evaluation_requires_references_for_every_screen_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = RUN.build_protocol()
    monkeypatch.setattr(
        MaterialViability, "accepted_protocol", lambda self, **_: protocol
    )
    (tmp_path / "index.json").write_text(json.dumps({"cases": {"x": {}}}))
    with pytest.raises(ValueError, match="do not cover"):
        RUN.evaluate(
            s10_run_root=tmp_path, checkpoint_path=tmp_path, references=tmp_path
        )
    cases = {case_id: {} for case_id in protocol["viability"]["references"]["cases"]}
    (tmp_path / "index.json").write_text(json.dumps({"cases": cases}))
    committed = tmp_path / "repo" / RUN.references_path
    committed.parent.mkdir(parents=True)
    committed.write_text(json.dumps({"cases": {}}))
    monkeypatch.setattr(material_run, "REPOSITORY", tmp_path / "repo")
    with pytest.raises(ValueError, match="committed index"):
        RUN.evaluate(
            s10_run_root=tmp_path, checkpoint_path=tmp_path, references=tmp_path
        )


def test_the_protocol_command_prints_the_rebuild(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert s26_hgo.main(["protocol"]) == 0
    assert json.loads(capsys.readouterr().out) == RUN.build_protocol()
